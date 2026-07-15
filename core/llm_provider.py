"""LLM providers for local, API-based, and authenticated CLI runtimes.

Supports Ollama, OpenAI-compatible APIs, Anthropic, Codex CLI, and Claude CLI.

Architecture:
  - Chat / generation  → ollama_model   (e.g. qwen2.5:7b)
  - Embeddings         → ollama_embed_model (e.g. nomic-embed-text:latest) via a dedicated
                         Ollama request that does NOT share state with the
                         chat model, avoiding concurrency 400 errors.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.config import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """Minimal interface every LLM backend must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a list of chat messages and return the assistant reply."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for the given text."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""

    def complete(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper: single user message."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def stream_chat(self, messages: list[dict], *, temperature=None, max_tokens=None, token_callback=None) -> str:
        """Compatibility stream for providers that only expose complete responses."""
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if token_callback and response:
            token_callback(response)
        return response


# ──────────────────────────────────────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """
    Calls Ollama's REST API:
      POST /api/chat   → chat completions
      POST /api/embed  → embeddings
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        import requests as _req  # local import to allow mocking

        self._requests = _req
        cfg = Config()
        self.base_url = (
            base_url or cfg.get("llm", "ollama_base_url", default="http://localhost:11434")
        ).rstrip("/")
        self.model = model or cfg.get("llm", "ollama_model", default="llama3")
        # Dedicated embedding model — separate from chat model to avoid
        # concurrency conflicts when both are called close together.
        self.embed_model = cfg.get("llm", "ollama_embed_model", default=self.model)
        self.temperature = cfg.get("llm", "temperature", default=0.2)
        self.max_tokens = cfg.get("llm", "max_tokens", default=16384)
        self.context_window = cfg.get("llm", "context_window", default=16384)
        self.think = cfg.get("llm", "think", default=False)
        # Generation on local hardware can legitimately take several minutes.
        # Keep the connection timeout short while allowing a configurable read
        # timeout for the model response.
        self.request_timeout_seconds = cfg.get(
            "llm", "request_timeout_seconds", default=600
        )
        self.embedding_timeout_seconds = cfg.get(
            "llm", "embedding_timeout_seconds", default=120
        )
        self._embedding_dim: Optional[int] = None  # Cache embedding dimension

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": self.context_window,
            },
        }
        if format:
            payload["format"] = format
        try:
            resp = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(10, self.request_timeout_seconds),
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as exc:
            logger.error("Ollama chat failed: %s", exc)
            raise

    def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        token_callback: Optional[callable] = None,
    ) -> str:
        """
        Stream chat response from Ollama, calling token_callback for each token.
        
        Args:
            messages: Chat messages
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            token_callback: Callable(token: str) invoked for each token streamed
            
        Returns:
            Full assembled response string
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": self.context_window,
            },
        }
        full_response = ""
        try:
            logger.debug("stream_chat: Starting request to %s/api/chat", self.base_url)
            resp = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(10, self.request_timeout_seconds),
                stream=True,
            )
            resp.raise_for_status()
            logger.debug("stream_chat: Response status %d, beginning iteration", resp.status_code)
            
            line_count = 0
            token_count = 0
            # Ollama emits one NDJSON object per token. A one-byte read chunk
            # prevents requests from buffering several token events together.
            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                line_count += 1
                if line:
                    try:
                        import json as _json
                        chunk = _json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            token_count += 1
                            full_response += token
                            if token_callback:
                                token_callback(token)
                            if token_count % 10 == 0:  # Log every 10 tokens
                                logger.debug("stream_chat: Received %d tokens so far", token_count)
                    except _json.JSONDecodeError as je:
                        logger.debug("stream_chat: JSONDecodeError on line %d: %s (line=%s)", line_count, je, line[:100])
                        pass  # Skip malformed JSON lines
            
            logger.debug("stream_chat: Completed - %d lines, %d tokens", line_count, token_count)
            return full_response
        except Exception as exc:
            logger.error("Ollama stream_chat failed: %s", exc)
            raise

    def embed(self, text: str) -> list[float]:
        """Embed using the dedicated embed model (separate from the chat model)."""
        payload = {"model": self.embed_model, "input": text}
        try:
            resp = self._requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=(10, self.embedding_timeout_seconds),
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings") or data.get("embedding")
            if isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
        except Exception as exc:
            logger.error("Ollama embed failed (model=%s): %s", self.embed_model, exc)
            raise

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider.
        
        Caches the dimension after first detection to avoid repeated API calls.
        """
        if self._embedding_dim is None:
            try:
                test_embed = self.embed("test")
                self._embedding_dim = len(test_embed)
                logger.info("Detected embedding dimension: %d", self._embedding_dim)
            except Exception as exc:
                logger.error("Could not detect embedding dimension: %s", exc)
                # Fallback to a reasonable default
                self._embedding_dim = 768
                logger.info("Using fallback embedding dimension: %d", self._embedding_dim)
        return self._embedding_dim


class RemoteChatProvider(BaseLLMProvider):
    """Base for remote/CLI chat providers while retaining local RAG embeddings."""

    def __init__(self) -> None:
        self._embedding_provider = OllamaProvider()

    def embed(self, text: str) -> list[float]:
        return self._embedding_provider.embed(text)

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_provider.embedding_dimension


class OpenAICompatibleProvider(RemoteChatProvider):
    """OpenAI Chat Completions API, including compatible self-hosted gateways."""

    def __init__(self, *, base_url=None, api_key=None, model=None, openai_native=False) -> None:
        import requests as _req
        super().__init__()
        cfg = Config()
        self._requests = _req
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or cfg.get("llm", "openai_base_url", default="https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL") or cfg.get("llm", "openai_model", default="gpt-5")
        self.openai_native = openai_native
        self.temperature = cfg.get("llm", "temperature", default=0.2)
        self.max_tokens = cfg.get("llm", "max_tokens", default=16384)
        self.timeout = cfg.get("llm", "request_timeout_seconds", default=600)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "stream": stream}
        token_field = "max_completion_tokens" if self.openai_native else "max_tokens"
        payload[token_field] = max_tokens or self.max_tokens
        if not self.openai_native:
            payload["temperature"] = self.temperature if temperature is None else temperature
        return payload

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        response = self._requests.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=self._payload(messages, temperature, max_tokens), timeout=(10, self.timeout))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream_chat(self, messages, *, temperature=None, max_tokens=None, token_callback=None) -> str:
        response = self._requests.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=self._payload(messages, temperature, max_tokens, stream=True), timeout=(10, self.timeout), stream=True)
        response.raise_for_status()
        complete = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            token = json.loads(payload).get("choices", [{}])[0].get("delta", {}).get("content") or ""
            complete += token
            if token_callback and token:
                token_callback(token)
        return complete


class AnthropicProvider(RemoteChatProvider):
    """Anthropic Messages API provider."""

    def __init__(self) -> None:
        import requests as _req
        super().__init__()
        cfg = Config()
        self._requests = _req
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
        self.model = os.getenv("ANTHROPIC_MODEL") or cfg.get("llm", "anthropic_model", default="claude-sonnet-4-5")
        self.max_tokens = cfg.get("llm", "max_tokens", default=16384)
        self.timeout = cfg.get("llm", "request_timeout_seconds", default=600)

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        payload = self._payload(messages, temperature, max_tokens)
        response = self._requests.post(f"{self.base_url}/messages", headers=self._headers(), json=payload, timeout=(10, self.timeout))
        response.raise_for_status()
        return "".join(block.get("text", "") for block in response.json().get("content", []) if block.get("type") == "text")

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict[str, Any]:
        system = "\n\n".join(item["content"] for item in messages if item.get("role") == "system")
        conversation = [item for item in messages if item.get("role") in {"user", "assistant"}]
        payload = {"model": self.model, "messages": conversation, "max_tokens": max_tokens or self.max_tokens, "stream": stream}
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def stream_chat(self, messages, *, temperature=None, max_tokens=None, token_callback=None) -> str:
        response = self._requests.post(f"{self.base_url}/messages", headers=self._headers(), json=self._payload(messages, temperature, max_tokens, stream=True), timeout=(10, self.timeout), stream=True)
        response.raise_for_status()
        complete = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            token = event.get("delta", {}).get("text", "") if event.get("type") == "content_block_delta" else ""
            complete += token
            if token_callback and token:
                token_callback(token)
        return complete


class CLIProvider(RemoteChatProvider):
    """Authenticated Codex or Claude CLI used as a non-interactive chat backend."""

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command
        self.executable = os.getenv(f"{command.upper()}_CLI_PATH", command)
        self.timeout = Config().get("llm", "request_timeout_seconds", default=600)

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        prompt = "\n\n".join(f"{item.get('role', 'user').upper()}: {item.get('content', '')}" for item in messages)
        args = [self.executable, "exec", "--skip-git-repo-check", prompt] if self.command == "codex" else [self.executable, "--print", prompt]
        completed = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout, check=True)
        return completed.stdout.strip()





# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_llm_provider(provider: Optional[str] = None) -> BaseLLMProvider:
    """
    Build the Ollama LLM provider from config (or explicit override).
    """
    cfg = Config()
    selected = (provider or os.getenv("LLM_PROVIDER") or cfg.get("llm", "provider", default="ollama")).lower()
    if selected == "ollama":
        return OllamaProvider()
    if selected in {"openai", "chatgpt"}:
        return OpenAICompatibleProvider(openai_native=True)
    if selected == "openai_compatible":
        return OpenAICompatibleProvider()
    if selected in {"anthropic", "claude_api"}:
        return AnthropicProvider()
    if selected == "codex_cli":
        return CLIProvider("codex")
    if selected == "claude_cli":
        return CLIProvider("claude")
    raise ValueError(f"Unsupported LLM provider: {selected}")
