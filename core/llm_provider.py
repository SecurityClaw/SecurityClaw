"""core/llm_provider.py — Ollama LLM provider.

Supports:
  - Ollama (via HTTP REST)

Architecture:
  - Chat / generation  → ollama_model   (e.g. qwen2.5:7b)
  - Embeddings         → ollama_embed_model (e.g. nomic-embed-text:latest) via a dedicated
                         Ollama request that does NOT share state with the
                         chat model, avoiding concurrency 400 errors.
"""
from __future__ import annotations

import logging
import os
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
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": max_tokens or self.max_tokens,  # Context window matches max_tokens from config
            },
        }
        if format:
            payload["format"] = format
        try:
            resp = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
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
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": max_tokens or self.max_tokens,
            },
        }
        full_response = ""
        try:
            logger.debug("stream_chat: Starting request to %s/api/chat", self.base_url)
            resp = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
            logger.debug("stream_chat: Response status %d, beginning iteration", resp.status_code)
            
            line_count = 0
            token_count = 0
            for line in resp.iter_lines():
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
                timeout=60,
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





# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_llm_provider(provider: Optional[str] = None) -> BaseLLMProvider:
    """
    Build the Ollama LLM provider from config (or explicit override).
    """
    cfg = Config()
    provider or cfg.get("llm", "provider", default="ollama")
    return OllamaProvider()
