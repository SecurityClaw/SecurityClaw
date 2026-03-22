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
