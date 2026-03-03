"""
core/config.py — Loads config.yaml and merges with env overrides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Singleton configuration loader."""

    _instance: "Config | None" = None
    _data: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        with open(_CONFIG_PATH) as f:
            self._data = yaml.safe_load(f)

        # Env overrides (credentials from .env)
        env_overrides: dict = {}
        if os.getenv("DB_USERNAME"):
            env_overrides.setdefault("db", {})["username"] = os.getenv("DB_USERNAME")
        if os.getenv("DB_PASSWORD"):
            env_overrides.setdefault("db", {})["password"] = os.getenv("DB_PASSWORD")
        if os.getenv("OPENAI_API_KEY"):
            env_overrides.setdefault("llm", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OLLAMA_BASE_URL"):
            env_overrides.setdefault("llm", {})["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL")

        self._data = _deep_merge(self._data, env_overrides)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Dot-path access: config.get('db', 'host')."""
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def section(self, key: str) -> dict:
        return self._data.get(key, {})

    @classmethod
    def reset(cls) -> None:
        """Force reload (useful in tests)."""
        cls._instance = None
