"""
core/config.py — Loads config.yaml and merges with env overrides.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
        # Try to load config.yaml; fall back to config.yaml.example if missing
        config_path = _CONFIG_PATH
        example_path = _ROOT / "config.yaml.example"
        
        if not config_path.exists() and example_path.exists():
            logger.warning(
                "config.yaml not found; using config.yaml.example as fallback. "
                "For production use, copy config.yaml.example to config.yaml and customize."
            )
            config_path = example_path
        
        with open(config_path) as f:
            self._data = yaml.safe_load(f) or {}

        # Env overrides (credentials from .env)
        env_overrides: dict = {}
        if os.getenv("DB_USERNAME"):
            env_overrides.setdefault("db", {})["username"] = os.getenv("DB_USERNAME")
        if os.getenv("DB_PASSWORD"):
            env_overrides.setdefault("db", {})["password"] = os.getenv("DB_PASSWORD")
        if os.getenv("OLLAMA_BASE_URL"):
            env_overrides.setdefault("llm", {})["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL")
        
        # External reputation intelligence API keys
        if os.getenv("ABUSEIPDB_API_KEY"):
            env_overrides.setdefault("apis", {})["abuseipdb_key"] = os.getenv("ABUSEIPDB_API_KEY")
        if os.getenv("ALIENVAULT_API_KEY"):
            env_overrides.setdefault("apis", {})["alienvault_key"] = os.getenv("ALIENVAULT_API_KEY")
        if os.getenv("VIRUSTOTAL_API_KEY"):
            env_overrides.setdefault("apis", {})["virustotal_key"] = os.getenv("VIRUSTOTAL_API_KEY")
        if os.getenv("TALOS_CLIENT_ID"):
            env_overrides.setdefault("apis", {})["talos_client_id"] = os.getenv("TALOS_CLIENT_ID")
        if os.getenv("TALOS_CLIENT_SECRET"):
            env_overrides.setdefault("apis", {})["talos_client_secret"] = os.getenv("TALOS_CLIENT_SECRET")
        if os.getenv("MAXMIND_LICENSE_KEY"):
            env_overrides.setdefault("geoip", {})["license_key"] = os.getenv("MAXMIND_LICENSE_KEY")
        if os.getenv("MAXMIND_EDITION_ID"):
            env_overrides.setdefault("geoip", {})["edition_id"] = os.getenv("MAXMIND_EDITION_ID")
        if os.getenv("MAXMIND_DB_PATH"):
            env_overrides.setdefault("geoip", {})["db_path"] = os.getenv("MAXMIND_DB_PATH")

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
