from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml
from click.testing import CliRunner

import main


def _write_example_config(root: Path) -> None:
    (root / "config.yaml.example").write_text(
        """
agent:
  name: SecurityClaw
  version: 1.0.0
  skills_dir: skills
  log_level: INFO
db:
  provider: opensearch
  host: localhost
  port: 9200
  logs_index: logstash-*
  anomaly_index: securityclaw-anomalies
  vector_index: securityclaw-vectors
  use_ssl: false
  verify_certs: false
llm:
  provider: ollama
  ollama_base_url: http://localhost:11434
  ollama_model: llama3
  ollama_embed_model: nomic-embed-text:latest
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _patch_interactive_answers(monkeypatch, prompt_answers: list[str], confirm_answers: list[bool]) -> None:
    prompt_iter = iter(prompt_answers)
    confirm_iter = iter(confirm_answers)

    monkeypatch.setattr(main.Prompt, "ask", lambda *args, **kwargs: next(prompt_iter))
    monkeypatch.setattr(main.Confirm, "ask", lambda *args, **kwargs: next(confirm_iter))


def test_onboard_cli_writes_config_and_credentials(monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    _write_example_config(app_root)

    monkeypatch.setattr(main, "__file__", str(app_root / "main.py"))
    monkeypatch.setattr(main, "_test_opensearch_connection", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "_test_ollama_connection", lambda *args, **kwargs: True)

    import core.skill_onboarding as skill_onboarding

    monkeypatch.setattr(skill_onboarding, "discover_skill_requirements", lambda: {})

    _patch_interactive_answers(
        monkeypatch,
        prompt_answers=[
            "elasticsearch",
            "db.internal.local",
            "9443",
            "analyst",
            "s3cret-pass",
            "net-logs-*",
            "soc-anomalies",
            "soc-vectors",
            "http://ollama.internal:11434",
            "llama3.2",
            "nomic-embed-text:latest",
        ],
        confirm_answers=[
            True,
            True,
            True,
            False,
        ],
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["onboard"])

    assert result.exit_code == 0, result.output

    config = yaml.safe_load((app_root / "config.yaml").read_text(encoding="utf-8"))
    env_text = (app_root / ".env").read_text(encoding="utf-8")

    assert config["db"]["provider"] == "elasticsearch"
    assert config["db"]["host"] == "db.internal.local"
    assert config["db"]["port"] == 9443
    assert config["db"]["logs_index"] == "net-logs-*"
    assert config["db"]["anomaly_index"] == "soc-anomalies"
    assert config["db"]["vector_index"] == "soc-vectors"
    assert config["db"]["use_ssl"] is True
    assert config["db"]["verify_certs"] is True
    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["ollama_base_url"] == "http://ollama.internal:11434"
    assert config["llm"]["ollama_model"] == "llama3.2"
    assert config["llm"]["ollama_embed_model"] == "nomic-embed-text:latest"
    assert "DB_USERNAME=analyst" in env_text
    assert "DB_PASSWORD=s3cret-pass" in env_text


def test_onboard_cli_prompts_for_skill_variables_when_requested(monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    _write_example_config(app_root)

    monkeypatch.setattr(main, "__file__", str(app_root / "main.py"))
    monkeypatch.setattr(main, "_test_opensearch_connection", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "_test_ollama_connection", lambda *args, **kwargs: True)

    import core.skill_onboarding as skill_onboarding

    fake_requirements = {
        "custom_skill": {
            "CUSTOM_API_KEY": {
                "description": "Custom API key",
                "env_key": "CUSTOM_API_KEY",
                "optional": False,
                "is_secret": True,
            }
        }
    }
    write_env_vars = MagicMock()

    monkeypatch.setattr(skill_onboarding, "discover_skill_requirements", lambda: fake_requirements)
    monkeypatch.setattr(skill_onboarding, "prompt_for_skill_variables", lambda requirements: {"CUSTOM_API_KEY": "abc123"})
    monkeypatch.setattr(skill_onboarding, "_write_env_vars", write_env_vars)

    _patch_interactive_answers(
        monkeypatch,
        prompt_answers=[
            "opensearch",
            "localhost",
            "9200",
            "securityclaw-logs",
            "securityclaw-anomalies",
            "securityclaw-vectors",
            "http://localhost:11434",
            "llama3",
            "nomic-embed-text:latest",
        ],
        confirm_answers=[
            False,
            False,
            False,
            True,
        ],
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["onboard"])

    assert result.exit_code == 0, result.output
    write_env_vars.assert_called_once_with({"CUSTOM_API_KEY": "abc123"})
