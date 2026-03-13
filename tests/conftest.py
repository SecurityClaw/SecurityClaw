"""
tests/conftest.py — Shared pytest fixtures.
"""
from __future__ import annotations

import pytest

from tests.mock_llm import MockLLMProvider
from tests.mock_opensearch import MockDBConnector
from tests.data_generator import (
    generate_anomaly_findings,
    generate_baseline_chunks,
    generate_normal_logs,
    generate_port_scan,
    generate_data_exfiltration,
    generate_lateral_movement,
    deterministic_embed,
)


@pytest.fixture
def mock_db() -> MockDBConnector:
    return MockDBConnector()


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider(dims=64)


@pytest.fixture
def seeded_db(mock_db) -> MockDBConnector:
    """DB populated with normal logs, anomalies, and baseline RAG chunks."""
    # Normal logs
    logs = generate_normal_logs(n=200)
    mock_db.seed_documents("securityclaw-logs", logs)

    # Anomalous logs
    mock_db.seed_documents("securityclaw-logs", generate_port_scan())
    mock_db.seed_documents("securityclaw-logs", generate_data_exfiltration())
    mock_db.seed_documents("securityclaw-logs", generate_lateral_movement())

    # AD findings
    findings = generate_anomaly_findings(
        detector_id="default-detector",
        n_normal=10, n_high=3, n_critical=2,
    )
    mock_db.seed_anomaly_findings(findings)

    # RAG baseline chunks (pre-embedded with deterministic embedder)
    chunks = generate_baseline_chunks()
    for chunk in chunks:
        chunk["embedding"] = deterministic_embed(chunk["text"], dims=64)
    mock_db.seed_documents("securityclaw-vectors", chunks)

    return mock_db


@pytest.fixture
def runner_context(seeded_db, mock_llm, tmp_path):
    """Full context dict as the Runner would provide to each skill."""
    from core.config import Config
    from core.memory import CheckpointBackedMemory

    memory = CheckpointBackedMemory(path=tmp_path / "runtime_memory.db")
    return {
        "db": seeded_db,
        "llm": mock_llm,
        "memory": memory,
        "config": Config(),
        "skills": {},
    }


@pytest.fixture
def tmp_memory(tmp_path):
    from core.memory import StateBackedMemory

    return StateBackedMemory()
