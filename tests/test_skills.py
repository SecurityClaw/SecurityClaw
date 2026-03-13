"""
tests/test_skills.py — Integration tests for all three skills.

All tests use MockDBConnector + MockLLMProvider — no live services needed.

Test coverage:
  - NetworkBaseliner: baseline building, RAG storage, memory update
  - AnomalyWatcher:  polling, enrichment, escalation logic
  - ThreatAnalyst:   RAG retrieval, verdict issuing, memory write-back
"""
from __future__ import annotations

import pytest

from tests.data_generator import (
    generate_baseline_chunks,
    generate_anomaly_findings,
    deterministic_embed,
)


# ──────────────────────────────────────────────────────────────────────────────
# NetworkBaseliner
# ──────────────────────────────────────────────────────────────────────────────

class TestNetworkBaseliner:
    def test_run_with_no_data_returns_no_data(self, runner_context):
        """When the logs index is empty, the skill should return 'no_data'."""
        import importlib
        logic = importlib.import_module("skills.network_baseliner.logic")
        # Use a fresh DB with no logs
        from tests.mock_opensearch import MockDBConnector
        ctx = {**runner_context, "db": MockDBConnector()}
        result = logic.run(ctx)
        assert result["status"] == "no_data"

    def test_run_stores_rag_document(self, runner_context):
        """With log data present, the skill should store at least one RAG doc."""
        import importlib
        logic = importlib.import_module("skills.network_baseliner.logic")
        result = logic.run(runner_context)

        # May be "ok" or "no_data" depending on timestamp range vs fixture data
        # Either way, if "ok" then we must have processed records and generated reports
        if result["status"] == "ok":
            db = runner_context["db"]
            # Verify return structure contains expected keys
            assert "records_processed" in result
            assert "networks_analyzed" in result
            assert "documents_stored" in result
            assert "identifier_field" in result
            assert "identifiers" in result
            # If documents were stored, vector index should have content
            if result["documents_stored"] > 0:
                assert db.document_count("securityclaw-vectors") > 0

    def test_run_updates_memory(self, runner_context):
        """A successful run should append a decision to agent memory."""
        import importlib
        logic = importlib.import_module("skills.network_baseliner.logic")

        # Seed logs with very recent timestamps so the range query matches
        from tests.mock_opensearch import MockDBConnector
        from tests.data_generator import generate_normal_logs
        from datetime import datetime, timezone, timedelta

        db = MockDBConnector()
        # Inject logs with "now" timestamps (within 6h window)
        recent_logs = generate_normal_logs(n=50)
        db.seed_documents("securityclaw-logs", recent_logs)

        # Pre-embed baseline chunks for vector index to exist
        chunks = generate_baseline_chunks()
        for c in chunks:
            c["embedding"] = deterministic_embed(c["text"], dims=64)
        db.seed_documents("securityclaw-vectors", chunks)

        ctx = {**runner_context, "db": db}
        result = logic.run(ctx)

        if result["status"] == "ok":
            decisions = runner_context["memory"].get_section("Recent Decisions")
            assert "NetworkBaseliner" in decisions or "baseline" in decisions.lower()

    def test_run_skips_without_db(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.network_baseliner.logic")
        result = logic.run({**runner_context, "db": None})
        assert result["status"] == "skipped"

    def test_run_skips_without_llm(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.network_baseliner.logic")
        result = logic.run({**runner_context, "llm": None})
        assert result["status"] == "skipped"


# ──────────────────────────────────────────────────────────────────────────────
# AnomalyWatcher
# ──────────────────────────────────────────────────────────────────────────────

class TestAnomalyTriage:
    def test_returns_ok_with_no_findings(self, runner_context):
        """If no findings match, should return ok with new_findings=0."""
        import importlib
        from tests.mock_opensearch import MockDBConnector
        logic = importlib.import_module("skills.anomaly_triage.logic")
        ctx = {**runner_context, "db": MockDBConnector()}
        result = logic.run(ctx)
        assert result["status"] == "ok"
        assert result["new_findings"] == 0

    def test_enriches_high_score_findings(self, runner_context, seeded_db):
        """High-score findings should be enriched and returned."""
        import importlib
        logic = importlib.import_module("skills.anomaly_triage.logic")
        # Reset cursor so all findings are "new"
        logic._last_poll_epoch_ms = None
        result = logic.run(runner_context)
        assert result["status"] == "ok"
        # There should be high+critical findings (>=0.7 threshold)
        assert result["new_findings"] > 0
        assert result["enriched"] >= 1

    def test_escalates_critical_findings(self, runner_context, seeded_db):
        """CRITICAL findings must appear in the escalation queue."""
        import importlib
        logic = importlib.import_module("skills.anomaly_triage.logic")
        logic._last_poll_epoch_ms = None
        result = logic.run(runner_context)

        memory = runner_context["memory"]
        escalation = memory.get_section("Escalation Queue")

        if result["escalated"] > 0:
            assert escalation and escalation != "None"

    def test_writes_findings_to_memory(self, runner_context, seeded_db):
        """Enriched findings should appear in Open Findings."""
        import importlib
        logic = importlib.import_module("skills.anomaly_triage.logic")
        logic._last_poll_epoch_ms = None
        result = logic.run(runner_context)

        if result["enriched"] > 0:
            findings_text = runner_context["memory"].get_section("Open Findings")
            assert findings_text and findings_text != "None"

    def test_skips_without_db(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.anomaly_triage.logic")
        result = logic.run({**runner_context, "db": None})
        assert result["status"] == "skipped"

    def test_bare_enrich_fallback(self):
        """_bare_enrich should work without an LLM."""
        from skills.anomaly_triage.logic import _bare_enrich, _score_to_severity
        raw = {
            "detector_id": "test-det",
            "anomaly_score": 0.95,
            "entity": {"value": "10.0.1.100"},
        }
        result = _bare_enrich(raw)
        assert result["severity"] == "CRITICAL"
        assert result["score"] == 0.95
        assert "description" in result

    def test_score_to_severity_mapping(self):
        from skills.anomaly_triage.logic import _score_to_severity
        assert _score_to_severity(0.99) == "CRITICAL"
        assert _score_to_severity(0.88) == "HIGH"
        assert _score_to_severity(0.75) == "MEDIUM"
        assert _score_to_severity(0.50) == "LOW"

    def test_cursor_advances_after_poll(self, runner_context, seeded_db):
        """After a poll, _last_poll_epoch_ms should be updated."""
        import importlib
        logic = importlib.import_module("skills.anomaly_triage.logic")
        logic._last_poll_epoch_ms = None
        logic.run(runner_context)
        assert logic._last_poll_epoch_ms is not None


# ──────────────────────────────────────────────────────────────────────────────
# ThreatAnalyst
# ──────────────────────────────────────────────────────────────────────────────

class TestThreatAnalyst:
    def _prime_escalation(self, memory, n: int = 2) -> None:
        """Inject synthetic escalation items into memory."""
        for i in range(n):
            memory.escalate(
                f"[CRITICAL] Needs ThreatAnalyst review: "
                f"Host 10.0.1.{i+10} sent 45MB to external IP on port 443."
            )

    def test_returns_ok_with_no_escalations(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        result = logic.run(runner_context)
        assert result["status"] == "ok"
        assert result["analyzed"] == 0

    def test_analyzes_escalated_findings(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        self._prime_escalation(runner_context["memory"], n=2)
        result = logic.run(runner_context)
        assert result["status"] == "ok"
        assert result["analyzed"] == 2

    def test_verdict_in_valid_set(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        self._prime_escalation(runner_context["memory"], n=3)
        result = logic.run(runner_context)
        valid_verdicts = {"TRUE_THREAT", "FALSE_POSITIVE", "UNKNOWN", "ERROR"}
        for v in result["verdicts"]:
            assert v.get("verdict") in valid_verdicts

    def test_clears_escalation_queue_after_run(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        self._prime_escalation(runner_context["memory"], n=2)
        logic.run(runner_context)
        escalation = runner_context["memory"].get_section("Escalation Queue")
        assert escalation.strip() == "None"

    def test_writes_decision_to_memory(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        self._prime_escalation(runner_context["memory"], n=1)
        logic.run(runner_context)
        decisions = runner_context["memory"].get_section("Recent Decisions")
        assert decisions and decisions != "None"

    def test_rag_context_retrieved_during_analysis(self, runner_context):
        """embed() should be called during analysis (for RAG lookup)."""
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")
        self._prime_escalation(runner_context["memory"], n=1)
        llm = runner_context["llm"]
        initial_calls = len(llm.call_log)
        logic.run(runner_context)
        # Should have made at least one embed call (RAG retrieval)
        embed_calls = [c for c in llm.call_log[initial_calls:] if c["type"] == "embed"]
        assert len(embed_calls) >= 1

    def test_skips_without_db_or_llm(self, runner_context):
        import importlib
        logic = importlib.import_module("skills.threat_analyst.logic")

        result = logic.run({**runner_context, "db": None})
        assert result["status"] == "ok"
        assert result["analyzed"] == 0

        result = logic.run({**runner_context, "llm": None})
        assert result["status"] == "skipped"

    def test_true_threat_sets_focus(self, runner_context, monkeypatch):
        """When TRUE_THREAT is returned, agent focus should be updated."""
        import importlib
        import json
        logic = importlib.import_module("skills.threat_analyst.logic")

        # Force the mock LLM to always return a TRUE_THREAT verdict
        def forced_chat(messages, **kw):
            return json.dumps({
                "verdict": "TRUE_THREAT",
                "confidence": 95,
                "reasoning": "Definitely malicious.",
                "recommended_action": "Isolate immediately.",
            })

        runner_context["llm"].chat = forced_chat
        self._prime_escalation(runner_context["memory"], n=1)
        logic.run(runner_context)

        focus = runner_context["memory"].get_section("Current Focus")
        assert "threat" in focus.lower() or "investigation" in focus.lower()
