"""
tests/test_russia_traffic_e2e.py

End-to-end tests for "Any traffic from russia in the past 30 days."

Validates that:
  1. Grounding returns {} (no IPs in the question — no heuristics, pure IP extractor)
  2. route_question routes to opensearch_querier via LLM (no keyword/regex shortcuts)
  3. opensearch_querier builds a match_phrase/term query with "Russia" / "RU"
  4. Supervisor is satisfied immediately when records_count > 0
  5. Supervisor loop terminates after 1 step when Russia traffic records are found
  6. LLM-mocked orchestrate_with_supervisor resolves the full question end-to-end

All tests use the simulator (MockDBConnector) or mock LLMs. No live data.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ── Fixtures ──────────────────────────────────────────────────────────────────

RUSSIA_RECORD_1 = {
    "_id": "rA1BVpwBvRK9Zm0gxCc2",
    "@timestamp": "2026-01-15T08:14:22.100Z",
    "src_ip": "37.230.117.113",
    "dest_ip": "192.168.0.50",
    "dest_port": 443,
    "proto": "TCP",
    "geoip": {
        "ip": "37.230.117.113",
        "country_name": "Russia",
        "country_code2": "RU",
        "country_code3": "RUS",
        "city_name": "Moscow",
    },
    "alert": {
        "signature": "ET SCAN Potential SSH Scan",
        "category": "Attempted Information Leak",
        "severity": 2,
    },
}

RUSSIA_RECORD_2 = {
    "_id": "sB2CVpwBvRK9Zm0gwigc",
    "@timestamp": "2026-01-16T14:02:55.300Z",
    "src_ip": "92.63.103.84",
    "dest_ip": "192.168.0.50",
    "dest_port": 22,
    "proto": "TCP",
    "geoip": {
        "ip": "92.63.103.84",
        "country_name": "Russia",
        "country_code2": "RU",
    },
}

OPENSEARCH_QUERIER_RESULT = {
    "status": "ok",
    "results_count": 2,
    "results": [RUSSIA_RECORD_1, RUSSIA_RECORD_2],
    "countries": ["Russia"],
    "ports": [],
    "protocols": [],
    "time_range": "now-30d",
    "reasoning": "User wants traffic from Russia in the past 30 days",
}

FIELD_MAPPINGS = {
    "all_fields": [
        "src_ip", "dest_ip", "@timestamp",
        "geoip.country_name", "geoip.country_code2", "geoip.country_code3",
        "dest_port", "proto",
    ],
    "port_fields": ["dest_port", "src_port"],
    "ip_fields": ["src_ip", "dest_ip"],
    "source_ip_fields": ["src_ip"],
    "destination_ip_fields": ["dest_ip"],
    "country_fields": ["geoip.country_name", "geoip.country_code2"],
    "text_fields": [],
}


# ── Test 1: Grounding returns {} — no heuristics, pure IP extractor ────────────

class TestRussiaGrounding:
    """Grounding must ignore country names; it only extracts IPv4 addresses."""

    def test_grounding_returns_empty_for_russia_question_no_ip(self):
        """The Russia question has no IP → grounding returns empty (LLM owns routing)."""
        from core.chat_router.logic import _ground_supervisor_question_with_llm

        grounding = _ground_supervisor_question_with_llm(
            user_question="Any traffic from russia in the past 30 days",
            llm=MagicMock(),
            instruction="test",
        )

        assert grounding == {}, (
            f"Expected empty grounding for country-only question, got: {grounding!r}. "
            "All routing decisions must be LLM-owned — no heuristics."
        )

    def test_grounding_does_not_set_preferred_routing_for_russia(self):
        """Grounding must not inject preferred_routing_groups or similar hints."""
        from core.chat_router.logic import _ground_supervisor_question_with_llm

        grounding = _ground_supervisor_question_with_llm(
            user_question="Any traffic from russia in the past 30 days",
            llm=MagicMock(),
            instruction="test",
        )

        assert "preferred_routing_groups" not in grounding
        assert "countries" not in grounding
        assert "time_window" not in grounding


# ── Test 2: route_question routes LLM-driven, not keyword-matched ─────────────

class TestRussiaRouting:
    """route_question must route via LLM decision, not pattern matching."""

    def test_route_question_russia_goes_to_opensearch_querier_via_llm(self):
        from core.chat_router.logic import route_question

        class _LLM:
            def chat(self, messages: list[dict]):
                return json.dumps({
                    "reasoning": "User wants to find network traffic originating from Russia over the past 30 days. This is a log search query.",
                    "skills": ["fields_querier", "opensearch_querier"],
                    "parameters": {"question": "Any traffic from russia in the past 30 days"},
                })

        result = route_question(
            user_question="Any traffic from russia in the past 30 days",
            available_skills=[
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
                {"name": "geoip_lookup", "description": "GeoIP enrichment"},
                {"name": "threat_analyst", "description": "Reputation analysis"},
            ],
            llm=_LLM(),
            instruction="test",
            conversation_history=[],
        )

        assert "opensearch_querier" in result["skills"], (
            f"Expected opensearch_querier in skills, got: {result['skills']}"
        )
        assert "threat_analyst" not in result["skills"], (
            "threat_analyst must not be included for a plain country traffic question"
        )

    def test_route_question_russia_question_preserved_in_parameters(self):
        """The original user question must be preserved in parameters."""
        from core.chat_router.logic import route_question

        question = "Any traffic from russia in the past 30 days"

        class _LLM:
            def chat(self, messages: list[dict]):
                return json.dumps({
                    "reasoning": "Country traffic log search.",
                    "skills": ["fields_querier", "opensearch_querier"],
                    "parameters": {"question": question},
                })

        result = route_question(
            user_question=question,
            available_skills=[
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
            ],
            llm=_LLM(),
            instruction="test",
            conversation_history=[],
        )

        assert result["parameters"].get("question") == question


# ── Test 3: opensearch_querier builds correct query for Russia ─────────────────

class TestRussiaOpenSearchQuery:
    """opensearch_querier must build a query with Russia GeoIP country filter."""

    def test_build_opensearch_query_contains_russia_country_filter(self):
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-30d",
            size=200,
            src_ip_field="src_ip",
            ip_direction="source",
            countries=["Russia"],
        )

        query_str = json.dumps(query)
        assert "Russia" in query_str, "Country name 'Russia' must be in the query"
        # Must use geo field(s)
        assert "geoip.country_name" in query_str or "source.geo.country_name" in query_str

    def test_build_opensearch_query_russia_has_time_filter_30d(self):
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-30d",
            size=200,
            ip_direction="source",
            countries=["Russia"],
        )

        query_str = json.dumps(query)
        assert "now-30d" in query_str, "Time range now-30d must be in the query"
        assert "@timestamp" in query_str, "@timestamp range must be in the query"

    def test_build_opensearch_query_russia_uses_bool_should_for_country(self):
        """Country must appear in a bool/should clause for geo field matching."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-30d",
            size=200,
            ip_direction="source",
            countries=["Russia"],
        )

        must = query["query"]["bool"].get("must", [])
        country_clause = next(
            (
                c for c in must
                if isinstance(c, dict) and "bool" in c
                and "should" in c["bool"]
                and "Russia" in json.dumps(c)
            ),
            None,
        )

        assert country_clause is not None, (
            "Country matching must use bool/should with geo field match"
        )


# ── Test 4: opensearch_querier.run() resolves Russia country query ─────────────

class TestRussiaOpenSearchRun:
    """run() with a mock LLM plan that returns Russia country filter."""

    def test_run_with_llm_plan_returns_russia_records(self):
        from skills.opensearch_querier import logic

        mock_db = MagicMock()
        mock_db.search.return_value = [RUSSIA_RECORD_1, RUSSIA_RECORD_2]
        mock_db.aggregate.return_value = {}

        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": [],
            "countries": ["Russia"],
            "ip_direction": "source",
            "aggregation_type": "none",
            "time_range": "now-30d",
            "ports": [],
            "protocols": [],
        })

        result = logic.run({
            "db": mock_db,
            "llm": mock_llm,
            "config": {},
            "parameters": {"question": "Any traffic from russia in the past 30 days"},
            "previous_results": {
                "fields_querier": {
                    "status": "ok",
                    "field_mappings": {
                        "destination_ip_fields": ["dest_ip"],
                        "destination_port_fields": ["dest_port"],
                        "source_ip_fields": ["src_ip"],
                        "protocol_fields": ["proto"],
                    }
                }
            },
        })

        assert result["status"] == "ok"
        assert result["results_count"] == 2
        # Verify the query included Russia country filter
        search_call = mock_db.search.call_args
        assert search_call is not None
        query_body = json.dumps(search_call[0][1])
        assert "Russia" in query_body


# ── Test 5: Supervisor satisfaction fast-path for Russia records ───────────────

class TestRussiaSupervisorEvaluation:
    """Supervisor must be satisfied immediately when Russia traffic records are found."""

    def test_supervisor_satisfied_immediately_on_russia_traffic(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="Any traffic from russia in the past 30 days",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={"opensearch_querier": OPENSEARCH_QUERIER_RESULT},
            step=1,
            max_steps=4,
        )

        mock_llm.chat.assert_not_called()
        assert eval_result["satisfied"] is True, f"Should be satisfied, got: {eval_result}"
        assert eval_result["confidence"] >= 0.8

    def test_supervisor_not_satisfied_when_no_russia_records(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="Any traffic from russia in the past 30 days",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={
                "opensearch_querier": {
                    "status": "ok",
                    "results_count": 0,
                    "results": [],
                }
            },
            step=1,
            max_steps=4,
        )

        mock_llm.chat.assert_not_called()
        assert eval_result["satisfied"] is False
        assert "No matching log records" in eval_result["reasoning"]


# ── Test 6: Full supervisor loop terminates after 1 step with Russia data ──────

class TestRussiaSupervisorLoop:
    """Supervisor loop must stop immediately when Russia traffic is found."""

    def test_supervisor_stops_after_first_step_with_russia_data(self):
        from core.chat_router.logic import orchestrate_with_supervisor

        mock_llm = MagicMock()
        mock_runner = MagicMock()

        mock_llm.chat.return_value = json.dumps({
            "reasoning": "Search for traffic from Russia in the past 30 days.",
            "skills": ["opensearch_querier"],
            "parameters": {"question": "Any traffic from russia in the past 30 days"},
        })

        mock_runner.dispatch.return_value = OPENSEARCH_QUERIER_RESULT

        available_skills = [
            {"name": "opensearch_querier", "description": "Search raw logs by query"},
            {"name": "threat_analyst", "description": "Check IP reputation"},
        ]

        steps_executed = []

        def callback(event, data, step, max_steps):
            steps_executed.append((event, step))

        orchestration = orchestrate_with_supervisor(
            user_question="Any traffic from russia in the past 30 days",
            available_skills=available_skills,
            runner=mock_runner,
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            step_callback=callback,
        )

        trace = orchestration["trace"]

        assert len(trace) == 1, (
            f"Supervisor should stop after 1 step when Russia traffic is found, "
            f"got {len(trace)} steps: {trace}"
        )
        assert trace[0]["evaluation"]["satisfied"] is True
        assert trace[0]["evaluation"]["confidence"] >= 0.8

        deciding_steps = [s for e, s in steps_executed if e == "deciding"]
        assert deciding_steps == [1], (
            f"Only step 1 should fire a 'deciding' event, got: {deciding_steps}"
        )

    def test_supervisor_russia_question_correct_skill_dispatched(self):
        """The runner must be called with opensearch_querier for Russia traffic."""
        from core.chat_router.logic import orchestrate_with_supervisor

        mock_llm = MagicMock()
        mock_runner = MagicMock()

        mock_llm.chat.return_value = json.dumps({
            "reasoning": "Country traffic search.",
            "skills": ["opensearch_querier"],
            "parameters": {"question": "Any traffic from russia in the past 30 days"},
        })

        mock_runner.dispatch.return_value = OPENSEARCH_QUERIER_RESULT

        orchestrate_with_supervisor(
            user_question="Any traffic from russia in the past 30 days",
            available_skills=[
                {"name": "opensearch_querier", "description": "Search raw logs by query"},
            ],
            runner=mock_runner,
            llm=mock_llm,
            instruction="You are a SOC analyst.",
        )

        dispatched_skill = mock_runner.dispatch.call_args[0][0]
        assert dispatched_skill == "opensearch_querier", (
            f"Expected opensearch_querier to be dispatched, got: {dispatched_skill!r}"
        )
