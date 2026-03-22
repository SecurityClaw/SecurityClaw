"""
tests/test_iran_traffic_e2e.py

End-to-end tests for Iran traffic query:
  - opensearch_querier uses match_phrase/term for country names (not fragile keyword search)
  - format_response correctly renders opensearch_querier results
  - supervisor evaluation marks satisfied immediately when records_count > 0
  - supervisor loop stops after 1 step (not 4) when data is found
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────
IRAN_RECORD_1 = {
    "_id": "sA6FVpwBvMK9Zm0gxCc1",
    "@timestamp": "2026-02-13T10:22:10.224Z",
    "src_ip": "62.60.131.168",
    "dest_ip": "192.168.0.16",
    "dest_port": 1194,
    "proto": "TCP",
    "geoip": {
        "ip": "62.60.131.168",
        "country_name": "Iran",
        "country_code2": "IR",
        "country_code3": "IRN",
        "city_name": "Tehran",
    },
    "alert": {
        "signature": "ET DROP Spamhaus DROP Listed Traffic Inbound group 8",
        "category": "Misc Attack",
        "severity": 2,
    },
}

IRAN_RECORD_2 = {
    "_id": "vA6GVpwBvMK9Zm0gwigb",
    "@timestamp": "2026-02-13T10:23:13.495Z",
    "src_ip": "62.60.131.168",
    "dest_ip": "192.168.0.16",
    "dest_port": 1194,
    "proto": "TCP",
    "geoip": {
        "ip": "62.60.131.168",
        "country_name": "Iran",
        "country_code2": "IR",
    },
}

OPENSEARCH_QUERIER_RESULT = {
    "status": "ok",
    "results_count": 2,
    "results": [IRAN_RECORD_1, IRAN_RECORD_2],
    "countries": ["Iran"],
    "ports": [],
    "protocols": [],
    "time_range": "now-3M",
    "reasoning": "User wants traffic from Iran in past 3 months",
}


# ── Test 1: opensearch_querier builds GeoIP country filter query ──────────────
class TestOpenSearchQuerierCountryQuery:
    """opensearch_querier must include GeoIP country filters when countries are specified."""

    def test_build_opensearch_query_includes_iran_country_filter(self):
        """Country name 'Iran' must appear in the query against geo fields."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-3M",
            size=200,
            src_ip_field="src_ip",
            ip_direction="source",
            countries=["Iran"],
        )

        query_str = json.dumps(query)
        assert "Iran" in query_str, "Country name 'Iran' must be in query"
        # Must use geo field(s) for country lookup
        assert "geoip.country_name" in query_str or "source.geo.country_name" in query_str

    def test_build_opensearch_query_includes_time_filter(self):
        """Query must always include a @timestamp range clause."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-3M",
            size=200,
            ip_direction="source",
            countries=["Iran"],
        )

        query_str = json.dumps(query)
        assert "now-3M" in query_str, "Time range now-3M must appear in query"
        assert "@timestamp" in query_str, "@timestamp range must be in query"


class TestOpenSearchQuerierIpFieldSelection:
    """IP direction and country filter must be applied correctly."""

    def test_source_direction_builds_src_ip_clauses(self):
        """ip_direction=source should include only src_ip field in IP clauses."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=["1.1.1.1"],
            dest_ip_field="dest_ip",
            time_range="now-90d",
            size=200,
            src_ip_field="src_ip",
            ip_direction="source",
        )

        body = json.dumps(query)
        assert "src_ip" in body, "src_ip field must appear for source direction"
        must = query["query"]["bool"]["must"]
        ip_clause = next((c for c in must if "bool" in c and "should" in c["bool"]), None)
        if ip_clause:
            fields_used = set()
            for s in ip_clause["bool"]["should"]:
                if "term" in s:
                    fields_used.update(s["term"].keys())
            assert "src_ip" in fields_used, "src_ip must be in IP term fields"
            assert "dest_ip" not in fields_used, "dest_ip must NOT be in IP terms for source direction"

    def test_any_direction_uses_both_ip_fields(self):
        """ip_direction=any should include both src_ip and dest_ip in IP clauses."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=["1.1.1.1"],
            dest_ip_field="dest_ip",
            time_range="now-90d",
            size=200,
            src_ip_field="src_ip",
            ip_direction="any",
        )

        body = json.dumps(query)
        assert "dest_ip" in body, "dest_ip must appear for any direction"
        assert "src_ip" in body, "src_ip must appear for any direction"

    def test_country_filter_uses_geo_fields(self):
        """Country filter must use standardised GeoIP fields."""
        from skills.opensearch_querier.logic import _build_opensearch_query

        query = _build_opensearch_query(
            search_terms=[],
            dest_ip_field="dest_ip",
            time_range="now-90d",
            size=200,
            ip_direction="source",
            countries=["Iran"],
        )

        body = json.dumps(query)
        geo_fields_used = [f for f in ["geoip.country_name", "source.geo.country_name", "destination.geo.country_name"] if f in body]
        assert len(geo_fields_used) > 0, f"No geo fields found in query. Query: {body}"


class TestQueryBuilderFieldDiscovery:
    """Field discovery should not classify geo_point containers as IP fields."""

    def test_discover_field_mappings_excludes_geo_point_from_ip_fields(self):
        from core.query_builder import discover_field_mappings

        db = MagicMock()
        db._client.indices.get_mapping.return_value = {
            "logstash-2026.03.10": {
                "mappings": {
                    "properties": {
                        "geoip": {
                            "properties": {
                                "country_name": {"type": "keyword"},
                                "country_code2": {"type": "keyword"},
                                "geohash": {"type": "geo_point"},
                            }
                        },
                        "source": {
                            "properties": {
                                "ip": {"type": "ip"},
                            }
                        },
                        "destination": {
                            "properties": {
                                "ip": {"type": "ip"},
                            }
                        },
                    }
                }
            }
        }

        mappings = discover_field_mappings(db, llm=None)

        assert "geoip" not in mappings["ip_fields"]
        assert "source.ip" in mappings["ip_fields"]
        assert "destination.ip" in mappings["ip_fields"]
        assert "source.ip" in mappings["source_ip_fields"]
        assert "destination.ip" in mappings["destination_ip_fields"]
        assert "geoip.country_name" in mappings["country_fields"]
        assert "geoip.country_code2" in mappings["country_fields"]


class TestValidationSampleExtraction:
    """GeoIP country data in records should be accessible for downstream use."""

    def test_iran_record_has_nested_country_field(self):
        """Test fixture: geoip.country_name must be Iran in IRAN_RECORD_1."""
        assert IRAN_RECORD_1["geoip"]["country_name"] == "Iran"
        assert IRAN_RECORD_1["src_ip"] == "62.60.131.168"



# ── Test 2: _format_opensearch_response correctly summarises Iran traffic ──────
class TestFormatOpenSearchResponse:
    """format_response must render opensearch_querier results — not say 'no traffic'."""

    def test_formats_iran_traffic_correctly(self):
        from core.chat_router.logic import _format_opensearch_response

        result = _format_opensearch_response(
            "any traffic from iran in the past 3 months",
            OPENSEARCH_QUERIER_RESULT,
        )

        # Must mention records found
        assert "2" in result, f"Should mention 2 records, got: {result}"
        # Must mention Iran in output
        assert "Iran" in result, f"Should mention Iran, got: {result}"
        # Must NOT say no traffic
        assert "no traffic" not in result.lower(), f"Must not say 'no traffic': {result}"
        assert "not contain" not in result.lower(), f"Must not deny traffic: {result}"

    def test_formats_correct_timestamps(self):
        from core.chat_router.logic import _format_opensearch_response

        result = _format_opensearch_response(
            "any traffic from iran",
            OPENSEARCH_QUERIER_RESULT,
        )
        # Should mention Feb 2026 dates
        assert "2026-02-13" in result, f"Should include timestamp, got: {result}"

    def test_formats_correct_ips(self):
        from core.chat_router.logic import _format_opensearch_response

        result = _format_opensearch_response(
            "any traffic from iran",
            OPENSEARCH_QUERIER_RESULT,
        )
        # Should mention the Iranian IP
        assert "62.60.131.168" in result, f"Should include source IP, got: {result}"

    def test_formats_flat_geoip_country_fields(self):
        from core.chat_router.logic import _format_opensearch_response

        flat_result = {
            "status": "ok",
            "results_count": 1,
            "results": [
                {
                    "@timestamp": "2026-02-13T10:22:10.224Z",
                    "source.ip": "62.60.131.168",
                    "destination.ip": "192.168.0.16",
                    "geoip.country_name": "Iran",
                }
            ],
            "countries": [],
            "ports": [],
            "protocols": [],
            "time_range": "now-30d",
            "search_terms": ["62.60.131.168"],
        }

        result = _format_opensearch_response(
            "What countries are these IPs from?",
            flat_result,
        )

        assert "Iran" in result, f"Should include flat geoip country fields, got: {result}"
        assert "62.60.131.168" in result, f"Should include flat source IP, got: {result}"

    def test_formats_total_hits_with_edge_window_sampling_note(self):
        from core.chat_router.logic import _format_opensearch_response

        sampled_result = {
            "status": "ok",
            "results_count": 842,
            "sampled_results_count": 400,
            "sample_strategy": "edge_windows",
            "oldest_sample_count": 200,
            "newest_sample_count": 200,
            "results": [
                {
                    "@timestamp": "2025-12-30T22:05:25.936Z",
                    "src_ip": "104.156.155.5",
                    "dest_ip": "192.168.0.85",
                    "destination.port": 1194,
                    "geoip.country_name": "China",
                }
            ],
            "summary_results": [
                {
                    "@timestamp": "2025-09-30T00:00:00.000Z",
                    "src_ip": "18.218.174.114",
                    "dest_ip": "192.168.0.85",
                    "destination.port": 1194,
                    "geoip.country_name": "Singapore",
                },
                {
                    "@timestamp": "2025-12-30T22:05:25.936Z",
                    "src_ip": "104.156.155.5",
                    "dest_ip": "192.168.0.85",
                    "destination.port": 1194,
                    "geoip.country_name": "China",
                },
            ],
            "countries": [],
            "ports": [1194],
            "protocols": [],
            "time_range_label": "now-90d",
            "search_terms": [],
        }

        result = _format_opensearch_response(
            "What countries aside from the usa have hit the network at 1194 port",
            sampled_result,
        )

        assert "Found 842 total record(s) matching port 1194 in the now-90d window." in result
        assert "sampled from up to 200 earliest and 200 latest matching records" in result
        assert "Earliest: 2025-09-30T00:00:00.000Z. Latest: 2025-12-30T22:05:25.936Z." in result
        assert "Countries seen: China, Singapore." in result

    def test_empty_results_says_no_records(self):
        from core.chat_router.logic import _format_opensearch_response

        empty_result = {
            "status": "ok",
            "results_count": 0,
            "results": [],
            "countries": ["Iran"],
            "ports": [],
            "protocols": [],
            "time_range": "now-3M",
        }
        result = _format_opensearch_response("traffic from iran", empty_result)
        assert "no matching" in result.lower() or "0" in result, f"Should say no records: {result}"

    def test_empty_results_with_directional_alternative_explains_opposite_hits(self):
        from core.chat_router.logic import _format_opensearch_response

        result = _format_opensearch_response(
            "any traffic from 1.1.1.1 today?",
            {
                "status": "ok",
                "results_count": 0,
                "results": [],
                "search_terms": ["1.1.1.1"],
                "time_range_label": "today",
                "ip_direction": "source",
                "directional_alternative": {
                    "direction": "destination",
                    "results_count": 2,
                    "time_range_label": "today",
                    "sample_peers": ["192.168.0.130", "192.168.0.85"],
                    "earliest": "2026-03-10T00:53:39.546Z",
                    "latest": "2026-03-10T00:53:49.424Z",
                },
            },
        )

        assert "No traffic source 1.1.1.1" in result
        assert "2 record(s) were found in the destination direction" in result
        assert "192.168.0.130" in result


# ── Test 3: format_response dispatches to _format_opensearch_response ─────────
class TestFormatResponseDispatching:
    """format_response must route opensearch_querier results through _format_opensearch_response."""

    def test_format_response_uses_opensearch_result(self):
        from core.chat_router.logic import format_response

        mock_llm = MagicMock()
        routing = {"skills": ["opensearch_querier"], "parameters": {}}

        result = format_response(
            "any traffic from iran in the past 3 months",
            routing,
            {"opensearch_querier": OPENSEARCH_QUERIER_RESULT},
            mock_llm,
        )

        # LLM should NOT have been called (deterministic renderer took over)
        mock_llm.chat.assert_not_called()
        # Result must mention records found / Iran
        assert "Iran" in result or "2" in result, f"Should mention Iran traffic, got: {result}"
        assert "no traffic" not in result.lower(), f"Must not deny traffic: {result}"

    def test_format_response_skips_validation_failed_opensearch_result(self):
        from core.chat_router.logic import format_response

        mock_llm = MagicMock()

        result = format_response(
            "Aside from the private IPs, what is the reputation of the others?",
            {"skills": ["opensearch_querier", "threat_analyst"], "parameters": {}},
            {
                "opensearch_querier": {
                    "status": "ok",
                    "results_count": 5,
                    "validation_failed": True,
                    "results": [{"src_ip": "75.75.75.75"}],
                },
                "threat_analyst": {
                    "status": "ok",
                    "verdicts": [{"verdict": "FALSE_POSITIVE", "confidence": 85}],
                },
            },
            mock_llm,
        )

        mock_llm.chat.assert_not_called()
        assert "75.75.75.75" not in result
        assert "FALSE_POSITIVE" in result


# ── Test 4: supervisor evaluation satisfies immediately when records found ─────
class TestSupervisorEvaluationFastPath:
    """Supervisor must mark satisfied after first skill run when records_count > 0."""

    def test_satisfied_immediately_when_records_found(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="any traffic from iran in the past 3 months",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={"opensearch_querier": OPENSEARCH_QUERIER_RESULT},
            step=1,
            max_steps=4,
        )

        # LLM should NOT be called — fast path triggers
        mock_llm.chat.assert_not_called()
        assert eval_result["satisfied"] is True, f"Should be satisfied, got: {eval_result}"
        assert eval_result["confidence"] >= 0.8, f"Confidence should be high, got: {eval_result}"
        assert "2" in eval_result["reasoning"], f"Reasoning should mention record count: {eval_result}"

    def test_not_satisfied_when_no_records(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="any traffic from iran",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={"opensearch_querier": {"status": "ok", "results_count": 0, "results": []}},
            step=1,
            max_steps=4,
        )

        mock_llm.chat.assert_not_called()
        assert eval_result["satisfied"] is False
        assert "No matching log records" in eval_result["reasoning"]

    def test_invalid_nonzero_results_do_not_trigger_fast_path(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({
            "satisfied": False,
            "confidence": 0.2,
            "reasoning": "Results were invalid and should not satisfy the question.",
            "missing": ["valid matching records"],
        })

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="is there any traffic from 1.1.1.1?",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={
                "opensearch_querier": {
                    "status": "ok",
                    "results_count": 200,
                    "results": [{"src_ip": "192.168.0.85"}],
                    "validation_failed": True,
                }
            },
            step=1,
            max_steps=4,
        )

        assert eval_result["satisfied"] is False
        mock_llm.chat.assert_called_once()

    def test_directional_alternative_satisfies_supervisor(self):
        from core.chat_router.logic import _supervisor_evaluate_satisfaction

        mock_llm = MagicMock()

        eval_result = _supervisor_evaluate_satisfaction(
            user_question="any traffic from 1.1.1.1 today?",
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            conversation_history=[],
            skill_results={
                "opensearch_querier": {
                    "status": "ok",
                    "results_count": 0,
                    "results": [],
                    "ip_direction": "source",
                    "directional_alternative": {
                        "direction": "destination",
                        "results_count": 25,
                    },
                }
            },
            step=1,
            max_steps=4,
        )

        mock_llm.chat.assert_not_called()
        assert eval_result["satisfied"] is True
        assert "destination-direction records" in eval_result["reasoning"]


# ── Test 5: Supervisor loop stops after 1 step when opensearch_querier finds data
class TestSupervisorLoopTermination:
    """Supervisor must stop looping after the first step when data is already found."""

    def test_supervisor_stops_after_first_step_with_data(self):
        from core.chat_router.logic import orchestrate_with_supervisor

        mock_llm = MagicMock()
        mock_runner = MagicMock()

        # Supervisor decides to run opensearch_querier
        mock_llm.chat.return_value = json.dumps({
            "reasoning": "Search for Iran traffic",
            "skills": ["opensearch_querier"],
            "parameters": {"question": "any traffic from iran in the past 3 months"},
        })

        # Runner returns Iran traffic records
        mock_runner.dispatch.return_value = OPENSEARCH_QUERIER_RESULT

        available_skills = [
            {"name": "opensearch_querier", "description": "Search raw logs by query"},
            {"name": "threat_analyst", "description": "Check IP reputation"},
        ]

        steps_executed = []

        def callback(event, data, step, max_steps):
            steps_executed.append((event, step))

        orchestration = orchestrate_with_supervisor(
            user_question="any traffic from iran in the past 3 months",
            available_skills=available_skills,
            runner=mock_runner,
            llm=mock_llm,
            instruction="You are a SOC analyst.",
            step_callback=callback,
        )

        trace = orchestration["trace"]

        # Should stop after just 1 step (fast path satisfaction)
        assert len(trace) == 1, f"Should stop after 1 step, got {len(trace)} steps: {trace}"
        assert trace[0]["evaluation"]["satisfied"] is True
        assert trace[0]["evaluation"]["confidence"] >= 0.8

        # Callback should have fired for deciding + evaluated (not multiple steps)
        deciding_steps = [s for e, s in steps_executed if e == "deciding"]
        assert deciding_steps == [1], f"Only step 1 should fire, got: {deciding_steps}"

    def test_supervisor_does_not_repeat_same_skill_when_already_satisfied(self):
        """Anti-repeat: same skill list chosen twice → forces finalization."""
        from core.chat_router.logic import orchestrate_with_supervisor

        mock_llm = MagicMock()
        mock_runner = MagicMock()

        call_count = 0

        def llm_chat_side_effect(messages):
            nonlocal call_count
            call_count += 1
            # Always choose opensearch_querier (would normally loop)
            return json.dumps({
                "reasoning": "Search again",
                "skills": ["opensearch_querier"],
                "parameters": {"question": "any traffic from iran"},
            })

        mock_llm.chat.side_effect = llm_chat_side_effect

        # Runner returns 0 results so fast path doesn't trigger
        mock_runner.dispatch.return_value = {
            "status": "ok",
            "results_count": 0,
            "results": [],
            "countries": ["Iran"],
            "time_range": "now-3M",
        }

        available_skills = [{"name": "opensearch_querier", "description": "Search logs"}]

        orchestration = orchestrate_with_supervisor(
            user_question="any traffic from iran",
            available_skills=available_skills,
            runner=mock_runner,
            llm=mock_llm,
            instruction="You are a SOC analyst.",
        )

        trace = orchestration["trace"]
        # Anti-repeat should kick in on step 2 (same skill selected twice)
        assert len(trace) <= 2, f"Anti-repeat should stop by step 2, got {len(trace)} steps"
