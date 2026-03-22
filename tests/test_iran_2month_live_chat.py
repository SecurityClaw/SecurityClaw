"""
tests/test_iran_2month_live_chat.py

Test for the exact live chat query: "any traffic from iran in the past 2 months"

This simulates what happens when a user types this query into the chat interface.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


IRAN_RECORD_FEB = {
    "_id": "iA1BVpwBvMK9Zm0g1",
    "@timestamp": "2026-02-10T08:14:22.100Z",
    "src_ip": "62.60.131.168",
    "dest_ip": "192.168.0.50",
    "dest_port": 443,
    "proto": "TCP",
    "geoip": {
        "ip": "62.60.131.168",
        "country_name": "Iran",
        "country_code2": "IR",
    },
}

IRAN_RECORD_JAN = {
    "_id": "iB2CVpwBvMK9Zm0g2",
    "@timestamp": "2026-01-25T14:02:55.300Z",
    "src_ip": "62.60.131.168",
    "dest_ip": "192.168.0.50",
    "dest_port": 22,
    "proto": "TCP",
    "geoip": {
        "ip": "62.60.131.168",
        "country_name": "Iran",
        "country_code2": "IR",
    },
}


def test_route_question_iran_past_2_months_via_llm():
    """Live chat query: 'any traffic from iran in the past 2 months'"""
    from core.chat_router.logic import route_question

    class _LLM:
        def chat(self, messages: list[dict]):
            return json.dumps({
                "reasoning": "User wants network traffic from Iran over the past 2 months. Query the logs.",
                "skills": ["fields_querier", "opensearch_querier"],
                "parameters": {"question": "any traffic from iran in the past 2 months"},
            })

    result = route_question(
        user_question="any traffic from iran in the past 2 months",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        llm=_LLM(),
        instruction="test",
        conversation_history=[],
    )

    assert "opensearch_querier" in result["skills"]
    assert result["parameters"]["question"] == "any traffic from iran in the past 2 months"


def test_opensearch_querier_iran_2months():
    """opensearch_querier should find Iran records when searching past 2 months."""
    from skills.opensearch_querier import logic
    from tests.mock_opensearch import MockDBConnector

    class _Cfg:
        def get(self, section: str, key: str, default=None):
            if (section, key) == ("db", "logs_index"):
                return "logstash*"
            return default

    db = MockDBConnector()
    llm = MagicMock()

    # Pre-seed Iran records so db.search() finds them
    db.seed_documents("logstash*", [IRAN_RECORD_FEB, IRAN_RECORD_JAN])

    field_mappings = {
        "all_fields": [
            "src_ip", "dest_ip", "dest_port", "@timestamp",
            "geoip.country_name", "geoip.country_code2",
        ],
        "ip_fields": ["src_ip", "dest_ip"],
        "source_ip_fields": ["src_ip"],
        "destination_ip_fields": ["dest_ip"],
        "destination_port_fields": ["dest_port"],
        "country_fields": ["geoip.country_name", "geoip.country_code2"],
        "port_fields": ["dest_port"],
        "text_fields": [],
    }

    with patch.object(
        logic,
        "_plan_with_llm",
        return_value={
            "search_terms": [],
            "countries": ["Iran"],
            "time_range": "now-2M",
            "aggregation_type": "none",
            "ip_direction": "any",
        },
    ):
        result = logic.run({
            "db": db,
            "llm": llm,
            "config": _Cfg(),
            "parameters": {"question": "any traffic from iran in the past 2 months"},
            "previous_results": {
                "fields_querier": {
                    "status": "ok",
                    "field_mappings": field_mappings,
                }
            },
        })

    assert result["status"] == "ok"
    assert result["results_count"] == 2
    assert "Iran" in result.get("countries", [])
    assert result["time_range"] == "now-2M"


def test_supervisor_evaluates_iran_2month_query_satisfied():
    """Supervisor should be satisfied when Iran traffic is found."""
    from core.chat_router.logic import _supervisor_evaluate_satisfaction

    mock_llm = MagicMock()

    eval_result = _supervisor_evaluate_satisfaction(
        user_question="any traffic from iran in the past 2 months",
        llm=mock_llm,
        instruction="You are a SOC analyst.",
        conversation_history=[],
        skill_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 2,
                "results": [IRAN_RECORD_FEB, IRAN_RECORD_JAN],
                "countries": ["Iran"],
                "time_range": "now-2M",
            }
        },
        step=1,
        max_steps=4,
    )

    mock_llm.chat.assert_not_called()
    assert eval_result["satisfied"] is True
    assert eval_result["confidence"] >= 0.8


def test_full_supervisor_loop_iran_2months():
    """Full orchestration for 'any traffic from iran in the past 2 months'."""
    from core.chat_router.logic import orchestrate_with_supervisor

    mock_llm = MagicMock()
    mock_runner = MagicMock()

    mock_llm.chat.return_value = json.dumps({
        "reasoning": "Search for Iran traffic in the past 2 months.",
        "skills": ["opensearch_querier"],
        "parameters": {"question": "any traffic from iran in the past 2 months"},
    })

    mock_runner.dispatch.return_value = {
        "status": "ok",
        "results_count": 2,
        "results": [IRAN_RECORD_FEB, IRAN_RECORD_JAN],
        "countries": ["Iran"],
        "time_range": "now-2M",
    }

    orchestration = orchestrate_with_supervisor(
        user_question="any traffic from iran in the past 2 months",
        available_skills=[
            {"name": "opensearch_querier", "description": "Search raw logs"},
        ],
        runner=mock_runner,
        llm=mock_llm,
        instruction="You are a SOC analyst.",
    )

    trace = orchestration["trace"]
    assert len(trace) == 1
    assert trace[0]["evaluation"]["satisfied"] is True
    assert trace[0]["evaluation"]["confidence"] >= 0.8
