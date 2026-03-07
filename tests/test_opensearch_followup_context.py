from __future__ import annotations

from skills.opensearch_querier.logic import _recover_followup_plan_from_context


def test_recovers_ip_search_terms_from_previous_results_for_country_followup():
    query_plan = {
        "reasoning": "The question asks about countries for prior IPs.",
        "search_type": "ip",
        "search_terms": [],
        "countries": [],
        "ports": [],
        "protocols": [],
        "time_range": "now-30d",
        "matching_strategy": "token",
    }
    previous_results = {
        "opensearch_querier": {
            "status": "ok",
            "results": [
                {"source.ip": "100.29.192.116"},
                {"destination.ip": "147.185.132.112"},
            ],
        }
    }

    recovered = _recover_followup_plan_from_context(
        "What countries are these IPs from?",
        query_plan,
        previous_results,
        conversation_history=[],
    )

    assert recovered["matching_strategy"] == "term"
    assert recovered["search_terms"] == ["100.29.192.116", "147.185.132.112"]
    assert "Recovered 2 IP(s)" in recovered["reasoning"]


def test_recovers_ip_search_terms_from_conversation_when_previous_results_empty():
    query_plan = {
        "reasoning": "The current question is referential.",
        "search_type": "ip",
        "search_terms": [],
        "countries": [],
        "ports": [],
        "protocols": [],
        "time_range": "now-30d",
        "matching_strategy": "token",
    }
    history = [
        {
            "role": "assistant",
            "content": (
                "IPs seen in matching alerts: 100.29.192.116, 100.29.192.35, 147.185.132.112. "
                "Earliest: 2026-02-18T05:35:15.022Z."
            ),
        }
    ]

    recovered = _recover_followup_plan_from_context(
        "What countries are these IPs from?",
        query_plan,
        previous_results={},
        conversation_history=history,
    )

    assert recovered["matching_strategy"] == "term"
    assert recovered["search_terms"] == ["100.29.192.116", "100.29.192.35", "147.185.132.112"]


def test_does_not_override_existing_query_criteria():
    query_plan = {
        "reasoning": "Already extracted a concrete IP.",
        "search_type": "ip",
        "search_terms": ["1.2.3.4"],
        "countries": [],
        "ports": [],
        "protocols": [],
        "time_range": "now-30d",
        "matching_strategy": "term",
    }

    recovered = _recover_followup_plan_from_context(
        "What countries are these IPs from?",
        query_plan,
        previous_results={},
        conversation_history=[],
    )

    assert recovered == query_plan