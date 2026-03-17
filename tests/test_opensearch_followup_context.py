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


def test_detects_port_followup_with_modified_article_phrase():
    """Test the fix for: 'What port is the above traffic associated with?'
    
    This bug occurred because the detector was looking for "the traffic" (exact match)
    but the question had "the above traffic" (with "above" in between).
    
    The fix adds:
    1. Explicit phrase matches for "the above traffic", etc.
    2. Regex pattern to catch "the [modifiers] traffic" 
    3. Recovery function now overrides bad LLM plans for follow-ups
    """
    # Simulate LLM plan that incorrectly extracted "traffic" as search term
    # (this is what was happening before the fix)
    bad_llm_plan = {
        "reasoning": "Simplified LLM planning",
        "search_type": "traffic",
        "search_terms": ["traffic"],  # Wrong! Should be IPs, not the word "traffic"
        "countries": [],
        "ports": [],  # This was causing malformed queries
        "protocols": [],
        "time_range": "now-90d",
        "matching_strategy": "token",
    }
    
    # Previous results from IP search
    previous_results = {
        "opensearch_querier": {
            "status": "ok",
            "description": "Found 3301314 total record(s) matching 1.1.1.1 in the now-90d window. "
                          "Source/destination IPs: 1.1.1.1, 192.168.0.130, 192.168.0.180, 192.168.0.80, 192.168.0.85.",
            "total_hits": 3301314,
        }
    }

    # Apply recovery with the actual user question
    recovered = _recover_followup_plan_from_context(
        "What port is the above traffic associated with?",
        bad_llm_plan,
        previous_results,
        conversation_history=[],
    )

    # Verify the recovery function overrode the bad LLM plan
    assert recovered["search_type"] == "traffic"
    assert "traffic" not in recovered["search_terms"]  # Removed the bad "traffic" term
    assert set(recovered["search_terms"]) == {"1.1.1.1", "192.168.0.130", "192.168.0.180", "192.168.0.80", "192.168.0.85"}
    assert recovered["ports"] == []  # Empty means "search for all ports on these IPs"
    assert recovered["matching_strategy"] == "term"
    assert "Recovered 5 IP(s)" in recovered["reasoning"]