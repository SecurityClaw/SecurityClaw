"""Tests for opensearch_querier run() with grounding context merging."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

from skills.opensearch_querier.logic import _plan_with_llm, _build_opensearch_query


def test_plan_with_llm_merges_grounding_ips_into_prompt():
    """Grounding IPs must be embedded in the LLM prompt."""
    mock_llm = Mock()
    mock_llm.complete.return_value = json.dumps({
        "search_terms": ["1.1.1.1"],
        "countries": [],
        "ip_direction": "any",
        "aggregation_type": "none",
        "time_range": "now-90d",
    })

    _plan_with_llm(
        "any traffic from 1.1.1.1?",
        mock_llm,
        grounding_context={"ips": ["1.1.1.1"]},
    )

    prompt = mock_llm.complete.call_args[0][0]
    assert "1.1.1.1" in prompt
    assert "PRE-EXTRACTED ENTITIES" in prompt


def test_plan_with_llm_no_grounding_still_works():
    """Plan with no grounding context should call LLM without grounding section."""
    mock_llm = Mock()
    mock_llm.complete.return_value = json.dumps({
        "search_terms": [],
        "countries": ["Iran"],
        "ip_direction": "source",
        "aggregation_type": "none",
        "time_range": "now-2M",
    })

    result = _plan_with_llm("any traffic from iran", mock_llm, grounding_context=None)

    assert result is not None
    assert result["countries"] == ["Iran"]
    prompt = mock_llm.complete.call_args[0][0]
    assert "PRE-EXTRACTED ENTITIES" not in prompt


def test_build_opensearch_query_country_filter_with_no_search_terms():
    """A country-only query should build a valid query even with no IP search terms."""
    query = _build_opensearch_query(
        search_terms=[],
        dest_ip_field="dest_ip",
        time_range="now-30d",
        size=200,
        ip_direction="source",
        countries=["Russia"],
    )
    body = json.dumps(query)
    assert "Russia" in body
    assert "geoip.country_name" in body or "source.geo.country_name" in body
    # Must contain time range
    assert "now-30d" in body
    assert "@timestamp" in body


