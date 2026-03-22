"""
Tests for _plan_with_llm — LLM-pure query planner.

Validates that the planner correctly identifies request types and returns
structured plans with aggregation_type, countries, ip_direction, time_range.

Run with: python -m pytest tests/test_simplified_llm_planning.py -v -s
"""

import pytest
from unittest.mock import Mock
import json
import sys
from pathlib import Path

from skills.opensearch_querier.logic import _plan_with_llm


class TestLLMPlannerFingerprintDetection:
    """Fingerprinting intent must produce aggregation_type=fingerprint_ports."""

    def test_fingerprint_ip_detection(self):
        """'fingerprint 192.168.0.17' must trigger fingerprint_ports aggregation."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["192.168.0.17"],
            "countries": [],
            "ip_direction": "destination",
            "aggregation_type": "fingerprint_ports",
            "time_range": "now-90d",
        })

        result = _plan_with_llm("fingerprint 192.168.0.17", mock_llm)

        assert result is not None
        assert result["aggregation_type"] == "fingerprint_ports"
        assert "192.168.0.17" in result["search_terms"]

    def test_what_ports_detection(self):
        """'what ports on this IP' must trigger fingerprint_ports."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["192.168.1.100"],
            "countries": [],
            "ip_direction": "destination",
            "aggregation_type": "fingerprint_ports",
            "time_range": "now-90d",
        })

        result = _plan_with_llm("what ports on 192.168.1.100", mock_llm)

        assert result is not None
        assert result["aggregation_type"] == "fingerprint_ports"

    def test_traffic_from_ip_is_not_fingerprinting(self):
        """'any traffic from 1.1.1.1' is NOT fingerprinting — aggregation_type=none."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["1.1.1.1"],
            "countries": [],
            "ip_direction": "any",
            "aggregation_type": "none",
            "time_range": "now-90d",
        })

        result = _plan_with_llm("any traffic from 1.1.1.1", mock_llm)

        assert result is not None
        assert result["aggregation_type"] == "none"


class TestLLMPlannerCountryDetection:
    """Country traffic questions must include country + ip_direction in plan."""

    def test_iran_traffic_returns_country_and_direction(self):
        """'traffic from iran' → countries=[Iran], ip_direction=source."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": [],
            "countries": ["Iran"],
            "ip_direction": "source",
            "aggregation_type": "none",
            "time_range": "now-2M",
        })

        result = _plan_with_llm("any traffic from iran the past 2 months", mock_llm)

        assert result is not None
        assert result["countries"] == ["Iran"]
        assert result["ip_direction"] == "source"
        assert result["aggregation_type"] == "none"

    def test_russia_traffic_30d(self):
        """'traffic from russia the past 30 days' → countries=[Russia], time=now-30d."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": [],
            "countries": ["Russia"],
            "ip_direction": "source",
            "aggregation_type": "none",
            "time_range": "now-30d",
        })

        result = _plan_with_llm("any traffic from russia the past 30 days", mock_llm)

        assert result is not None
        assert result["countries"] == ["Russia"]
        assert result["time_range"] == "now-30d"


class TestLLMPlannerRobustness:
    """The planner must handle edge cases gracefully."""

    def test_missing_aggregation_type_defaults_to_none(self):
        """LLM response without aggregation_type should return what was given."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["test"],
            "countries": [],
            "ip_direction": "any",
            "time_range": "now-90d",
            # Note: no aggregation_type
        })

        result = _plan_with_llm("test query", mock_llm)
        assert result is not None
        # aggregation_type defaults to "none" when missing
        assert result.get("aggregation_type", "none") == "none"

    def test_invalid_json_returns_none(self):
        mock_llm = Mock()
        mock_llm.complete.return_value = "not json at all"

        result = _plan_with_llm("any query", mock_llm)
        assert result is None

    def test_markdown_fenced_json_is_parsed(self):
        mock_llm = Mock()
        mock_llm.complete.return_value = '```json\n{"search_terms":[],"countries":["Russia"],"ip_direction":"source","aggregation_type":"none","time_range":"now-30d"}\n```'

        result = _plan_with_llm("any traffic from russia?", mock_llm)
        assert result is not None
        assert result["countries"] == ["Russia"]

    def test_no_llm_returns_none(self):
        result = _plan_with_llm("any query", llm=None)
        assert result is None

    def test_grounding_context_in_prompt(self):
        """grounding_context must be embedded in the prompt passed to LLM."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["192.168.0.16"],
            "countries": [],
            "ip_direction": "destination",
            "aggregation_type": "fingerprint_ports",
            "time_range": "now-90d",
        })

        _plan_with_llm(
            "fingerprint 192.168.0.16",
            mock_llm,
            grounding_context={"ips": ["192.168.0.16"]},
        )

        prompt = mock_llm.complete.call_args[0][0]
        assert "192.168.0.16" in prompt
        assert "PRE-EXTRACTED ENTITIES" in prompt

