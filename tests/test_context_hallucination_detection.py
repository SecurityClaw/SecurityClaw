"""
Test for context hallucination detection in opensearch_querier.

This test validates that the skill can detect when LLM planning mismatches
the user's actual question intent due to conversation context bleeding.

Example: User asks "any traffic from 1.1.1.1" but LLM plans country_terms
aggregation based on prior conversation about "non-US countries".

Run with: python -m pytest tests/test_context_hallucination_detection.py -v -s
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch
import json
from skills.opensearch_querier.logic import (
    _extract_countries_from_text,
    _llm_ground_question_intent,
    _plan_opensearch_query_with_llm,
)


class TestContextHallucinationDetection:
    """Test detection of LLM hallucinations from conversation context."""

    def test_traffic_ip_question_should_reject_country_agg(self):
        """
        Test that "any traffic from 1.1.1.1" rejects country_terms aggregation.
        
        Scenario: Prior conversation was about "non-US source countries".
        LLM hallucinates this context and plans country aggregation for
        an unrelated question asking for traffic from a specific IP.
        """
        # Create two different mocks: one for main planning (bad), one for simplified (good)
        main_llm = Mock()
        simplified_llm = Mock()
        
        # Main LLM first returns a bad plan, then its own review rejects the plan as context bleed.
        main_llm.complete.side_effect = [
            json.dumps({
                "summary": "User wants traffic records for 1.1.1.1.",
                "search_type": "traffic",
                "countries": [],
                "ips": ["1.1.1.1"],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "aggregation_type": "none",
                "entity_scope": "Traffic involving IP 1.1.1.1 only.",
            }),
            json.dumps({
                "reasoning": "User wants a distinct list of non-US source countries seen in traffic",
                "search_type": "traffic",
                "search_terms": ["1.1.1.1"],
                "countries": [],
                "exclude_countries": ["United States"],
                "ports": [],
                "protocols": [],
                "time_range": "now-30d",
                "matching_strategy": "term",
                "aggregation_type": "country_terms",
                "aggregation_field": "country",
                "result_limit": 10,
                "field_analysis": "Country distribution analysis",
                "skip_search": False,
            }),
            json.dumps({
                "is_valid": False,
                "issue": "The plan drifted into country aggregation from prior context instead of answering the current traffic-from-IP question.",
                "suggestion": "Use a traffic search with no country aggregation.",
                "confidence": 0.99,
            }),
        ]

        simplified_response = {
            "search_terms": ["1.1.1.1"],
            "ports": [],
            "search_type": "traffic",
            "matching_strategy": "term",
            "aggregation_type": "none",
        }
        simplified_llm.complete.return_value = json.dumps(simplified_response)

        question = "any traffic from 1.1.1.1"
        conversation_history = [
            {"role": "user", "content": "what countries are generating traffic other than US?"},
            {"role": "assistant", "content": "Countries with most traffic: ..."},
        ]
        
        # Mock field mappings
        field_mappings = {
            "ip_fields": ["source.ip", "dest_ip"],
            "country_fields": ["geoip.country_name"],
        }
        
        print(f"\n[TEST] Question: '{question}'")
        print(f"[TEST] Prior context: User asked about 'non-US countries'")
        print(f"[TEST] Main LLM incorrectly planned: country_terms aggregation")
        print(f"[TEST] Simplified LLM should correctly plan: none aggregation")

        with patch('skills.opensearch_querier.logic._plan_opensearch_query_with_llm_simplified') as mock_simplified:
            mock_simplified.return_value = simplified_response
            
            result = _plan_opensearch_query_with_llm(
                question, conversation_history, field_mappings, main_llm
            )
            
            print(f"[TEST] Final plan aggregation_type: {result.get('aggregation_type')}")
            
            # After mismatch detection and fallback, aggregation_type should NOT be country_terms
            assert result.get("aggregation_type") == "none", \
                f"Expected aggregation_type='none' after hallucination detection, got '{result.get('aggregation_type')}'"
            
            # Verify the model was used for planning and grounding review.
            assert main_llm.complete.call_count >= 3, "Main LLM should ground the question, plan, and review grounding before fallback"
            
            print("[TEST] ✓ PASS: Context hallucination was detected and corrected via fallback\n")

    def test_explicit_country_question_allows_country_agg(self):
        """
        Verify that legitimate country aggregation requests still work.
        
        Scenario: User explicitly asks "what countries have traffic?"
        LLM plans country_terms aggregation - this is CORRECT.
        """
        mock_llm = Mock()

        mock_llm.complete.side_effect = [
            json.dumps({
                "summary": "User wants a country distribution of traffic.",
                "search_type": "traffic",
                "countries": [],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "aggregation_type": "country_terms",
                "entity_scope": "Return countries seen in traffic.",
            }),
            json.dumps({
                "reasoning": "User wants to know what countries are generating traffic",
                "search_type": "traffic",
                "search_terms": [],
                "countries": [],
                "exclude_countries": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "matching_strategy": "token",
                "aggregation_type": "country_terms",
                "aggregation_field": "country",
                "result_limit": 10,
                "field_analysis": "Country distribution",
                "skip_search": False,
            }),
            json.dumps({
                "is_valid": True,
                "issue": "",
                "suggestion": "",
                "confidence": 0.97,
            }),
        ]

        question = "what countries are we getting traffic from?"
        conversation_history = []
        
        field_mappings = {
            "ip_fields": ["source.ip", "dest_ip"],
            "country_fields": ["geoip.country_name"],
        }
        
        print(f"\n[TEST] Question: '{question}'")
        print(f"[TEST] LLM planned: country_terms aggregation")
        
        result = _plan_opensearch_query_with_llm(
            question, conversation_history, field_mappings, mock_llm
        )
        
        print(f"[TEST] Final plan aggregation_type: {result.get('aggregation_type')}")
        
        # This should NOT trigger mismatch detection
        # because the question explicitly asks for countries
        assert result.get("aggregation_type") == "country_terms", \
            f"Expected country_terms for explicit country question, got {result.get('aggregation_type')}"
        
        print("[TEST] ✓ PASS: Legitimate country aggregation request accepted\n")

    def test_country_question_should_reject_different_country_plan(self):
        mock_llm = Mock()

        mock_llm.complete.side_effect = [
            json.dumps({
                "summary": "User wants traffic from Russia.",
                "search_type": "traffic",
                "countries": ["Russia"],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "aggregation_type": "none",
                "entity_scope": "Traffic originating from Russia.",
            }),
            json.dumps({
                "reasoning": "User wants to see network traffic originating from Iran.",
                "search_type": "traffic",
                "search_terms": [],
                "countries": ["Iran"],
                "exclude_countries": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "matching_strategy": "term",
                "aggregation_type": "none",
                "aggregation_field": "none",
                "result_limit": 10,
                "field_analysis": "Use country and timestamp fields.",
                "skip_search": False,
            }),
            json.dumps({
                "is_valid": False,
                "issue": "The plan targets Iran even though the current question explicitly asks for Russia.",
                "suggestion": "Keep the country filter aligned to Russia.",
                "confidence": 0.99,
            }),
        ]

        fallback_plan = {
            "reasoning": "Simplified LLM planning",
            "search_type": "traffic",
            "search_terms": [],
            "countries": ["Russia"],
            "ports": [],
            "protocols": [],
            "time_range": "now-90d",
            "matching_strategy": "term",
            "aggregation_type": "none",
        }

        with patch('skills.opensearch_querier.logic._plan_opensearch_query_with_llm_simplified') as mock_simplified:
            mock_simplified.return_value = fallback_plan
            result = _plan_opensearch_query_with_llm(
                "any traffic from russia",
                [{"role": "assistant", "content": "Earlier we discussed Iran traffic."}],
                {"country_fields": ["geoip.country_name"]},
                mock_llm,
            )

        assert result.get("countries") == ["Russia"]
        assert mock_llm.complete.call_count >= 3

    def test_simplified_plan_review_rejects_scope_drift_and_falls_back(self):
        mock_llm = Mock()

        mock_llm.complete.side_effect = [
            json.dumps({
                "summary": "User wants to know whether there is any traffic from Russia in the past 30 days.",
                "search_type": "traffic",
                "countries": ["Russia"],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-30d",
                "aggregation_type": "none",
                "entity_scope": "Traffic from Russia in the past 30 days.",
            }),
            "not valid json",
            json.dumps({
                "search_terms": [],
                "countries": [],
                "ports": [],
                "protocols": [],
                "search_type": "traffic",
                "matching_strategy": "token",
                "aggregation_type": "country_terms",
                "time_range": "now-30d",
            }),
            json.dumps({
                "is_valid": False,
                "reasoning": "A country distribution does not answer whether there was traffic from Russia specifically.",
                "issue": "The simplified plan broadened the question into a top-countries aggregation.",
                "suggestion": "Search directly for traffic filtered to Russia.",
                "confidence": 0.99,
            }),
        ]

        result = _plan_opensearch_query_with_llm(
            "any traffic from russia in the past 30 days?",
            [{"role": "assistant", "content": "We were previously discussing traffic from many countries."}],
            {"country_fields": ["geoip.country_name"]},
            mock_llm,
        )

        assert result.get("aggregation_type") == "none"
        assert result.get("countries") == ["Russia"]
        assert mock_llm.complete.call_count >= 4

    def test_country_extraction_recognizes_iran_in_traffic_question(self):
        assert _extract_countries_from_text("any traffic from iran in the past 2 months") == ["Iran"]

    def test_country_extraction_uses_rag_values_for_greece(self):
        assert _extract_countries_from_text(
            "any traffic from greece in the past 2 months",
            {
                "country_fields": ["geoip.country_name"],
                "country_values": ["Greece", "Iran", "Germany"],
                "field_value_examples": {"geoip.country_name": ["Greece", "Iran", "Germany"]},
            },
        ) == ["Greece"]

    def test_question_grounding_corrects_bad_llm_output_for_iran_traffic(self):
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "User wants IP threat analysis.",
            "search_type": "ip",
            "countries": [],
            "ips": [],
            "ports": [],
            "protocols": [],
            "time_range": "now",
            "aggregation_type": "fingerprint_ports",
            "entity_scope": "Threat analysis on an IP.",
        })

        grounding = _llm_ground_question_intent(
            "any traffic from iran in the past 2 months",
            mock_llm,
        )

        assert grounding["search_type"] == "traffic"
        assert grounding["aggregation_type"] == "none"
        assert grounding["countries"] == ["Iran"]
        assert grounding["time_range"] == "now-2M"

    def test_question_grounding_uses_rag_values_for_greece_traffic(self):
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "User wants generic traffic.",
            "search_type": "general",
            "countries": [],
            "ips": [],
            "ports": [],
            "protocols": [],
            "time_range": "now-90d",
            "aggregation_type": "none",
            "entity_scope": "",
        })

        grounding = _llm_ground_question_intent(
            "any traffic from greece in the past 2 months",
            mock_llm,
            {
                "country_fields": ["geoip.country_name"],
                "country_values": ["Greece", "Iran"],
                "field_value_examples": {"geoip.country_name": ["Greece", "Iran"]},
            },
        )

        assert grounding["search_type"] == "traffic"
        assert grounding["aggregation_type"] == "none"
        assert grounding["countries"] == ["Greece"]
        assert grounding["time_range"] == "now-2M"

    def test_question_grounding_recovers_greece_today_without_rag_values(self):
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "Geolocate traffic sources.",
            "search_type": "ip",
            "countries": [],
            "ips": [],
            "ports": [],
            "protocols": [],
            "time_range": "now-90d",
            "aggregation_type": "fingerprint_ports",
            "entity_scope": "Find the relevant IP address.",
        })

        grounding = _llm_ground_question_intent(
            "any traffic from greece today?",
            mock_llm,
        )

        assert grounding["search_type"] == "traffic"
        assert grounding["aggregation_type"] == "none"
        assert grounding["countries"] == ["Greece"]
        assert grounding["time_range"] == "now/d"

    def test_country_traffic_question_rejects_fingerprint_drift(self):
        mock_llm = Mock()
        mock_llm.complete.side_effect = [
            json.dumps({
                "summary": "User wants IP threat analysis.",
                "search_type": "ip",
                "countries": [],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now",
                "aggregation_type": "fingerprint_ports",
                "entity_scope": "Threat analysis on an IP.",
            }),
            json.dumps({
                "reasoning": "The user wants to profile a target IP for ports.",
                "search_type": "ip",
                "search_terms": ["threat", "ip"],
                "countries": [],
                "ports": [],
                "protocols": [],
                "time_range": "now",
                "matching_strategy": "token",
                "aggregation_type": "fingerprint_ports",
                "aggregation_field": "destination.port",
                "result_limit": 256,
                "field_analysis": "Use IP and port fields.",
                "skip_search": False,
            }),
            json.dumps({
                "is_valid": True,
                "issue": "",
                "suggestion": "",
                "confidence": 0.99,
            }),
        ]

        result = _plan_opensearch_query_with_llm(
            "any traffic from iran in the past 2 months",
            [],
            {"country_fields": ["geoip.country_name"]},
            mock_llm,
        )

        assert result["search_type"] == "traffic"
        assert result["aggregation_type"] == "none"
        assert result["countries"] == ["Iran"]
        assert result["time_range"] == "now-2M"
        assert result["search_terms"] == []

    def test_country_traffic_today_strips_generic_ip_placeholder_terms(self):
        mock_llm = Mock()
        mock_llm.complete.side_effect = [
            json.dumps({
                "summary": "Traffic from Greece today.",
                "search_type": "traffic",
                "countries": ["Greece"],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now/d",
                "aggregation_type": "none",
                "entity_scope": "Traffic from Greece today.",
            }),
            "not valid json",
            json.dumps({
                "reasoning": "Look up the IP address seen in Greece traffic.",
                "search_type": "ip",
                "search_terms": ["IP address", "Greece"],
                "countries": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-90d",
                "matching_strategy": "token",
                "aggregation_type": "fingerprint_ports",
                "aggregation_field": "destination.port",
                "result_limit": 10,
                "field_analysis": "Use IP and port fields.",
                "skip_search": False,
            }),
            json.dumps({
                "is_valid": True,
                "issue": "",
                "suggestion": "",
                "confidence": 0.99,
            }),
        ]

        result = _plan_opensearch_query_with_llm(
            "any traffic from greece today?",
            [],
            {"country_fields": ["geoip.country_name"]},
            mock_llm,
        )

        assert result["search_type"] == "traffic"
        assert result["aggregation_type"] == "none"
        assert result["countries"] == ["Greece"]
        assert result["time_range"] == "now/d"
        assert result["search_terms"] == []


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 70)
    print("CONTEXT HALLUCINATION DETECTION TESTS")
    print("=" * 70)
    
    try:
        pytest.main([__file__, "-v", "-s"])
    except:
        print("\nRunning tests manually...\n")
        test_obj = TestContextHallucinationDetection()
        test_obj.test_traffic_ip_question_should_reject_country_agg()
        test_obj.test_explicit_country_question_allows_country_agg()
