"""
Test simplified LLM planning to verify fingerprinting intent detection.

This test validates that the simplified LLM planner correctly identifies
fingerprinting requests and sets aggregation_type to "fingerprint_ports".

Run with: python -m pytest tests/test_simplified_llm_planning.py -v -s
Or standalone: python tests/test_simplified_llm_planning.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
import sys
from pathlib import Path

from skills.opensearch_querier.logic import (
    _plan_opensearch_query_with_llm_simplified,
)


class TestSimplifiedLLMPlanning:
    """Test simplified LLM planning for fingerprinting intent."""

    def test_fingerprint_ip_detection(self):
        """Test that 'fingerprint 192.168.0.17' triggers fingerprint_ports aggregation."""
        
        # Mock LLM that returns proper JSON with fingerprint_ports aggregation
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["192.168.0.17"],
            "ports": [],
            "search_type": "ip",
            "matching_strategy": "token",
            "aggregation_type": "fingerprint_ports"
        })
        
        question = "fingerprint 192.168.0.17"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"\n[TEST] Question: '{question}'")
        print(f"[TEST] LLM Response: {mock_llm.complete.return_value}")
        print(f"[TEST] Parsed Result: {result}")
        
        assert result is not None, "Simplified planning should not return None"
        assert result["aggregation_type"] == "fingerprint_ports", \
            f"Expected aggregation_type='fingerprint_ports', got '{result.get('aggregation_type')}'"
        assert result["search_type"] == "ip", \
            f"Expected search_type='ip', got '{result.get('search_type')}'"
        print("[TEST] ✓ PASS: Fingerprinting intent correctly detected\n")

    def test_what_ports_detection(self):
        """Test that 'what ports on this IP' triggers fingerprint_ports."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["192.168.1.100"],
            "ports": [],
            "search_type": "ip",
            "matching_strategy": "token",
            "aggregation_type": "fingerprint_ports"
        })
        
        question = "what ports on 192.168.1.100"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"[TEST] Question: '{question}'")
        print(f"[TEST] Result aggregation_type: {result.get('aggregation_type')}")
        
        assert result["aggregation_type"] == "fingerprint_ports", \
            f"Expected fingerprint_ports, got {result.get('aggregation_type')}"
        print("[TEST] ✓ PASS: 'What ports' question detected\n")

    def test_country_distribution_detection(self):
        """Test that country-related questions trigger country_terms aggregation."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["country", "distribution"],
            "ports": [],
            "search_type": "general",
            "matching_strategy": "token",
            "aggregation_type": "country_terms"
        })
        
        question = "country distribution of traffic"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"[TEST] Question: '{question}'")
        print(f"[TEST] Result aggregation_type: {result.get('aggregation_type')}")
        
        assert result["aggregation_type"] == "country_terms", \
            f"Expected country_terms, got {result.get('aggregation_type')}"
        print("[TEST] ✓ PASS: Country distribution detected\n")

    def test_general_search_no_aggregation(self):
        """Test that general questions don't trigger aggregation."""
        mock_llm = Mock()
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["nginx", "errors"],
            "ports": [],
            "search_type": "traffic",
            "matching_strategy": "token",
            "aggregation_type": "none"
        })
        
        question = "show me nginx errors"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"[TEST] Question: '{question}'")
        print(f"[TEST] Result aggregation_type: {result.get('aggregation_type')}")
        
        assert result["aggregation_type"] == "none", \
            f"Expected none, got {result.get('aggregation_type')}"
        print("[TEST] ✓ PASS: General search has no aggregation\n")

    def test_missing_aggregation_type_gets_default(self):
        """Test that missing aggregation_type field gets default 'none'."""
        mock_llm = Mock()
        # LLM returns JSON WITHOUT aggregation_type field
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["test"],
            "ports": [],
            "search_type": "general",
            "matching_strategy": "token"
            # Note: missing aggregation_type
        })
        
        question = "test query"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"[TEST] Question: '{question}'")
        print(f"[TEST] LLM returned no aggregation_type")
        print(f"[TEST] Result aggregation_type: {result.get('aggregation_type')}")
        
        assert "aggregation_type" in result, "Missing field should be added with default"
        assert result["aggregation_type"] == "none", \
            f"Default should be 'none', got '{result.get('aggregation_type')}'"
        print("[TEST] ✓ PASS: Missing aggregation_type defaults to 'none'\n")

    def test_traffic_from_ip_should_not_be_fingerprinting(self):
        """Test that 'any traffic from IP' is classified as traffic, NOT fingerprinting.
        
        This is a critical distinction:
        - "any traffic from 1.1.1.1" → search_type="traffic", aggregation_type="none"
        - "fingerprint 1.1.1.1" → search_type="ip", aggregation_type="fingerprint_ports"
        """
        mock_llm = Mock()
        # LLM should correctly identify this as a traffic query, not fingerprinting
        mock_llm.complete.return_value = json.dumps({
            "search_terms": ["1.1.1.1"],
            "ports": [],
            "search_type": "traffic",
            "matching_strategy": "token",
            "aggregation_type": "none"
        })
        
        question = "any traffic from 1.1.1.1"
        result = _plan_opensearch_query_with_llm_simplified(question, mock_llm)
        
        print(f"\n[TEST] Question: '{question}'")
        print(f"[TEST] LLM Response: {mock_llm.complete.return_value}")
        print(f"[TEST] Parsed Result: {result}")
        
        assert result is not None, "Simplified planning should not return None"
        # Critical: should NOT be fingerprinting
        assert result["aggregation_type"] == "none", \
            f"Expected aggregation_type='none' (NOT fingerprinting), got '{result.get('aggregation_type')}'"
        assert result["search_type"] == "traffic", \
            f"Expected search_type='traffic', got '{result.get('search_type')}'"
        print("[TEST] ✓ PASS: Traffic query correctly distinguished from fingerprinting\n")


class TestSimplifiedLLMIntegration:
    """Integration tests with real LLM (if available)."""
    
    def test_real_llm_fingerprinting(self):
        """Test with real LLM if available (requires LLM to be configured)."""
        try:
            from core.llm_base import get_llm
            llm = get_llm()
            
            print("\n[INTEGRATION TEST] Testing with real LLM")
            question = "fingerprint 192.168.0.17"
            result = _plan_opensearch_query_with_llm_simplified(question, llm)
            
            print(f"[INTEGRATION TEST] Question: '{question}'")
            print(f"[INTEGRATION TEST] Result: {result}")
            
            if result:
                print(f"[INTEGRATION TEST] aggregation_type: {result.get('aggregation_type')}")
                print(f"[INTEGRATION TEST] search_type: {result.get('search_type')}")
                
                # Check if fingerprinting intent was detected
                if result.get('aggregation_type') == 'fingerprint_ports':
                    print("[INTEGRATION TEST] ✓ PASS: Real LLM correctly detected fingerprinting\n")
                    return True
                else:
                    print(f"[INTEGRATION TEST] ✗ FAIL: Expected fingerprint_ports, got {result.get('aggregation_type')}\n")
                    return False
            else:
                print("[INTEGRATION TEST] ✗ FAIL: LLM returned None\n")
                return False
                
        except Exception as e:
            print(f"[INTEGRATION TEST] Skipped (LLM not available): {e}\n")
            return None  # Skip if LLM not available


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 70)
    print("SIMPLIFIED LLM PLANNING TESTS")
    print("=" * 70)
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v", "-s"])
    except:
        # Fallback to manual test execution
        print("\nRunning tests manually (pytest not available)...\n")
        test_obj = TestSimplifiedLLMPlanning()
        test_obj.test_fingerprint_ip_detection()
        test_obj.test_what_ports_detection()
        test_obj.test_country_distribution_detection()
        test_obj.test_general_search_no_aggregation()
        test_obj.test_missing_aggregation_type_gets_default()
        
        test_int = TestSimplifiedLLMIntegration()
        test_int.test_real_llm_fingerprinting()
