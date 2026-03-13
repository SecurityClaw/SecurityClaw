"""
Test fallback mechanisms for opensearch_querier when LLM planning fails.

Validates:
1. Fallback heuristic extraction when LLM returns invalid JSON
2. Simplified LLM retry with minimal prompt
3. Context recovery for port/protocol follow-up questions
"""

import pytest
from unittest.mock import Mock, MagicMock

from skills.opensearch_querier.logic import (
    _fallback_plan_from_question,
    _extract_ports_from_text,
    _extract_countries_from_text,
    _question_asks_for_country_aggregation,
    _question_asks_for_followup_details,
    _recover_followup_plan_from_context,
    _plan_opensearch_query_with_llm_simplified,
)


class TestFallbackExtraction:
    """Test heuristic-based extraction when LLM fails."""
    
    def test_extract_ports_from_question(self):
        """Extract port numbers from natural language questions."""
        assert _extract_ports_from_text("What traffic on port 443?") == [443]
        assert _extract_ports_from_text("Check ports 80 and 443") == [80, 443]
        assert _extract_ports_from_text("Connection to :8080 failed") == [8080]
        assert _extract_ports_from_text("No ports mentioned here") == []
    
    def test_extract_countries_from_question(self):
        """Extract country names from natural language questions."""
        assert "China" in _extract_countries_from_text("Where is this IP from China?")
        assert "Russia" in _extract_countries_from_text("Check traffic from Russia")
        assert _extract_countries_from_text("No countries here") == []
    
    def test_fallback_plan_port_question(self):
        """Generate fallback plan for port-related question."""
        question = "What port was associated with this traffic?"
        plan = _fallback_plan_from_question(question)
        
        assert plan["skip_search"] == False, "Fallback should not skip search"
        assert plan["search_type"] == "traffic"
        assert "fallback" in plan["reasoning"].lower()
    
    def test_fallback_plan_with_explicit_port(self):
        """Fallback extraction should find explicit port numbers."""
        question = "What was happening on port 3306?"
        plan = _fallback_plan_from_question(question)
        
        assert 3306 in plan["ports"]
        assert plan["search_type"] == "traffic"
    
    def test_fallback_plan_extracts_from_previous_results(self):
        """Fallback extraction should use IPs from previous results."""
        question = "What countries are these IPs from?"
        previous_results = {
            "opensearch_querier": {
                "results": [
                    {"src_ip": "1.2.3.4"},
                    {"src_ip": "5.6.7.8"},
                ]
            }
        }
        
        plan = _fallback_plan_from_question(question, previous_results)
        
        # Should have extracted IPs from previous_results
        assert "1.2.3.4" in plan["search_terms"]
        assert "5.6.7.8" in plan["search_terms"]


class TestQuestionClassification:
    """Test classification of follow-up questions."""

    def test_identifies_country_aggregation_question(self):
        assert _question_asks_for_country_aggregation(
            "What countries other than the USA do we get traffic from in the past month"
        )
    
    def test_identifies_port_followup(self):
        """Recognize questions asking about port details."""
        assert _question_asks_for_followup_details("What port was associated with this traffic?")
        assert _question_asks_for_followup_details("What protocol did that connection use?")
        assert _question_asks_for_followup_details("What traffic was on these IPs?")
    
    def test_rejects_non_followup_questions(self):
        """Non-followup questions should not match."""
        assert not _question_asks_for_followup_details("What is this IP?")
        assert not _question_asks_for_followup_details("Show me alerts")


class TestContextRecovery:
    """Test recovery of context for follow-up questions."""
    
    def test_recover_ips_for_port_question(self):
        """Recover IP context when asked about ports on previous traffic."""
        question = "What port was associated with this traffic?"
        previous_results = {
            "opensearch_querier": {
                "results": [
                    {"src_ip": "147.185.132.112", "dest_ip": "192.168.0.16"},
                ]
            }
        }
        
        original_plan = {
            "search_type": "general",
            "search_terms": [],
            "countries": [],
            "ports": [],
            "protocols": [],
        }
        
        recovered = _recover_followup_plan_from_context(
            question, original_plan, previous_results, []
        )
        
        assert recovered["search_type"] == "traffic"
        assert "147.185.132.112" in recovered["search_terms"]
        assert "192.168.0.16" in recovered["search_terms"]
        assert "traffic detail" in recovered["reasoning"].lower()
    
    def test_recover_from_conversation_history(self):
        """Recover IPs from conversation history if not in previous_results."""
        question = "What port was associated with this traffic?"
        conversation_history = [
            {"role": "agent", "content": "Found IP 10.0.0.1 in logs"}
        ]
        
        original_plan = {
            "search_type": "general",
            "search_terms": [],
            "countries": [],
            "ports": [],
            "protocols": [],
        }
        
        recovered = _recover_followup_plan_from_context(
            question, original_plan, {}, conversation_history
        )
        
        assert "10.0.0.1" in recovered["search_terms"]
    
    def test_respects_existing_criteria(self):
        """Don't override if query plan already has criteria."""
        question = "What port was associated with this traffic?"
        previous_results = {
            "opensearch_querier": {
                "results": [{"src_ip": "1.2.3.4"}]
            }
        }
        
        original_plan = {
            "search_type": "traffic",
            "search_terms": ["malware"],
            "countries": [],
            "ports": [],
            "protocols": [],
        }
        
        recovered = _recover_followup_plan_from_context(
            question, original_plan, previous_results, []
        )
        
        # Should keep existing search_terms
        assert recovered["search_terms"] == ["malware"]


class TestSimplifiedLLMPlanning:
    """Test simplified fallback LLM with minimal prompt."""
    
    def test_simplified_llm_planning_valid_response(self):
        """Simplified LLM should parse valid JSON response."""
        mock_llm = Mock()
        mock_llm.complete.return_value = '''{
            "search_terms": ["port", "443"],
            "ports": [443],
            "search_type": "traffic",
            "matching_strategy": "token"
        }'''
        
        result = _plan_opensearch_query_with_llm_simplified("traffic on port 443", mock_llm)
        
        assert result is not None
        assert 443 in result["ports"]
        assert result["search_type"] == "traffic"
    
    def test_simplified_llm_returns_none_on_failure(self):
        """Simplified LLM should return None if JSON parsing fails."""
        mock_llm = Mock()
        mock_llm.complete.return_value = "This is not JSON at all"
        
        result = _plan_opensearch_query_with_llm_simplified("question", mock_llm)
        
        assert result is None
    
    def test_simplified_llm_handles_markdown_json(self):
        """Simplified LLM should extract JSON from markdown code blocks."""
        mock_llm = Mock()
        mock_llm.complete.return_value = '''
Here's the plan:
```json
{
    "search_terms": ["alert"],
    "ports": [],
    "search_type": "alert",
    "matching_strategy": "phrase"
}
```
        '''
        
        result = _plan_opensearch_query_with_llm_simplified("show alerts", mock_llm)
        
        assert result is not None
        assert result["search_type"] == "alert"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_run_uses_country_aggregation_for_non_us_country_summary():
    from skills.opensearch_querier import logic
    from tests.mock_opensearch import MockDBConnector

    class _Cfg:
        def get(self, section: str, key: str, default=None):
            if (section, key) == ("db", "logs_index"):
                return "logstash*"
            return default

    db = MockDBConnector()
    db._client = MagicMock()
    db._client.search.return_value = {
        "aggregations": {
            "country_counts": {
                "buckets": [
                    {"key": "United States", "doc_count": 15},
                    {"key": "Iran", "doc_count": 2},
                    {"key": "Russia", "doc_count": 1},
                ]
            }
        }
    }

    llm = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "core.query_builder.discover_field_mappings",
            lambda db, llm: {
                "country_fields": ["geoip.country_name"],
                "all_fields": ["geoip.country_name", "@timestamp"],
                "timestamp_fields": ["@timestamp"],
                "ip_fields": [],
                "source_ip_fields": [],
                "destination_ip_fields": [],
                "port_fields": [],
                "text_fields": [],
            },
        )
        mp.setattr(
            logic,
            "_plan_opensearch_query_with_llm",
            lambda question, conversation_history, field_mappings, llm: {
                "reasoning": "User wants distinct non-US countries.",
                "search_type": "traffic",
                "search_terms": [],
                "countries": [],
                "ports": [],
                "protocols": [],
                "time_range": "custom",
                "matching_strategy": "token",
            },
        )

        result = logic.run(
            {
                "db": db,
                "llm": llm,
                "config": _Cfg(),
                "parameters": {"question": "What countries other than the USA do we get traffic from in the past month"},
                "previous_results": {},
            }
        )

    assert result["status"] == "ok"
    assert result["aggregation_type"] == "country_terms"
    assert result["time_range_label"] == "past month"
    assert result["excluded_countries"] == ["United States"]
    assert result["country_buckets"] == [
        {"country": "Iran", "count": 2},
        {"country": "Russia", "count": 1},
    ]


def test_country_aggregation_tries_keyword_variant_before_hit_fallback():
    from skills.opensearch_querier import logic

    class _Client:
        def __init__(self):
            self.fields = []

        def search(self, index=None, body=None, size=0):
            field_name = body["aggs"]["country_counts"]["terms"]["field"]
            self.fields.append(field_name)
            if field_name == "geoip.country_name.keyword":
                return {
                    "aggregations": {
                        "country_counts": {
                            "buckets": [
                                {"key": "Iran", "doc_count": 2},
                                {"key": "United Kingdom", "doc_count": 1},
                            ]
                        }
                    }
                }
            if field_name == "geoip.country_name":
                raise RuntimeError("fielddata disabled")
            return {"aggregations": {"country_counts": {"buckets": []}}}

    class _DB:
        def __init__(self):
            self._client = _Client()
            self.search_called = False

        def search(self, index, query, size=10):
            self.search_called = True
            raise AssertionError("hit fallback should not run when a direct aggregation succeeds")

    db = _DB()
    result = logic._execute_country_aggregation_query(
        db=db,
        index="logstash*",
        field_mappings={
            "country_fields": ["geoip.country_name", "geoip.country_code2"],
            "all_fields": ["geoip.country_name", "geoip.country_code2", "@timestamp"],
        },
        time_range="now-30d",
        exclude_countries=["United States"],
        result_limit=5,
    )

    assert result["status"] == "ok"
    assert result["aggregation_type"] == "country_terms"
    assert result["aggregation_field"] == "geoip.country_name.keyword"
    assert result["country_buckets"] == [
        {"country": "Iran", "count": 2},
        {"country": "United Kingdom", "count": 1},
    ]
    assert db.search_called is False
    assert db._client.fields[0] == "geoip.country_name.keyword"
