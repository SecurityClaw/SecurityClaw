"""
Comprehensive tests for intelligent query repair with learning and retries.

Tests will show:
1. Memory records successful fixes
2. Retries with exponential backoff work
3. LLM repairs queries correctly
4. System is data-agnostic (learns fields)
5. No silent failures (errors are caught and reported)
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from core.query_repair_memory import QueryRepairMemory, get_memory
import core.query_repair_memory as query_repair_memory_module
from core.query_repair import IntelligentQueryRepair, QueryRepairStrategy


class TestQueryRepairMemory:
    """Test the persistent learning system."""
    
    def test_memory_records_and_retrieves_fixes(self):
        """Memory should record and retrieve successful fixes."""
        memory = QueryRepairMemory()
        
        error = "parsing_exception: something wrong"
        original = {"query": {"bool": {"must": []}}}
        fixed = {"query": {"bool": {"must": [{"term": {"field": "value"}}]}}}
        
        memory.record_error_fix(error, original, fixed)
        
        retrieved = memory.get_known_fix(error)
        assert retrieved is not None
        assert retrieved["fixed"] == fixed
    
    def test_memory_learns_field_types(self):
        """Memory should learn and recall field types from mappings."""
        memory = QueryRepairMemory()
        
        memory.record_field_type("source.ip", "ip")
        memory.record_field_type("destination.port", "integer")
        
        assert memory.get_field_type("source.ip") == "ip"
        assert memory.get_field_type("destination.port") == "integer"
        assert memory.get_field_type("unknown.field") is None
    
    def test_memory_learns_from_mappings(self):
        """Memory should extract field types from index mappings."""
        memory = QueryRepairMemory()
        
        mapping = {
            "properties": {
                "source": {
                    "properties": {
                        "ip": {"type": "ip"},
                        "port": {"type": "integer"}
                    }
                },
                "geoip": {
                    "properties": {
                        "country_code2": {"type": "keyword"}
                    }
                }
            }
        }
        
        memory.learn_from_mapping(mapping)
        
        assert memory.get_field_type("source.ip") == "ip"
        assert memory.get_field_type("source.port") == "integer"
        assert memory.get_field_type("geoip.country_code2") == "keyword"

    def test_memory_compacts_repairs_to_recent_limit(self, tmp_path):
        memory_file = tmp_path / "query_repair_memory.json"

        with patch.object(query_repair_memory_module, "MEMORY_FILE", memory_file):
            with patch.object(query_repair_memory_module.Config, "get", side_effect=lambda section, key, default=None: 2 if key == "query_repair_max_repairs" else default):
                memory = QueryRepairMemory()
                memory.record_error_fix("error-one", {"query": 1}, {"fixed": 1})
                memory.record_error_fix("error-two", {"query": 2}, {"fixed": 2})
                memory.record_error_fix("error-three", {"query": 3}, {"fixed": 3})

                reloaded = QueryRepairMemory()

        assert len(reloaded.repairs) == 2
        assert reloaded.get_known_fix("error-three") is not None
        assert reloaded.get_known_fix("error-two") is not None

    def test_memory_compacts_field_types_to_limit(self, tmp_path):
        memory_file = tmp_path / "query_repair_memory.json"

        with patch.object(query_repair_memory_module, "MEMORY_FILE", memory_file):
            with patch.object(query_repair_memory_module.Config, "get", side_effect=lambda section, key, default=None: 2 if key == "query_repair_max_field_types" else default):
                memory = QueryRepairMemory()
                memory.record_field_type("field.one", "keyword")
                memory.record_field_type("field.two", "ip")
                memory.record_field_type("field.three", "date")

        assert memory.get_field_type("field.one") is None
        assert memory.get_field_type("field.two") == "ip"
        assert memory.get_field_type("field.three") == "date"


class TestQueryRepairStrategy:
    """Test individual repair strategies."""
    
    def test_python_fix_converts_range_with_strings_to_match(self):
        """Range queries with string values should become match queries."""
        query = {
            "query": {
                "bool": {
                    "filter": {"range": {"country": {"gte": "Iran"}}}
                }
            }
        }
        
        fixed = QueryRepairStrategy.apply_python_fix(query)
        
        # Should convert to match query
        assert "range" not in json.dumps(fixed)
        assert "match" in json.dumps(fixed)
    
    def test_python_fix_ensures_arrays(self):
        """must/should/filter should be arrays."""
        query = {
            "query": {
                "bool": {
                    "must": {"term": {"field": "value"}},
                    "filter": {"range": {"date": {"gte": "2026-01-01"}}}
                }
            }
        }
        
        fixed = QueryRepairStrategy.apply_python_fix(query)
        
        assert isinstance(fixed["query"]["bool"]["must"], list)
        assert isinstance(fixed["query"]["bool"]["filter"], list)

    def test_python_fix_moves_size_out_of_bool_clause(self):
        """OpenSearch rejects size inside bool; it must be lifted to the query root."""
        query = {
            "query": {
                "bool": {
                    "must": [{"term": {"source.ip": "1.1.1.1"}}],
                    "size": 200,
                }
            }
        }

        fixed = QueryRepairStrategy.apply_python_fix(query)

        assert fixed["size"] == 200
        assert "size" not in fixed["query"]["bool"]

    def test_python_fix_removes_placeholder_timestamp_terms(self):
        """Bogus placeholder timestamp clauses should be dropped deterministically."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source.ip": "1.1.1.1"}},
                        {"term": {"@timestamp": "custom"}},
                    ]
                }
            }
        }

        fixed = QueryRepairStrategy.apply_python_fix(query)

        must_clauses = fixed["query"]["bool"]["must"]
        assert must_clauses == [{"term": {"source.ip": "1.1.1.1"}}]

    def test_python_fix_preserves_valid_timestamp_range(self):
        """Valid @timestamp range filters must not be rewritten into match queries."""
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"@timestamp": {"gte": "now-24h"}}}
                    ]
                }
            }
        }

        fixed = QueryRepairStrategy.apply_python_fix(query)

        assert fixed["query"]["bool"]["filter"] == [
            {"range": {"@timestamp": {"gte": "now-24h"}}}
        ]
    
    def test_llm_fix_with_known_pattern(self):
        """Should return known fix if pattern was seen before."""
        mock_llm = MagicMock()
        memory = QueryRepairMemory()
        
        error = "parsing_exception"
        original = {"query": {"bool": {"should": "invalid"}}}
        fixed = {"query": {"bool": {"should": []}}}
        
        # Record a known fix
        memory.record_error_fix(error, original, fixed)
        
        # Mock get_memory to return our memory
        with patch('core.query_repair.get_memory', return_value=memory):
            result = QueryRepairStrategy.apply_llm_fix(original, error, mock_llm)
        
        # Should return the known fix
        assert result == fixed
        # Should NOT call LLM since we had a known fix
        mock_llm.complete.assert_not_called()


class TestIntelligentQueryRepair:
    """Test the main repair executor with retries."""
    
    def test_repair_succeeds_on_first_try(self):
        """If query works, should return immediately."""
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        expected_results = [{"_id": "1", "field": "value"}]
        mock_db.search.return_value = expected_results
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        query = {"query": {"bool": {"must": []}}}
        
        success, results, message = repair.repair_and_retry("test-index", query)
        
        assert success is True
        assert results == expected_results
        assert mock_db.search.call_count == 1
        assert mock_llm.complete.call_count == 0  # LLM not needed
    
    def test_repair_retries_on_malformed_error(self):
        """Should retry with fixes if QueryMalformedException is raised."""
        from core.db_connector import QueryMalformedException
        
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        # First call fails, second succeeds
        query_error = QueryMalformedException(
            "test-index",
            {"query": {"bool": {"should": "invalid"}}},
            "parsing_exception"
        )
        expected_results = [{"_id": "1"}]
        mock_db.search.side_effect = [query_error, expected_results]
        
        # LLM returns a fixed query
        mock_llm.complete.return_value = json.dumps({
            "query": {"bool": {"should": [{"match": {"field": "value"}}]}}
        })
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        original = {"query": {"bool": {"should": "invalid"}}}
        
        success, results, message = repair.repair_and_retry("test-index", original)
        
        assert success is True
        assert results == expected_results
        assert mock_db.search.call_count == 2  # First failed, second succeeded
    
    def test_repair_fails_after_max_retries(self):
        """Should fail if max retries exceeded."""
        from core.db_connector import QueryMalformedException
        
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        # Always fail
        query_error = QueryMalformedException(
            "test-index",
            {"query": {"bool": {"should": "invalid"}}},
            "parsing_exception"
        )
        mock_db.search.side_effect = query_error
        
        # LLM always fails to return valid JSON
        mock_llm.complete.return_value = "This is not JSON"
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        original = {"query": {"bool": {"should": "invalid"}}}
        
        success, results, message = repair.repair_and_retry("test-index", original)
        
        assert success is False
        assert results is None
        # The repair loop may stop early once it detects identical failures repeating.
        assert mock_db.search.call_count > 1
        assert mock_db.search.call_count <= repair.max_retries + 1
        assert "Repeated identical malformed query failure" in message or "Failed after" in message
    
    def test_repair_with_progressively_detailed_prompts(self):
        """Each LLM attempt should have more detailed instructions."""
        from core.db_connector import QueryMalformedException
        
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        # Always fail so LLM keeps being called
        query_error = QueryMalformedException(
            "test-index",
            {"query": {"bool": {"should": "invalid"}}},
            "parsing_exception"
        )
        mock_db.search.side_effect = query_error
        mock_llm.complete.return_value = "invalid"  # Always fail
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        original = {"query": {"bool": {"should": "invalid"}}}
        
        success, results, message = repair.repair_and_retry("test-index", original)
        
        # LLM should have been called multiple times with different prompts
        assert mock_llm.complete.call_count > 0
        
        # Each call should have different content (increasingly detailed)
        calls = mock_llm.complete.call_args_list
        prompts = [call[0][0] for call in calls]
        
        # Prompts should get progressively more specific
        assert len(prompts[0]) < len(prompts[-1])  # Later prompts have more detail

    def test_repair_stops_repeating_identical_llm_payloads(self):
        """The loop should not burn all retries on the same repaired payload."""
        from core.db_connector import QueryMalformedException

        mock_db = MagicMock()
        mock_llm = MagicMock()

        malformed = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source.ip": "1.1.1.1"}},
                        {"term": {"@timestamp": "custom"}},
                    ],
                    "size": 200,
                }
            }
        }

        query_error = QueryMalformedException(
            "logstash*",
            malformed,
            "RequestError(400, 'x_content_parse_exception', '[1:94] [bool] unknown field [size]')",
        )
        mock_db.search.side_effect = query_error
        mock_llm.complete.return_value = json.dumps(malformed)

        repair = IntelligentQueryRepair(mock_db, mock_llm)

        success, results, message = repair.repair_and_retry("logstash*", malformed)

        assert success is False
        assert results is None
        assert mock_llm.complete.call_count < repair.max_retries
        assert "Repeated identical malformed query failure" in message or "Unable" in message


class TestEndToEndRepair:
    """End-to-end tests showing real repair scenarios."""
    
    def test_iran_country_search_repair(self):
        """
        Real scenario: Searching for "Iran" in country field.
        LLM creates query with range instead of match.
        """
        from core.db_connector import QueryMalformedException
        
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        # Original query fails (range with string)
        malformed = {
            "query": {
                "bool": {
                    "filter": {
                        "range": {
                            "geoip.country_code2.keyword": {"gte": "Iran"}
                        }
                    }
                }
            }
        }
        
        query_error = QueryMalformedException(
            "logstash*", malformed, "For input string: \"Iran\""
        )
        
        # Python fix should convert range to match
        # Then retry succeeds
        results = [{"_id": "1", "geoip": {"country_code2": "IR"}}]
        mock_db.search.side_effect = [query_error, results]
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        success, found, message = repair.repair_and_retry("logstash*", malformed)
        
        assert success is True
        assert len(found) == 1
        assert "Iran" not in str(found[0]) or found[0]["geoip"]["country_code2"] == "IR"
    
    def test_protocol_search_repair(self):
        """Protocol search (TCP, UDP) should be handled correctly."""
        from core.db_connector import QueryMalformedException
        
        mock_db = MagicMock()
        mock_llm = MagicMock()
        
        # Query with protocol search
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {"protocol": "TCP"}}]
                }
            }
        }
        
        results = [{"_id": "1", "protocol": "tcp"}]
        mock_db.search.return_value = results
        
        repair = IntelligentQueryRepair(mock_db, mock_llm)
        success, found, message = repair.repair_and_retry("logstash*", query)
        
        # Should work without needing LLM
        assert success is True
        assert mock_llm.complete.call_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
