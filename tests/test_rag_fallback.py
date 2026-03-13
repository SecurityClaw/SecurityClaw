"""
tests/test_rag_fallback.py — Tests for RAG KNN fallback and threat_analyst history extraction
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch


class TestRAGKNNFallback:
    """Test RAGEngine fallback from KNN to keyword search when KNN fails."""

    def test_retrieve_falls_back_to_keyword_when_knn_fails(self):
        """When KNN search fails, should fall back to keyword search."""
        from core.rag_engine import RAGEngine
        
        # Mock db that fails on knn_search but works on search
        mock_db = Mock()
        mock_db.knn_search.side_effect = Exception("Engine [NMSLIB] does not support filters")
        mock_db.search.return_value = [
            {
                "text": "Normal baseline traffic includes HTTP on port 80",
                "source": "network.baseline",
                "category": "field_documentation"
            }
        ]
        
        # Mock LLM for embedding
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2, 0.3]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        
        # Should fall back to keyword search
        results = rag.retrieve("What is normal traffic?", k=5, category="field_documentation")
        
        # Should have called search after knn_search failed
        assert mock_db.knn_search.called
        assert mock_db.search.called
        assert len(results) > 0
        assert results[0]["text"] == "Normal baseline traffic includes HTTP on port 80"

    def test_retrieve_category_filter_in_fallback(self):
        """When falling back to keyword search, category filter should apply."""
        from core.rag_engine import RAGEngine
        
        mock_db = Mock()
        mock_db.knn_search.side_effect = Exception("KNN not supported")
        mock_db.search.return_value = [
            {"text": "Field doc", "category": "field_documentation"}
        ]
        
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        results = rag.retrieve("What fields?", category="field_documentation")
        
        # Verify search was called with category filter
        assert mock_db.search.called
        call_args = mock_db.search.call_args
        query_dict = call_args[1]["query"]
        
        # Should have filter for category
        assert "bool" in query_dict["query"]
        assert "filter" in query_dict["query"]["bool"]

    def test_retrieve_returns_empty_when_all_fail(self):
        """When both KNN and keyword search fail, return empty list."""
        from core.rag_engine import RAGEngine
        
        mock_db = Mock()
        mock_db.knn_search.side_effect = Exception("KNN failed")
        mock_db.search.side_effect = Exception("Search also failed")
        
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        results = rag.retrieve("Query")
        
        assert results == []

    def test_build_context_string_with_fallback(self):
        """build_context_string should work with fallback search."""
        from core.rag_engine import RAGEngine
        
        mock_db = Mock()
        mock_db.knn_search.side_effect = Exception("KNN failed")
        mock_db.search.return_value = [
            {
                "text": "Port 22 is typically used for SSH",
                "source": "baseline",
                "category": "field_documentation"
            }
        ]
        
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        context = rag.build_context_string("Port information")
        
        assert "Port 22" in context
        assert "baseline" in context


class TestThreatAnalystHistoryExtraction:
    """Test threat_analyst extracts IPs from conversation history."""

    def test_enrich_extracts_ip_from_history(self):
        """_enrich_with_reputation should extract IP from conversation history."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        # Conversation history with an IP address
        history = [
            {"role": "user", "content": "The IP address from Iran was 62.60.131.168"},
            {"role": "assistant", "content": "I found something unusual about that IP"},
            {"role": "user", "content": "can you pull threat intel on this ip"}  # No IP in this message
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            mock_intel.return_value = {
                "ip": "62.60.131.168",
                "combined_risk": "HIGH",
                "queries": ["abuseipdb"]
            }
            
            # This question has no IP, but history does
            result_string, queried_apis = _enrich_with_reputation("can you pull threat intel on this ip", history)
            
            # Should have extracted IP from history and queried it
            assert mock_intel.called
            mock_intel.assert_called_with("62.60.131.168")
            assert "62.60.131.168" in result_string
            assert "HIGH" in result_string
            assert "abuseipdb" in queried_apis

    def test_enrich_extracts_domain_from_history(self):
        """_enrich_with_reputation should extract domain from history."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        history = [
            {"role": "user", "content": "Traffic to malware.example.com detected"},
            {"role": "user", "content": "What threat intel is available?"}  # No domain here
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_domain_reputation") as mock_intel:
            mock_intel.return_value = {
                "domain": "malware.example.com",
                "combined_risk": "CRITICAL",
                "queries": ["virustotal"]
            }
            
            result_string, queried_apis = _enrich_with_reputation("What threat intel is available?", history)
            
            assert mock_intel.called
            mock_intel.assert_called_with("malware.example.com")
            assert "malware.example.com" in result_string
            assert "CRITICAL" in result_string
            assert "virustotal" in queried_apis

    def test_enrich_prefers_question_over_history(self):
        """If IP is in question, use it — but private IPs are always filtered since
        they have no external threat reputation regardless of where they appear."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        history = [
            {"role": "user", "content": "Old IP was 1.2.3.4"}
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            mock_intel.return_value = {
                "ip": "1.2.3.4",
                "combined_risk": "LOW",
                "queries": ["abuseipdb"]
            }
            
            # Question has a private IP explicitly — should be filtered (no external reputation for RFC-1918)
            result = _enrich_with_reputation("Check 192.168.1.1", history)
            
            # Private IP in question is filtered; history IP 1.2.3.4 is not in the finding text
            # so mock should not be called at all
            mock_intel.assert_not_called()

    def test_enrich_no_error_when_no_history(self):
        """_enrich_with_reputation should work fine with None history."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation"):
            # Should not crash with None history
            result_string, queried_apis = _enrich_with_reputation("No IP here", None)
            
            assert isinstance(result_string, str)
            assert isinstance(queried_apis, list)

    def test_enrich_excludes_private_ips_when_question_requests_public_only(self):
        """Private/internal IPs should be dropped for follow-up reputation questions that exclude them."""
        from skills.threat_analyst.logic import _enrich_with_reputation

        history = [
            {
                "role": "assistant",
                "content": "Source/destination IPs: 192.168.0.156, 37.230.117.113, 82.146.61.17.",
            }
        ]

        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            mock_intel.side_effect = [
                {"ip": "37.230.117.113", "combined_risk": "HIGH", "queries": ["abuseipdb"]},
                {"ip": "82.146.61.17", "combined_risk": "MEDIUM", "queries": ["abuseipdb"]},
            ]

            result_string, queried_apis = _enrich_with_reputation(
                "Aside from the private IPs, what is the reputation of the others?",
                history,
            )

            looked_up = [call.args[0] for call in mock_intel.call_args_list]
            assert looked_up == ["37.230.117.113", "82.146.61.17"]
            assert "192.168.0.156" not in result_string
            assert "abuseipdb" in queried_apis

    def test_threat_analyst_passes_history_to_enrich(self):
        """threat_analyst.run() should pass conversation_history to _enrich_with_reputation."""
        from skills.threat_analyst.logic import _analyze_finding
        
        mock_rag = Mock()
        mock_rag.build_context_string.return_value = "Baseline context"
        
        mock_llm = Mock()
        mock_llm.chat.return_value = '{"verdict": "TRUE_THREAT", "confidence": 80}'
        
        history = [
            {"role": "user", "content": "Incident at 62.60.131.168"}
        ]
        
        with patch("skills.threat_analyst.logic._enrich_with_reputation") as mock_enrich:
            mock_enrich.return_value = ("IP: 62.60.131.168 - Risk: HIGH", ["abuseipdb"])
            
            result = _analyze_finding(
                "can you pull threat intel",
                "instruction",
                mock_rag,
                mock_llm,
                history
            )
            
            # Should have passed history to _enrich_with_reputation
            assert mock_enrich.called
            call_args = mock_enrich.call_args
            assert call_args[0][1] == history  # Second arg is history


class TestRAGBuildContextWithFallback:
    """Test build_context_string works with KNN fallback."""

    def test_build_context_returns_results_on_fallback(self):
        """build_context_string should format results even when from fallback search."""
        from core.rag_engine import RAGEngine
        
        mock_db = Mock()
        mock_db.knn_search.side_effect = Exception("KNN not supported")
        mock_db.search.return_value = [
            {
                "text": "Normal DNS queries on port 53",
                "source": "baseline",
                "category": "network_baseline"
            },
            {
                "text": "HTTP traffic on port 80 is expected",
                "source": "baseline",
                "category": "network_baseline"
            }
        ]
        
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        context = rag.build_context_string("Traffic on standard ports")
        
        # Should have both results formatted
        assert "Normal DNS" in context
        assert "HTTP traffic" in context
        assert "network_baseline" in context
        assert "1." in context  # Should be numbered
        assert "2." in context
