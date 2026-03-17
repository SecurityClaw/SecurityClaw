"""
tests/test_integration_threat_intel_workflow.py — End-to-end threat intel workflow
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, patch


class TestThreatIntelWorkflow:
    """Integration tests for complete threat intelligence workflow."""

    def test_reputation_only_run_skips_rag_engine_initialization(self):
        """Reputation-only threat analysis must not initialize RAGEngine or use the DB."""
        from skills.threat_analyst.logic import run

        mock_llm = Mock()
        mock_llm.chat.return_value = (
            '{"verdict": "TRUE_THREAT", "confidence": 85, '
            '"reasoning": "37.230.117.113 and 82.146.61.17 have elevated abuse history."}'
        )
        context = {
            "db": None,
            "llm": mock_llm,
            "memory": None,
            "config": Mock(),
            "parameters": {
                "question": (
                    "What is the reputation of these IPs? Previously discovered entities from log search: "
                    "IPs: 37.230.117.113, 82.146.61.17"
                )
            },
            "conversation_history": [],
        }

        with patch(
            "core.rag_engine.RAGEngine",
            side_effect=AssertionError("RAGEngine must not be initialized for reputation-only analysis"),
        ):
            with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_get_ip:
                mock_get_ip.side_effect = [
                    {
                        "ip": "37.230.117.113",
                        "abuseipdb": {"abuse_score": 91, "reports": 11},
                        "combined_risk": "HIGH",
                        "queries": ["abuseipdb"],
                    },
                    {
                        "ip": "82.146.61.17",
                        "abuseipdb": {"abuse_score": 76, "reports": 5},
                        "combined_risk": "HIGH",
                        "queries": ["abuseipdb"],
                    },
                ]

                result = run(context)

        assert result["status"] == "ok"
        assert len(result["verdicts"]) == 1
        assert mock_get_ip.call_count == 2

    def test_conversation_history_to_threat_intel_complete_flow(self):
        """Complete workflow: history -> extract IP -> query APIs -> format response."""
        from skills.threat_analyst.logic import run, _analyze_finding, _enrich_with_reputation
        
        # Simulate conversation history with an IP mentioned
        conversation_history = [
            {
                "role": "user",
                "content": "The suspicious IP address 62.60.131.168 connected from Iran"
            },
            {
                "role": "assistant",
                "content": "I found network flows to that IP on port 1194"
            },
            {
                "role": "user",
                "content": "can you pull threat intel on this ip?"  # No IP in this message
            }
        ]
        
        # Mock context
        mock_db = Mock()
        mock_llm = Mock()
        mock_llm.chat.return_value = '{"verdict": "TRUE_THREAT", "confidence": 85}'
        context = {
            "db": mock_db,
            "llm": mock_llm,
            "memory": None,
            "config": Mock(),
            "parameters": {"question": "can you pull threat intel on this ip?"},
            "conversation_history": conversation_history
        }
        
        with patch("core.rag_engine.RAGEngine") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.build_context_string.return_value = "Baseline: normal traffic"
            mock_rag_class.return_value = mock_rag
            
            with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_get_ip:
                mock_get_ip.return_value = {
                    "ip": "62.60.131.168",
                    "abuseipdb": {"abuse_score": 75, "reports": 42},
                    "alienvault": {"reputation": "malicious", "pulses": 5},
                    "virustotal": {"malicious": 3},
                    "combined_risk": "HIGH",
                    "queries": ["abuseipdb", "alienvault", "virustotal"]
                }
                
                # Run threat_analyst
                result = run(context)
                
                # Verify the IP was found and queried
                assert mock_get_ip.called
                mock_get_ip.assert_called_with("62.60.131.168")
                
                # Verify verdict was generated
                assert result["status"] == "ok"
                assert len(result["verdicts"]) > 0

    def test_knn_search_falls_back_to_keyword_search(self):
        """When KNN search fails, system falls back to keyword search."""
        from core.rag_engine import RAGEngine
        
        mock_db = Mock()
        # KNN fails with NMSLIB error
        mock_db.knn_search.side_effect = Exception(
            "Engine [NMSLIB] does not support filters"
        )
        # Keyword search succeeds
        mock_db.search.return_value = [
            {
                "text": "Normal baseline includes HTTP on port 80",
                "category": "network_baseline",
                "source": "documentation"
            }
        ]
        
        mock_llm = Mock()
        mock_llm.embed.return_value = [0.1, 0.2, 0.3]
        
        rag = RAGEngine(db=mock_db, llm=mock_llm)
        
        # Request with category filter
        results = rag.retrieve("Network baseline", category="network_baseline")
        
        # Should still get results from fallback search
        assert len(results) > 0
        assert results[0]["text"] == "Normal baseline includes HTTP on port 80"
        
        # Both searches should have been attempted
        assert mock_db.knn_search.called
        assert mock_db.search.called

    def test_threat_intel_with_multiple_ips_in_history(self):
        """threat_analyst handles multiple IPs in conversation history."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        conversation_history = [
            {"role": "user", "content": "Traffic from 62.60.131.168 and 192.168.0.1"},
            {"role": "user", "content": "Get threat intel"}  # No IPs here
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            # Mock different risks for different IPs
            def get_intel_side_effect(ip):
                if ip == "62.60.131.168":
                    return {
                        "ip": ip,
                        "abuseipdb": {"abuse_score": 75},
                        "combined_risk": "HIGH",
                        "queries": ["abuseipdb"]
                    }
                elif ip == "192.168.0.1":
                    return {
                        "ip": ip,
                        "abuseipdb": {"abuse_score": 0},
                        "combined_risk": "LOW",
                        "queries": ["abuseipdb"]
                    }
            
            mock_intel.side_effect = get_intel_side_effect
            
            # Question has no IP, should extract public IPs from history
            result_string, queried_apis = _enrich_with_reputation("Get threat intel", conversation_history)
            
            # Private IPs (192.168.0.1) should be filtered — only the public IP is checked
            assert mock_intel.call_count == 1
            assert mock_intel.call_args[0][0] == "62.60.131.168"
            
            # Result should mention the public IP
            assert "62.60.131.168" in result_string
            # Private IP should NOT be queried for external reputation
            assert "192.168.0.1" not in result_string
            
            # Should track which APIs were queried
            assert "abuseipdb" in queried_apis

    def test_reputation_data_formatted_for_llm(self):
        """Reputation data should be properly formatted for LLM consumption."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            mock_intel.return_value = {
                "ip": "8.8.8.8",
                "abuseipdb": {
                    "abuse_score": 0,
                    "reports": 0,
                    "is_whitelisted": True
                },
                "alienvault": {
                    "reputation": "clean",
                    "pulses": 0,
                    "tags": []
                },
                "virustotal": {
                    "malicious": 0,
                    "suspicious": 0,
                    "undetected": 71
                },
                "combined_risk": "LOW",
                "queries": ["abuseipdb", "alienvault", "virustotal"]
            }
            
            result_string, queried_apis = _enrich_with_reputation("What about 8.8.8.8?")
            
            # Result should be readable string for LLM
            assert isinstance(result_string, str)
            assert "8.8.8.8" in result_string
            assert "AbuseIPDB" in result_string
            assert "AlienVault" in result_string
            # VirusTotal is only shown if malicious > 0 (optimization)
            # For 8.8.8.8 it's not shown since no malicious detections
            assert "LOW" in result_string  # Risk level should be visible
            assert "abuseipdb" in queried_apis
            assert "alienvault" in queried_apis
            assert "virustotal" in queried_apis

    def test_threat_analyst_private_ip_returns_grounded_no_external_reputation_message(self):
        from skills.threat_analyst.hooks import format_response

        response = format_response(
            "Any threat intel on 192.168.0.16",
            {
                "status": "ok",
                "verdicts": [
                    {
                        "verdict": "UNKNOWN",
                        "confidence": 0,
                        "reasoning": "No external reputation data needed after excluding private/internal IPs.",
                        "_requested_ips": ["192.168.0.16"],
                        "_queried_apis": [],
                    }
                ],
            },
        )

        assert "private/internal IP address" in response
        assert "external threat-intelligence feeds do not apply directly" in response

    def test_no_ips_in_question_or_history(self):
        """When no IPs found anywhere, enrichment returns graceful message."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        history = [
            {"role": "user", "content": "What is normal baseline traffic?"}
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation"):
            # Question and history have no IPs
            result_string, queried_apis = _enrich_with_reputation(
                "Can you explain the baseline?",
                history
            )
            
            # Should return graceful message, not error
            assert isinstance(result_string, str)
            assert "No external reputation data" in result_string or "no IPs" in result_string.lower()
            assert queried_apis == []  # No APIs queried

    def test_threat_analyst_prefers_question_ips_over_history(self):
        """If question has IP, don't search history (avoid stale context)."""
        from skills.threat_analyst.logic import _enrich_with_reputation
        
        history = [
            {"role": "user", "content": "Old IP: 1.2.3.4"}
        ]
        
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_intel:
            mock_intel.return_value = {
                "ip": "8.8.8.8",
                "combined_risk": "LOW",
                "queries": ["abuseipdb"]
            }
            
            # Question has explicit IP - should use that
            result = _enrich_with_reputation("Check 8.8.8.8", history)
            
            # Should only query the question's IP, not history's
            mock_intel.assert_called_once_with("8.8.8.8")


def test_end_to_end_followup_reputation_uses_visible_ips_without_db_queries():
    """Follow-up reputation analysis should use the listed IPs from history and never touch RAG/DB."""
    from core.chat_router.logic import orchestrate_with_supervisor
    from skills.threat_analyst.logic import run as threat_run

    class _Cfg:
        def get(self, section: str, key: str, default=None):
            values = {
                ("chat", "supervisor_max_steps"): 4,
                ("llm", "anti_hallucination_check"): False,
            }
            return values.get((section, key), default)

    class _StrictDB:
        def __getattr__(self, name):
            raise AssertionError(f"DB should not be touched for reputation-only follow-ups: {name}")

    class _UnifiedLLM:
        def __init__(self):
            self.threat_prompts: list[str] = []
            self.final_format_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Follow-up reputation question anchored to entities from the previous answer",
                        "skills": ["threat_analyst"],
                        "parameters": {},
                    }
                )
            if "Evaluate whether the current skill outputs are sufficient" in prompt:
                return json.dumps(
                    {
                        "satisfied": True,
                        "confidence": 0.95,
                        "reasoning": "Threat intelligence verdicts were produced for the requested IPs.",
                        "missing": [],
                    }
                )
            if "Based on these skill execution results" in prompt:
                self.final_format_calls += 1
                return "Eight IP addresses were identified: 8.8.8.8, 1.1.1.1, 203.0.113.5, and five others across 45 connection records."

            self.threat_prompts.append(prompt)
            return json.dumps(
                {
                    "verdict": "TRUE_THREAT",
                    "confidence": 88,
                    "reasoning": (
                        "Eight IP addresses were identified: 8.8.8.8, 1.1.1.1, 203.0.113.5, "
                        "37.230.117.113, 82.146.61.17, 82.202.197.102, 92.63.103.84, and 94.139.250.252."
                    ),
                    "recommended_action": "Block or monitor the listed IPs.",
                }
            )

    class _Runner:
        def __init__(self, llm):
            self.llm = llm
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name != "threat_analyst":
                raise AssertionError(f"Unexpected skill call in reputation follow-up: {skill_name}")

            skill_context = {
                "db": _StrictDB(),
                "llm": self.llm,
                "memory": None,
                "config": Mock(),
                "parameters": context.get("parameters", {}),
                "conversation_history": context.get("conversation_history", []),
            }
            return threat_run(skill_context)

    history = [
        {
            "role": "assistant",
            "content": (
                "Found 200 record(s) matching Russia in the last 30 days window. Countries seen: Russia. "
                "Source IPs: 37.230.117.113, 82.146.61.17, 82.202.197.102, 92.63.103.84, 94.139.250.252. "
                "Earliest: 2026-02-17T07:29:17.210Z. Latest: 2026-02-28T19:16:19.262Z."
            ),
        }
    ]

    llm = _UnifiedLLM()
    runner = _Runner(llm)

    with patch(
        "core.rag_engine.RAGEngine",
        side_effect=AssertionError("RAGEngine must not be initialized for reputation-only follow-ups"),
    ):
        with patch("skills.threat_analyst.reputation_intel.get_ip_reputation") as mock_get_ip:
            mock_get_ip.side_effect = lambda ip: {
                "ip": ip,
                "abuseipdb": {"abuse_score": 80, "reports": 7},
                "combined_risk": "HIGH",
                "queries": ["abuseipdb"],
            }

            out = orchestrate_with_supervisor(
                user_question="What is the reputation of these IPs?",
                available_skills=[
                    {"name": "fields_querier", "description": "Field schema discovery"},
                    {"name": "opensearch_querier", "description": "Direct log search"},
                    {"name": "threat_analyst", "description": "Reputation analysis"},
                ],
                runner=runner,
                llm=llm,
                instruction="You are a SOC assistant.",
                cfg=_Cfg(),
                conversation_history=history,
            )

    assert runner.calls == ["threat_analyst"]
    enriched_question = runner.contexts["threat_analyst"]["parameters"]["question"]
    assert "37.230.117.113" in enriched_question
    assert "82.146.61.17" in enriched_question
    assert "82.202.197.102" in enriched_question
    assert "92.63.103.84" in enriched_question
    assert "94.139.250.252" in enriched_question
    assert mock_get_ip.call_count == 5
    assert out["evaluation"]["satisfied"] is True
    assert llm.final_format_calls == 0
    assert "37.230.117.113" in out["response"]
    assert "82.146.61.17" in out["response"]
    assert "92.63.103.84" in out["response"]
    assert "1.1.1.1" not in out["response"]
    assert "8.8.8.8" not in out["response"]
    assert "203.0.113.5" not in out["response"]
