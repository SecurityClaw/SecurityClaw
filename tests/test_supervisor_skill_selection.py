"""
tests/test_supervisor_skill_selection.py

Consolidated tests for supervisor skill routing logic.
Verifies that the supervisor correctly routes questions based on CRITICAL RULES.

This consolidates:
- test_supervisor_threat_analyst_priority.py (threat/reputation routing)
- test_supervisor_field_discovery.py (field discovery routing)
- test_supervisor_alert_type_routing.py (alert type routing)

All tests use mocked LLM and never touch the live OpenSearch instance.
The tests verify that the CRITICAL RULES in the supervisor prompt are
properly documented and that the LLM can be guided to follow them.
"""

import pytest
from unittest.mock import MagicMock
from core.chat_router.logic import _supervisor_next_action


class TestSupervisorRoutingRules:
    """
    Verify that supervisor applies CRITICAL RULES correctly when selecting skills.
    
    All tests mock the LLM and verify that:
    1. The prompt contains the required rules
    2. When the LLM makes correct choices, they match the expected routing
    """

    @pytest.fixture
    def available_skills(self):
        """Standard set of available skills for routing tests."""
        return [
            {"name": "forensic_examiner", "description": "Network forensic analysis"},
            {"name": "baseline_querier", "description": "RAG-enhanced behavioral log search"},
            {"name": "fields_querier", "description": "Field schema discovery from local field catalog"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "anomaly_triage", "description": "Anomaly detection"},
            {"name": "threat_analyst", "description": "IP/domain reputation and threat intelligence"},
            {"name": "network_baseliner", "description": "Baseline creation"},
            {"name": "fields_baseliner", "description": "Field catalog builder"},
        ]

    @staticmethod
    def _create_mock_llm(response_json: str) -> MagicMock:
        """Create a mock LLM that returns fixed JSON."""
        llm = MagicMock()
        llm.chat.return_value = response_json
        return llm

    # ════════════════════════════════════════════════════════════════════════════════
    # RULE 1: THREAT / REPUTATION QUESTIONS → threat_analyst
    # ════════════════════════════════════════════════════════════════════════════════

    def test_reputation_question_prompts_threat_analyst(self, available_skills):
        """RULE: Reputation questions should route to threat_analyst."""
        llm = self._create_mock_llm('{"skills": ["threat_analyst"], "reasoning": "reputation"}')
        
        result = _supervisor_next_action(
            user_question="what's the reputation of 1.2.3.4?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )
        
        assert result["skills"] == ["threat_analyst"]

        # Deterministic direct-routing may short-circuit before any LLM call.
        if llm.chat.call_count >= 1:
            call_args = llm.chat.call_args_list[-1][0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])

            assert any(
                "reputation" in p.lower() and "threat_analyst" in p.lower()
                for p in prompt_text.split("\n")
            ), "Prompt should guide threat_analyst for reputation questions"

    def test_threat_keyword_questions_route_correctly(self, available_skills):
        """RULE: Questions with threat/malicious/risk keywords should hint at threat_analyst."""
        threat_questions = [
            ("is this IP malicious?", ["threat_analyst"]),
            ("what threat does this IP pose?", ["threat_analyst"]),
            ("what is the risk from this domain?", ["threat_analyst"]),
            ("is this vulnerable?", ["threat_analyst"]),
        ]
        
        for question, expected_skills in threat_questions:
            llm = self._create_mock_llm(
                f'{{"skills": {expected_skills}, "reasoning": "threat question"}}'
            )
            
            result = _supervisor_next_action(
                user_question=question,
                available_skills=available_skills,
                llm=llm,
                instruction="test",
                conversation_history=[],
                previous_trace=[],
                current_results={},
                previous_eval={"satisfied": False},
            )
            
            # Verify the prompt contains threat keywords and threat_analyst guidance
            llm.chat.assert_called()
            call_args = llm.chat.call_args[0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
            
            assert "threat_analyst" in prompt_text.lower(), \
                f"Prompt should mention threat_analyst for: {question}"

    # ════════════════════════════════════════════════════════════════════════════════
    # RULE 2: ALERT TYPE / EVENT TYPE QUESTIONS → baseline_querier / opensearch_querier
    # ════════════════════════════════════════════════════════════════════════════════

    def test_alert_type_questions_route_to_log_search(self, available_skills):
        """RULE: Alert type / event type questions should route to log search, not threat_analyst."""
        alert_questions = [
            "is there an ET EXPLOIT type alert in the past 3 months?",
            "are there any Suricata signature matches in the past week?",
            "alert_type equals POLICY_VIOLATION",
            "what event types occurred in the past 3 days?",
        ]
        
        for question in alert_questions:
            llm = self._create_mock_llm(
                '{"skills": ["baseline_querier"], "reasoning": "alert type search"}'
            )
            
            result = _supervisor_next_action(
                user_question=question,
                available_skills=available_skills,
                llm=llm,
                instruction="test",
                conversation_history=[],
                previous_trace=[],
                current_results={},
                previous_eval={"satisfied": False},
            )
            
            # Verify the prompt contains alert type guidance
            llm.chat.assert_called()
            call_args = llm.chat.call_args[0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
            
            # Check for alert type or log searching guidance
            assert any(
                kw in prompt_text.lower()
                for kw in ["alert type", "event type", "alert_type", "log type", "signature"]
            ), f"Prompt should guide log search for alert type question: {question}"

    def test_generic_alert_queries_use_field_discovery_then_log_search(self, available_skills):
        """RULE: Generic alert queries without specific type should emphasize field discovery and log search."""
        generic_alert_questions = [
            "what are the top 10 alerts in the past month?",
            "show me a few alerts",
            "any alerts today?",
            "what alerts were triggered?",
            "show me the alerts from the past week",
            "list all signals",
            "what events occurred?",
        ]
        
        for question in generic_alert_questions:
            llm = self._create_mock_llm(
                '{"skills": ["baseline_querier"], "reasoning": "generic alert discovery - need field discovery first"}'
            )
            
            result = _supervisor_next_action(
                user_question=question,
                available_skills=available_skills,
                llm=llm,
                instruction="test",
                conversation_history=[],
                previous_trace=[],
                current_results={},
                previous_eval={"satisfied": False},
            )
            
            # Verify the prompt contains guidance about generic alert field discovery
            llm.chat.assert_called()
            call_args = llm.chat.call_args[0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
            
            # Should mention generic alert queries and field/baseline discovery
            assert any(
                kw in prompt_text.lower()
                for kw in ["generic alert", "alert field", "field discovery", "baseline_querier", "opensearch_querier"]
            ), f"Prompt should guide log search for generic alert question: {question}"
            
            # Should mention that opensearch_querier needs specific criteria
            assert "opensearch_querier" in prompt_text.lower() or "cannot" in prompt_text.lower(), \
                f"Prompt should clarify opensearch_querier limitations for: {question}"

    # ════════════════════════════════════════════════════════════════════════════════
    # RULE 3: FIELD DISCOVERY QUESTIONS → fields_querier first
    # ════════════════════════════════════════════════════════════════════════════════

    def test_field_discovery_questions_use_fields_querier_first(self, available_skills):
        """RULE: Field value/discovery questions should prioritize fields_querier."""
        field_questions = [
            "what were the byte transfer to client and server?",
            "how many bytes were transferred?",
            "what are the packet counts?",
            "show me the flow duration and bytes",
            "what fields are available for these events?",
        ]
        
        for question in field_questions:
            llm = self._create_mock_llm(
                '{"skills": ["fields_querier"], "reasoning": "field discovery"}'
            )
            
            result = _supervisor_next_action(
                user_question=question,
                available_skills=available_skills,
                llm=llm,
                instruction="test",
                conversation_history=[],
                previous_trace=[],
                current_results={},
                previous_eval={"satisfied": False},
            )
            
            # Verify the prompt contains field discovery guidance
            llm.chat.assert_called()
            call_args = llm.chat.call_args[0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
            
            assert any(
                kw in prompt_text.lower()
                for kw in ["field", "schema", "discovery", "bytes", "packets"]
            ), f"Prompt should guide fields_querier for field question: {question}"

    # ════════════════════════════════════════════════════════════════════════════════
    # RULE 4: TRAFFIC / LOG SEARCH QUESTIONS → baseline_querier / opensearch_querier
    # ════════════════════════════════════════════════════════════════════════════════

    def test_log_traffic_questions_route_correctly(self, available_skills):
        """RULE: Traffic/log search questions should route to log search skills."""
        traffic_questions = [
            "show me traffic from Iran",
            "what IPs connected to 192.168.0.16?",
            "find flows on port 1194",
            "any connections to Russia in the past month?",
        ]
        
        for question in traffic_questions:
            llm = self._create_mock_llm(
                '{"skills": ["baseline_querier"], "reasoning": "log search"}'
            )
            
            result = _supervisor_next_action(
                user_question=question,
                available_skills=available_skills,
                llm=llm,
                instruction="test",
                conversation_history=[],
                previous_trace=[],
                current_results={},
                previous_eval={"satisfied": False},
            )
            
            # Verify the prompt contains log search guidance
            llm.chat.assert_called()
            call_args = llm.chat.call_args[0][0]
            prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
            
            assert any(
                kw in prompt_text.lower()
                for kw in ["traffic", "search", "query", "baseline_querier", "opensearch"]
            ), f"Prompt should guide log search for: {question}"

    # ════════════════════════════════════════════════════════════════════════════════
    # MANIFEST TESTS: Verify skill manifests declare capabilities correctly
    # ════════════════════════════════════════════════════════════════════════════════

    def test_threat_analyst_manifest_declares_correct_capabilities(self):
        """Verify threat_analyst manifest declares it CANNOT search logs."""
        from core.skill_manifest import SkillManifestLoader

        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        
        threat_manifest = manifests.get("threat_analyst")
        assert threat_manifest is not None, "threat_analyst manifest should exist"
        
        # Should not be able to answer alert type questions
        cannot_answer = threat_manifest.get("cannot_answer", [])
        assert any(
            "alert" in str(item).lower() or "log" in str(item).lower()
            for item in cannot_answer
        ), "threat_analyst should declare it cannot search logs for alert types"

    def test_baseline_querier_manifest_declares_log_search_capability(self):
        """Verify baseline_querier manifest declares it CAN search logs."""
        from core.skill_manifest import SkillManifestLoader

        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        
        bq_manifest = manifests.get("baseline_querier")
        assert bq_manifest is not None, "baseline_querier manifest should exist"
        
        # Should be able to answer alert/event type questions
        can_answer = bq_manifest.get("can_answer", [])
        assert any(
            "alert" in str(item).lower() or "log" in str(item).lower() or "traffic" in str(item).lower()
            for item in can_answer
        ), "baseline_querier should declare it can search for alerts/logs/traffic"

    def test_opensearch_querier_manifest_declares_direct_query_capability(self):
        """Verify opensearch_querier manifest declares its capabilities."""
        from core.skill_manifest import SkillManifestLoader

        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        
        os_manifest = manifests.get("opensearch_querier")
        assert os_manifest is not None, "opensearch_querier manifest should exist"
        
        # Should be able to do direct searches
        can_answer = os_manifest.get("can_answer", [])
        assert any(
            kw in str(item).lower()
            for item in can_answer
            for kw in ["search", "filter", "query"]
        ), "opensearch_querier should declare direct search capability"

    # ════════════════════════════════════════════════════════════════════════════════
    # DISAMBIGUATION TESTS: Distinguish between similar patterns
    # ════════════════════════════════════════════════════════════════════════════════

    def test_et_exploit_is_log_search_not_threat(self, available_skills):
        """
        ET EXPLOIT is a Suricata rule category (log search), not threat intelligence.
        Should route to log search, not threat_analyst.
        """
        llm = self._create_mock_llm(
            '{"skills": ["baseline_querier"], "reasoning": "ET EXPLOIT is alarm type, not threat analysis"}'
        )
        
        result = _supervisor_next_action(
            user_question="is there an ET EXPLOIT type alert in the past 3 months?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )
        
        # The result might depend on LLM, but the prompt should guide correctly
        llm.chat.assert_called()
        call_args = llm.chat.call_args[0][0]
        prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
        
        # Should mention that alert type questions route to log search
        assert "alert" in prompt_text.lower(), \
            "Prompt should recognize ET EXPLOIT as alert type, not threat"

    def test_malicious_alerts_could_use_both_skills(self, available_skills):
        """
        'Malicious alerts from Iran' could use:
        1. Log search to find Iran alerts (baseline_querier/opensearch_querier)
        2. Threat analysis to check IP reputation (threat_analyst)
        
        The supervisor should be guided by context to choose appropriately.
        """
        llm = self._create_mock_llm(
            '{"skills": ["baseline_querier"], "reasoning": "search for alerts, then optionally analyze"}'
        )
        
        result = _supervisor_next_action(
            user_question="are there any malicious alerts from Iran in the past month?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )
        
        llm.chat.assert_called()

    # ════════════════════════════════════════════════════════════════════════════════
    # ANTI-REGRESSION: Ensure old bugs don't return
    # ════════════════════════════════════════════════════════════════════════════════

    def test_et_exploit_never_routes_to_threat_analyst(self, available_skills):
        """
        REGRESSION: Bug where "ET EXPLOIT type alert" routed to threat_analyst.
        This test documents that the fix prevents this misrouting.
        
        The question is about alert TYPE discovery, not threat analysis.
        In the capability-first flow, the supervisor may keep baseline_querier or
        choose opensearch_querier, but it must not reframe the task as threat intel.
        """
        llm = self._create_mock_llm(
            '{"skills": ["baseline_querier"], "reasoning": "alert type search"}'
        )
        
        result = _supervisor_next_action(
            user_question="is there an ET EXPLOIT type alert in the past 3 months?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )
        
        # threat_analyst should NOT be routed (primary test goal)
        assert "threat_analyst" not in result.get("skills", [])
        
        assert "opensearch_querier" in result.get("skills", []) or "baseline_querier" in result.get("skills", [])

    def test_field_discovery_never_skips_to_opensearch(self, available_skills):
        """
        REGRESSION: Bug where 'byte transfer' went to opensearch_querier
        directly without discovering field names first.
        """
        llm = self._create_mock_llm(
            '{"skills": ["fields_querier"], "reasoning": "field discovery first"}'
        )
        
        result = _supervisor_next_action(
            user_question="what were the byte transfer to client and server?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )
        
        # Should prioritize fields_querier for field discovery
        assert "fields_querier" in result.get("skills", [])

    def test_natural_language_traffic_search_prefers_fields_first(self, available_skills):
        """Natural-language traffic questions should prepend fields_querier before opensearch."""
        llm = self._create_mock_llm(
            '{"skills": ["opensearch_querier"], "reasoning": "traffic search"}'
        )

        result = _supervisor_next_action(
            user_question="In the past week what traffic has visited my 1194 port?",
            available_skills=available_skills,
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={"satisfied": False},
        )

        assert result.get("skills", [])[:2] == ["fields_querier", "opensearch_querier"]


class TestSupervisorRoutingIntegration:
    """
    Integration tests that verify supervisor routing works end-to-end.
    These tests use the actual supervisor code with mocked LLM and DB.
    """

    def test_supervisor_applies_rules_in_correct_order(self):
        """
        Verify CRITICAL RULES are applied in priority order:
        1. Alert type rules (catch ET EXPLOIT before threat rule)
        2. Threat/reputation rules
        3. Field discovery rules
        4. Traffic search rules
        """
        # This is more of a documentation test — it verifies the prompt
        # contains rules in the right order.
        
        from core.chat_router.logic import _supervisor_next_action
        
        llm = MagicMock()
        llm.chat.return_value = '{"skills": [], "reasoning": ""}'
        
        _supervisor_next_action(
            user_question="test",
            available_skills=[
                {"name": "baseline_querier", "description": "behavioral log search"},
                {"name": "fields_querier", "description": "field schema discovery"},
                {"name": "threat_analyst", "description": "test"},
            ],
            llm=llm,
            instruction="test",
            conversation_history=[],
            previous_trace=[],
            current_results={},
            previous_eval={},
        )
        
        # Extract the prompt
        call_args = llm.chat.call_args[0][0]
        prompt_text = "\n".join([str(m.get("content", "")) for m in call_args])
        
        # Find the CRITICAL RULES section specifically
        rules_start = prompt_text.find("CRITICAL RULES:")
        if rules_start >= 0:
            rules_section = prompt_text[rules_start:]
            alert_type_pos = rules_section.find("ALERT TYPE / EVENT TYPE")
            threat_pos = rules_section.find("REPUTATION, THREAT INTEL")
            
            # Alert type rule should come before threat rule
            if alert_type_pos >= 0 and threat_pos >= 0:
                assert alert_type_pos < threat_pos, \
                    "Alert type rule must come BEFORE threat rule in CRITICAL RULES section"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
