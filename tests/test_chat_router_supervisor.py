from __future__ import annotations

import json
from unittest.mock import MagicMock

from core.chat_router.logic import (
    _ground_supervisor_question_with_llm,
    _supervisor_next_action,
    decide_node,
    execute_skill_workflow,
    orchestrate_with_supervisor,
    route_question,
)


class _Cfg:
    def get(self, section: str, key: str, default=None):
        values = {
            ("chat", "supervisor_max_steps"): 4,
            ("llm", "anti_hallucination_check"): False,
        }
        return values.get((section, key), default)


class _RunnerStub:
    def __init__(self):
        self.calls: list[str] = []

    def _build_context(self):
        return {}

    def dispatch(self, skill_name: str, context: dict):
        self.calls.append(skill_name)
        if skill_name == "opensearch_querier":
            # First call returns no results to allow supervisor to continue
            # Second call (if context has 'retry' flag) returns results
            if len([c for c in self.calls if c == "opensearch_querier"]) == 1:
                return {
                    "status": "ok",
                    "results": [],
                    "results_count": 0,
                    "countries": [],
                    "ports": [],
                }
            # Subsequent calls return results
            return {
                "status": "ok",
                "results": [
                    {
                        "source.ip": "62.60.131.168",
                        "destination.ip": "192.168.0.16",
                        "destination.port": 1194,
                        "geoip.country_code2": "IR",
                        "@timestamp": "2026-02-13T10:22:10.224Z",
                    }
                ],
                "results_count": 1,
                "countries": ["Iran"],
                "ports": ["1194"],
            }
        if skill_name == "threat_analyst":
            return {
                "status": "ok",
                "verdicts": [
                    {
                        "verdict": "TRUE_THREAT",
                        "confidence": 84,
                        "reasoning": "Abuse history and recurring probe pattern indicate malicious behavior.",
                    }
                ],
            }
        return {"status": "ok"}


class _SupervisorLLM:
    def __init__(self):
        self.next_action_calls = 0
        self.eval_calls = 0

    def chat(self, messages: list[dict]):
        prompt = messages[-1].get("content", "")

        if "SOC supervisor orchestrator" in prompt:
            self.next_action_calls += 1
            if self.next_action_calls == 1:
                return json.dumps(
                    {
                        "reasoning": "Need traffic evidence first.",
                        "skills": ["opensearch_querier"],
                        "parameters": {},
                    }
                )
            return json.dumps(
                {
                    "reasoning": "Need threat reputation after evidence.",
                    "skills": ["threat_analyst"],
                    "parameters": {},
                }
            )

        if "Evaluate whether the current skill outputs are sufficient" in prompt:
            self.eval_calls += 1
            if self.eval_calls == 1:
                return json.dumps(
                    {
                        "satisfied": False,
                        "confidence": 0.5,
                        "reasoning": "Need threat confidence to answer fully.",
                        "missing": ["threat score"],
                    }
                )
            return json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.9,
                    "reasoning": "Now sufficient with evidence and threat verdict.",
                    "missing": [],
                }
            )

        if "Based on these skill execution results" in prompt:
            return "Traffic is from Iran and threat scoring indicates elevated risk."

        return json.dumps({"response": "ok"})


def test_supervisor_orchestrator_runs_multiple_skill_rounds_until_satisfied():
    llm = _SupervisorLLM()
    runner = _RunnerStub()
    available_skills = [
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
        {"name": "forensic_examiner", "description": "Timeline reconstruction"},
    ]

    out = orchestrate_with_supervisor(
        user_question="What countries is this traffic coming from and what is their threat score?",
        available_skills=available_skills,
        runner=runner,
        llm=llm,
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[{"role": "assistant", "content": "Earlier we saw Iran traffic to 192.168.0.16:1194"}],
    )

    assert "response" in out
    assert len(out.get("trace", [])) >= 2
    assert out.get("evaluation", {}).get("satisfied") is True
    assert "opensearch_querier" in out.get("skill_results", {})
    assert "threat_analyst" in out.get("skill_results", {})
    assert runner.calls[:2] == ["opensearch_querier", "threat_analyst"]


def test_route_question_chains_field_discovery_into_opensearch_for_alert_search():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need field discovery first for ET POLICY alerts.",
                    "skills": ["fields_querier"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
    ]

    result = route_question(
        user_question="check for ET POLICY alerts and their ips",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier"]


def test_route_question_shortcuts_explicit_private_ip_threat_intel_to_threat_analyst():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            raise AssertionError("LLM should not be called for direct threat-intel IP routing")

    result = route_question(
        user_question="Any threat intel on 192.168.0.16",
        available_skills=[
            {"name": "geoip_lookup", "description": "GeoIP enrichment"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["threat_analyst"]
    assert result["parameters"]["question"] == "Any threat intel on 192.168.0.16"


def test_route_question_prepends_fields_for_natural_language_port_search():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Port traffic search.",
                    "skills": ["opensearch_querier"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
    ]

    result = route_question(
        user_question="In the past week what traffic has visited my 1194 port?",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier"]


def test_route_question_prepends_fingerprint_prerequisites():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Direct IP fingerprint request.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"ip": "192.168.0.16"},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
    ]

    result = route_question(
        user_question="fingerprint 192.168.0.16",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]


def test_route_question_fallback_path_prepends_fingerprint_prerequisites():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return "planner text before json {\"reasoning\": \"Direct IP fingerprint request.\", \"skills\": [\"ip_fingerprinter\"], \"parameters\": {\"ip\": \"192.168.0.16\"}} trailing text"

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
    ]

    result = route_question(
        user_question="fingerprint 192.168.0.16",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]


def test_supervisor_next_action_prepends_fingerprint_prerequisites():
    class _SupervisorFingerprintLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Use the fingerprinting skill.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"ip": "192.168.0.16"},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
    ]

    decision = _supervisor_next_action(
        user_question="fingerprint 192.168.0.16",
        available_skills=available_skills,
        llm=_SupervisorFingerprintLLM(),
        instruction="test",
        conversation_history=[],
        previous_trace=[],
        current_results={},
        previous_eval={},
    )

    assert set(decision["skills"]) == {"fields_querier", "opensearch_querier", "ip_fingerprinter"}
    assert decision["skills"][-1] == "ip_fingerprinter"


def test_decide_node_prepends_fingerprint_prerequisites_for_chat_flow():
    class _SupervisorFingerprintLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Use the fingerprinting skill.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"ip": "192.168.0.16"},
                }
            )

    state = {
        "user_question": "fingerprint 192.168.0.16",
        "messages": [],
        "skill_results": {},
        "previously_run_skills": [],
        "step_count": 0,
        "max_steps": 4,
        "evaluation": {},
        "trace": [],
    }
    config = {
        "configurable": {
            "available_skills": [
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
                {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
            ],
            "llm": _SupervisorFingerprintLLM(),
            "instruction": "test",
        }
    }

    decision = decide_node(state, config)

    assert decision["skill_plan"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]


def test_supervisor_next_action_anchors_fingerprint_followup_to_previous_user_ip():
    class _FingerprintFollowupLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "Analyze ONLY the current user question below." in prompt:
                return json.dumps(
                    {
                        "summary": "Fingerprint the previously discussed IP using ports and services.",
                        "requested_capability": "composite workflow",
                        "immediate_need": "gather evidence for host fingerprinting",
                        "preferred_routing_groups": ["host_fingerprinting"],
                        "disallowed_routing_groups": ["geo_enrichment"],
                        "must_preserve": ["ports", "services"],
                        "must_not_reframe_as": ["geolocation lookup"],
                        "confidence": 0.98,
                    }
                )
            return json.dumps(
                {
                    "reasoning": "Use the fingerprinting skill for the referenced IP.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"question": "i want to fingerprit thee ip like ports and servivrd"},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
    ]

    decision = _supervisor_next_action(
        user_question="i want to fingerprit thee ip like ports and servivrd",
        available_skills=available_skills,
        llm=_FingerprintFollowupLLM(),
        instruction="test",
        conversation_history=[
            {"role": "user", "content": "fingerprint 192.168.0.16"},
            {"role": "assistant", "content": "Found 200 total record(s) matching 192.168.0.16 in the now-90d window."},
        ],
        previous_trace=[],
        current_results={},
        previous_eval={},
    )

    assert set(decision["skills"]) == {"fields_querier", "opensearch_querier", "ip_fingerprinter"}
    assert decision["skills"][-1] == "ip_fingerprinter"
    assert "fingerprint 192.168.0.16" in decision["parameters"]["question"]
    assert "ports, services" in decision["parameters"]["question"]


def test_decide_node_preserves_anchored_fingerprint_followup_for_routing_guards():
    class _FingerprintFollowupLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "Analyze ONLY the current user question below." in prompt:
                return json.dumps(
                    {
                        "summary": "Fingerprint the previously discussed IP using ports and services.",
                        "requested_capability": "composite workflow",
                        "immediate_need": "gather evidence for host fingerprinting",
                        "preferred_routing_groups": ["host_fingerprinting"],
                        "disallowed_routing_groups": ["geo_enrichment"],
                        "must_preserve": ["ports", "services"],
                        "must_not_reframe_as": ["geolocation lookup"],
                        "confidence": 0.98,
                    }
                )
            return json.dumps(
                {
                    "reasoning": "Use the fingerprinting skill for the referenced IP.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"question": "i want to fingerprit thee ip like ports and servivrd"},
                }
            )

    state = {
        "user_question": "i want to fingerprit thee ip like ports and servivrd",
        "messages": [
            {"role": "user", "content": "fingerprint 192.168.0.16"},
            {"role": "assistant", "content": "Passive fingerprint for 192.168.0.16: likely_server."},
        ],
        "skill_results": {},
        "previously_run_skills": [],
        "step_count": 0,
        "max_steps": 4,
        "evaluation": {},
        "trace": [],
    }
    config = {
        "configurable": {
            "available_skills": [
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
                {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
            ],
            "llm": _FingerprintFollowupLLM(),
            "instruction": "test",
        }
    }

    decision = decide_node(state, config)

    assert decision["skill_plan"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert "fingerprint 192.168.0.16" in decision["pending_parameters"]["question"]
    assert decision["pending_parameters"].get("ip", "192.168.0.16") == "192.168.0.16"


def test_supervisor_next_action_promotes_host_fingerprint_skill_when_plan_only_has_evidence_search():
    class _EvidenceOnlyFingerprintLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "Analyze ONLY the current user question below." in prompt:
                return json.dumps(
                    {
                        "summary": "Passive fingerprint the IP using observed behavior and port evidence.",
                        "requested_capability": "composite workflow",
                        "immediate_need": "gather evidence for host fingerprinting",
                        "preferred_routing_groups": ["schema_discovery", "evidence_search", "host_fingerprinting"],
                        "disallowed_routing_groups": ["geo_enrichment"],
                        "must_preserve": ["192.168.0.16", "fingerprinting"],
                        "must_not_reframe_as": ["geolocation lookup", "traffic analysis"],
                        "confidence": 0.99,
                    }
                )
            return json.dumps(
                {
                    "reasoning": "Gather fingerprint evidence from logs first.",
                    "skills": ["opensearch_querier"],
                    "parameters": {"question": "fingerprint 192.168.0.16"},
                }
            )

    decision = _supervisor_next_action(
        user_question="fingerprint 192.168.0.16",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
        ],
        llm=_EvidenceOnlyFingerprintLLM(),
        instruction="test",
        conversation_history=[],
        previous_trace=[],
        current_results={},
        previous_eval={},
    )

    assert set(decision["skills"]) == {"fields_querier", "opensearch_querier", "ip_fingerprinter"}
    assert decision["skills"][-1] == "ip_fingerprinter"


def test_decide_node_reviews_and_repairs_invalid_first_step_before_execution():
    class _ReviewingSupervisorLLM:
        def __init__(self):
            self.next_action_calls = 0
            self.review_calls = 0
            self.repair_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_action_calls += 1
                return json.dumps(
                    {
                        "reasoning": "The user wants to know the country for this IP.",
                        "skills": ["geoip_lookup"],
                        "parameters": {"question": "fingerprint 192.168.0.16"},
                    }
                )
            if "Review whether this proposed next supervisor step is the best grounded immediate action" in prompt:
                self.review_calls += 1
                if self.review_calls == 1:
                    return json.dumps(
                        {
                            "is_valid": False,
                            "should_execute": False,
                            "confidence": 0.98,
                            "reasoning": "The proposed step pivots to geolocation even though the user asked for passive fingerprinting.",
                            "issue": "GeoIP enrichment does not answer the requested fingerprinting task.",
                            "suggestion": "Choose the evidence-gathering and fingerprint workflow instead of geolocation.",
                        }
                    )
                return json.dumps(
                    {
                        "is_valid": True,
                        "should_execute": True,
                        "confidence": 0.92,
                        "reasoning": "This repaired plan is grounded because it gathers evidence for passive fingerprinting before analysis.",
                        "issue": "",
                        "suggestion": "",
                    }
                )
            if "# Supervisor Plan Repair" in prompt or "You are repairing a supervisor plan that was invalid, unavailable, or not viable." in prompt:
                self.repair_calls += 1
                return json.dumps(
                    {
                        "reasoning": "Use the passive fingerprint workflow.",
                        "skills": ["ip_fingerprinter"],
                        "parameters": {"question": "fingerprint 192.168.0.16"},
                    }
                )
            return json.dumps({"response": "ok"})

    state = {
        "user_question": "fingerprint 192.168.0.16",
        "messages": [],
        "skill_results": {},
        "previously_run_skills": [],
        "step_count": 0,
        "max_steps": 4,
        "evaluation": {},
        "trace": [],
    }
    llm = _ReviewingSupervisorLLM()
    config = {
        "configurable": {
            "available_skills": [
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
                {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
                {"name": "geoip_lookup", "description": "GeoIP enrichment"},
            ],
            "llm": llm,
            "instruction": "test",
        }
    }

    decision = decide_node(state, config)

    assert decision["skill_plan"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert llm.next_action_calls == 1
    assert llm.review_calls >= 2
    assert llm.repair_calls == 1


def test_decide_node_repairs_low_confidence_review_before_execution():
    class _LowConfidenceReviewSupervisorLLM:
        def __init__(self):
            self.next_action_calls = 0
            self.review_calls = 0
            self.repair_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_action_calls += 1
                return json.dumps(
                    {
                        "reasoning": "This looks like IP look-up work.",
                        "skills": ["geoip_lookup"],
                        "parameters": {"question": "fingerprint 192.168.0.16"},
                    }
                )
            if "Review whether this proposed next supervisor step is the best grounded immediate action" in prompt:
                self.review_calls += 1
                if self.review_calls == 1:
                    return json.dumps(
                        {
                            "is_valid": True,
                            "should_execute": True,
                            "confidence": 0.0,
                            "reasoning": "This might be viable, but I am not confident it directly answers passive fingerprinting.",
                            "issue": "",
                            "suggestion": "Prefer the passive fingerprint workflow.",
                        }
                    )
                return json.dumps(
                    {
                        "is_valid": True,
                        "should_execute": True,
                        "confidence": 0.95,
                        "reasoning": "This repaired plan is the grounded fingerprint workflow.",
                        "issue": "",
                        "suggestion": "",
                    }
                )
            if "# Supervisor Plan Repair" in prompt or "QUESTION GROUNDING:" in prompt:
                self.repair_calls += 1
                return json.dumps(
                    {
                        "reasoning": "Use the passive fingerprint workflow.",
                        "skills": ["ip_fingerprinter"],
                        "parameters": {"question": "fingerprint 192.168.0.16"},
                    }
                )
            return json.dumps({"response": "ok"})

    state = {
        "user_question": "fingerprint 192.168.0.16",
        "messages": [],
        "skill_results": {},
        "previously_run_skills": [],
        "step_count": 0,
        "max_steps": 4,
        "evaluation": {},
        "trace": [],
    }
    llm = _LowConfidenceReviewSupervisorLLM()
    config = {
        "configurable": {
            "available_skills": [
                {"name": "fields_querier", "description": "Field schema discovery"},
                {"name": "opensearch_querier", "description": "Direct log search"},
                {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
                {"name": "geoip_lookup", "description": "GeoIP enrichment"},
            ],
            "llm": llm,
            "instruction": "test",
        }
    }

    decision = decide_node(state, config)

    assert decision["skill_plan"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert llm.review_calls >= 2
    assert llm.repair_calls == 1


def test_orchestrate_with_supervisor_executes_fingerprint_prerequisites_before_fingerprinter():
    class _FingerprintRunnerStub:
        def __init__(self):
            self.calls: list[str] = []

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            if skill_name == "fields_querier":
                return {"status": "ok", "field_mappings": {"destination.port": "number"}}
            if skill_name == "opensearch_querier":
                return {
                    "status": "ok",
                    "results_count": 3,
                    "results": [
                        {"destination.port": 53, "src_ip": "192.168.0.16"},
                        {"destination.port": 443, "src_ip": "192.168.0.16"},
                    ],
                    "ports": [53, 443],
                }
            if skill_name == "ip_fingerprinter":
                return {
                    "status": "ok",
                    "ip": "192.168.0.16",
                    "ports": [{"port": 53}, {"port": 443}],
                    "likely_role": "server",
                }
            return {"status": "ok"}

    class _FingerprintSupervisorLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Use fingerprinting.",
                        "skills": ["ip_fingerprinter"],
                        "parameters": {"ip": "192.168.0.16"},
                    }
                )
            if "Evaluate whether the current skill outputs are sufficient" in prompt:
                return json.dumps(
                    {
                        "satisfied": True,
                        "confidence": 0.9,
                        "reasoning": "Fingerprint and supporting evidence are available.",
                        "missing": [],
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Observed ports 53 and 443 for 192.168.0.16."
            return json.dumps({"response": "ok"})

    runner = _FingerprintRunnerStub()
    llm = _FingerprintSupervisorLLM()
    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
    ]

    out = orchestrate_with_supervisor(
        user_question="fingerprint 192.168.0.16",
        available_skills=available_skills,
        runner=runner,
        llm=llm,
        instruction="test",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls[:3] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert out.get("skill_results", {}).get("ip_fingerprinter", {}).get("status") == "ok"


def test_route_question_preserves_original_question_when_fields_and_opensearch_are_combined():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need field discovery first.",
                    "skills": ["fields_querier", "opensearch_querier"],
                    "parameters": {"question": "What fields contain source country information?"},
                }
            )

    result = route_question(
        user_question="What countries other than the USA do we get traffic from in the past month",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier"]
    assert result["parameters"]["question"] == "What countries other than the USA do we get traffic from in the past month"


def test_route_question_keeps_direct_opensearch_for_explicit_field_query():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Explicit field query.",
                    "skills": ["opensearch_querier"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
    ]

    result = route_question(
        user_question="show logs where destination.port=1194 and source.ip=1.2.3.4",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["opensearch_querier"]


def test_route_question_anchors_followup_reputation_to_previous_public_ips_only():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need more data first.",
                    "skills": ["fields_querier", "opensearch_querier", "threat_analyst"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
    ]
    history = [
        {
            "role": "assistant",
            "content": "Found 200 record(s) matching Russia. Countries seen: Russia. Source/destination IPs: 192.168.0.156, 37.230.117.113, 82.146.61.17.",
        }
    ]

    result = route_question(
        user_question="Aside from the private IPs, what is the reputation of the others?",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=history,
    )

    assert result["skills"] == ["threat_analyst"]
    enriched_question = result["parameters"]["question"]
    assert "37.230.117.113" in enriched_question
    assert "82.146.61.17" in enriched_question
    assert "192.168.0.156" not in enriched_question


def test_route_question_anchors_just_mentioned_non_private_ip_followup():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need more data first.",
                    "skills": ["opensearch_querier", "threat_analyst"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
    ]
    history = [
        {
            "role": "assistant",
            "content": "Found 14 record(s) matching Russia in the past 7 days window. Countries seen: Russia. Source/destination IPs: 192.168.0.85, 37.230.117.113, 92.63.103.84. Earliest: 2026-03-09T21:04:10.437Z. Latest: 2026-03-09T21:08:07.670Z.",
        }
    ]

    result = route_question(
        user_question="Run threat intelligence to the non private IPs you've just mentioned",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=history,
    )

    assert result["skills"] == ["threat_analyst"]
    enriched_question = result["parameters"]["question"]
    assert "37.230.117.113" in enriched_question
    assert "92.63.103.84" in enriched_question
    assert "192.168.0.85" not in enriched_question


def test_route_question_anchors_above_ips_reputation_followup():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need more data first.",
                    "skills": ["fields_querier", "opensearch_querier", "threat_analyst"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
    ]
    history = [
        {
            "role": "assistant",
            "content": "Found 14 record(s) matching Russia in the past 7 days window. Countries seen: Russia. Source/destination IPs: 192.168.0.85, 37.230.117.113, 92.63.103.84. Earliest: 2026-03-09T21:04:10.437Z. Latest: 2026-03-09T21:08:07.670Z.",
        }
    ]

    result = route_question(
        user_question="What is the reputation of the above IPs?",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=history,
    )

    assert result["skills"] == ["threat_analyst"]
    enriched_question = result["parameters"]["question"]
    assert "37.230.117.113" in enriched_question
    assert "92.63.103.84" in enriched_question
    # Private IPs should be filtered out of reputation requests — they have no external threat intel
    assert "192.168.0.85" not in enriched_question


def test_route_question_anchors_singular_ip_reputation_followup_without_new_search():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need more data first.",
                    "skills": ["fields_querier", "opensearch_querier", "threat_analyst"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
    ]
    history = [
        {
            "role": "assistant",
            "content": (
                "Found 2 record(s) matching Iran in the past 30 days window. Countries seen: "
                "Iran. Source IPs: 62.60.131.168. Earliest: 2026-02-13T10:22:10.224Z. "
                "Latest: 2026-02-13T10:23:13.495Z."
            ),
        }
    ]

    result = route_question(
        user_question="what is the reputation of the ip",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=history,
    )

    assert result["skills"] == ["threat_analyst"]
    enriched_question = result["parameters"]["question"]
    assert "62.60.131.168" in enriched_question
    assert "Countries: Iran" in enriched_question


def test_supervisor_reputation_followup_uses_single_visible_ip_from_history():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name == "threat_analyst":
                return {
                    "status": "ok",
                    "verdicts": [
                        {
                            "verdict": "TRUE_THREAT",
                            "confidence": 90,
                            "reasoning": context["parameters"]["question"],
                            "_requested_ips": ["62.60.131.168"],
                        }
                    ],
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Search logs before checking reputation.",
                        "skills": ["fields_querier", "opensearch_querier"],
                        "parameters": {},
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Threat intel completed for the previously mentioned IP."
            return json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.95,
                    "reasoning": "Threat intelligence verdict was produced for the anchored IP.",
                    "missing": [],
                }
            )

    history = [
        {
            "role": "assistant",
            "content": (
                "Found 2 record(s) matching Iran in the past 30 days window. Countries seen: "
                "Iran. Source IPs: 62.60.131.168. Earliest: 2026-02-13T10:22:10.224Z. "
                "Latest: 2026-02-13T10:23:13.495Z."
            ),
        }
    ]

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="what is the reputation of the ip",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=history,
    )

    assert runner.calls == ["threat_analyst"]
    enriched_question = runner.contexts["threat_analyst"]["parameters"]["question"]
    assert "62.60.131.168" in enriched_question
    assert "Countries: Iran" in enriched_question
    assert out.get("evaluation", {}).get("satisfied") is True


def test_execute_skill_workflow_threat_analyst_falls_back_to_history_when_same_turn_has_no_action():
    class _Runner:
        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            if skill_name == "opensearch_querier":
                return {"status": "no_action"}
            if skill_name == "threat_analyst":
                return {
                    "status": "ok",
                    "verdicts": [
                        {
                            "verdict": "FALSE_POSITIVE",
                            "confidence": 90,
                            "reasoning": context["parameters"]["question"],
                        }
                    ],
                }
            return {"status": "ok"}

    history = [
        {
            "role": "assistant",
            "content": "Found 14 record(s) matching Russia in the past 7 days window. Countries seen: Russia. Source/destination IPs: 192.168.0.85, 37.230.117.113, 92.63.103.84. Earliest: 2026-03-09T21:04:10.437Z. Latest: 2026-03-09T21:08:07.670Z.",
        }
    ]

    results = execute_skill_workflow(
        skills=["opensearch_querier", "threat_analyst"],
        runner=_Runner(),
        context={},
        routing_decision={
            "parameters": {
                "question": "Run threat intelligence to the non private IPs you've just mentioned",
            }
        },
        conversation_history=history,
        aggregated_results={},
    )

    threat_reasoning = results["threat_analyst"]["verdicts"][0]["reasoning"]
    assert "37.230.117.113" in threat_reasoning
    assert "92.63.103.84" in threat_reasoning
    assert "192.168.0.85" not in threat_reasoning


def test_execute_skill_workflow_enriches_ip_fingerprinter_from_history():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            return {"status": "no_data", "ip": "192.168.0.16"}

    runner = _Runner()
    execute_skill_workflow(
        skills=["ip_fingerprinter"],
        runner=runner,
        context={},
        routing_decision={"parameters": {"question": "fingerprint it like ports and services"}},
        conversation_history=[
            {"role": "assistant", "content": "Passive fingerprint for 192.168.0.16: hybrid (88% confidence)."}
        ],
        aggregated_results={},
        memory=None,
    )

    assert runner.calls == ["ip_fingerprinter"]
    enriched_question = runner.contexts["ip_fingerprinter"]["parameters"]["question"]
    assert "192.168.0.16" in enriched_question


def test_format_response_ignores_validation_failed_opensearch_hits():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "Aside from the private IPs, what is the reputation of the others?",
        {"skills": ["opensearch_querier", "threat_analyst"], "parameters": {}},
        {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 5,
                "validation_failed": True,
                "results": [{"src_ip": "75.75.75.75"}],
            },
            "threat_analyst": {
                "status": "ok",
                "verdicts": [{"verdict": "FALSE_POSITIVE", "confidence": 85, "reasoning": "Low risk."}],
            },
        },
        mock_llm,
        cfg=_Cfg(),
    )

    assert "75.75.75.75" not in response
    mock_llm.chat.assert_not_called()
    assert "FALSE_POSITIVE" in response


def test_supervisor_evaluation_does_not_satisfy_country_mismatch_results():
    from core.chat_router.logic import _supervisor_evaluate_satisfaction

    mock_llm = MagicMock()
    mock_llm.chat.return_value = json.dumps(
        {
            "satisfied": False,
            "confidence": 0.2,
            "reasoning": "The returned traffic is for the wrong country.",
            "missing": ["traffic for Russia"],
        }
    )

    eval_result = _supervisor_evaluate_satisfaction(
        user_question="any traffic from russia in the past 30 days",
        llm=mock_llm,
        instruction="You are a SOC analyst.",
        conversation_history=[],
        skill_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 136,
                "results": [
                    {"geoip": {"country_name": "China"}, "src_ip": "118.190.162.28"},
                    {"geoip": {"country_name": "China"}, "src_ip": "223.4.221.28"},
                ],
                "validation_failed": True,
                "validation_issue": "Requested countries ['Russia'] but sampled results only show ['China'].",
            }
        },
        step=1,
        max_steps=4,
    )

    assert eval_result["satisfied"] is False
    assert "wrong country" in eval_result["reasoning"].lower() or "missing" in eval_result


def test_opensearch_validation_rejects_country_mismatch_samples():
    from skills.opensearch_querier.logic import _llm_validate_results

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        json.dumps(
            {
                "summary": "User wants traffic from Russia in the past 30 days.",
                "search_type": "traffic",
                "countries": ["Russia"],
                "ips": [],
                "ports": [],
                "protocols": [],
                "time_range": "now-30d",
                "aggregation_type": "none",
                "entity_scope": "Traffic from Russia in the past 30 days.",
            }
        ),
        json.dumps(
            {
                "is_valid": False,
                "issue": "The question asks for Russia but the sampled results are for China.",
                "suggestion": "Re-run the search with country grounding preserved from the question.",
                "confidence": 0.98,
            }
        ),
    ]

    validation = _llm_validate_results(
        question="any traffic from russia in the past 30 days",
        search_terms=[],
        requested_filters={"countries": ["Russia"], "ports": [], "protocols": [], "time_range": "now-30d"},
        results=[
            {"geoip": {"country_name": "China"}, "src_ip": "118.190.162.28"},
            {"geoip": {"country_name": "China"}, "src_ip": "223.4.221.28"},
        ],
        field_mappings={"country_fields": ["geoip.country_name"]},
        llm=mock_llm,
    )

    assert validation["is_valid"] is False
    assert "Russia" in validation["issue"]
    assert mock_llm.complete.call_count == 2


def test_route_question_strips_threat_analyst_for_plain_country_traffic_question():
    class _RouteLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "Need country search plus intel.",
                    "skills": ["fields_querier", "opensearch_querier", "threat_analyst"],
                    "parameters": {},
                }
            )

    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "threat_analyst", "description": "Reputation analysis"},
    ]

    result = route_question(
        user_question="Any traffic from Russia this past week?",
        available_skills=available_skills,
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier"]


def test_supervisor_grounding_keeps_country_traffic_as_evidence_search():
    grounding = _ground_supervisor_question_with_llm(
        user_question="Any traffic from Greece today?",
        llm=MagicMock(),
        instruction="test",
    )

    assert grounding["requested_capability"] == "evidence search"
    assert grounding["preferred_routing_groups"] == ["schema_discovery", "evidence_search"]
    assert "geo_enrichment" in grounding["disallowed_routing_groups"]
    assert "host_fingerprinting" in grounding["disallowed_routing_groups"]
    assert "Greece" in grounding["must_preserve"]
    assert "today" in [item.lower() for item in grounding["must_preserve"]]


def test_supervisor_grounding_routes_private_ip_threat_intel_to_threat_analysis():
    grounding = _ground_supervisor_question_with_llm(
        user_question="Any threat intel on 192.168.0.16",
        llm=MagicMock(),
        instruction="test",
    )

    assert grounding["requested_capability"] == "enrichment"
    assert grounding["preferred_routing_groups"] == ["threat_enrichment"]
    assert "geo_enrichment" in grounding["disallowed_routing_groups"]
    assert "host_fingerprinting" in grounding["disallowed_routing_groups"]
    assert "192.168.0.16" in grounding["must_preserve"]


def test_supervisor_next_action_shortcuts_explicit_private_ip_threat_intel():
    class _LLM:
        def chat(self, messages: list[dict]):
            raise AssertionError("LLM planning should not run for direct threat-intel IP shortcut")

    decision = _supervisor_next_action(
        user_question="Any threat intel on 192.168.0.16",
        available_skills=[
            {"name": "geoip_lookup", "description": "GeoIP enrichment"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        llm=_LLM(),
        instruction="test",
        conversation_history=[],
        previous_trace=[],
        current_results={},
        previous_eval={},
    )

    assert decision["skills"] == ["threat_analyst"]
    assert decision["parameters"]["question"] == "Any threat intel on 192.168.0.16"


def test_postprocess_recovers_fields_then_opensearch_after_schema_validation_failure():
    from core.chat_router.logic import _postprocess_selected_skills
    from core.skill_manifest import SkillManifestLoader

    manifests = SkillManifestLoader().load_all_manifests()
    skills = _postprocess_selected_skills(
        user_question="Any traffic from Russia this past week?",
        selected_skills=["opensearch_querier"],
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        current_results={
            "opensearch_querier": {
                "status": "ok",
                "validation_failed": True,
                "validation_issue": "The results do not contain the required fields for country information.",
            }
        },
        manifests=manifests,
    )

    assert skills == ["fields_querier", "opensearch_querier"]


def test_postprocess_promotes_geoip_for_country_question_when_ips_already_exist():
    from core.chat_router.logic import _postprocess_selected_skills
    from core.skill_manifest import SkillManifestLoader

    manifests = SkillManifestLoader().load_all_manifests()
    skills = _postprocess_selected_skills(
        user_question="What country are those IPs from?",
        selected_skills=["opensearch_querier"],
        available_skills=[
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "geoip_lookup", "description": "GeoIP enrichment"},
        ],
        current_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 1,
                "results": [{"src_ip": "37.230.117.113"}],
            }
        },
        manifests=manifests,
    )

    assert skills == ["opensearch_querier", "geoip_lookup"]


def test_postprocess_adds_threat_analyst_only_after_evidence_exists_for_reputation_question():
    from core.chat_router.logic import _postprocess_selected_skills
    from core.skill_manifest import SkillManifestLoader

    manifests = SkillManifestLoader().load_all_manifests()
    skills = _postprocess_selected_skills(
        user_question="What is the reputation of the above IPs?",
        selected_skills=["opensearch_querier"],
        available_skills=[
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        current_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 2,
                "results": [{"src_ip": "37.230.117.113"}, {"src_ip": "92.63.103.84"}],
            }
        },
        manifests=manifests,
    )

    assert skills == ["opensearch_querier", "threat_analyst"]


def test_supervisor_upgrades_repeated_field_discovery_to_opensearch_after_schema_results():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {
                        "source_ip_fields": ["src_ip"],
                        "destination_ip_fields": ["dest_ip"],
                        "text_fields": ["alert.signature"],
                    },
                }
            if skill_name == "opensearch_querier":
                return {
                    "status": "ok",
                    "results_count": 1,
                    "results": [
                        {
                            "alert.signature": "ET POLICY Dropbox.com Offsite File Backup in Use",
                            "src_ip": "8.8.8.8",
                            "dest_ip": "192.168.0.16",
                        }
                    ],
                }
            return {"status": "ok"}

    class _SupervisorLLMRepeatFields:
        def __init__(self):
            self.next_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_calls += 1
                return json.dumps(
                    {
                        "reasoning": "Discover alert fields first.",
                        "skills": ["fields_querier"],
                        "parameters": {},
                    }
                )
            if "Evaluate whether the current skill outputs are sufficient" in prompt:
                return json.dumps(
                    {
                        "satisfied": False if self.next_calls == 1 else True,
                        "confidence": 0.6,
                        "reasoning": "Need actual alert records after field discovery.",
                        "missing": ["matching alert records"] if self.next_calls == 1 else [],
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Found ET POLICY alert records and extracted the IPs."
            return json.dumps({"response": "ok"})

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="check for ET POLICY alerts and their ips",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        runner=runner,
        llm=_SupervisorLLMRepeatFields(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls == ["fields_querier", "opensearch_querier"]
    assert out.get("skill_results", {}).get("opensearch_querier", {}).get("results_count") == 1


def test_supervisor_reputation_followup_uses_listed_ips_from_visible_history():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name == "threat_analyst":
                return {
                    "status": "ok",
                    "verdicts": [
                        {
                            "verdict": "TRUE_THREAT",
                            "confidence": 91,
                            "reasoning": context["parameters"]["question"],
                        }
                    ],
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Search for the IPs first.",
                        "skills": ["fields_querier", "opensearch_querier"],
                        "parameters": {},
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Threat intel completed for the listed public IPs."
            return json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.95,
                    "reasoning": "Threat intelligence verdicts were produced for the listed IPs.",
                    "missing": [],
                }
            )

    history = [
        {
            "role": "assistant",
            "content": (
                "Found 200 record(s) matching Russia in the past 30 days window. "
                "Countries seen: Russia. Source IPs: 37.230.117.113, 82.146.61.17, "
                "82.202.197.102, 92.63.103.84, 94.139.250.252."
            ),
        }
    ]

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="The IPs listed, what is their reputation?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
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
    assert out.get("evaluation", {}).get("satisfied") is True


def test_graph_preserves_supervisor_enriched_question_between_nodes():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            return {
                "status": "ok",
                "verdicts": [
                    {
                        "verdict": "TRUE_THREAT",
                        "confidence": 88,
                        "reasoning": context["parameters"]["question"],
                    }
                ],
            }

    class _SupervisorLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Threat question already has the entities we need.",
                        "skills": ["threat_analyst"],
                        "parameters": {
                            "question": "Analyze only these IPs for reputation: 37.230.117.113, 82.146.61.17",
                        },
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Threat intel completed."
            return json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.9,
                    "reasoning": "Threat intelligence verdicts were produced.",
                    "missing": [],
                }
            )

    runner = _Runner()
    orchestrate_with_supervisor(
        user_question="What is their reputation?",
        available_skills=[
            {"name": "threat_analyst", "description": "Reputation analysis"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls == ["threat_analyst"]
    assert runner.contexts["threat_analyst"]["parameters"]["question"] == (
        "Analyze only these IPs for reputation: 37.230.117.113, 82.146.61.17"
    )


def test_supervisor_field_recovery_uses_original_question_for_opensearch_and_stays_grounded():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {
                        "country_fields": ["geoip.country_name"],
                        "source_ip_fields": ["src_ip"],
                        "destination_ip_fields": ["dest_ip"],
                    },
                }
            if skill_name == "opensearch_querier":
                return {
                    "status": "no_action",
                    "results_count": 0,
                    "results": [],
                    "time_range_label": "past 30 days",
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def __init__(self):
            self.next_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_calls += 1
                if self.next_calls == 1:
                    return json.dumps(
                        {
                            "reasoning": "Need to identify the country fields first.",
                            "skills": ["fields_querier"],
                            "parameters": {
                                "question": "What fields indicate source country and source IP?",
                            },
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "reasoning": "Now search for traffic using discovered fields.",
                            "skills": ["opensearch_querier"],
                            "parameters": {
                                "question": "Any traffic from russia in the past 30 days?",
                            },
                        }
                    )
            if "Evaluate whether the current skill outputs are sufficient" in prompt:
                return json.dumps(
                    {
                        "satisfied": False,
                        "confidence": 0.4,
                        "reasoning": "Need matching traffic records.",
                        "missing": ["matching traffic records"],
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Eight IP addresses were identified: 8.8.8.8, 1.1.1.1, 203.0.113.5, and five others across 45 connection records."
            return json.dumps({"response": "ok"})

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="Any traffic from russia in the past 30 days?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls.count("fields_querier") >= 1
    assert runner.calls[-1] == "opensearch_querier"
    assert runner.contexts["opensearch_querier"]["parameters"]["question"] == "Any traffic from russia in the past 30 days?"
    assert "8.8.8.8" not in out["response"]
    assert "1.1.1.1" not in out["response"]
    assert "grounded OpenSearch query" in out["response"]


def test_supervisor_combined_fields_and_opensearch_preserves_original_question():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {
                        "country_fields": ["geoip.country_name"],
                        "source_ip_fields": ["src_ip"],
                        "destination_ip_fields": ["dest_ip"],
                    },
                }
            if skill_name == "opensearch_querier":
                return {
                    "status": "ok",
                    "results_count": 3,
                    "results": [],
                    "aggregation_type": "country_terms",
                    "country_buckets": [
                        {"country": "Iran", "count": 2},
                        {"country": "Russia", "count": 1},
                    ],
                    "excluded_countries": ["United States"],
                    "time_range_label": "past month",
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Discover fields first, then search.",
                        "skills": ["fields_querier", "opensearch_querier"],
                        "parameters": {
                            "question": "What fields identify source countries in traffic logs?",
                        },
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Observed traffic from Iran and Russia."
            return json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.9,
                    "reasoning": "Aggregated country buckets were returned.",
                    "missing": [],
                }
            )

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="What countries other than the USA do we get traffic from in the past month",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls == ["fields_querier", "opensearch_querier"]
    assert runner.contexts["opensearch_querier"]["parameters"]["question"] == (
        "What countries other than the USA do we get traffic from in the past month"
    )
    assert "Iran (2), Russia (1)" in out["response"]


def test_baseline_followup_uses_grounded_baseline_response_and_stops_after_evidence():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            self.contexts[skill_name] = context
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {
                        "ip_fields": ["src_ip", "dest_ip"],
                        "timestamp_fields": ["@timestamp"],
                    },
                }
            if skill_name == "baseline_querier":
                return {
                    "status": "ok",
                    "findings": {
                        "answer": "1.1.1.1 appears to be routine destination-side DNS traffic rather than anomalous source activity.",
                        "rag_sources": 2,
                        "log_records": 12,
                        "evidence": {
                            "ips": ["1.1.1.1", "192.168.0.85"],
                            "ports": ["53"],
                            "timestamps": ["2026-03-11T23:43:41.898Z", "2026-03-11T23:43:56.274Z"],
                        },
                    },
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def __init__(self):
            self.next_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_calls += 1
                if self.next_calls == 1:
                    return json.dumps(
                        {
                            "reasoning": "Need field discovery first.",
                            "skills": ["fields_querier"],
                            "parameters": {},
                        }
                    )
                return json.dumps(
                    {
                        "reasoning": "Try a baseline query.",
                        "skills": ["baseline_querier"],
                        "parameters": {
                            "question": "Discover potential field names related to network traffic and system events that might be relevant to identifying anomalous behavior.",
                        },
                    }
                )
            if "# Supervisor Reflection Repair" in prompt or "# Supervisor Plan Repair" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Field discovery already ran; move to the baseline query using the observed IP.",
                        "skills": ["baseline_querier"],
                        "parameters": {
                            "question": "Is 1.1.1.1 normal behavior in this network?",
                        },
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Eight IP addresses were identified: 8.8.8.8, 1.1.1.1, 203.0.113.5, and five others across 45 connection records."
            return json.dumps(
                {
                    "satisfied": False,
                    "confidence": 0.4,
                    "reasoning": "Need more analysis.",
                    "missing": ["baseline evidence"],
                }
            )

    history = [
        {
            "role": "assistant",
            "content": (
                "No traffic source 1.1.1.1 was found in the last 24 hours window. "
                "However, 200 record(s) were found in the destination direction for the same IP in the last 24 hours window. "
                "Peers seen: 192.168.0.142, 192.168.0.85. Earliest: 2026-03-11T23:43:41.898Z. "
                "Latest: 2026-03-11T23:43:56.274Z."
            ),
        }
    ]

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="Is 1.1.1.1 normal behavior in this network?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "baseline_querier", "description": "Behavioral baseline search"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=history,
    )

    assert runner.calls == ["baseline_querier"]
    baseline_question = runner.contexts["baseline_querier"]["parameters"]["question"]
    assert "1.1.1.1" in baseline_question
    assert "Recent observed traffic" in baseline_question
    assert out.get("evaluation", {}).get("satisfied") is True
    assert "routine destination-side DNS traffic" in out["response"]
    assert "Observed records: 12." in out["response"]
    assert "8.8.8.8" not in out["response"]
    assert "203.0.113.5" not in out["response"]


def test_explicit_ip_search_advances_from_fields_to_opensearch_instead_of_repeating_schema():
    class _Runner:
        def __init__(self):
            self.calls: list[str] = []

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.calls.append(skill_name)
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {
                        "ip_fields": ["src_ip", "dest_ip"],
                        "source_ip_fields": ["src_ip"],
                        "destination_ip_fields": ["dest_ip"],
                    },
                }
            if skill_name == "opensearch_querier":
                return {
                    "status": "ok",
                    "results_count": 0,
                    "results": [],
                    "search_terms": ["1.1.1.1"],
                    "time_range_label": "today",
                    "ip_direction": "source",
                    "directional_alternative": {
                        "direction": "destination",
                        "results_count": 200,
                        "time_range_label": "today",
                        "sample_peers": ["192.168.0.85"],
                        "earliest": "2026-03-11T23:43:41.898Z",
                        "latest": "2026-03-11T23:43:56.274Z",
                    },
                }
            return {"status": "ok"}

    class _SupervisorLLM:
        def __init__(self):
            self.next_calls = 0

        def chat(self, messages: list[dict]):
            prompt = messages[-1].get("content", "")
            if "SOC supervisor orchestrator" in prompt:
                self.next_calls += 1
                return json.dumps(
                    {
                        "reasoning": "Need fields first.",
                        "skills": ["fields_querier"],
                        "parameters": {},
                    }
                )
            if "# Supervisor Reflection Repair" in prompt or "# Supervisor Plan Repair" in prompt:
                return json.dumps(
                    {
                        "reasoning": "Field discovery already ran; move to OpenSearch for the explicit IP search.",
                        "skills": ["opensearch_querier"],
                        "parameters": {"question": "Any traffic from 1.1.1.1 today?"},
                    }
                )
            if "Based on these skill execution results" in prompt:
                return "Eight IP addresses were identified: 8.8.8.8, 1.1.1.1, 203.0.113.5, and five others across 45 connection records."
            return json.dumps(
                {
                    "satisfied": False,
                    "confidence": 0.4,
                    "reasoning": "Need grounded evidence.",
                    "missing": ["matching records"],
                }
            )

    runner = _Runner()
    out = orchestrate_with_supervisor(
        user_question="Any traffic from 1.1.1.1 today?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        runner=runner,
        llm=_SupervisorLLM(),
        instruction="You are a SOC assistant.",
        cfg=_Cfg(),
        conversation_history=[],
    )

    assert runner.calls == ["fields_querier", "opensearch_querier"]
    assert "No traffic source 1.1.1.1" in out["response"]
    assert "8.8.8.8" not in out["response"]


def test_format_response_prefers_grounded_baseline_over_prior_no_action_opensearch():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "Is 1.1.1.1 normal behavior in this network?",
        {"skills": ["opensearch_querier", "baseline_querier"], "parameters": {}},
        {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 200,
                "results": [{"src_ip": "192.168.0.85", "dest_ip": "1.1.1.1"}],
                "search_terms": ["1.1.1.1"],
                "time_range_label": "now-1d",
            },
            "baseline_querier": {
                "status": "ok",
                "findings": {
                    "answer": "Here’s a summary of the network traffic data across all retrieved records.",
                    "grounded_assessment": (
                        "1.1.1.1 appears to be routine destination-side DNS traffic in this network. "
                        "It matched 2 log record(s) in the sampled baseline search (0 as source, 2 as destination)."
                    ),
                    "rag_sources": 2,
                    "log_records": 50,
                    "evidence": {
                        "ips": ["1.1.1.1", "192.168.0.85"],
                        "timestamps": ["2026-03-11T23:43:41.898Z", "2026-03-11T23:43:56.274Z"],
                    },
                },
            },
        },
        mock_llm,
        cfg=_Cfg(),
    )

    mock_llm.chat.assert_not_called()
    assert "routine destination-side DNS traffic" in response
    assert "It matched 2 log record(s)" in response
    assert "Here’s a summary of the network traffic data" not in response
    assert "Found 200 record(s) matching" not in response


def test_format_response_prefers_preferred_skill_over_prior_geoip():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "fingerprint 192.168.0.16",
        {
            "skills": ["geoip_lookup", "ip_fingerprinter"],
            "preferred_skills": ["ip_fingerprinter"],
            "parameters": {},
        },
        {
            "geoip_lookup": {
                "status": "not_found",
                "ip": "192.168.0.16",
                "db_path": "/tmp/GeoLite2-City.mmdb",
            },
            "ip_fingerprinter": {
                "status": "ok",
                "ip": "192.168.0.16",
                "likely_role": {"classification": "server", "confidence": 91},
                "ports": [
                    {"port": 53, "service_name": "domain", "observations": 20, "registered": True},
                    {"port": 443, "service_name": "https", "observations": 12, "registered": True},
                ],
                "os_family_likelihoods": [{"family": "Linux", "confidence": 63}],
            },
        },
        mock_llm,
        cfg=_Cfg(),
    )

    mock_llm.chat.assert_not_called()
    assert "Passive fingerprint for 192.168.0.16" in response
    assert "Listening on ports: 53 (domain), 443 (https)." in response
    assert "No MaxMind geolocation record" not in response


def test_format_response_prefers_threat_analyst_over_geoip_for_private_ip_reputation_question():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "Any threat intel on 192.168.0.16",
        {
            "skills": ["geoip_lookup", "threat_analyst"],
            "preferred_skills": ["threat_analyst"],
            "parameters": {},
        },
        {
            "geoip_lookup": {
                "status": "not_found",
                "ip": "192.168.0.16",
                "db_path": "/tmp/GeoLite2-City.mmdb",
            },
            "threat_analyst": {
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
        },
        mock_llm,
        cfg=_Cfg(),
    )

    mock_llm.chat.assert_not_called()
    assert "private/internal IP address" in response
    assert "No MaxMind geolocation record" not in response


def test_format_response_prefers_last_preferred_skill_in_chain_for_fingerprint():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "fingerprint 192.168.0.16",
        {
            "skills": ["fields_querier", "opensearch_querier", "ip_fingerprinter"],
            "preferred_skills": ["fields_querier", "opensearch_querier", "ip_fingerprinter"],
            "parameters": {},
        },
        {
            "opensearch_querier": {
                "status": "ok",
                "aggregation_type": "fingerprint_ports",
                "search_terms": ["192.168.0.16"],
                "observed_ports": [53, 443, 1900],
                "results_count": 24233,
                "time_range_label": "now-30d",
            },
            "ip_fingerprinter": {
                "status": "ok",
                "ip": "192.168.0.16",
                "likely_role": {"classification": "server", "confidence": 91},
                "ports": [
                    {"port": 53, "service_name": "domain", "observations": 20, "registered": True},
                    {"port": 443, "service_name": "https", "observations": 12, "registered": True},
                ],
                "os_family_likelihoods": [{"family": "Linux", "confidence": 63}],
            },
        },
        mock_llm,
        cfg=_Cfg(),
    )

    assert "Passive fingerprint for 192.168.0.16" in response
    assert "Likely OS: Linux" in response
    assert "Observed 3 target-owned port(s)" not in response


def test_format_response_renders_country_aggregation_results():
    from core.chat_router.logic import format_response

    mock_llm = MagicMock()

    response = format_response(
        "What countries other than the USA do we get traffic from in the past month",
        {"skills": ["opensearch_querier"], "parameters": {}},
        {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 3,
                "results": [],
                "aggregation_type": "country_terms",
                "country_buckets": [
                    {"country": "Iran", "count": 2},
                    {"country": "Russia", "count": 1},
                ],
                "excluded_countries": ["United States"],
                "time_range_label": "past month",
            },
        },
        mock_llm,
        cfg=_Cfg(),
    )

    mock_llm.chat.assert_not_called()
    assert "Observed traffic from 2 country(s) in the past month window excluding United States" in response
    assert "Iran (2), Russia (1)" in response


def test_supervisor_does_not_satisfy_invalid_country_aggregation():
    from core.chat_router.logic import _supervisor_evaluate_satisfaction

    mock_llm = MagicMock()

    eval_result = _supervisor_evaluate_satisfaction(
        user_question="any traffic from russia in the past 30 days?",
        llm=mock_llm,
        instruction="You are a SOC analyst.",
        conversation_history=[],
        skill_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 3599983,
                "aggregation_type": "country_terms",
                "country_buckets": [
                    {"country": "United States", "count": 3441698},
                    {"country": "Türkiye", "count": 74635},
                ],
                "validation_failed": True,
                "validation_issue": "The aggregation broadened the question into a top-countries list.",
                "validation_reasoning": "The user asked whether there was traffic from Russia specifically, not for a country distribution.",
            }
        },
        step=2,
        max_steps=4,
    )

    assert eval_result["satisfied"] is False
    assert "russia" in eval_result["reasoning"].lower() or "question" in eval_result["reasoning"].lower()
