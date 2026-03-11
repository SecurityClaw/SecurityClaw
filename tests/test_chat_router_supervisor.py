from __future__ import annotations

import json
from unittest.mock import MagicMock

from skills.chat_router.logic import execute_skill_workflow, orchestrate_with_supervisor, route_question


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
    assert "192.168.0.85" in enriched_question


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


def test_format_response_ignores_validation_failed_opensearch_hits():
    from skills.chat_router.logic import format_response

    mock_llm = MagicMock()
    mock_llm.chat.return_value = "Threat intel points to the public IPs from the prior results, not the invalid fallback search."

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
    assert mock_llm.chat.called


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
