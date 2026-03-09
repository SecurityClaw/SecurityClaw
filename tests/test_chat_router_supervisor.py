from __future__ import annotations

import json

from skills.chat_router.logic import orchestrate_with_supervisor, route_question


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
