from __future__ import annotations

import json

from core.chat_router.logic import _supervisor_next_action, decide_node


class _GroundingLLM:
    def __init__(self):
        self.calls: list[str] = []

    def chat(self, messages: list[dict]):
        prompt = messages[-1].get("content", "")
        self.calls.append(prompt)

        if "Analyze ONLY the current user question below." in prompt:
            return json.dumps(
                {
                    "summary": "Find traffic involving 1.1.1.1.",
                    "requested_capability": "evidence search",
                    "immediate_need": "gather grounded log evidence",
                    "preferred_routing_groups": ["schema_discovery", "evidence_search"],
                    "disallowed_routing_groups": ["geo_enrichment"],
                    "must_preserve": ["1.1.1.1", "traffic"],
                    "must_not_reframe_as": ["geolocation lookup"],
                    "confidence": 0.95,
                }
            )

        if "Supervisor Skill Routing Orchestrator" in prompt:
            return json.dumps(
                {
                    "reasoning": "Need to search logs for traffic from 1.1.1.1.",
                    "skills": ["log_searcher"],
                    "parameters": {"question": "any traffic from 1.1.1.1?"},
                }
            )

        if "Supervisor Plan Repair" in prompt:
            return json.dumps(
                {
                    "reasoning": "log_searcher is not a loaded skill; grounded traffic lookup should run through schema discovery and evidence search.",
                    "skills": ["opensearch_querier"],
                    "parameters": {"question": "any traffic from 1.1.1.1?"},
                }
            )

        if "Review whether this proposed next supervisor step is the best grounded immediate action." in prompt:
            return json.dumps(
                {
                    "is_valid": True,
                    "should_execute": True,
                    "confidence": 0.95,
                    "reasoning": "The repaired schema-plus-evidence plan is grounded and executable.",
                    "issue": "",
                    "suggestion": "",
                }
            )

        if "Supervisor Reflection Repair" in prompt:
            return json.dumps(
                {
                    "reasoning": "Retry with the grounded schema-plus-OpenSearch path instead of an unavailable skill.",
                    "skills": ["opensearch_querier"],
                    "parameters": {"question": "any traffic from 1.1.1.1?"},
                }
            )

        raise AssertionError(f"Unexpected prompt: {prompt[:160]}")


class _Cfg:
    def get(self, section: str, key: str, default=None):
        values = {
            ("chat", "supervisor_max_steps"): 4,
        }
        return values.get((section, key), default)


def test_supervisor_next_action_repairs_hallucinated_skill_to_loaded_plan():
    llm = _GroundingLLM()
    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Grounded log search"},
        {"name": "geoip_lookup", "description": "GeoIP enrichment"},
    ]

    result = _supervisor_next_action(
        user_question="any traffic from 1.1.1.1?",
        available_skills=available_skills,
        llm=llm,
        instruction="You are a SOC assistant.",
        conversation_history=[],
        previous_trace=[],
        current_results={},
        previous_eval={"satisfied": False, "reasoning": "Need grounded evidence."},
    )

    assert result["skills"] == ["fields_querier", "opensearch_querier"]
    assert any("Supervisor Plan Repair" in prompt for prompt in llm.calls)


def test_decide_node_repairs_invalid_plan_before_execution():
    llm = _GroundingLLM()
    available_skills = [
        {"name": "fields_querier", "description": "Field schema discovery"},
        {"name": "opensearch_querier", "description": "Grounded log search"},
    ]
    state = {
        "user_question": "any traffic from 1.1.1.1?",
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
            "available_skills": available_skills,
            "llm": llm,
            "instruction": "You are a SOC assistant.",
            "cfg": _Cfg(),
        }
    }

    decision = decide_node(state, config)

    assert decision["skill_plan"] == ["fields_querier", "opensearch_querier"]
    assert decision["pending_parameters"]["question"] == "any traffic from 1.1.1.1?"
    assert any("Supervisor Plan Repair" in prompt for prompt in llm.calls)
