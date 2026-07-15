from core.chat_router.logic import (
    _build_conversation_context,
    _build_result_context,
    _supervisor_next_action,
    execute_skill_workflow,
)
from core.chat_router import logic
from web.api import server


def test_conversation_context_preserves_followup_entities_and_recent_answer():
    context = _build_conversation_context([
        {"role": "user", "content": "Investigate traffic to 203.0.113.8"},
        {"role": "assistant", "content": "The host exposed ports 22 and 443."},
        {"role": "user", "content": "Is that IP malicious?"},
    ])

    assert "203.0.113.8" in context
    assert "ports 22 and 443" in context
    assert "Is that IP malicious?" in context


def test_result_context_contains_evidence_not_only_status_and_count():
    context = _build_result_context({
        "opensearch_querier": {
            "status": "ok",
            "results_count": 1,
            "results": [{"source.ip": "203.0.113.8", "destination.port": 443}],
        }
    })

    assert "203.0.113.8" in context
    assert "destination.port" in context
    assert '"results_count": 1' in context


class _Runner:
    def __init__(self):
        self.contexts = {}

    def _build_context(self):
        return {}

    def dispatch(self, skill_name, *, context):
        self.contexts[skill_name] = context
        return {"status": "ok"}


def test_skill_specific_parameters_do_not_leak_between_tools():
    runner = _Runner()
    execute_skill_workflow(
        ["geoip_lookup", "opensearch_querier"],
        runner,
        {},
        {
            "parameters": {
                "question": "Investigate the host",
                "by_skill": {
                    "geoip_lookup": {"ip": "203.0.113.8"},
                    "opensearch_querier": {"time_range": "24h"},
                },
            }
        },
    )

    geo = runner.contexts["geoip_lookup"]["parameters"]
    search = runner.contexts["opensearch_querier"]["parameters"]
    assert geo == {"question": "Investigate the host", "ip": "203.0.113.8"}
    assert search == {"question": "Investigate the host", "time_range": "24h"}


def test_router_history_excludes_current_persisted_user_message(monkeypatch):
    monkeypatch.setattr(server, "load_conversation_history", lambda _: [
        {"role": "user", "content": "Check 203.0.113.8"},
        {"role": "assistant", "content": "It exposed port 443"},
        {"role": "user", "content": "Is it malicious?"},
    ])

    history = server._chat_history_for_router("conversation")

    assert [message["content"] for message in history] == [
        "Check 203.0.113.8",
        "It exposed port 443",
    ]


class _CapturingLLM:
    def __init__(self):
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return '{"reasoning":"Use prior evidence","skills":["custom_tool"],"parameters":{"question":"Inspect 203.0.113.8"}}'


def test_supervisor_prompt_receives_history_and_actual_tool_evidence(monkeypatch):
    llm = _CapturingLLM()
    monkeypatch.setattr(
        logic,
        "_review_and_refine_supervisor_plan",
        lambda *, decision, **kwargs: decision,
    )

    _supervisor_next_action(
        user_question="What should we do with that host?",
        available_skills=[{"name": "custom_tool", "description": "Inspect a host"}],
        llm=llm,
        instruction="Route using grounded evidence.",
        conversation_history=[
            {"role": "user", "content": "Check 203.0.113.8"},
            {"role": "assistant", "content": "It exposed port 443"},
        ],
        previous_trace=[],
        current_results={
            "search": {
                "status": "ok",
                "results": [{"source.ip": "203.0.113.8", "destination.port": 443}],
            }
        },
        previous_eval={},
    )

    prompt = llm.messages[0][1]["content"]
    assert "Check 203.0.113.8" in prompt
    assert "It exposed port 443" in prompt
    assert '"source.ip": "203.0.113.8"' in prompt
    assert '"destination.port": 443' in prompt


def test_decide_node_short_circuits_direct_response_without_plan_review(monkeypatch):
    monkeypatch.setattr(logic, "_supervisor_next_action", lambda **kwargs: {
        "response_mode": "direct",
        "reasoning": "Explain the existing result",
        "skills": [],
        "parameters": {"question": kwargs["user_question"]},
    })
    monkeypatch.setattr(
        logic,
        "_review_and_refine_supervisor_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("direct response must not be reviewed")
        ),
    )
    state = {
        "user_question": "What does that result mean?",
        "messages": [{"role": "assistant", "content": "Found port 443"}],
        "skill_results": {},
        "previously_run_skills": [],
        "step_count": 0,
        "max_steps": 4,
        "evaluation": {},
        "trace": [],
    }
    config = {"configurable": {
        "available_skills": [],
        "llm": object(),
        "instruction": "",
    }}

    result = logic.decide_node(state, config)

    assert result["response_mode"] == "direct"
    assert result["skill_plan"] == []
    assert result["plan_exhausted"] is True


def test_decide_node_can_request_operator_clarification(monkeypatch):
    monkeypatch.setattr(logic, "_supervisor_next_action", lambda **kwargs: {
        "response_mode": "clarify",
        "reasoning": "The target path is required",
        "skills": [],
        "parameters": {"question": "Which path should be inspected?"},
    })
    result = logic.decide_node(
        {
            "user_question": "Inspect it",
            "messages": [], "skill_results": {}, "previously_run_skills": [],
            "step_count": 0, "max_steps": 4, "evaluation": {}, "trace": [],
        },
        {"configurable": {"available_skills": [], "llm": object(), "instruction": ""}},
    )
    assert result["response_mode"] == "clarify"
    assert result["pending_parameters"]["question"] == "Which path should be inspected?"


def test_decide_node_applies_operator_guidance_on_next_round(monkeypatch):
    captured = {}

    def next_action(**kwargs):
        captured["history"] = kwargs["conversation_history"]
        return {
            "response_mode": "direct",
            "reasoning": "Operator requested completion",
            "skills": [],
            "parameters": {"question": kwargs["user_question"]},
        }

    monkeypatch.setattr(logic, "_supervisor_next_action", next_action)
    result = logic.decide_node(
        {
            "user_question": "Inspect this host",
            "messages": [], "skill_results": {}, "previously_run_skills": [],
            "step_count": 1, "max_steps": 4, "evaluation": {"satisfied": False}, "trace": [],
        },
        {"configurable": {
            "available_skills": [], "llm": object(), "instruction": "",
            "guidance_provider": lambda: "Focus on persistence and then finish",
        }},
    )
    assert result["response_mode"] == "direct"
    assert "Focus on persistence" in captured["history"][-1]["content"]


def test_endpoint_investigation_forces_complete_evidence_synthesis(monkeypatch):
    captured = {}
    monkeypatch.setattr(logic, "format_response", lambda *args, **kwargs: captured.update(kwargs) or "assessment")
    result = logic.format_response_node(
        {
            "user_question": "Assess this PC",
            "messages": [],
            "skill_results": {
                "host_inventory": {"status": "ok", "platform": "linux"},
                "process_monitor": {"status": "ok", "count": 10},
            },
            "trace": [], "evaluation": {}, "step_count": 1, "max_steps": 4,
            "response_mode": "tools",
        },
        {"configurable": {"llm": object(), "cfg": None, "available_skills": []}},
    )
    assert result["response"] == "assessment"
    assert captured["force_agent_synthesis"] is True
