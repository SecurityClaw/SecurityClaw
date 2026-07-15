from core.chat_router.logic import _chat_response, format_response


class _StreamingLLM:
    def __init__(self):
        self.stream_calls = 0
        self.chat_calls = 0

    def stream_chat(self, messages, *, token_callback):
        self.stream_calls += 1
        token_callback("Security")
        token_callback("Claw")
        return "SecurityClaw"

    def chat(self, messages):
        self.chat_calls += 1
        return "blocking"


def test_chat_response_forwards_streamed_tokens_with_phase():
    llm = _StreamingLLM()
    events = []

    response = _chat_response(
        llm,
        [{"role": "user", "content": "hello"}],
        token_callback=lambda phase, token: events.append((phase, token)),
        phase="answer",
    )

    assert response == "SecurityClaw"
    assert events == [("answer", "Security"), ("answer", "Claw")]
    assert llm.stream_calls == 1
    assert llm.chat_calls == 0


def test_chat_response_falls_back_to_blocking_without_callback():
    llm = _StreamingLLM()

    response = _chat_response(
        llm,
        [{"role": "user", "content": "hello"}],
    )

    assert response == "blocking"
    assert llm.stream_calls == 0
    assert llm.chat_calls == 1


class _DirectAnswerLLM:
    def __init__(self):
        self.messages = None

    def stream_chat(self, messages, *, token_callback):
        self.messages = messages
        tokens = ["I am SecurityClaw. ", "I help investigate security events."]
        for token in tokens:
            token_callback(token)
        return "".join(tokens)


def test_no_tool_plan_answers_capability_question_instead_of_returning_error():
    llm = _DirectAnswerLLM()
    events = []

    response = format_response(
        "What are you and which skills are available?",
        {"skills": []},
        {},
        llm,
        available_skills=[
            {"name": "geoip_lookup", "description": "Resolve IP geolocation"},
            {"name": "threat_analyst", "description": "Assess threat reputation"},
        ],
        token_callback=lambda phase, token: events.append((phase, token)),
        conversation_history=[
            {"role": "assistant", "content": "The previous scan found port 443."},
        ],
    )

    assert response.startswith("I am SecurityClaw")
    assert events[0] == ("answer", "I am SecurityClaw. ")
    assert "same language as the user" in llm.messages[0]["content"]
    assert "geoip_lookup" in llm.messages[1]["content"]
    assert "threat_analyst" in llm.messages[1]["content"]
    assert "The previous scan found port 443" in llm.messages[1]["content"]


def test_agent_synthesis_uses_all_actions_and_observations():
    llm = _DirectAnswerLLM()

    response = format_response(
        "Assess this host",
        {"skills": ["search"]},
        {"search": {"status": "ok", "results_count": 3}},
        llm,
        execution_trace=[
            {
                "step": 1,
                "decision": {"skills": ["search"]},
                "observations": {"search": {"results_count": 3}},
            }
        ],
        force_agent_synthesis=True,
        token_callback=lambda *_: None,
    )

    assert response.startswith("I am SecurityClaw")
    assert "Agent actions and observations" in llm.messages[1]["content"]
    assert '"results_count": 3' in llm.messages[1]["content"]
    assert "complete investigation" in llm.messages[0]["content"]
