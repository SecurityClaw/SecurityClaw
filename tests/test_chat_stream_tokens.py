from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


class _DummyService:
    def __init__(self, enable_scheduler: bool = True):
        self.enable_scheduler = enable_scheduler
        self.context = SimpleNamespace(
            runner=SimpleNamespace(),
            llm=SimpleNamespace(),
            cfg=SimpleNamespace(),
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def restart(self) -> None:
        return None


def test_chat_stream_forwards_token_events_and_step_events():
    from web.api import server

    def _fake_run_graph(**kwargs):
        step_callback = kwargs["step_callback"]
        step_callback("deciding", {"reasoning": "Need a lookup", "skills": ["opensearch_querier"]}, 1, 4)
        step_callback("token", {"phase": "think", "token": "Thought"}, 1, 4)
        step_callback("token", {"phase": "answer", "token": "Answer"}, 1, 4)
        return {
            "response": "ThoughtAnswer",
            "routing": {"skills": ["opensearch_querier"]},
            "trace": [],
            "skill_results": {},
        }

    with patch.object(server, "SecurityClawService", _DummyService), patch.object(server, "run_graph", _fake_run_graph):
        app = server.create_app(enable_scheduler=False)
        with TestClient(app) as client:
            response = client.post("/api/chat/stream", json={"message": "test stream"})

    body = response.text
    assert response.status_code == 200
    assert "event: meta" in body
    assert "event: step" in body
    assert '"kind": "thinking"' in body
    assert "event: token" in body
    assert '"phase": "think"' in body
    assert '"token": "Thought"' in body
    assert '"phase": "answer"' in body
    assert '"token": "Answer"' in body
    assert "event: response" in body
    assert '"response": "ThoughtAnswer"' in body