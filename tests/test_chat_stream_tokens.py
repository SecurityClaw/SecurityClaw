from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException


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


async def _request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


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
        app.state.service = _DummyService(enable_scheduler=False)
        app.state.checkpointer = SimpleNamespace()
        response = asyncio.run(_request(app, "POST", "/api/chat/stream", json={"message": "test stream"}))

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


def test_spa_routes_return_index_and_unknown_api_remains_404():
    from web.api import server

    with patch.object(server, "SecurityClawService", _DummyService):
        app = server.create_app(enable_scheduler=False)
        fallback = next(route.endpoint for route in app.routes if route.path == "/{full_path:path}")
        favicon = next(route.endpoint for route in app.routes if route.path == "/favicon.ico")

    chat_response = asyncio.run(fallback("chat/5ceb87d3"))
    status_response = asyncio.run(fallback("status"))
    favicon_response = asyncio.run(favicon())

    assert chat_response.path == server.DIST_DIR / "index.html"
    assert status_response.path == server.DIST_DIR / "index.html"
    assert favicon_response.status_code == 204
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(fallback("api/does-not-exist"))
    assert exc_info.value.status_code == 404


def test_chat_stream_persists_failed_turn(monkeypatch, tmp_path):
    from core.chat_router import logic
    from web.api import server

    monkeypatch.setattr(logic, "CONVERSATIONS_DIR", tmp_path)
    monkeypatch.setattr(server, "add_user_message_to_history", logic.add_user_message_to_history)
    monkeypatch.setattr(server, "add_assistant_message_to_history", logic.add_assistant_message_to_history)

    def _failing_run_graph(**kwargs):
        raise TimeoutError("model timed out")

    with patch.object(server, "SecurityClawService", _DummyService), patch.object(server, "run_graph", _failing_run_graph):
        app = server.create_app(enable_scheduler=False)
        app.state.service = _DummyService(enable_scheduler=False)
        app.state.checkpointer = SimpleNamespace()
        response = asyncio.run(_request(
            app,
            "POST",
            "/api/chat/stream",
            json={"message": "keep this question", "conversation_id": "failed123"},
        ))
        conversation = asyncio.run(_request(app, "GET", "/api/conversations/failed123")).json()

    assert response.status_code == 200
    assert "event: error" in response.text
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][0]["content"] == "keep this question"
    assert conversation["messages"][1]["error"] is True
