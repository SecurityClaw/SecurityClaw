from core.llm_provider import OllamaProvider


class _Config:
    values = {
        ("llm", "ollama_base_url"): "http://localhost:11434",
        ("llm", "ollama_model"): "test-chat",
        ("llm", "ollama_embed_model"): "test-embed",
        ("llm", "temperature"): 0.2,
        ("llm", "think"): False,
        ("llm", "max_tokens"): 256,
        ("llm", "context_window"): 8192,
        ("llm", "request_timeout_seconds"): 600,
        ("llm", "embedding_timeout_seconds"): 120,
    }

    def get(self, *keys, default=None):
        return self.values.get(keys, default)


class _Response:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "ok"}, "embeddings": [[0.1, 0.2]]}

    def iter_lines(self, **kwargs):
        self.iter_lines_kwargs = kwargs
        yield '{"message":{"content":"Security"}}'
        yield '{"message":{"content":"Claw"}}'


class _Requests:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response()


def test_chat_uses_configurable_local_generation_timeout(monkeypatch):
    monkeypatch.setattr("core.llm_provider.Config", _Config)
    provider = OllamaProvider()
    requests = _Requests()
    provider._requests = requests

    assert provider.chat([{"role": "user", "content": "hello"}]) == "ok"
    assert requests.calls[0][1]["timeout"] == (10, 600)
    assert requests.calls[0][1]["json"]["think"] is False
    assert requests.calls[0][1]["json"]["options"]["num_predict"] == 256
    assert requests.calls[0][1]["json"]["options"]["num_ctx"] == 8192


def test_embed_uses_separate_configurable_timeout(monkeypatch):
    monkeypatch.setattr("core.llm_provider.Config", _Config)
    provider = OllamaProvider()
    requests = _Requests()
    provider._requests = requests

    assert provider.embed("hello") == [0.1, 0.2]
    assert requests.calls[0][1]["timeout"] == (10, 120)


def test_stream_chat_forwards_each_ollama_token_without_buffering(monkeypatch):
    monkeypatch.setattr("core.llm_provider.Config", _Config)
    provider = OllamaProvider()
    requests = _Requests()
    provider._requests = requests
    tokens = []

    response = provider.stream_chat(
        [{"role": "user", "content": "hello"}],
        token_callback=tokens.append,
    )

    assert response == "SecurityClaw"
    assert tokens == ["Security", "Claw"]
    assert requests.calls[0][1]["stream"] is True
    assert requests.calls[0][1]["timeout"] == (10, 600)
