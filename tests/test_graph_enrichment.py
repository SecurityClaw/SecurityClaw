from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock


class _Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, section, key, default=None):
        return self.values.get((section, key), default)


def test_graph_enrichment_uses_specialist_context_before_synthesis(monkeypatch, tmp_path):
    from core import graph_enrichment as module

    monkeypatch.setattr(module, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(module, "_cache", {})
    specialist = Mock(return_value={"sources": ["local_telemetry", "abuseipdb"], "reputation": {"combined_risk": "LOW"}})
    monkeypatch.setattr(module, "_specialist_context", specialist)

    llm = Mock()
    llm.chat.return_value = """[
      {"id":"network:one","description":"Public TLS peer with low reported reputation risk.","why":"Observed by the connection sensor.","security_relevance":"Validate ownership and expected process use before trusting the peer."}
    ]"""
    nodes = [{
        "id": "network:one",
        "name": "1.1.1.1:443",
        "type": "network",
        "evidence": {"remote": "1.1.1.1:443", "protocol": "tcp"},
    }]

    result = module.enrich_graph_nodes(nodes, llm=llm, cfg=_Config())

    specialist.assert_called_once()
    llm.chat.assert_called_once()
    assert result[0]["id"] == "network:one"
    assert result[0]["enrichmentSources"] == ["local_telemetry", "abuseipdb"]


def test_duckduckgo_is_only_used_when_specialist_cve_context_is_empty(monkeypatch):
    from core import graph_enrichment as module

    monkeypatch.setattr(module, "_osv", lambda cve, timeout: {})
    fallback = Mock(return_value={"summary": "Technical advisory context", "source": "DuckDuckGo Instant Answer"})
    monkeypatch.setattr(module, "_duckduckgo", fallback)
    node = {"id": "vulnerability:CVE-2026-12345", "name": "CVE-2026-12345", "type": "vulnerability", "evidence": {}}

    context = module._specialist_context(node, _Config(), 2, True)

    fallback.assert_called_once()
    assert context["sources"] == ["local_telemetry", "duckduckgo_instant_answer"]


def test_duckduckgo_is_not_used_when_osv_has_an_advisory(monkeypatch):
    from core import graph_enrichment as module

    monkeypatch.setattr(module, "_osv", lambda cve, timeout: {"id": cve, "summary": "Grounded OSV advisory"})
    fallback = Mock()
    monkeypatch.setattr(module, "_duckduckgo", fallback)
    node = {"id": "vulnerability:CVE-2026-12345", "name": "CVE-2026-12345", "type": "vulnerability", "evidence": {}}

    context = module._specialist_context(node, _Config(), 2, True)

    fallback.assert_not_called()
    assert "osv" in context["sources"]


def test_private_network_nodes_never_trigger_public_reputation_queries(monkeypatch, tmp_path):
    from core import graph_enrichment as module

    monkeypatch.setattr(module, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(module, "_cache", {})
    reputation = Mock()
    monkeypatch.setattr(module, "_enrichers", {"geoip": None, "ip_reputation": reputation, "domain_reputation": None})
    llm = SimpleNamespace(chat=lambda *args, **kwargs: "[]")
    nodes = [{"id": "network:private", "name": "10.0.0.5:443", "type": "network", "evidence": {"remote": "10.0.0.5:443"}}]

    module.enrich_graph_nodes(nodes, llm=llm, cfg=_Config())

    reputation.assert_not_called()
