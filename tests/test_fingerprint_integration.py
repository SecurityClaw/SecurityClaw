from __future__ import annotations

import json
from typing import Any

from core.chat_router.logic import execute_skill_workflow, format_response, route_question
from core.db_connector import BaseDBConnector
from skills.ip_fingerprinter.logic import run as run_ip_fingerprinter
from skills.opensearch_querier.logic import run as run_opensearch


FIELD_MAPPINGS = {
    "ip_fields": ["destination.ip", "source.ip"],
    "destination_ip_fields": ["destination.ip"],
    "source_ip_fields": ["source.ip"],
    "port_fields": ["destination.port", "source.port"],
    "destination_port_fields": ["destination.port"],
    "source_port_fields": ["source.port"],
    "all_fields": ["destination.ip", "source.ip", "destination.port", "source.port", "@timestamp"],
    "field_types": {
        "destination.ip": "ip",
        "source.ip": "ip",
        "destination.port": "long",
        "source.port": "long",
        "@timestamp": "date",
    },
}


class _Cfg:
    def get(self, section: str, key: str, default=None):
        values = {
            ("db", "logs_index"): "logstash*",
            ("llm", "anti_hallucination_check"): False,
        }
        return values.get((section, key), default)


class _Indices:
    def get_mapping(self, index: str | None = None) -> dict[str, Any]:
        return {
            "logstash-test": {
                "mappings": {
                    "properties": {
                        "destination": {
                            "properties": {
                                "ip": {"type": "ip"},
                                "port": {"type": "long"},
                            }
                        },
                        "source": {
                            "properties": {
                                "ip": {"type": "ip"},
                                "port": {"type": "long"},
                            }
                        },
                        "@timestamp": {"type": "date"},
                    }
                }
            }
        }


class _Client:
    def __init__(self):
        self.indices = _Indices()
        self.search_calls: list[dict[str, Any]] = []

    def search(self, index: str | None = None, body: dict | None = None, size: int = 0) -> dict[str, Any]:
        payload = body or {}
        self.search_calls.append(payload)

        aggs = payload.get("aggs") or {}
        assert any(key.startswith("service_ports_target_destination_") for key in aggs)

        return {
            "hits": {"total": {"value": 46}},
            "aggregations": {
                "service_ports_target_destination_0": {
                    "ports": {
                        "buckets": [
                            {"key": 22, "doc_count": 30},
                            {"key": 9200, "doc_count": 12},
                            {"key": 3389, "doc_count": 4},
                        ]
                    }
                }
            },
        }


class _MockDBConnector(BaseDBConnector):
    def __init__(self):
        self._client = _Client()

    def search(self, index: str, query: dict, size: int = 100) -> list[dict]:
        return []

    def search_with_metadata(self, index: str, query: dict, size: int = 100) -> dict[str, Any]:
        return {"results": [], "total": 0}

    def aggregate(self, index: str, query: dict) -> dict[str, Any]:
        return self._client.search(index=index, body=query, size=0)

    def index_document(self, index: str, doc_id: str, body: dict) -> dict:
        raise NotImplementedError()

    def bulk_index(self, index: str, documents: list[dict]) -> dict:
        raise NotImplementedError()

    def get_anomaly_findings(self, detector_id: str, from_epoch_ms: int | None = None, size: int = 200) -> list[dict]:
        return []

    def knn_search(self, index: str, vector: list[float], k: int = 5, filters: dict | None = None) -> list[dict]:
        return []

    def ensure_index(self, index: str, mappings: dict, settings: dict | None = None) -> None:
        return None


class _RouteLLM:
    def chat(self, messages: list[dict]) -> str:
        return json.dumps(
            {
                "reasoning": "Direct passive fingerprint request for a specific IP.",
                "skills": ["ip_fingerprinter"],
                "parameters": {"ip": "192.168.0.17"},
            }
        )


class _SkillLLM:
    def __init__(self):
        self.field_plan_calls = 0
        self.fingerprint_summary_calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        if "Choose the best discovered fields for a passive IP fingerprint aggregation." in prompt:
            self.field_plan_calls += 1
            return json.dumps(
                {
                    "ip_fields": ["destination.ip"],
                    "port_fields": ["destination.port"],
                    "reasoning": "Use the destination-side IP and port fields for target-owned service evidence.",
                }
            )
        if "Interpret this passive IP fingerprint from aggregated network evidence." in prompt:
            self.fingerprint_summary_calls += 1
            return json.dumps(
                {
                    "summary": "The host behaves like a server exposing SSH, OpenSearch, and RDP.",
                    "likely_role": "server",
                    "confidence": 0.91,
                    "evidence": [
                        "Port 22 is consistently observed",
                        "Port 9200 suggests an OpenSearch service",
                        "Port 3389 indicates remote desktop exposure",
                    ],
                }
            )
        raise AssertionError(f"Unexpected prompt sent to skill LLM: {prompt[:200]}")


class _NoLLM:
    def chat(self, messages: list[dict]) -> str:
        raise AssertionError("format_response should use deterministic formatting for fingerprint results")


class _Registry:
    action = "mock"
    source = "test"
    cache_path = ""
    warning = ""

    def classify(self, port: int, protocol: str | None = None) -> dict[str, Any]:
        service_names = {
            22: "ssh",
            9200: "opensearch",
            3389: "ms-wbt-server",
        }
        return {
            "service_name": service_names.get(port, str(port)),
            "description": "",
            "registered": True,
            "range_class": "system",
            "ephemeral_likelihood": "unlikely",
            "ephemeral_reason": "",
        }


class _Runner:
    def __init__(self, db: BaseDBConnector, llm: Any):
        self.db = db
        self.llm = llm
        self.calls: list[str] = []

    def _build_context(self) -> dict[str, Any]:
        return {
            "db": self.db,
            "llm": self.llm,
            "config": _Cfg(),
        }

    def dispatch(self, skill_name: str, context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(skill_name)
        if skill_name == "fields_querier":
            return {
                "status": "ok",
                "field_mappings": FIELD_MAPPINGS,
                "findings": {"field_mappings": FIELD_MAPPINGS},
            }
        if skill_name == "opensearch_querier":
            return run_opensearch(context)
        if skill_name == "ip_fingerprinter":
            return run_ip_fingerprinter(context)
        raise AssertionError(f"Unexpected skill dispatch: {skill_name}")


def test_fingerprinting_integration_uses_mock_db(monkeypatch):
    monkeypatch.setattr(
        "skills.ip_fingerprinter.logic.load_port_registry",
        lambda cfg, force_update=False: _Registry(),
    )

    routing_decision = route_question(
        user_question="fingerprint 192.168.0.17",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "ip_fingerprinter", "description": "Passive IP fingerprinting"},
        ],
        llm=_RouteLLM(),
        instruction="test",
        conversation_history=[],
    )

    db = _MockDBConnector()
    skill_llm = _SkillLLM()
    runner = _Runner(db=db, llm=skill_llm)

    skill_results = execute_skill_workflow(
        skills=routing_decision["skills"],
        runner=runner,
        context={},
        routing_decision=routing_decision,
        conversation_history=[],
        aggregated_results={},
    )

    assert routing_decision["skills"] == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert runner.calls == ["fields_querier", "opensearch_querier", "ip_fingerprinter"]
    assert len(db._client.search_calls) == 1

    opensearch_result = skill_results["opensearch_querier"]
    assert opensearch_result["status"] == "ok"
    assert opensearch_result["aggregation_type"] == "fingerprint_ports"
    assert opensearch_result["results_count"] == 46
    assert opensearch_result["observed_ports"] == [22, 3389, 9200]
    assert opensearch_result["aggregated_ports"][22]["observations"] == 30
    assert opensearch_result["fingerprint_likely_role"] == "server"

    fingerprint_result = skill_results["ip_fingerprinter"]
    assert fingerprint_result["status"] == "ok"
    assert fingerprint_result["ip"] == "192.168.0.17"
    assert set(fingerprint_result["port_summary"]["listening_ports"]) == {22, 3389, 9200}
    assert fingerprint_result["likely_role"]["classification"] == "likely_server"

    assert skill_llm.field_plan_calls == 1
    assert skill_llm.fingerprint_summary_calls == 1

    rendered = format_response(
        "fingerprint 192.168.0.17",
        routing_decision,
        skill_results,
        llm=_NoLLM(),
        cfg=_Cfg(),
    )

    assert "Passive fingerprint for 192.168.0.17" in rendered
    assert "22 (ssh)" in rendered
    assert "9200 (opensearch)" in rendered
