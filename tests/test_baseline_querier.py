from __future__ import annotations

from unittest.mock import MagicMock, patch

from skills.baseline_querier import logic


def test_baseline_querier_heuristic_plan_falls_back_to_ip_search_when_llm_plan_fails():
    llm = MagicMock()
    llm.complete.return_value = "not json"
    llm.chat.return_value = "1.1.1.1 appears regularly as destination-side traffic in this network."

    db = MagicMock()
    captured_query = {}

    def _search(index: str, query: dict, size: int = 50):
        captured_query["value"] = query
        return [
            {
                "_id": "evt-1",
                "@timestamp": "2026-03-11T23:43:41.898Z",
                "src_ip": "192.168.0.85",
                "dest_ip": "1.1.1.1",
                "dest_port": 53,
            }
        ]

    db.search.side_effect = _search

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = []

    with patch("core.query_builder.discover_field_mappings", return_value={
        "all_fields": ["src_ip", "dest_ip", "dest_port", "@timestamp"],
        "ip_fields": ["src_ip", "dest_ip"],
        "port_fields": ["dest_port"],
        "timestamp_fields": ["@timestamp"],
        "text_fields": [],
    }), patch("core.rag_engine.RAGEngine", return_value=mock_rag):
        result = logic.run(
            {
                "db": db,
                "llm": llm,
                "memory": None,
                "config": MagicMock(get=lambda section, key, default=None: "logstash*" if (section, key) == ("db", "logs_index") else "securityclaw-vectors" if (section, key) == ("db", "vector_index") else default),
                "parameters": {"question": "Is 1.1.1.1 normal behavior in this network?"},
            }
        )

    must_clause = captured_query["value"]["query"]["bool"]["must"][0]["bool"]["should"]
    queried_fields = {
        next(iter(clause["term"].keys()))
        for clause in must_clause
        if "term" in clause
    }

    assert queried_fields == {"src_ip", "dest_ip"}
    assert result["status"] == "ok"
    assert result["findings"]["log_records"] == 1
    assert "1.1.1.1" in result["findings"]["evidence"]["ips"]


def test_baseline_querier_emits_grounded_assessment_for_focus_ip():
    llm = MagicMock()
    llm.complete.return_value = "not json"
    llm.chat.return_value = "Generic summary that should not drive the baseline response."

    db = MagicMock()

    def _search(index: str, query: dict, size: int = 50):
        return [
            {
                "_id": "evt-1",
                "@timestamp": "2026-03-11T23:43:41.898Z",
                "src_ip": "192.168.0.85",
                "dest_ip": "1.1.1.1",
                "dest_port": 53,
                "protocol": "dns",
            },
            {
                "_id": "evt-2",
                "@timestamp": "2026-03-11T23:43:56.274Z",
                "src_ip": "192.168.0.142",
                "dest_ip": "1.1.1.1",
                "dest_port": 53,
                "protocol": "dns",
            },
        ]

    db.search.side_effect = _search

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [{"text": "DNS to public resolvers is common.", "category": "dns_baseline"}]

    with patch("core.query_builder.discover_field_mappings", return_value={
        "all_fields": ["src_ip", "dest_ip", "dest_port", "protocol", "@timestamp"],
        "ip_fields": ["src_ip", "dest_ip"],
        "port_fields": ["dest_port"],
        "timestamp_fields": ["@timestamp"],
        "text_fields": [],
    }), patch("core.rag_engine.RAGEngine", return_value=mock_rag):
        result = logic.run(
            {
                "db": db,
                "llm": llm,
                "memory": None,
                "config": MagicMock(get=lambda section, key, default=None: "logstash*" if (section, key) == ("db", "logs_index") else "securityclaw-vectors" if (section, key) == ("db", "vector_index") else default),
                "parameters": {"question": "Is 1.1.1.1 normal behavior in this network?"},
            }
        )

    grounded_assessment = result["findings"]["grounded_assessment"]
    observations = result["findings"]["observations"]["entities"]["1.1.1.1"]

    assert "routine destination-side DNS traffic" in grounded_assessment
    assert "It matched 2 log record(s)" in grounded_assessment
    assert observations["source_records"] == 0
    assert observations["destination_records"] == 2
    assert observations["peer_ips"] == ["192.168.0.142", "192.168.0.85"]


def test_baseline_querier_grounded_assessment_uses_discovered_nested_fields():
    llm = MagicMock()
    llm.complete.return_value = "not json"
    llm.chat.return_value = "Generic summary"

    db = MagicMock()
    db.search.return_value = [
        {
            "_id": "evt-1",
            "@timestamp": "2026-03-11T23:43:41.898Z",
            "source": {"ip": "192.168.0.85", "port": 55321},
            "destination": {"ip": "1.1.1.1", "port": 53},
            "network": {"protocol": "dns"},
        },
        {
            "_id": "evt-2",
            "@timestamp": "2026-03-11T23:43:56.274Z",
            "source": {"ip": "192.168.0.142", "port": 55322},
            "destination": {"ip": "1.1.1.1", "port": 53},
            "network": {"protocol": "dns"},
        },
    ]

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = []

    with patch("core.query_builder.discover_field_mappings", return_value={
        "all_fields": [
            "source.ip", "destination.ip", "source.port", "destination.port", "network.protocol", "@timestamp",
        ],
        "ip_fields": ["source.ip", "destination.ip"],
        "port_fields": ["source.port", "destination.port"],
        "timestamp_fields": ["@timestamp"],
        "text_fields": [],
    }), patch("core.rag_engine.RAGEngine", return_value=mock_rag):
        result = logic.run(
            {
                "db": db,
                "llm": llm,
                "memory": None,
                "config": MagicMock(get=lambda section, key, default=None: "logstash*" if (section, key) == ("db", "logs_index") else "securityclaw-vectors" if (section, key) == ("db", "vector_index") else default),
                "parameters": {"question": "Is 1.1.1.1 normal behavior in this network?"},
            }
        )

    grounded_assessment = result["findings"]["grounded_assessment"]
    observations = result["findings"]["observations"]["entities"]["1.1.1.1"]

    assert "routine destination-side DNS traffic" in grounded_assessment
    assert observations["source_records"] == 0
    assert observations["destination_records"] == 2
    assert observations["peer_ips"] == ["192.168.0.142", "192.168.0.85"]