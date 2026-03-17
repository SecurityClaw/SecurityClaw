from __future__ import annotations

import json
from unittest.mock import MagicMock

from skills.fields_baseliner import logic as baseliner_logic
from skills.fields_querier import logic as fields_querier_logic
from tests.mock_opensearch import MockDBConnector


def test_fields_baseliner_records_aggregated_country_values(tmp_path, monkeypatch):
    db = MockDBConnector()
    db.seed_documents(
        "securityclaw-logs",
        [
            {"@timestamp": "2026-03-01T00:00:00Z", "geoip": {"country_name": "Greece"}, "proto": "TCP"},
            {"@timestamp": "2026-03-02T00:00:00Z", "geoip": {"country_name": "Greece"}, "proto": "UDP"},
            {"@timestamp": "2026-03-03T00:00:00Z", "geoip": {"country_name": "Iran"}, "proto": "TCP"},
        ],
    )

    output_file = tmp_path / "fields_rag.json"
    state_file = tmp_path / "fields_state.json"
    monkeypatch.setattr(baseliner_logic, "DATA_DIR", tmp_path)
    monkeypatch.setattr(baseliner_logic, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(baseliner_logic, "STATE_FILE", state_file)

    result = baseliner_logic.run(
        {
            "db": db,
            "config": MagicMock(get=lambda section, key, default=None: "securityclaw-logs" if (section, key) == ("db", "logs_index") else default),
            "parameters": {"force_refresh": True},
        }
    )

    docs = json.loads(output_file.read_text(encoding="utf-8"))
    field_doc = next(doc for doc in docs if doc.get("category") == "field_documentation")
    country_info = field_doc["fields"]["geoip.country_name"]

    assert result["status"] == "ok"
    assert result["values_profiled"] >= 1
    assert country_info["aggregation_field"] in {"geoip.country_name", "geoip.country_name.keyword"}
    assert country_info["top_values"][0]["value"] == "Greece"
    assert country_info["top_values"][0]["count"] == 2
    assert any(entry["value"] == "Iran" for entry in country_info["top_values"])


def test_fields_querier_exposes_country_values_from_fields_rag(tmp_path, monkeypatch):
    fields_file = tmp_path / "fields_rag.json"
    fields_file.write_text(
        json.dumps(
            [
                {
                    "category": "field_documentation",
                    "fields": {
                        "geoip.country_name": {
                            "inferred_type": "geo/string",
                            "examples": ["Greece", "Iran"],
                            "top_values": [
                                {"value": "Greece", "count": 7},
                                {"value": "Iran", "count": 3},
                            ],
                        }
                    },
                    "text": "FIELD: geoip.country_name\n  Top Values: Greece (7), Iran (3)",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fields_querier_logic, "FIELDS_FILE", fields_file)

    result = fields_querier_logic.run(
        {
            "llm": None,
            "parameters": {"question": "what country field values exist?"},
        }
    )

    mappings = result["field_mappings"]
    assert mappings["country_fields"] == ["geoip.country_name"]
    assert mappings["country_values"] == ["Greece", "Iran"]
    assert mappings["field_value_examples"]["geoip.country_name"] == ["Greece", "Iran"]