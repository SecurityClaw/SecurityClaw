from __future__ import annotations

import json
from unittest.mock import Mock, patch

from core.query_repair import _is_valid_query_structure
from core.chat_router.logic import route_question
from skills.forensic_examiner import logic as forensic_logic


class _RoutingLLM:
    """Returns a deterministic routing decision for chat_router prompts."""

    def chat(self, messages: list[dict]) -> str:
        user_prompt = messages[-1].get("content", "")
        if "Analyze this security question" in user_prompt:
            return json.dumps(
                {
                    "reasoning": "explicit forensic follow-up request",
                    "skills": ["forensic_examiner"],
                    "parameters": {},
                }
            )
        return json.dumps({"reasoning": "fallback", "skills": [], "parameters": {}})


class _ForensicLLM:
    """Deterministic forensic LLM for search strategy and timeline synthesis."""

    def chat(self, messages: list[dict]) -> str:
        prompt = messages[-1].get("content", "")

        if "Design a search strategy" in prompt:
            return json.dumps(
                {
                    "summary": "Investigate prior Iran-to-target VPN traffic context",
                    "search_queries": [
                        {
                            "description": "Find Iran-origin activity to the known destination over VPN port",
                            "keywords": ["Iran", "192.168.0.16", "1194"],
                        }
                    ],
                    "time_window": "2025-12-01 to 2026-03-04",
                    "reasoning": "Use conversation context and known entities for reconstruction",
                }
            )

        if "iterative workflow" in prompt:
            return json.dumps(
                {
                    "summary": "Investigate anchored VPN probing with stepwise TODOs",
                    "time_window": "2025-12-01 to 2026-03-04",
                    "todos": [
                        {
                            "title": "Validate known Iran -> target flow",
                            "goal": "Confirm provided IP/port evidence in logs",
                            "search_queries": [
                                {
                                    "description": "Find traffic from Iran source to destination on 1194",
                                    "keywords": ["62.60.131.168", "192.168.0.16", "1194", "tcp"],
                                }
                            ],
                        }
                    ],
                    "stop_criteria": "Evidence is anchored and sufficient for narrative",
                }
            )

        if "Re-evaluate this forensic investigation progress" in prompt:
            return json.dumps(
                {
                    "is_relevant": True,
                    "is_sufficient": True,
                    "confidence": 0.9,
                    "reasoning": "Anchored evidence found for known entities",
                    "gaps": [],
                    "next_action": None,
                }
            )

        if "Build a comprehensive forensic timeline" in prompt:
            return (
                "Forensic Timeline\n"
                "- 2026-01-10 08:30:45 UTC: Iran-attributed source connected to 192.168.0.16:1194.\n"
                "- 2026-01-15 14:22:10 UTC: Repeat attempt to the same target/port.\n"
                "- 2026-01-20 09:15:30 UTC: Third recurring event with same behavior.\n"
                "Pattern Analysis: periodic 5-day interval suggests automation rather than ad-hoc human use.\n"
                "Risk Assessment: medium risk reconnaissance against exposed VPN surface; repeat cadence and stable target indicate persistence.\n"
                "Recommendations: keep destination under enhanced monitoring, block source, and correlate with auth/firewall logs."
            )

        if "Design FOLLOW-UP searches" in prompt:
            return json.dumps(
                {
                    "summary": "No additional searches needed",
                    "search_queries": [],
                    "rationale": "Enough evidence from initial run",
                }
            )

        return json.dumps({"summary": "fallback", "search_queries": []})

    def embed(self, text: str) -> list[float]:
        """Mock embedding method required by RAG engine."""
        # Return a deterministic embedding for test purposes
        return [0.1] * 384  # Standard embedding size


class _Config:
    def get(self, section: str, key: str, default=None):
        values = {
            ("db", "logs_index"): "logstash*",
            ("db", "vector_index"): "securityclaw-vectors",
        }
        return values.get((section, key), default)


def test_followup_forensic_analysis_routes_to_forensic_examiner():
    routing_llm = _RoutingLLM()
    available_skills = [
        {"name": "baseline_querier", "description": "Search behavioral logs"},
        {"name": "forensic_examiner", "description": "Build forensic timeline"},
    ]

    history = [
        {
            "role": "assistant",
            "content": (
                "The provided data shows traffic from Iran to destination IP 192.168.0.16 "
                "in the past 3 months, latest record February 13, 2026."
            ),
        }
    ]

    decision = route_question(
        "forensic analysis",
        available_skills,
        routing_llm,
        "You are a SOC assistant.",
        history,
    )

    assert decision["skills"] == ["forensic_examiner"]


def test_followup_forensic_analysis_builds_valid_queries_and_meaningful_timeline():
    forensic_llm = _ForensicLLM()
    cfg = _Config()

    captured_queries: list[dict] = []

    db = Mock()

    logs = [
        {
            "_id": f"log-{i}",
            "@timestamp": ts,
            "source.ip": "1.1.1.100",
            "destination.ip": "192.168.0.16",
            "destination.port": 1194,
            "geoip.country_code2": "IR",
            "event.message": "connection attempt",
        }
        for i, ts in enumerate(
            [
                "2026-01-10T08:30:45Z",
                "2026-01-15T14:22:10Z",
                "2026-01-20T09:15:30Z",
                "2026-01-25T11:00:00Z",
                "2026-01-30T07:10:00Z",
            ],
            start=1,
        )
    ]

    def _search(index: str, query: dict, size: int = 100):
        captured_queries.append(query)
        ok, reason = _is_valid_query_structure(query)
        assert ok, f"Invalid query structure: {reason} | query={json.dumps(query)}"
        return logs

    db.search = Mock(side_effect=_search)

    field_docs = """
    - source.ip (IPv4 address): Source IP
    - destination.ip (IPv4 address): Destination IP
    - destination.port (Port number): Destination port
    - geoip.country_code2 (Text): Country code
    - event.message (Text): Event message
    - @timestamp (Timestamp): Event time
    """

    context = {
        "db": db,
        "llm": forensic_llm,
        "config": cfg,
        "parameters": {"question": "forensic analysis"},
        "conversation_history": [
            {
                "role": "assistant",
                "content": (
                    "The provided data shows traffic from Iran to destination IP 192.168.0.16 "
                    "on port 1194 in the past 3 months."
                ),
            }
        ],
    }

    with patch("skills.forensic_examiner.logic._fetch_field_documentation", return_value=field_docs):
        result = forensic_logic.run(context)

    assert result["status"] == "ok"
    report = result["forensic_report"]
    assert report["results_found"] >= 5
    assert len(captured_queries) >= 1

    timeline = report["timeline_narrative"]
    assert len(timeline) > 250
    assert "2026-01-10" in timeline
    assert "192.168.0.16" in timeline
    assert "1194" in timeline
    assert "Pattern" in timeline or "pattern" in timeline
    assert "Risk" in timeline or "risk" in timeline
