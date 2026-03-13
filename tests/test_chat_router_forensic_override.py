from __future__ import annotations

import json

from core.chat_router.logic import route_question


class _StaticRoutingLLM:
    def __init__(self, skills: list[str]) -> None:
        self._skills = skills

    def chat(self, messages: list[dict]) -> str:
        return json.dumps(
            {
                "reasoning": "test routing response",
                "skills": self._skills,
                "parameters": {},
            }
        )


def _available_skills() -> list[dict]:
    return [
        {"name": "opensearch_querier", "description": "Direct log search"},
        {"name": "forensic_examiner", "description": "Build forensic timeline"},
    ]


def test_explicit_forensic_analysis_forces_forensic_examiner():
    llm = _StaticRoutingLLM(["baseline_querier"])

    result = route_question(
        "forensic analysis",
        _available_skills(),
        llm,
        "You are a SOC assistant.",
    )

    assert result["skills"] == ["forensic_examiner"]


def test_forensic_with_search_filters_chains_baseline_then_forensic():
    llm = _StaticRoutingLLM(["baseline_querier"])

    result = route_question(
        "forensic analysis of traffic from Iran on port 1194",
        _available_skills(),
        llm,
        "You are a SOC assistant.",
    )

    assert result["skills"] == ["opensearch_querier", "forensic_examiner"]


def test_non_forensic_query_keeps_selected_skills():
    llm = _StaticRoutingLLM(["baseline_querier"])

    result = route_question(
        "traffic from Iran in the past 3 months",
        _available_skills(),
        llm,
        "You are a SOC assistant.",
    )

    assert result["skills"] == ["opensearch_querier"]
