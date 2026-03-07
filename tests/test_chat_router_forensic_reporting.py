from __future__ import annotations

from skills.chat_router.logic import execute_skill_workflow, format_response


class _RunnerStub:
    def __init__(self):
        self.calls: list[str] = []
        self.contexts: dict[str, dict] = {}

    def _build_context(self):
        return {}

    def dispatch(self, skill_name: str, context: dict):
        self.calls.append(skill_name)
        self.contexts[skill_name] = context
        if skill_name == "forensic_examiner":
            return {
                "status": "ok",
                "forensic_report": {
                    "incident_summary": "Iran traffic from 62.60.131.168 to 192.168.0.16 port 1194",
                    "context_anchors": {
                        "ips": ["62.60.131.168", "192.168.0.16"],
                        "ports": ["1194"],
                        "countries": ["iran"],
                        "protocols": ["tcp"],
                    },
                    "results_found": 102,
                    "refinement_rounds": 1,
                    "timeline_narrative": (
                        "2026-02-10 12:10:00 UTC first event from 62.60.131.168 to 192.168.0.16:1194. "
                        "2026-02-13 13:10:00 UTC repeat event. Pattern appears periodic every 3 days with medium risk."
                    ),
                },
            }
        if skill_name == "threat_analyst":
            return {
                "status": "ok",
                "verdicts": [
                    {
                        "verdict": "TRUE_THREAT",
                        "confidence": 84,
                        "reasoning": "IP reputation and repeated cadence indicate likely malicious probing.",
                    }
                ],
            }
        return {"status": "ok"}


class _LLMUnused:
    def chat(self, messages):
        return "unused"


def test_execute_skill_workflow_auto_chains_threat_analyst():
    runner = _RunnerStub()
    routing_decision = {"parameters": {"question": "forensic analysis"}}

    results = execute_skill_workflow(
        ["forensic_examiner"],
        runner,
        {},
        routing_decision,
        conversation_history=[{"role": "assistant", "content": "prior Iran findings"}],
    )

    assert "forensic_examiner" in results
    assert "threat_analyst" in results
    assert runner.calls == ["forensic_examiner", "threat_analyst"]
    threat_question = runner.contexts["threat_analyst"]["parameters"]["question"]
    assert "62.60.131.168" in threat_question
    assert "192.168.0.16" in threat_question
    assert "1194" in threat_question


def test_execute_skill_workflow_passes_same_step_results_to_later_skills():
    class _SequentialRunner:
        def __init__(self):
            self.contexts: dict[str, dict] = {}

        def _build_context(self):
            return {}

        def dispatch(self, skill_name: str, context: dict):
            self.contexts[skill_name] = context
            if skill_name == "fields_querier":
                return {
                    "status": "ok",
                    "field_mappings": {"country_fields": ["geoip.country_name"]},
                }
            if skill_name == "opensearch_querier":
                return {"status": "ok", "results": [], "results_count": 0}
            return {"status": "ok"}

    runner = _SequentialRunner()
    routing_decision = {"parameters": {"question": "What countries are these IPs from?"}}

    execute_skill_workflow(
        ["fields_querier", "opensearch_querier"],
        runner,
        {},
        routing_decision,
        conversation_history=[{"role": "assistant", "content": "IPs: 1.2.3.4"}],
    )

    previous_results = runner.contexts["opensearch_querier"].get("previous_results", {})
    assert "fields_querier" in previous_results
    assert previous_results["fields_querier"]["field_mappings"]["country_fields"] == ["geoip.country_name"]


def test_format_response_forensic_is_detailed_and_multi_paragraph():
    routing = {"skills": ["forensic_examiner"]}
    skill_results = {
        "forensic_examiner": {
            "status": "ok",
            "forensic_report": {
                "incident_summary": "Iran traffic from 62.60.131.168 to 192.168.0.16 on port 1194",
                "results_found": 102,
                "refinement_rounds": 2,
                "timeline_narrative": (
                    "2026-02-10 12:10:00 UTC: initial connection observed.\n"
                    "2026-02-13 13:10:00 UTC: second connection observed.\n"
                    "Pattern is periodic with 3-day intervals and medium risk posture."
                ),
            },
        },
        "threat_analyst": {
            "status": "ok",
            "verdicts": [
                {
                    "verdict": "TRUE_THREAT",
                    "confidence": 90,
                    "reasoning": "Abuse history plus recurring traffic pattern indicates coordinated probing.",
                }
            ],
        },
    }

    output = format_response("forensic analysis", routing, skill_results, _LLMUnused(), cfg=None)

    paragraphs = [p for p in output.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 3
    assert "Timeline" in output
    assert "Pattern" in output or "pattern" in output
    assert "IPs involved" in output
    assert "Ports involved" in output
    assert "Reputation and threat intel" in output
