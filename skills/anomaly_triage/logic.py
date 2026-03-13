"""
skills/anomaly_triage/logic.py

Polls the anomaly detection index every minute.
Enriches raw findings with LLM descriptions and queues them
for ThreatAnalyst via agent memory.

Context keys consumed:
    context["db"]     -> BaseDBConnector
    context["llm"]    -> BaseLLMProvider
    context["memory"] -> Memory instance (StateBackedMemory or CheckpointBackedMemory)
    context["config"] -> Config
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "anomaly_triage"

# In-process last-seen cursor (persists across scheduler ticks for the
# lifetime of the process; production would persist this to disk/DB).
_last_poll_epoch_ms: Optional[int] = None


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    global _last_poll_epoch_ms

    db = context.get("db")
    llm = context.get("llm")
    memory = context.get("memory")
    cfg = context.get("config")

    if db is None:
        logger.warning("[%s] db not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db"}

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    detector_id = cfg.get("anomaly", "detector_id", default="default-detector")
    max_findings = cfg.get("anomaly", "max_findings_per_poll", default=50)
    severity_threshold = cfg.get("anomaly", "severity_threshold", default=0.7)

    # ── 1. Poll for new findings ───────────────────────────────────────────────
    from_ms = _last_poll_epoch_ms
    findings = db.get_anomaly_findings(
        detector_id=detector_id,
        from_epoch_ms=from_ms,
        size=int(max_findings),
    )

    if not findings:
        logger.debug("[%s] No new findings since %s.", SKILL_NAME, from_ms)
        return {"status": "ok", "new_findings": 0}

    logger.info("[%s] %d new finding(s) found.", SKILL_NAME, len(findings))

    # ── 2. Update cursor ───────────────────────────────────────────────────────
    _last_poll_epoch_ms = _epoch_ms_now()

    enriched = []
    escalated = []

    for raw in findings:
        score = raw.get("anomaly_score", raw.get("score", 0.0))
        if score < severity_threshold:
            logger.debug("[%s] Skipping low-score finding (%.2f)", SKILL_NAME, score)
            continue

        # ── 3. LLM enrichment ─────────────────────────────────────────────────
        if llm:
            enriched_finding = _enrich_with_llm(raw, instruction, llm)
        else:
            enriched_finding = _bare_enrich(raw)

        enriched.append(enriched_finding)

        # ── 4. Write to agent memory ───────────────────────────────────────────
        if memory:
            desc = enriched_finding.get("description", str(raw)[:120])
            sev = enriched_finding.get("severity", "UNKNOWN")
            memory.add_finding(f"[{sev}] {desc}")

            if sev in ("HIGH", "CRITICAL"):
                memory.escalate(f"[{sev}] Needs ThreatAnalyst review: {desc}")
                escalated.append(enriched_finding)

    return {
        "status": "ok",
        "new_findings": len(findings),
        "enriched": len(enriched),
        "escalated": len(escalated),
        "findings": enriched,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_with_llm(raw: dict, instruction: str, llm) -> dict:
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                "Enrich this raw anomaly detection finding:\n\n"
                f"```json\n{json.dumps(raw, indent=2)}\n```"
            ),
        },
    ]
    try:
        response = llm.chat(messages)
        parsed = _parse_json(response)
        if parsed:
            parsed["_raw"] = raw
            return parsed
        return {**_bare_enrich(raw), "_llm_raw": response}
    except Exception as exc:
        logger.error("[%s] LLM enrichment failed: %s", SKILL_NAME, exc)
        return _bare_enrich(raw)


def _bare_enrich(raw: dict) -> dict:
    """Minimal enrichment without LLM."""
    score = raw.get("anomaly_score", raw.get("score", 0.0))
    severity = _score_to_severity(float(score))
    return {
        "detector": raw.get("detector_id", "unknown"),
        "entity": raw.get("entity", {}).get("value", raw.get("entity", "unknown")),
        "score": score,
        "severity": severity,
        "description": f"Anomaly score {score:.2f} detected by {raw.get('detector_id', 'unknown')}.",
        "_raw": raw,
    }


def _score_to_severity(score: float) -> str:
    if score >= 0.95:
        return "CRITICAL"
    if score >= 0.85:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    return "LOW"


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _epoch_ms_now() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
