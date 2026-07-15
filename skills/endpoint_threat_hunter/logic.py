"""Scheduled, read-only endpoint change detection and threat hunting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.endpoint_security import (
    collect_file_integrity,
    collect_network_connections,
    collect_persistence,
    collect_processes,
    collect_security_posture,
)

STATE_PATH = Path("data/endpoint_threat_hunter_state.json")


def _keys(items: list[dict], fields: tuple[str, ...]) -> set[str]:
    return {"|".join(str(item.get(field) or "") for field in fields) for item in items}


def _load_previous() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(snapshot: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    temporary.replace(STATE_PATH)


def run(context: dict) -> dict:
    processes = collect_processes(limit=1000)
    network = collect_network_connections(limit=1000)
    persistence = collect_persistence(limit=1000)
    integrity = collect_file_integrity(max_files=2000)
    posture = collect_security_posture(limit=1000)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processes": processes.get("processes") or [],
        "connections": network.get("connections") or [],
        "persistence": persistence.get("persistence") or [],
        "files": integrity.get("files") or [],
        "checks": posture.get("checks") or {},
    }
    previous = _load_previous()
    if not previous:
        _save(snapshot)
        return {
            "status": "initialized",
            "message": "Endpoint threat-hunting baseline created.",
            "findings": [],
            "finding_count": 0,
            "coverage": ["processes", "network_connections", "persistence", "file_integrity", "security_posture"],
            "baseline_counts": {key: len(value) for key, value in snapshot.items() if isinstance(value, list)},
        }

    comparisons = {
        "new_processes": (_keys(snapshot["processes"], ("pid", "name", "executable")) - _keys(previous.get("processes") or [], ("pid", "name", "executable"))),
        "new_connections": (_keys(snapshot["connections"], ("protocol", "local", "remote", "process")) - _keys(previous.get("connections") or [], ("protocol", "local", "remote", "process"))),
        "new_persistence": (_keys(snapshot["persistence"], ("type", "path", "target", "TaskName")) - _keys(previous.get("persistence") or [], ("type", "path", "target", "TaskName"))),
    }
    previous_hashes = {item.get("path"): item.get("sha256") for item in previous.get("files") or []}
    changed_files = [item for item in snapshot["files"] if item.get("path") in previous_hashes and previous_hashes[item.get("path")] != item.get("sha256")]
    findings = [{"type": name, "count": len(values), "evidence": sorted(values)[:100]} for name, values in comparisons.items() if values]
    if changed_files:
        findings.append({"type": "file_integrity_changes", "count": len(changed_files), "evidence": changed_files[:100]})
    if snapshot["checks"] != previous.get("checks", {}):
        findings.append({"type": "security_posture_change", "count": 1, "evidence": {"before": previous.get("checks", {}), "after": snapshot["checks"]}})

    llm_analysis = None
    llm = context.get("llm")
    if findings and llm is not None:
        prompt = (
            "You are performing defensive endpoint threat hunting. Analyze only the supplied change evidence. "
            "Identify which changes merit investigation, explain why, and state uncertainty. Do not infer malware "
            "from a process name alone and do not request containment without corroborating evidence. Return JSON "
            "with keys severity, summary, suspicious_entities, recommended_follow_up.\n\nEvidence:\n"
            + json.dumps(findings, default=str)[:24000]
        )
        try:
            llm_analysis = llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            llm_analysis = f"LLM analysis unavailable: {exc}"
    _save(snapshot)
    return {"status": "ok", "timestamp": snapshot["timestamp"], "findings": findings, "finding_count": sum(item["count"] for item in findings), "analysis": llm_analysis, "coverage": ["processes", "network_connections", "persistence", "file_integrity", "security_posture"]}
