"""Persistent notifications emitted by manifest-declared passive skills."""

from __future__ import annotations

import json
import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AlertStore:
    def __init__(self, path: Path | str = Path("data/passive_alerts.json"), limit: int = 1000) -> None:
        self.path = Path(path)
        self.limit = limit
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, alerts: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(alerts[-self.limit :], indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _fingerprint(skill: str, severity: str, findings: Any) -> str:
        """Build stable incident identity while ignoring volatile collection metadata."""
        volatile = {"timestamp", "created_at", "updated_at", "observed_at", "last_seen", "collected_at"}

        def normalize(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in sorted(value.items()) if key.lower() not in volatile}
            if isinstance(value, list):
                normalized = [normalize(item) for item in value]
                return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
            return value

        canonical = json.dumps({"skill": skill, "severity": severity, "findings": normalize(findings)}, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def emit(self, skill: str, result: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any] | None:
        count_field = str(contract.get("count_field") or "finding_count")
        findings_field = str(contract.get("findings_field") or "findings")
        findings = result.get(findings_field)
        count = result.get(count_field)
        if count is None:
            count = len(findings) if isinstance(findings, list) else 0
        if int(count or 0) <= 0:
            return None
        severity = str(result.get(contract.get("severity_field", "severity")) or contract.get("default_severity") or "medium").lower()
        fingerprint = self._fingerprint(skill, severity, findings or [])
        alert = {
            "id": uuid.uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
            "skill": skill, "severity": severity, "fingerprint": fingerprint,
            "title": str(contract.get("title") or f"Passive finding from {skill}"),
            "count": int(count), "findings": findings or [], "analysis": result.get("analysis"),
            "coverage": result.get("coverage"), "status": "unread",
        }
        with self._lock:
            alerts = self._read()
            previous = next((
                item for item in reversed(alerts)
                if (item.get("fingerprint") or self._fingerprint(str(item.get("skill") or ""), str(item.get("severity") or "medium"), item.get("findings") or [])) == fingerprint
            ), None)
            if previous is not None:
                previous["fingerprint"] = fingerprint
                previous["last_seen_at"] = datetime.now(timezone.utc).isoformat()
                previous["suppressed_repeats"] = int(previous.get("suppressed_repeats") or 0) + 1
                self._write(alerts)
                return None
            alerts.append(alert)
            self._write(alerts)
        return alert

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._read()))

    def update_status(self, alert_id: str, status: str) -> dict[str, Any] | None:
        with self._lock:
            alerts = self._read()
            selected = None
            for alert in alerts:
                if alert.get("id") == alert_id:
                    alert["status"] = status
                    selected = alert
                    break
            self._write(alerts)
            return selected


alert_store = AlertStore()
