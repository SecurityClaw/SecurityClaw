"""Endpoint defensive posture skill."""

from core.endpoint_security import collect_security_posture


def run(context: dict) -> dict:
    parameters = (context.get("routing_decision") or {}).get("parameters") or {}
    return collect_security_posture(limit=int(parameters.get("limit", 500)))
