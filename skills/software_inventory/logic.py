"""Versioned software inventory skill."""

from skills.endpoint_telemetry.logic import collect_software_inventory


def run(context: dict) -> dict:
    parameters = (context.get("routing_decision") or {}).get("parameters") or {}
    return collect_software_inventory(limit=int(parameters.get("limit", 2000)))
