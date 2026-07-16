from skills.endpoint_telemetry.logic import collect_persistence


def run(context: dict) -> dict:
    parameters = context.get("parameters") or {}
    return collect_persistence(limit=min(int(parameters.get("limit", 500)), 2000))
