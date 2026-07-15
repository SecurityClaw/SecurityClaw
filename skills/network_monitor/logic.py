from core.endpoint_security import collect_network_connections

_previous_connections: set[str] | None = None


def run(context: dict) -> dict:
    global _previous_connections
    parameters = context.get("parameters") or {}
    result = collect_network_connections(limit=min(int(parameters.get("limit", 500)), 2000))
    keyed = {repr(sorted(item.items())): item for item in result.get("connections", [])}
    result["new_connections"] = [] if _previous_connections is None else [
        item for key, item in keyed.items() if key not in _previous_connections
    ]
    _previous_connections = set(keyed)
    return result
