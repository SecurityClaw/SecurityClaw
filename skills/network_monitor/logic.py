from skills.endpoint_telemetry.logic import collect_network_connections, filter_securityclaw_connections

_previous_connections: set[str] | None = None


def run(context: dict) -> dict:
    global _previous_connections
    parameters = context.get("parameters") or {}
    result = collect_network_connections(limit=min(int(parameters.get("limit", 500)), 2000))
    visible, excluded, owned_ports = filter_securityclaw_connections(
        result.get("connections") or [],
        context.get("config"),
    )
    result["connections"] = visible
    result["count"] = len(visible)
    result["excluded_own_connection_count"] = len(excluded)
    result["excluded_owned_ports"] = sorted(owned_ports)
    keyed = {repr(sorted(item.items())): item for item in result.get("connections", [])}
    result["new_connections"] = [] if _previous_connections is None else [
        item for key, item in keyed.items() if key not in _previous_connections
    ]
    _previous_connections = set(keyed)
    return result
