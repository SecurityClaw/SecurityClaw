from core.endpoint_security import collect_processes

_previous_processes: set[tuple] | None = None


def run(context: dict) -> dict:
    global _previous_processes
    parameters = context.get("parameters") or {}
    result = collect_processes(limit=min(int(parameters.get("limit", 500)), 2000))
    current = {(item.get("pid"), item.get("executable"), item.get("command")) for item in result.get("processes", [])}
    result["new_processes"] = [] if _previous_processes is None else [
        item for item in result.get("processes", [])
        if (item.get("pid"), item.get("executable"), item.get("command")) not in _previous_processes
    ]
    _previous_processes = current
    return result
