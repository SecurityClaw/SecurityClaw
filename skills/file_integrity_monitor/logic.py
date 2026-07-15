from core.endpoint_security import collect_file_integrity

_previous_hashes: dict[str, str] | None = None


def run(context: dict) -> dict:
    global _previous_hashes
    parameters = context.get("parameters") or {}
    paths = parameters.get("paths")
    if isinstance(paths, str):
        paths = [paths]
    result = collect_file_integrity(paths=paths, max_files=min(int(parameters.get("max_files", 1000)), 5000))
    current = {item["path"]: item["sha256"] for item in result.get("files", [])}
    if _previous_hashes is None:
        result["baseline_initialized"] = True
        result["changes"] = []
    else:
        result["changes"] = [
            {"path": path, "change": "created" if path not in _previous_hashes else "modified"}
            for path, digest in current.items()
            if _previous_hashes.get(path) != digest
        ] + [
            {"path": path, "change": "deleted"}
            for path in _previous_hashes
            if path not in current
        ]
    _previous_hashes = current
    memory = context.get("memory")
    if memory and result["changes"]:
        memory.add_finding(f"File integrity changes detected: {result['changes'][:10]}")
    return result
