from __future__ import annotations


def enrich_question_for_followup(
    original_question: str,
    conversation_history: list[dict] | None,
    previous_results: dict | None = None,
) -> str:
    """Enrich fingerprint follow-ups with the recovered target entity."""
    from core.chat_router.logic import (
        _build_context_aware_fingerprint_question,
        _recover_fingerprint_followup_entities,
    )

    entities = _recover_fingerprint_followup_entities(
        original_question,
        conversation_history,
        previous_results,
    )
    if not entities.get("ips"):
        return original_question

    enriched = _build_context_aware_fingerprint_question(
        original_question,
        entities,
        conversation_history,
    )
    return enriched if enriched != original_question else original_question


def evaluate_satisfaction(user_question: str, result: dict, skill_results: dict | None = None, conversation_history: list[dict] | None = None) -> dict | None:
    question_lower = str(user_question or "").lower()
    if not any(
        term in question_lower
        for term in ["fingerprint", "port profile", "ports associated", "ephemeral", "client or server", "server or client", "os likelihood"]
    ):
        return None

    if not result or result.get("status") not in {"ok", "no_data"}:
        return None

    if result.get("status") == "ok":
        observed_ports = len(result.get("ports") or [])
        return {
            "satisfied": True,
            "confidence": 0.95,
            "reasoning": f"Fingerprinting completed using {observed_ports} observed port(s).",
            "missing": [],
        }

    return {
        "satisfied": True,
        "confidence": 0.8,
        "reasoning": result.get("reason") or "Fingerprinting completed but no matching port observations were found.",
        "missing": [],
    }


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    if not result or result.get("status") not in {"ok", "no_data"}:
        return "No passive fingerprint could be produced."

    if result.get("status") == "no_data":
        ip = result.get("ip", "the requested IP")
        return f"No matching port observations were found for {ip} in the available records."

    ip = result.get("ip", "the requested IP")
    role = (result.get("likely_role") or {}).get("classification", "inconclusive")
    confidence = (result.get("likely_role") or {}).get("confidence", 0)
    
    ports = result.get("ports") or []
    os_likelihoods = result.get("os_family_likelihoods") or []
    top_os = os_likelihoods[0] if os_likelihoods else {}

    parts = [f"Passive fingerprint for {ip}: {role} ({confidence}% confidence)."]
    
    # Format listening ports (all destination ports, no filtering needed)
    if ports:
        notable_ports = sorted(
            ports,
            key=lambda entry: (
                0 if entry.get("registered") else 1,
                -(int(entry.get("observations", 0) or 0)),
                int(entry.get("port", 0) or 0),
            ),
        )
        port_descriptions = []
        for port_entry in notable_ports[:15]:
            port_num = port_entry.get("port")
            service_name = port_entry.get("service_name", "unknown")
            port_descriptions.append(f"{port_num} ({service_name})")
        parts.append(f"Listening on ports: {', '.join(port_descriptions)}.")
    
    if top_os.get("family"):
        parts.append(f"Likely OS: {top_os.get('family')} ({top_os.get('confidence', 0)}%).")
    
    return " ".join(parts)