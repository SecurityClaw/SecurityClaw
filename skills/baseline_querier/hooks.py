from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    findings = result.get("findings") or {}
    grounded_assessment = " ".join(str(findings.get("grounded_assessment", "") or "").split())
    # Use grounded assessment if available (LLM supervisor decides when baseline is needed)
    if grounded_assessment:
        return grounded_assessment

    answer = " ".join(str(findings.get("answer", "") or "").split())
    if not answer:
        return "No grounded baseline analysis was produced."

    log_records = int(findings.get("log_records", 0) or 0)
    rag_sources = int(findings.get("rag_sources", 0) or 0)
    evidence = findings.get("evidence") or {}
    timestamps = evidence.get("timestamps") or []
    ips = evidence.get("ips") or []
    ports = evidence.get("ports") or []

    details = []
    if log_records > 0:
        details.append(f"Observed records: {log_records}.")
    elif rag_sources > 0:
        details.append(f"Baseline documents consulted: {rag_sources}.")
    if ips:
        details.append(f"IPs referenced: {', '.join(ips[:10])}.")
    if ports:
        details.append(f"Ports referenced: {', '.join(str(port) for port in ports[:10])}.")
    if timestamps:
        ts_sorted = sorted(str(ts) for ts in timestamps)
        details.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")

    suffix = " " + " ".join(details) if details else ""
    return answer + suffix


# ─── ENTITY EXTRACTION FOR BASELINE FOLLOW-UPS ──────────────────────────
# NOTE: Keyword checking has been removed. The LLM supervisor decides whether
# baseline_querier should be invoked based on the question intent and manifest
# declarations. Baseline-related keywords are provided in supervisor prompts.
# See: SUPERVISOR_NEXT_ACTION_PROMPT.md for baseline question examples.


def extract_entities(
    user_question: str,
    conversation_history: list[dict] | None = None,
    aggregated_results: dict | None = None,
    **kwargs
) -> dict:
    """
    Extract entities from question, conversation history, and prior results for baseline follow-ups.
    
    This is the manifest-declared entity extractor for baseline_querier.
    Called by router when preparing enrichment for baseline follow-up questions.
    
    Args:
        user_question: Current question from user
        conversation_history: Prior conversation for entity recovery
        aggregated_results: Results from prior skill executions
        **kwargs: Additional context (compatibility)
    
    Returns:
        Dict with keys: ips, domains, countries, ports, sources
    """
    # Extract explicit IPs from question (deterministic structural extraction, not keyword matching)
    explicit_ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question or "")))
    entities = {
        "ips": explicit_ips,
        "domains": [],
        "countries": [],
        "ports": [],
        "sources": [],
    }

    # Extract entities from conversation history
    history_entities = _extract_entities_from_conversation_history(conversation_history)
    
    # Extract entities from prior skill results
    previous_entities = _extract_entities_from_previous_results(aggregated_results or {})

    # Merge entities from all sources, preserving uniqueness
    for key in ("ips", "domains", "countries", "ports"):
        combined = entities.get(key, []) + history_entities.get(key, []) + previous_entities.get(key, [])
        entities[key] = list(dict.fromkeys(combined))

    # Filter based on question context (e.g., exclude private IPs if requested)
    entities = _filter_entities_for_question(entities, user_question)
    
    return entities


def extract_followup_question(original_question: str, entities: dict, conversation_history: list[dict] | None = None, **kwargs) -> str:
    """
    Build context-aware question for baseline analysis using discovered entities.
    
    This is the manifest-declared followup question builder for baseline_querier.
    
    Args:
        original_question: Original user question
        entities: Extracted entities {ips, domains, countries, ports, sources}
        conversation_history: Prior conversation for additional context
        **kwargs: Additional context (compatibility)
    
    Returns:
        Enriched question string with discovered entities and baseline context
    """
    # LLM supervisor decides whether baseline enrichment is needed.
    # This function always enriches if called.
    entities = _filter_entities_for_question(entities or {}, original_question)
    ips = entities.get("ips", [])
    domains = entities.get("domains", [])
    countries = entities.get("countries", [])
    ports = [str(port) for port in entities.get("ports", [])]
    history_summary = _latest_assistant_observation(conversation_history)

    context_parts = []
    if ips:
        context_parts.append(f"Focus entities: IPs {', '.join(ips[:5])}")
    if domains:
        context_parts.append(f"Domains {', '.join(domains[:3])}")
    if countries:
        context_parts.append(f"Countries {', '.join(countries[:5])}")
    if ports:
        context_parts.append(f"Ports {', '.join(ports[:5])}")
    if history_summary:
        context_parts.append(f"Recent observed traffic: {history_summary}")

    if not context_parts:
        return original_question

    return (
        f"{original_question}\n\n"
        "Compare the requested behavior against known baseline patterns and any observed traffic evidence. "
        "If possible, quantify how often the entity appears and whether the behavior looks routine or unusual.\n"
        + " | ".join(context_parts)
    )


# ─── HELPERS FOR BASELINE EXTRACTORS ──────────────────────────────────

def _latest_assistant_observation(conversation_history: list[dict] | None) -> str:
    """Get the most recent grounded assistant observation from conversation."""
    if not conversation_history:
        return ""
    for msg in reversed(conversation_history[-8:]):
        if msg.get("role") == "assistant":
            content = " ".join(str(msg.get("content", "") or "").split())
            if content and any(
                marker in content
                for marker in [
                    "Found ",
                    "No traffic ",
                    "No matching records found",
                    "Countries seen:",
                    "Source IPs:",
                    "Source/destination IPs:",
                    "Remote peers:",
                ]
            ):
                return content[:600]
    return ""


def _extract_entities_from_conversation_history(conversation_history: list[dict] | None) -> dict:
    """Extract the most recent concrete entities from recent conversation history."""
    empty = {
        "ips": [],
        "domains": [],
        "countries": [],
        "ports": [],
        "sources": [],
    }
    if not conversation_history:
        return empty

    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"

    def _is_grounded_assistant_message(role: str, text: str) -> bool:
        if role != "assistant":
            return True
        grounded_markers = [
            "Found ",
            "No traffic ",
            "No matching records found",
            "Countries seen:",
            "Source IPs:",
            "Source/destination IPs:",
            "Remote peers:",
            "Reputation analysis for ",
        ]
        return any(marker in text for marker in grounded_markers)

    for msg in reversed(conversation_history[-8:]):
        text = str(msg.get("content", "") or "")
        if not text:
            continue
        role = str(msg.get("role", ""))
        if not _is_grounded_assistant_message(role, text):
            continue

        ips = list(dict.fromkeys(re.findall(ip_pattern, text)))
        domains = list(dict.fromkeys(re.findall(domain_pattern, text.lower())))
        countries: list[str] = []
        for match in re.finditer(r"Countries seen:\s*([A-Za-z ,_-]+?)(?:\.|$)", text, re.IGNORECASE):
            country_text = match.group(1).strip().rstrip(".")
            countries.extend([part.strip() for part in country_text.split(",") if part.strip()])

        ports: list[str] = []
        for match in re.finditer(r"Ports:\s*([0-9, ]+?)(?:\.|$)", text, re.IGNORECASE):
            port_text = match.group(1).strip().rstrip(".")
            ports.extend([part.strip() for part in port_text.split(",") if part.strip()])

        if ips or domains or countries or ports:
            return {
                "ips": ips,
                "domains": domains,
                "countries": countries,
                "ports": ports,
                "sources": [msg.get("role", "history")],
            }

    return empty


def _filter_entities_for_question(entities: dict, user_question: str) -> dict:
    """Filter extracted entities based on question context."""
    if not entities:
        return entities
    filtered = {
        "ips": list(entities.get("ips", [])),
        "domains": list(entities.get("domains", [])),
        "countries": list(entities.get("countries", [])),
        "ports": list(entities.get("ports", [])),
        "sources": list(entities.get("sources", [])),
    }
    return filtered


def _extract_entities_from_previous_results(aggregated_results: dict) -> dict:
    """Extract IPs, domains, countries, and ports from previous skill results."""
    entities = {
        "ips": set(),
        "domains": set(),
        "countries": set(),
        "ports": set(),
        "sources": [],
    }
    
    # Extract from opensearch_querier results
    if "opensearch_querier" in aggregated_results:
        result = aggregated_results["opensearch_querier"]
        entities["sources"].append("opensearch_querier")
        
        # Extract from raw results (log documents)
        results_list = result.get("results", [])
        if isinstance(results_list, list):
            for record in results_list:
                if isinstance(record, dict):
                    source_ips: set[str] = set()
                    destination_ips: set[str] = set()
                    record_countries: set[str] = set()
                    # Common IP field names
                    for ip_field in ["src_ip", "source_ip", "srcip", "src", "ip", "_source.src_ip", "source.ip"]:
                        if ip_field in record and record[ip_field]:
                            val = record[ip_field]
                            if isinstance(val, str):
                                source_ips.add(val)
                    for ip_field in ["dst_ip", "dest_ip", "destination_ip", "destination.ip"]:
                        if ip_field in record and record[ip_field]:
                            val = record[ip_field]
                            if isinstance(val, str):
                                destination_ips.add(val)
                    nested_source_ip = record.get("source", {}).get("ip") if isinstance(record.get("source"), dict) else None
                    nested_destination_ip = record.get("destination", {}).get("ip") if isinstance(record.get("destination"), dict) else None
                    if isinstance(nested_source_ip, str):
                        source_ips.add(nested_source_ip)
                    if isinstance(nested_destination_ip, str):
                        destination_ips.add(nested_destination_ip)
                    
                    # Common domain field names
                    for domain_field in ["domain", "hostname", "fqdn", "src_domain"]:
                        if domain_field in record and record[domain_field]:
                            val = record[domain_field]
                            if isinstance(val, str):
                                entities["domains"].add(val)
                    
                    # Country extraction
                    for country_field in [
                        "country", "src_country", "country_name", "geoip.country_name",
                        "source.geo.country_name", "destination.geo.country_name",
                    ]:
                        if country_field in record and record[country_field]:
                            val = record[country_field]
                            if isinstance(val, str):
                                entities["countries"].add(val)
                                record_countries.add(val)
                    geo = record.get("geoip") or {}
                    if isinstance(geo, dict):
                        for nested_country in (geo.get("country_name"), geo.get("country")):
                            if isinstance(nested_country, str):
                                entities["countries"].add(nested_country)
                                record_countries.add(nested_country)
                    has_country_info = bool(record_countries)

                    if source_ips and has_country_info:
                        entities["ips"].update(source_ips)
                    else:
                        entities["ips"].update(source_ips)
                        entities["ips"].update(destination_ips)
                    
                    # Port extraction
                    for port_field in ["port", "dst_port", "dest_port", "dport", "destination.port", "destination_port"]:
                        if port_field in record and record[port_field]:
                            val = record[port_field]
                            if isinstance(val, (int, str)):
                                entities["ports"].add(str(val))
                    nested_dest_port = record.get("destination", {}).get("port") if isinstance(record.get("destination"), dict) else None
                    if isinstance(nested_dest_port, (int, str)):
                        entities["ports"].add(str(nested_dest_port))
        
        # Only trust summary metadata when the opensearch result passed validation.
        if not result.get("validation_failed"):
            entities["countries"].update(result.get("countries", []))
            entities["ports"].update(result.get("ports", []))
    
    # Extract from baseline_querier / fields_querier results
    for rag_skill in ("baseline_querier", "fields_querier"):
        if rag_skill in aggregated_results:
            result = aggregated_results[rag_skill]
            entities["sources"].append(rag_skill)
            entities["ips"].update(result.get("ips", []))
            entities["ports"].update(result.get("ports", []))
    
    # Convert sets to lists
    return {
        "ips": list(entities["ips"]),
        "domains": list(entities["domains"]),
        "countries": list(entities["countries"]),
        "ports": list(entities["ports"]),
        "sources": entities["sources"],
    }


# ─── EVALUATION HOOK FOR BASELINE FINDINGS ──────────────────────────── 

def evaluate_satisfaction(
    user_question: str,
    result: dict,
    skill_results: dict,
    conversation_history: list[dict] | None = None,
    **kwargs
) -> dict | None:
    """
    Evaluate if this baseline_querier result satisfies the user question.
    
    This is the manifest-declared evaluation hook for baseline_querier.
    Returns dict with {satisfied, confidence, reasoning, missing} if evaluation is applicable,
    or None to let router use default evaluation logic.
    
    Validates that baseline findings include expected evidence and source documents.
    
    Args:
        user_question: User's original question
        result: Result dict from baseline_querier execution
        skill_results: All skill results executed so far
        conversation_history: Prior conversation for context
        **kwargs: Additional context (compatibility)
    
    Returns:
        {satisfied, confidence, reasoning, missing} dict, or None to skip evaluation
    """
    # Only evaluate if baseline_querier produced findings
    if not result or result.get("status") != "ok":
        return None
    
    findings = result.get("findings") or {}
    if not findings.get("answer"):
        return None
    
    baseline_log_records = int(findings.get("log_records", 0) or 0)
    baseline_sources = int(findings.get("rag_sources", 0) or 0)
    
    # If baseline_querier has log records or RAG sources, the answer is grounded
    if baseline_log_records > 0 or baseline_sources > 0:
        logger.info(
            "[baseline_querier] Found %d log records and %d baseline sources — marking satisfied",
            baseline_log_records, baseline_sources
        )
        return {
            "satisfied": True,
            "confidence": 0.85 if baseline_log_records > 0 else 0.75,
            "reasoning": (
                f"Baseline analysis returned {baseline_log_records} matching log record(s) "
                f"and {baseline_sources} relevant baseline document(s)."
            ),
            "missing": [],
        }
    
    # If baseline_querier ran but produced no evidence, it's still satisfied
    # (the LLM supervisor will not have routed to baseline_querier unless appropriate)
    logger.info("[baseline_querier] No log records or baseline sources found, but baseline_querier was routed by LLM")
    return {
        "satisfied": True,
        "confidence": 0.6,
        "reasoning": "Baseline querier was executed but found no matching evidence in available baselines or logs.",
        "missing": [],
    }


def enrich_question_for_followup(
    original_question: str,
    conversation_history: list[dict] | None,
    previous_results: dict | None = None,
) -> str:
    """Enrich baseline_querier question with baseline comparison context.
    
    This hook is called by the router before executing baseline_querier to enhance
    the question with baseline-specific analysis intent and extracted entities.
    
    Args:
        original_question: User's original question
        conversation_history: Prior conversation for context
        previous_results: Results from prior skills
    
    Returns:
        Enriched question string, or original if no enrichment needed
    """
    # Import helper functions from the router
    from core.chat_router.logic import (
        _recover_baseline_followup_entities,
        _build_context_aware_baseline_question,
    )
    
    # Only enrich if this appears to be a baseline follow-up query
    entities = _recover_baseline_followup_entities(
        original_question,
        conversation_history,
        previous_results,
    )
    
    if not entities:
        return original_question
    
    enriched_q = _build_context_aware_baseline_question(
        original_question,
        entities,
        conversation_history,
    )
    
    return enriched_q if enriched_q != original_question else original_question