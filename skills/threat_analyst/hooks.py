from __future__ import annotations

import ipaddress
import logging
import re

logger = logging.getLogger(__name__)


def _shorten_naturally(text: str, max_len: int = 180) -> str:
    def _clean_tail(value: str) -> str:
        value = value.rstrip(" ,;:-")
        value = re.sub(r"\b(and|or|but|because|which|that|while|with)$", "", value, flags=re.IGNORECASE).rstrip(" ,;:-")
        return value

    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= max_len:
        return _clean_tail(cleaned)

    sentence_window = cleaned[: max_len + 1]
    last_sentence_end = max(
        sentence_window.rfind(". "),
        sentence_window.rfind("! "),
        sentence_window.rfind("? "),
    )
    if last_sentence_end >= int(max_len * 0.6):
        return _clean_tail(sentence_window[: last_sentence_end + 1])

    word_window = cleaned[: max_len + 1]
    last_space = word_window.rfind(" ")
    if last_space >= int(max_len * 0.6):
        return _clean_tail(word_window[:last_space]) + "..."

    return _clean_tail(cleaned[:max_len]) + "..."


def append_summary(base_response: str, threat_result: dict) -> str:
    if not threat_result or threat_result.get("status") != "ok":
        return base_response

    verdicts = threat_result.get("verdicts") or []
    if not verdicts:
        return base_response

    per_verdict_limit = 600 if len(verdicts) == 1 else 350
    summary_parts = []
    for verdict in verdicts[:3]:
        label = verdict.get("verdict", "UNKNOWN")
        confidence = verdict.get("confidence", 0)
        reasoning = " ".join(str(verdict.get("reasoning", "")).split())
        if reasoning:
            shortened = _shorten_naturally(reasoning, per_verdict_limit)
            summary_parts.append(f"{label} ({confidence}%): {shortened}")
        else:
            summary_parts.append(f"{label} ({confidence}%)")

    all_apis = sorted({api for verdict in verdicts for api in verdict.get("_queried_apis", [])})
    suffix = f" Threat intel: {'; '.join(summary_parts)}."
    if all_apis:
        suffix += f" Sources queried: {', '.join(all_apis)}."
    return base_response + suffix


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    if not result or result.get("status") != "ok":
        return "No threat intelligence verdict was produced."

    verdicts = result.get("verdicts") or []
    if not verdicts:
        return "No threat intelligence verdict was produced."

    requested_ips: list[str] = []
    for verdict in verdicts:
        for ip in verdict.get("_requested_ips", []):
            if ip not in requested_ips:
                requested_ips.append(ip)

    if not requested_ips:
        requested_ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question or "")))

    primary = verdicts[0]
    verdict_label = str(primary.get("verdict", "UNKNOWN") or "UNKNOWN")
    confidence = int(primary.get("confidence", 0) or 0)
    reasoning = _shorten_naturally(" ".join(str(primary.get("reasoning", "") or "").split()), 320)
    subject = ", ".join(requested_ips) if requested_ips else "the requested IPs"

    all_apis = sorted({api for verdict in verdicts for api in verdict.get("_queried_apis", [])})
    if requested_ips and all(_is_private_ip(ip) for ip in requested_ips) and not all_apis:
        return (
            f"{subject} is a private/internal IP address, so public GeoIP and external threat-intelligence feeds do not apply directly. "
            "Use internal log evidence, asset ownership, and local detections to assess whether it is suspicious."
        )

    response = f"Reputation analysis for {subject}: {verdict_label} ({confidence}% confidence)."
    if reasoning:
        response += f" {reasoning}"

    if all_apis:
        response += f"\n\n_[Threat Intelligence Sources Queried: {', '.join(all_apis)}]_"
    return response


# ─── ENTITY EXTRACTION FOR THREAT FOLLOW-UPS ───────────────────────────────

def _is_private_ip(ip: str) -> bool:
    """Check if IP is private (RFC 1918, loopback, etc.)."""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def _question_asks_for_reputation(user_question: str) -> bool:
    """Check if question is asking for threat/reputation assessment."""
    question_lower = user_question.lower()
    return any(
        term in question_lower
        for term in ["reputation", "threat intel", "threat intelligence", "risk", "malicious", "verdict", "score"]
    )


def _question_has_explicit_entities(user_question: str) -> bool:
    """Check if question explicitly includes IPs or domains."""
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question):
        return True
    return bool(re.search(r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", user_question.lower()))


def _question_excludes_private_ips(user_question: str) -> bool:
    """Check if question explicitly excludes private IPs."""
    question_lower = user_question.lower()
    return any(
        term in question_lower
        for term in [
            "aside from the private ip",
            "aside from private ip",
            "excluding private ip",
            "exclude private ip",
            "except private ip",
            "other than private ip",
            "non-private ip",
            "public ip",
            "private ips",
            "internal ips",
        ]
    )


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
    if _question_excludes_private_ips(user_question):
        filtered["ips"] = [ip for ip in filtered["ips"] if not _is_private_ip(ip)]
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


def _recover_followup_reputation_entities(user_question: str, conversation_history: list[dict] | None) -> dict:
    """Recover prior entities for follow-up reputation questions like 'what about the others?'"""
    if not _question_asks_for_reputation(user_question):
        return {}
    if _question_has_explicit_entities(user_question):
        return {}

    question_lower = user_question.lower()
    plural_referential_cues = [
        "those", "them", "above", "listed", "list above", "listed above",
        "the ip listed", "the ips listed", "listed ip", "listed ips",
        "above ip", "above ips", "the others", "others", "these",
        "this ip", "that ip", "mentioned", "just mentioned",
        "you just mentioned", "you've just mentioned", "previously mentioned",
        "prior ip", "prior ips", "previous ip", "previous ips",
        "aside from", "excluding", "exclude", "except", "other than",
    ]
    singular_referential_cues = [
        "the ip", "that ip", "this ip", "the address",
        "that address", "this address", "the host",
        "that host", "this host",
    ]
    asks_for_new_evidence = bool(
        re.search(
            r"\b(?:traffic|country|countries|log|logs|port|protocol|connection|connections|flow|flows|when|where|search|show|find|list)\b",
            question_lower,
        )
    )
    if asks_for_new_evidence and not _question_excludes_private_ips(user_question):
        return {}

    history_entities = _extract_entities_from_conversation_history(conversation_history)
    history_entities = _filter_entities_for_question(history_entities, user_question)
    public_history_ips = [ip for ip in history_entities.get("ips", []) if not _is_private_ip(ip)]
    singular_reference = any(re.search(rf"\b{re.escape(cue)}\b", question_lower) for cue in singular_referential_cues)
    plural_reference = any(re.search(rf"\b{re.escape(cue)}\b", question_lower) for cue in plural_referential_cues)

    if singular_reference:
        if len(public_history_ips) == 1:
            singular_entities = dict(history_entities)
            singular_entities["ips"] = [public_history_ips[0]]
            singular_entities["domains"] = []
            return singular_entities
        return {}

    if not plural_reference:
        return {}

    if history_entities.get("ips") or history_entities.get("domains"):
        return history_entities
    
    return {}


def extract_entities(
    user_question: str,
    conversation_history: list[dict] | None = None,
    aggregated_results: dict | None = None,
    **kwargs
) -> dict:
    """
    Extract entities from question, conversation history, and prior results for threat follow-ups.
    
    This is the manifest-declared entity extractor for threat_analyst.
    Called by router when preparing enrichment for threat follow-up questions.
    
    Args:
        user_question: Current question from user
        conversation_history: Prior conversation for entity recovery
        aggregated_results: Results from prior skill executions
        **kwargs: Additional context (compatibility)
    
    Returns:
        Dict with keys: ips, domains, countries, ports, sources
    """
    entities = _recover_followup_reputation_entities(user_question, conversation_history)
    if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
        return entities

    if aggregated_results:
        entities = _extract_entities_from_previous_results(aggregated_results)
        entities = _filter_entities_for_question(entities, user_question)
        if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
            return entities

    return {"ips": [], "domains": [], "countries": [], "ports": [], "sources": []}


def extract_followup_question(original_question: str, entities: dict, **kwargs) -> str:
    """
    Build context-aware question for threat analysis using discovered entities.
    
    This is the manifest-declared followup question builder for threat_analyst.
    
    Args:
        original_question: Original user question
        entities: Extracted entities {ips, domains, countries, ports, sources}
        **kwargs: Additional context (compatibility)
    
    Returns:
        Enriched question string with discovered entities
    """
    entities = _filter_entities_for_question(entities, original_question)
    if not entities or not any([entities.get("ips"), entities.get("domains"), entities.get("countries")]):
        return original_question
    
    # Filter private IPs: they have no external threat reputation
    ips = [ip for ip in entities.get("ips", []) if not _is_private_ip(ip)]
    domains = entities.get("domains", [])
    countries = entities.get("countries", [])
    ports = [str(port) for port in entities.get("ports", [])]
    
    enriched = original_question
    
    # Build context string with discovered entities
    context_parts = []
    if ips:
        context_parts.append(f"IPs: {', '.join(ips[:5])}" + (" (and more)" if len(ips) > 5 else ""))
    if domains:
        context_parts.append(f"Domains: {', '.join(domains[:3])}" + (" (and more)" if len(domains) > 3 else ""))
    if countries:
        context_parts.append(f"Countries: {', '.join(countries)}")
    if ports:
        context_parts.append(f"Ports: {', '.join(ports[:5])}" + (" (and more)" if len(ports) > 5 else ""))
    
    if context_parts:
        context_str = " | ".join(context_parts)
        enriched = f"{original_question}\n\nPreviously discovered entities from log search: {context_str}"
        logger.info("[threat_analyst] Enriched question with discovered entities: %s", context_str)
    
    return enriched


# ─── EVALUATION HOOK FOR THREAT FOLLOW-UPS ──────────────────────────────

def evaluate_satisfaction(
    user_question: str,
    result: dict,
    skill_results: dict,
    conversation_history: list[dict] | None = None,
    **kwargs
) -> dict | None:
    """
    Evaluate if this threat_analyst result satisfies the user question.
    
    This is the manifest-declared evaluation hook for threat_analyst.
    Returns dict with {satisfied, confidence, reasoning, missing} if evaluation is applicable,
    or None to let router use default evaluation logic.
    
    Applies when threat verdicts have been produced and we need to validate 
    they're analyzing the correct entities (IP mismatch detection).
    
    Args:
        user_question: User's original question
        result: Result dict from threat_analyst execution
        skill_results: All skill results executed so far
        conversation_history: Prior conversation for entity recovery
        **kwargs: Additional context (compatibility)
    
    Returns:
        {satisfied, confidence, reasoning, missing} dict, or None to skip evaluation
    """
    # Only evaluate if threat_analyst produced verdicts
    if not result or result.get("status") != "ok":
        return None
    
    threat_verdicts = result.get("verdicts") or []
    if not threat_verdicts:
        return None
    
    # Validate that verdict IPs match question IPs when possible
    # Extract IPs from question to ensure response is about the right entities
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    question_ips = set(re.findall(ip_pattern, user_question))
    
    # Also look in recent conversation for mentioned IPs (for "these IPs" follow-ups)
    history_text = "\n".join(
        str(m.get("content", ""))
        for m in (conversation_history or [])[-3:]
    )
    discovered_ips = set(re.findall(ip_pattern, history_text))
    question_ips.update(discovered_ips)
    
    # If question had specific IPs, check that verdicts mention similar ones
    verdicts_mention_wrong_ips = False
    if question_ips:
        # Extract IPs from verdict reasoning
        verdict_text = "\n".join(
            str(v.get("reasoning", ""))
            for v in threat_verdicts
        )
        verdict_ips = set(re.findall(ip_pattern, verdict_text))
        
        # If verdict mentions IPs, at least one should overlap with question IPs
        if verdict_ips and not verdict_ips.intersection(question_ips):
            # Verdict IPs and question IPs don't overlap — likely analyzed wrong entities
            verdicts_mention_wrong_ips = True
            logger.warning(
                "[threat_analyst] Verdict mentions IPs %s but question asked about %s — possible IP mismatch",
                verdict_ips, question_ips
            )
    
    if verdicts_mention_wrong_ips:
        # Re-run to get correct verdicts
        logger.info(
            "[threat_analyst] Detected IP mismatch in verdicts — requesting re-analysis",
        )
        return {
            "satisfied": False,
            "confidence": 0.3,
            "reasoning": "Threat intel was produced but for wrong IPs; re-analyzing.",
            "missing": ["threat reputation for correct IPs"],
        }
    
    # If we get here, verdicts are valid—don't override router's default evaluation
    # Return None to let router determine overall satisfaction based on all skills
    return None


def enrich_question_for_followup(
    original_question: str,
    conversation_history: list[dict] | None,
    previous_results: dict | None = None,
) -> str:
    """Enrich threat_analyst question with discovered entities from prior results.
    
    This hook is called by the router before executing threat_analyst to enhance
    the question with concrete IPs, domains, and countries discovered by previous skills.
    
    The enrichment improves the quality of threat intelligence analysis by grounding
    the question in actual discovered entities rather than generic reputation lookups.
    
    Args:
        original_question: User's original question
        conversation_history: Prior conversation for context
        previous_results: Results from prior skills (forensic_examiner, opensearch_querier, etc.)
    
    Returns:
        Enriched question string, or original if no enrichment possible
    """
    # Import helper functions from the router
    # These functions handle complex entity extraction and context-aware question building
    from core.chat_router.logic import (
        _recover_threat_followup_entities,
        _build_context_aware_threat_question,
    )
    
    if not previous_results and not conversation_history:
        return original_question
    
    # Try to recover entities with previous results first (most valuable)
    if previous_results:
        entities = _recover_threat_followup_entities(
            original_question,
            conversation_history,
            previous_results,
        )
        if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
            enriched_q = _build_context_aware_threat_question(original_question, entities)
            return enriched_q
    
    # Fallback: recover entities from conversation history alone
    if conversation_history:
        entities = _recover_threat_followup_entities(
            original_question,
            conversation_history,
        )
        if entities:
            enriched_q = _build_context_aware_threat_question(original_question, entities)
            return enriched_q
    
    return original_question
    return None