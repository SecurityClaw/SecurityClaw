"""
skills/opensearch_querier/logic.py

Skill wrapper around core.query_builder utilities.

This skill provides:
1. A direct interface for user queries via chat
2. Shared query_builder utilities that other skills import

All query logic is in core.query_builder (DRY principle).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "opensearch_querier"
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _extract_json_from_response(response: str) -> dict | None:
    """
    Extract JSON from LLM response, handling markdown code blocks and extra text.
    
    Handles formats like:
    - Raw JSON: {"query": {...}}
    - Markdown: ```json\n{"query": {...}}\n```
    - With explanation: "Here's the fixed query: {"query": {...}}"
    """
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code blocks
    matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object in the response
    matches = re.findall(r'\{[\s\S]*\}', response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def _extract_ips_from_text(text: str) -> list[str]:
    """Extract unique IPv4 addresses from free text while preserving order."""
    if not text:
        return []

    seen: set[str] = set()
    ips: list[str] = []
    for ip in _IP_PATTERN.findall(str(text)):
        if ip not in seen:
            seen.add(ip)
            ips.append(ip)
    return ips


def _extract_ports_from_text(text: str) -> list[int]:
    """Extract port numbers from text (patterns like 'port 443' or 'port:443')."""
    if not text:
        return []
    
    ports = set()
    
    # Handle compound patterns like "ports 80 and 443" or "ports: 80, 443"
    compound_match = re.search(r'\bports?\s*:?\s*([\d\s,and]+)', text, re.IGNORECASE)
    if compound_match:
        port_list = compound_match.group(1)
        # Extract all numbers from the port list
        for num in re.findall(r'\d+', port_list):
            port = int(num)
            if 0 < port < 65536:
                ports.add(port)
    
    # Also match individual "port NNN" patterns
    for match in re.finditer(r'\bport\s*:?\s*(\d+)', text, re.IGNORECASE):
        port = int(match.group(1))
        if 0 < port < 65536:
            ports.add(port)
    
    # Also match colons like ":443"
    for match in re.finditer(r':(\d{4,5})\b', text):
        port = int(match.group(1))
        if 0 < port < 65536:
            ports.add(port)
    
    return sorted(list(ports))


def _extract_countries_from_text(text: str) -> list[str]:
    """Extract country names from text."""
    if not text:
        return []
    
    # Common country names that might appear
    countries_map = {
        'united states': 'United States',
        'us': 'United States',
        'usa': 'United States',
        'china': 'China',
        'russia': 'Russia',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        'germany': 'Germany',
        'india': 'India',
        'france': 'France',
        'japan': 'Japan',
        'brazil': 'Brazil',
        'canada': 'Canada',
        'mexico': 'Mexico',
        'australia': 'Australia',
    }
    
    countries = set()
    text_lower = text.lower()
    
    for country_key, country_name in countries_map.items():
        if country_key in text_lower:
            countries.add(country_name)
    
    return sorted(list(countries))


def _fallback_plan_from_question(
    question: str,
    previous_results: dict | None = None
) -> dict:
    """
    Fallback query planning when LLM fails: extract search parameters from question text.
    
    This allows the system to continue even when LLM planning fails with JSON parse errors.
    Uses regex and heuristics to extract:
    - Ports (from "port 443" or ":443" patterns)
    - IPs (from IP addresses in text)
    - Countries (from country names)
    - Search terms (words that aren't structural)
    - Search type (inferred from keywords)
    
    For follow-up questions, extracts IPs from previous results and adds them to search_terms.
    """
    question_lower = question.lower()
    
    # Extract structured data
    ports = _extract_ports_from_text(question)
    ips = _extract_ips_from_text(question)
    countries = _extract_countries_from_text(question)
    
    # Also extract IPs from previous results if available
    # For follow-up questions, add these directly to search_terms
    if previous_results:
        previous_ips = _extract_ips_from_previous_results(previous_results)
        for ip in previous_ips:
            if ip not in ips:
                ips.append(ip)
    
    # Infer search type from keywords
    search_type = "general"
    if any(kw in question_lower for kw in ["alert", "signature", "et policy", "et exploit", "et info", "et drop"]):
        search_type = "alert"
    elif any(kw in question_lower for kw in ["traffic", "connection", "flow", "packet", "network", "port", "protocol", "happening", "activity"]):
        search_type = "traffic"
    elif ips or "ip" in question_lower or "country" in question_lower:
        search_type = "ip"
    elif any(kw in question_lower for kw in ["domain", "host", "dns"]):
        search_type = "domain"
    
    # Extract key terms (remove common words and structural elements)
    stop_words = {
        "what", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an",
        "on", "in", "at", "to", "from", "with", "by", "was", "port", "ports", "associated",
        "traffic", "alert", "signature", "and", "or", "this", "that", "these", "those",
        "for", "of", "this", "country", "countries", "ip", "ips", "address", "addresses",
        "happening", "activity", "protocol", "protocols", "connection", "flow", "packet"
    }
    
    search_terms = []
    for word in question.split():
        clean_word = word.strip('.,!?;:"\'-').lower()
        if clean_word and clean_word not in stop_words and len(clean_word) > 2:
            # Exclude IPs and ports
            if not re.match(r'^\d+$', clean_word) and not _IP_PATTERN.match(clean_word):
                search_terms.append(clean_word)
    
    # Remove duplicates while preserving order
    seen = set()
    search_terms = [t for t in search_terms if not (t in seen or seen.add(t))]
    
    # For IP-type searches (or when IPs are extracted), add them to search_terms
    if (search_type == "ip" or ips) and ips:
        # IPs become search terms for IP-specific searches
        search_terms = ips + search_terms
    
    return {
        "reasoning": f"Fallback plan (LLM planning failed): Extracted ports={ports}, countries={countries}, ips={ips}, search_type={search_type}",
        "search_type": search_type,
        "search_terms": search_terms,
        "countries": countries,
        "ports": ports,
        "protocols": [],
        "time_range": "now-90d",
        "matching_strategy": "token",
        "field_analysis": "Using fallback heuristic extraction from question text",
        "skip_search": False,  # Important: don't skip the search
    }


def _extract_ips_from_previous_results(previous_results: dict) -> list[str]:
    """Extract IPs from previous skill results for follow-up questions."""
    if not previous_results:
        return []

    seen: set[str] = set()
    ips: list[str] = []
    field_candidates = (
        "src_ip",
        "dest_ip",
        "source.ip",
        "destination.ip",
        "source_ip",
        "destination_ip",
        "ip",
    )

    for skill_result in previous_results.values():
        if not isinstance(skill_result, dict):
            continue

        for record in skill_result.get("results", []) or []:
            if not isinstance(record, dict):
                continue

            for field in field_candidates:
                value = record.get(field)
                if isinstance(value, str) and value not in seen and _IP_PATTERN.fullmatch(value):
                    seen.add(value)
                    ips.append(value)

            for value in (
                record.get("source", {}).get("ip") if isinstance(record.get("source"), dict) else None,
                record.get("destination", {}).get("ip") if isinstance(record.get("destination"), dict) else None,
            ):
                if isinstance(value, str) and value not in seen and _IP_PATTERN.fullmatch(value):
                    seen.add(value)
                    ips.append(value)

    return ips


def _question_asks_for_ip_geolocation(question: str) -> bool:
    """Return True for follow-ups asking where referenced IPs are from."""
    q = str(question or "").lower()
    has_geo_intent = any(token in q for token in ("country", "countries", "origin", "where are", "where is", "from which country"))
    refers_to_prior_ips = "these ip" in q or "those ip" in q or "the ip" in q or "their country" in q
    return has_geo_intent and refers_to_prior_ips


def _question_asks_for_followup_details(question: str) -> bool:
    """Return True for follow-ups asking for details about previously mentioned traffic/IPs."""
    q = str(question or "").lower()
    # Check for follow-up patterns asking about traffic/connection details
    asks_for_details = any(token in q for token in (
        "what port", "which port", "what ports", 
        "what protocol", "which protocol",
        "traffic", "connection",
        "associated with", "associated with this",
        "from that", "from those", "from the"
    ))
    # More flexible pattern matching for previous context references
    refers_to_prior_context = any(phrase in q for phrase in (
        "this traffic", "these ips", "that ip", "the traffic",
        "that connection", "that traffic", "these connections"
    ))
    return asks_for_details and refers_to_prior_context


def _recover_followup_plan_from_context(
    question: str,
    query_plan: dict,
    previous_results: dict,
    conversation_history: list[dict],
) -> dict:
    """
    Recover concrete IP/traffic criteria for follow-up questions from context.

    Handles:
    - Geographic follow-ups: "What countries are these IPs from?"
    - Port/Protocol follow-ups: "What port was associated with this traffic?"
    - Traffic detail follow-ups: "What protocols were used?"
    
    Example:
      - Prior answer listed IPs from an alert search (147.185.132.112, 192.168.0.16)
      - User asks: "What countries are these IPs from?" → Recovered IPs from context
      - User asks: "What port was associated with this traffic?" → Search for ports on recovered IPs
    """
    plan = dict(query_plan or {})
    if plan.get("search_terms") or plan.get("countries") or plan.get("ports") or plan.get("protocols"):
        return plan

    # Check if this is a follow-up asking for details about previously mentioned traffic
    if _question_asks_for_ip_geolocation(question):
        # Existing geolocation recovery logic
        candidate_ips = _extract_ips_from_previous_results(previous_results)
        if not candidate_ips:
            for message in reversed(conversation_history or []):
                content = message.get("content", "") if isinstance(message, dict) else ""
                extracted = _extract_ips_from_text(content)
                for ip in extracted:
                    if ip not in candidate_ips:
                        candidate_ips.append(ip)
                if candidate_ips:
                    break

        if not candidate_ips:
            return plan

        recovered_ips = candidate_ips[:12]
        reasoning_prefix = (plan.get("reasoning") or "").strip()
        recovery_reason = f"Recovered {len(recovered_ips)} IP(s) from prior context for geographic follow-up lookup."

        plan["search_type"] = "ip"
        plan["search_terms"] = recovered_ips
        plan["countries"] = []
        plan["ports"] = plan.get("ports", []) if isinstance(plan.get("ports"), list) else []
        plan["protocols"] = plan.get("protocols", []) if isinstance(plan.get("protocols"), list) else []
        plan["matching_strategy"] = "term"
        plan["time_range"] = plan.get("time_range") or "now-90d"
        plan["field_analysis"] = "Using contextual IP addresses from previous results/history because the current question is a referential follow-up."
        plan["reasoning"] = f"{reasoning_prefix} {recovery_reason}".strip()
        return plan
    
    # Handle follow-ups asking about traffic details (ports, protocols, etc.)
    if _question_asks_for_followup_details(question):
        candidate_ips = _extract_ips_from_previous_results(previous_results)
        if not candidate_ips:
            for message in reversed(conversation_history or []):
                content = message.get("content", "") if isinstance(message, dict) else ""
                extracted = _extract_ips_from_text(content)
                for ip in extracted:
                    if ip not in candidate_ips:
                        candidate_ips.append(ip)
                if candidate_ips:
                    break

        if not candidate_ips:
            return plan

        recovered_ips = candidate_ips[:12]
        reasoning_prefix = (plan.get("reasoning") or "").strip()
        recovery_reason = f"Recovered {len(recovered_ips)} IP(s) from prior context for traffic detail follow-up."
        
        # Infer what details are being asked for from the question
        q_lower = question.lower()
        is_port_query = "port" in q_lower
        is_protocol_query = "protocol" in q_lower
        
        # Build search focused on the recovered IPs and the requested details
        plan["search_type"] = "traffic"
        plan["search_terms"] = recovered_ips
        if is_port_query:
            plan["ports"] = []  # Empty ports means "search for any port on these IPs"
        if is_protocol_query:
            plan["protocols"] = []  # Empty protocols means "search for any protocol on these IPs"
        plan["countries"] = plan.get("countries", []) if isinstance(plan.get("countries"), list) else []
        plan["matching_strategy"] = "term"
        plan["time_range"] = plan.get("time_range") or "now-90d"
        plan["field_analysis"] = f"Searching traffic details on recovered IPs: {', '.join(recovered_ips[:3])}"
        plan["reasoning"] = f"{reasoning_prefix} {recovery_reason}".strip()
        return plan

    return plan


def _execute_search_with_llm_repair(db: Any, llm: Any, index: str, query: dict, size: int = None) -> list[dict]:
    """
    Execute search with intelligent repair on malformed queries.
    
    Uses QueryRepairMemory to remember successful fixes and avoid redundant LLM calls.
    Retries up to 3 times with progressively detailed prompts.
    """
    if size is None:
        size = query.get("size", 200)
    
    try:
        logger.debug("[%s] Executing search query on index: %s", SKILL_NAME, index)
        return db.search(index, query, size=size)
    except Exception as exc:
        from core.db_connector import QueryMalformedException
        
        if isinstance(exc, QueryMalformedException):
            logger.warning("[%s] Query malformed: %s — attempting intelligent repair", SKILL_NAME, exc.error_message)
            
            from core.query_repair import IntelligentQueryRepair
            repair = IntelligentQueryRepair(db, llm)
            success, results, message = repair.repair_and_retry(index, exc.original_query, size=size)
            
            if success:
                logger.info("[%s] Repair successful! Got %d results", SKILL_NAME, len(results or []))
                return results or []
            else:
                logger.error("[%s] Repair failed: %s", SKILL_NAME, message)
                return []
        else:
            # Non-malformed errors
            logger.error("[%s] Unexpected search error (type: %s): %s", SKILL_NAME, type(exc).__name__, exc)
            return []

SKILL_NAME = "opensearch_querier"


def run(context: dict) -> dict:
    """Entry point for opensearch_querier skill."""
    from core.query_builder import (
        discover_field_mappings,
        build_keyword_query,
    )

    db = context.get("db")
    llm = context.get("llm")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    previous_results = context.get("previous_results", {})

    # Defensive check: ensure db is actually a database connector
    if db is None:
        logger.warning("[%s] db not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db"}
    
    from core.db_connector import BaseDBConnector
    if not isinstance(db, BaseDBConnector):
        logger.error("[%s] db is not a BaseDBConnector! Got type=%s. Context keys: %s", 
                    SKILL_NAME, type(db).__name__, list(context.keys()))
        return {"status": "error", "error": f"db is corrupted: {type(db).__name__}"}

    # Get query parameters - if provided explicitly, use them
    # Otherwise, use defaults and let LLM determine search strategy
    index = parameters.get("index", cfg.get("db", "logs_index", default="securityclaw-logs"))
    question = parameters.get("question", parameters.get("query"))
    
    # If neither question/query provided, this was likely a direct dispatch with
    # explicit parameters like keywords, query_type, etc.
    if not question and (parameters.get("keywords") or parameters.get("raw_query")):
        return _execute_explicit_query(context, index)
    
    if not question:
        logger.warning("[%s] No question provided in parameters", SKILL_NAME)
        return {"status": "skipped", "reason": "no question"}
    
    # ── LLM PLANNING PHASE (like rag_querier) ────────────────────────────────
    # Use LLM to understand what to search for
    if llm is None:
        logger.warning("[%s] LLM not available for query planning.", SKILL_NAME)
        return {"status": "skipped", "reason": "no llm"}
    
    conversation_history = parameters.get("conversation_history", [])
    
    # ── CHECK FOR PREVIOUSLY DISCOVERED FIELDS (from fields_querier) ──────────
    # If fields_querier ran before, use its discovered field mappings
    field_mappings = None
    if previous_results.get("fields_querier"):
        fields_result = previous_results["fields_querier"]
        if fields_result.get("status") == "ok":
            field_mappings = fields_result.get("field_mappings") or fields_result.get("findings", {}).get("field_mappings")
            if field_mappings:
                logger.info("[%s] Using field mappings discovered by fields_querier", SKILL_NAME)
    
    # Always ensure we have complete field mappings from DB schema
    # This handles cases where fields_querier returns partial mappings
    full_field_mappings = discover_field_mappings(db, llm)
    
    # Merge field_mappings: use fields_querier as primary, but supplement with DB discovery
    # This ensures we have both the curated fields AND any fields fields_querier might have missed
    if field_mappings:
        logger.debug("[%s] run: Before merge - text_fields count: %d, has_alert.signature: %s", 
                    SKILL_NAME, len(field_mappings.get("text_fields", [])), 
                    "alert.signature" in field_mappings.get("text_fields", []))
        # Merge: add any missing categories and fields from discovery
        for category, fields in full_field_mappings.items():
            if category not in field_mappings:
                field_mappings[category] = fields
            elif isinstance(fields, list) and isinstance(field_mappings.get(category), list):
                # Merge lists: fields_querier first, then add any from discovery that aren't there
                existing = set(field_mappings[category])
                for field in fields:
                    if field not in existing:
                        field_mappings[category].append(field)
        logger.info("[%s] Supplemented fields_querier mappings with full DB discovery", SKILL_NAME)
        logger.debug("[%s] run: After merge - text_fields count: %d, has_alert.signature: %s", 
                    SKILL_NAME, len(field_mappings.get("text_fields", [])), 
                    "alert.signature" in field_mappings.get("text_fields", []))
    else:
        field_mappings = full_field_mappings
        logger.info("[%s] Using full field mappings from DB discovery", SKILL_NAME)
    
    query_plan = _plan_opensearch_query_with_llm(
        question, conversation_history, field_mappings, llm
    )
    query_plan = _recover_followup_plan_from_context(
        question,
        query_plan,
        previous_results,
        conversation_history,
    )
    
    if not query_plan or query_plan.get("skip_search"):
        logger.info("[%s] LLM determined no search needed.", SKILL_NAME)
        return {"status": "no_action", "reason": "query not needed for raw logs"}
    
    # ── LOG REASONING STEP 1: Intent Analysis ──────────────────────────
    logger.info("[%s] REASONING CHAIN - Step 1: Intent Analysis", SKILL_NAME)
    logger.info("[%s]   Search Type: %s", SKILL_NAME, query_plan.get("search_type"))
    logger.info("[%s]   Matching Strategy: %s", SKILL_NAME, query_plan.get("matching_strategy"))
    logger.info("[%s]   Reasoning: %s", SKILL_NAME, query_plan.get("reasoning", "")[:150])
    
    search_terms = query_plan.get("search_terms", [])
    countries = query_plan.get("countries", [])
    ports = query_plan.get("ports", [])
    protocols = query_plan.get("protocols", [])
    time_range = query_plan.get("time_range", "now-90d")
    matching_strategy = query_plan.get("matching_strategy", "token")

    has_criteria = bool(search_terms or countries or ports or protocols)
    if not has_criteria:
        logger.info("[%s] LLM planning: no search criteria extracted.", SKILL_NAME)
        return {"status": "no_action"}

    # ── BUILD QUERY using LLM-determined strategy ──────────────────────────────
    # Let LLM decide all aspects including field selection and matching strategy
    query = _build_opensearch_query(
        search_terms=search_terms,
        countries=countries,
        ports=ports,
        protocols=protocols,
        time_range=time_range,
        field_mappings=field_mappings,
        matching_strategy=matching_strategy,
    )
    query["size"] = parameters.get("size", 200)

    logger.debug("[%s] Built query: %s", SKILL_NAME, str(query)[:500])
    logger.info(
        "[%s] Querying '%s': %s | Strategy=%s | Time: %s | Countries: %s | Ports: %s | Terms: %s | Field_mappings_type=%s",
        SKILL_NAME, index, query_plan.get("reasoning", ""), matching_strategy, time_range, countries, ports, search_terms,
        type(field_mappings).__name__
    )

    try:
        results = _execute_search_with_llm_repair(db, llm, index, query)
        logger.info("[%s] Raw results from opensearch: %d items", SKILL_NAME, len(results) if results else 0)
        
        # ── LOG REASONING STEP 2: Query Execution ──────────────────────────
        logger.info("[%s] REASONING CHAIN - Step 2: Query Execution", SKILL_NAME)
        logger.info("[%s]   Results Found: %d", SKILL_NAME, len(results) if results else 0)
        if not results:
            logger.info("[%s]   Status: ZERO RESULTS - will attempt multi-turn diagnosis", SKILL_NAME)
        
        # DEBUG: Log the actual query and first few results for investigation
        import json
        query_must = query.get("query", {}).get("bool", {}).get("must", [])
        if query_must and isinstance(query_must[0], dict) and "bool" in query_must[0]:
            should_clause = query_must[0]["bool"].get("should", [])
            field_names = set()
            for clause in should_clause:
                for key in clause:
                    if isinstance(clause[key], dict):
                        field_names.update(clause[key].keys())
            logger.debug("[%s] Query fields in should clause: %s", SKILL_NAME, field_names)
        
        if results:
            logger.debug("[%s] First result signature: %s", SKILL_NAME, 
                        results[0].get("alert", {}).get("signature", "N/A")[:80] if results[0].get("alert", {}).get("signature") else "N/A")

        # ── RESULT VALIDATION WITH REFLECTION ──────────────────────────
        # If we got results, validate that they actually match the intent
        if results:
            validation = _llm_validate_results_reflective(
                question=question,
                search_terms=search_terms,
                results=results,
                previous_validation_failed=False,
                llm=llm,
            )
            
            # ── LOG REASONING STEP 3: Validation & Reflection ──────────────────────────
            logger.info("[%s] REASONING CHAIN - Step 3: Validation & Reflection", SKILL_NAME)
            logger.info("[%s]   Valid: %s | Confidence: %.1f%%", SKILL_NAME, 
                       validation.get("is_valid"), validation.get("confidence", 0))
            if not validation.get("is_valid"):
                logger.warning("[%s]   Issue: %s", SKILL_NAME, validation.get("issue"))
                logger.info("[%s]   LLM Reflection: %s", SKILL_NAME, validation.get("reflection", "none")[:200])
            
            if not validation.get("is_valid"):
                logger.warning(
                    "[%s] LLM validation failed: %s | Reflection: %s",
                    SKILL_NAME, validation.get("issue"), validation.get("reflection", "none")
                )
                # Try recovery: switch matching strategy
                recovery_strategy = "token" if matching_strategy == "phrase" else "phrase"
                logger.info("[%s] Trying recovery with alternate strategy: %s", SKILL_NAME, recovery_strategy)
                
                recovery_query = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=time_range,
                    field_mappings=field_mappings,
                    matching_strategy=recovery_strategy,
                )
                recovery_query["size"] = parameters.get("size", 200)
                
                recovery_results = _execute_search_with_llm_repair(db, llm, index, recovery_query)
                
                if recovery_results:
                    recovery_validation = _llm_validate_results_reflective(
                        question=question,
                        search_terms=search_terms,
                        results=recovery_results,
                        previous_validation_failed=True,
                        llm=llm,
                    )
                    if recovery_validation.get("is_valid"):
                        logger.info("[%s] Recovery strategy succeeded after reflection", SKILL_NAME)
                        results = recovery_results
                        validation = recovery_validation
                    else:
                        logger.warning("[%s] Recovery failed, keeping original results", SKILL_NAME)

        # Recovery: if primary query returns 0 results, diagnose why and suggest recovery
        if not results:
            # Fast check: pattern-based diagnosis (keep for speed)
            logger.warning("[%s] Primary query returned 0 results - performing multi-turn diagnosis", SKILL_NAME)
            
            # ── LOG REASONING STEP 3: Multi-Turn Diagnosis ──────────────────────────
            logger.info("[%s] REASONING CHAIN - Step 3: Zero Results Diagnosis", SKILL_NAME)
            
            diagnosis = _diagnose_query_failure(
                question=question,
                search_terms=search_terms,
                field_mappings=field_mappings,
                last_strategy=matching_strategy,
                llm=llm,
            )
            
            logger.info("[%s]   Suggested Recovery: %s", SKILL_NAME, diagnosis.get("suggested_recovery", "none")[:200])
            
            # Try LLM-suggested recovery if it looks promising
            if diagnosis.get("should_try_recovery"):
                # First try: phrase → token recovery (most common fix)
                recovery_strategy = "token" if matching_strategy == "phrase" else "phrase"
                logger.info("[%s] Attempting strategy switch recovery: %s → %s", 
                           SKILL_NAME, matching_strategy, recovery_strategy)
                
                recovery_query = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=time_range,
                    field_mappings=field_mappings,
                    matching_strategy=recovery_strategy,
                )
                if recovery_query:
                    recovery_query["size"] = parameters.get("size", 200)
                    results = _execute_search_with_llm_repair(db, llm, index, recovery_query)
                    if results:
                        logger.info("[%s] Recovery successful: got %d results after strategy switch", 
                                   SKILL_NAME, len(results))
            
            # If still nothing and we have countries, try relaxed matching
            if not results and countries:
                logger.info("[%s] Still no results with countries filter - trying relaxed matching", SKILL_NAME)
                recovery = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=time_range,
                    field_mappings=field_mappings,
                    relaxed=True,
                )
                if recovery:
                    recovery["size"] = parameters.get("size", 200)
                    results = _execute_search_with_llm_repair(db, llm, index, recovery)
                    if results:
                        logger.info("[%s] Relaxed recovery succeeded: got %d results", SKILL_NAME, len(results))

        return {
            "status": "ok",
            "results_count": len(results) if results else 0,
            "results": results[:25],  # Return top 25 for display
            "search_terms": search_terms,
            "countries": countries,
            "ports": ports,
            "protocols": protocols,
            "time_range": time_range,
            "reasoning": query_plan.get("reasoning", ""),
            "reasoning_chain": {
                "planning": query_plan.get("reasoning"),
                "strategy_used": matching_strategy,
                "recovery_performed": not results if not results else False,
            }
        }
    except Exception as exc:
        logger.error("[%s] Search failed: %s", SKILL_NAME, exc)
        return {"status": "error", "error": str(exc)}


# Country code mapping (same as rag_querier)
_COUNTRY_CODE_MAP = {
    "iran": "IR", "iraq": "IQ", "syria": "SY", "north korea": "KP",
    "china": "CN", "russia": "RU", "united states": "US", "usa": "US",
    "uk": "GB", "united kingdom": "GB", "france": "FR", "germany": "DE",
    "india": "IN", "pakistan": "PK",
}


def _build_opensearch_query(
    search_terms: list,
    countries: list,
    ports: list,
    protocols: list,
    time_range: str,
    field_mappings: dict,
    matching_strategy: str = "token",
    relaxed: bool = False,
) -> dict:
    """
    Build a robust OpenSearch query using LLM-recommended matching strategy.
    
    Args:
        matching_strategy: "phrase" (exact phrase), "token" (tokenized), "term" (exact term)
                          - Chosen by LLM based on field analysis
    """
    # CRITICAL: Override matching strategy for IP addresses
    # IP addresses must use term matching (exact match), never phrase matching
    if matching_strategy == "phrase" and search_terms:
        import re
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$|^[a-f0-9:]{3,}$'  # IPv4 or IPv6-like
        if any(re.match(ip_pattern, str(t)) for t in search_terms):
            logger.info("[%s] _build_opensearch_query: Detected IP address in search terms — overriding strategy from 'phrase' to 'term'", SKILL_NAME)
            matching_strategy = "term"
    
    must_clauses = []
    all_fields = field_mappings.get("all_fields") or []
    country_fields = [f for f in all_fields if "country" in str(f).lower()][:10]
    port_fields = (field_mappings.get("port_fields") or [])[:6]
    
    # ── PRIORITIZE SPECIFIC FIELD TYPES ──────────────────────────────────
    # If specific text_fields are provided (e.g., for alert queries), use them
    # Otherwise, infer common fields that are likely to have the data
    # Skip timestamp/numeric fields that shouldn't be used for text matching
    if field_mappings.get("text_fields"):
        available_text = field_mappings["text_fields"]
    else:
        available_text = []
    
    logger.debug("[%s] _build_opensearch_query: available_text before injection: %s (length=%d, has_alert.signature=%s)", 
                SKILL_NAME, available_text[:15] if len(available_text) > 15 else available_text, 
                len(available_text), "alert.signature" in available_text)
    
    # Make sure alert fields are included if available
    if not available_text or "alert.signature" not in available_text:
        all_text_with_alerts = list(available_text) if available_text else []
        # Add alert-related fields from all_fields if they're not already there
        alert_fields_to_add = [f for f in all_fields if "alert" in str(f).lower()]
        logger.debug("[%s] _build_opensearch_query: Adding %d alert fields", SKILL_NAME, len(alert_fields_to_add))
        for af in alert_fields_to_add:
            if af not in all_text_with_alerts:
                all_text_with_alerts.append(af)
        available_text = all_text_with_alerts
        logger.debug("[%s] _build_opensearch_query: After injection: length=%d, has_alert.signature=%s", 
                    SKILL_NAME, len(available_text), "alert.signature" in available_text)
    
    # Now filter out timestamp/numeric fields
    filter_keywords = ("timestamp", "time", "date", "epoch", "_ms", "port", "bytes", "count", "length", "size")
    filtered_text = [
        f for f in available_text
        if not any(k in str(f).lower() for k in filter_keywords)
    ]
    
    # CRITICAL: Prioritize alert.* fields to ensure they're included in top 12
    # Separate alert fields from others
    alert_fields_list = [f for f in filtered_text if f.startswith("alert.")]
    non_alert_fields = [f for f in filtered_text if not f.startswith("alert.")]
    
    # Put alert fields first, then other fields, take first 12
    # This ensures alert.signature, alert.category, etc. are included
    prioritized = alert_fields_list + non_alert_fields
    text_fields = prioritized[:12]
    
    # CRITICAL FIX: If we have alert fields AND search terms that look like alert signatures,
    # use ONLY alert fields to avoid OpenSearch field mismatch issues  
    # (mixing alert.signature with flow.state causes "no results" in some cases)
    has_alert_fields = len(alert_fields_list) > 0
    search_looks_like_alert = any(
        term.upper() in ("EXPLOIT", "MALWARE", "POLICY", "CVE", "BACKDOOR", "TROJAN", "WORM")
        or "ET " in term.upper() 
        for term in search_terms
    )
    
    logger.debug("[%s] _build_opensearch_query: has_alert_fields=%s, search_looks_like_alert=%s, search_terms=%s",
                SKILL_NAME, has_alert_fields, search_looks_like_alert, search_terms)
    
    if has_alert_fields and search_looks_like_alert:
        # Use only alert fields for alert signature searches
        text_fields = alert_fields_list[:12] if alert_fields_list else text_fields
        logger.info("[%s] _build_opensearch_query: Using alert-only fields for alert signature search. Fields: %s", SKILL_NAME, text_fields)
    
    logger.debug("[%s] _build_opensearch_query: Prioritized fields with alerts first: %s (has_alert.signature=%s)", 
                SKILL_NAME, text_fields, "alert.signature" in text_fields)

    # Country matching — use match_phrase for full name AND term for ISO code
    # IMPORTANT: Skip country filters for alert signature searches - the alert itself is what matters
    # Alert records may not have geoip data, and filtering by country would lose alert results
    if countries and country_fields and not (search_looks_like_alert and has_alert_fields):
        country_should = []
        for field in country_fields:
            for country in countries:
                country_should.append({"match_phrase": {field: country}})
                code = _COUNTRY_CODE_MAP.get(country.lower())
                if code:
                    country_should.append({"term": {field: code}})
                    country_should.append({"term": {field: code.lower()}})
        if country_should:
            must_clauses.append({"bool": {"should": country_should, "minimum_should_match": 1}})
    elif countries and not country_fields:
        # No discovered country fields — fall back to multi_match on text fields
        for country in countries:
            must_clauses.append({"multi_match": {"query": country, "fields": text_fields or ["*"]}})

    # Port matching
    if ports and port_fields:
        port_should = []
        for field in port_fields:
            for p in ports:
                try:
                    port_should.append({"term": {field: int(p)}})
                except Exception:
                    pass
        if port_should:
            must_clauses.append({"bool": {"should": port_should, "minimum_should_match": 1}})

    # Protocol matching
    proto_fields = [f for f in all_fields if "proto" in str(f).lower()][:6]
    if protocols and proto_fields:
        proto_should = []
        for field in proto_fields:
            for proto in protocols:
                proto_should.append({"term": {field: str(proto).lower()}})
                proto_should.append({"match": {field: str(proto)}})
        if proto_should:
            must_clauses.append({"bool": {"should": proto_should, "minimum_should_match": 1}})

    # Keyword search - use LLM-recommended matching strategy
    if search_terms and (relaxed or not must_clauses):
        # CRITICAL: For IP addresses, search in IP fields, not text fields
        import re
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$|^[a-f0-9:]{3,}$'  # IPv4 or IPv6-like
        is_ip_search = any(re.match(ip_pattern, str(t)) for t in search_terms)
        
        if is_ip_search:
            # Use IP-specific fields for IP searches
            ip_fields = [f for f in all_fields if any(k in str(f).lower() for k in ("src_ip", "dest_ip", "ip"))][:10]
            fields = ip_fields if ip_fields else [f for f in all_fields if "ip" in str(f).lower()][:10]
            logger.info("[%s] _build_opensearch_query: Detected IP search - using IP fields: %s", SKILL_NAME, fields[:5] if len(fields) > 5 else fields)
        else:
            # Select text fields for keyword search
            # Priority 1: Use provided text_fields if available  (already prioritized with alerts first)
            # Priority 2: Fall back to fields containing "message", "alert", "signature", etc.
            
            # NOTE: text_fields was already carefully selected and prioritized above
            # to ensure alert fields come first. Don't re-extract or re-filter here!
            if not text_fields:
                # Only fall back if text_fields is empty (which shouldn't happen after prioritization above)
                if field_mappings.get("text_fields"):
                    text_fields = field_mappings["text_fields"][:12]
                    text_fields = [
                        f for f in text_fields
                        if not any(k in str(f).lower() for k in ("timestamp", "time", "date", "epoch", "_ms", "port", "id", "bytes", "count", "length", "size", "timeout"))
                    ]
                else:
                    text_fields = [
                        f for f in all_fields
                        if any(k in str(f).lower() for k in ("message", "alert", "signature", "event", "hostname", "domain"))
                        and not any(k in str(f).lower() for k in ("timestamp", "time", "date", "epoch", "_ms", "port", "id", "bytes", "count", "length", "size", "timeout"))
                    ][:12]

            fields = text_fields or ["*"]
        logger.debug("[%s] _build_opensearch_query: Building keyword search with fields: %s (first 5 shown)", 
                    SKILL_NAME, fields[:5] if len(fields) > 5 else fields)
        kw_should = []
        
        if matching_strategy == "phrase":
            # Exact phrase matching (no tokenization)
            # Use for structured fields like rule names
            for field in fields:
                for t in search_terms:
                    if t:
                        kw_should.append({"match_phrase": {field: str(t)}})
            logger.debug("[%s] _build_opensearch_query: Created %d phrase match clauses", SKILL_NAME, len(kw_should))
        
        elif matching_strategy == "term":
            # Exact term matching (case-insensitive, no analysis)
            # Use for keyword fields
            for field in fields:
                for t in search_terms:
                    if t:
                        kw_should.append({"term": {field: str(t).lower()}})
        
        else:  # "token" or default
            # Standard multi_match (tokenized text search)
            # Use for free-text fields
            kw_should = [{"multi_match": {"query": str(t), "fields": fields}} for t in search_terms if t]
        
        if kw_should:
            must_clauses.append({"bool": {"should": kw_should, "minimum_should_match": 1}})

    if not must_clauses:
        return {"query": {"match_none": {}}}

    time_filter = {"range": {"@timestamp": {"gte": time_range}}}
    return {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": [time_filter],
            }
        }
    }


def _diagnose_query_failure(
    question: str,
    search_terms: list,
    field_mappings: dict,
    last_strategy: str,
    llm: Any,
) -> dict:
    """
    When initial query returns 0 results, ask LLM to reason about why.
    Multi-turn reflection to understand the real issue.
    """
    logger.info("[%s] _diagnose_query_failure: Analyzing why query with strategy '%s' returned 0 results",
                SKILL_NAME, last_strategy)
    
    # First pass: initial diagnosis
    diagnosis_prompt = f"""The user asked: "{question}"

We searched for: {search_terms}
Using strategy: {last_strategy}
Available fields: {list(field_mappings.keys())[:10]}...

We got 0 results. Why might that be? Consider:
1. Is the search term likely to exist in this database?
2. Is the strategy (phrase/token/term) wrong for this data type?
3. Are we searching in the right kind of fields?
4. Is the term too specific or too general?
5. Could this be a data structure issue (nested vs flat)?

Diagnose the most likely root cause."""

    try:
        diagnosis = llm.complete(diagnosis_prompt)
        logger.info("[%s] _diagnose_query_failure DIAGNOSIS:\n%s", SKILL_NAME, diagnosis[:300])
        
        # Second pass: suggest recovery
        recovery_prompt = f"""Based on this diagnosis: {diagnosis}

What is ONE specific thing we should try next?
- Change matching strategy? (phrase→token or vice versa)
- Search in different fields?
- Relax field selection?
- Ask user for clarification?
- Accept that this data isn't in the database?

Return actionable next step with explanation."""
        
        recovery = llm.complete(recovery_prompt)
        logger.info("[%s] _diagnose_query_failure RECOVERY SUGGESTION:\n%s", SKILL_NAME, recovery[:300])
        
        return {
            "diagnosis": diagnosis,
            "suggested_recovery": recovery,
            "should_try_recovery": "change" in recovery.lower() or "search" in recovery.lower(),
        }
    except Exception as exc:
        logger.warning("[%s] Diagnosis failed: %s", SKILL_NAME, exc)
        return {
            "diagnosis": f"Diagnosis failed: {exc}",
            "suggested_recovery": "fallback to token matching",
            "should_try_recovery": True,
        }


def _llm_validate_results_reflective(
    question: str,
    search_terms: list,
    results: list,
    previous_validation_failed: bool,
    llm: Any,
) -> dict:
    """
    Validate results with reflection. If validation fails, reason about why before giving up.
    """
    # First pass: quick validation
    validation = _llm_validate_results(question, search_terms, results, llm)
    
    if validation.get("is_valid"):
        logger.info("[%s] Results validated on first pass", SKILL_NAME)
        return {**validation, "reflection": "Valid on first pass", "iterations": 1}
    
    # Validation failed - reflect on why
    logger.warning("[%s] Results failed validation: %s. Reflecting...", SKILL_NAME, validation.get("issue"))
    
    reflection_prompt = f"""We got results, but validation failed.

Failure reason: {validation.get("issue")}

Looking at the results more carefully:
- Are they the wrong type of data? (e.g., got traffic logs when looking for alerts?)
- Wrong values but right structure? (e.g., alerts for different signatures?)
- Could this be a misunderstanding of the question?
- Is the validation criterion too strict?

Why did validation fail?"""
    
    try:
        reflection = llm.complete(reflection_prompt)
        logger.info("[%s] REFLECTION ON VALIDATION FAILURE:\n%s", SKILL_NAME, reflection[:400])
        
        return {
            **validation,
            "reflection": reflection,
            "iterations": 2 if previous_validation_failed else 1,
            "should_retry": "strict" in reflection.lower() or "misunderstanding" in reflection.lower(),
        }
    except Exception as exc:
        logger.warning("[%s] Reflection failed: %s", SKILL_NAME, exc)
        return {**validation, "reflection": "Reflection attempt failed", "iterations": 1}


def _llm_validate_results(
    question: str,
    search_terms: list,
    results: list,
    llm: Any,
) -> dict:
    """
    Ask LLM to validate that returned results are actually relevant to the question.
    
    This catches cases like:
    - Searching for "ET EXPLOIT" but getting "ET INFO" or "ET POLICY"
    - Results that don't match the intent even if they match some criteria
    
    Returns dict with:
      - is_valid: bool (true if results match intent)
      - issue: str (description of what's wrong, if any)
      - suggestion: str (how to fix it)
    """
    if not results:
        return {"is_valid": False, "issue": "No results returned", "suggestion": "Try relaxed search or different terms"}
    
    # Sample first few results and extract relevant fields
    samples = []
    for result in results[:3]:
        sample_record = {}
        
        # Try to extract signature fields (handle nested alert object)
        if "alert" in result and isinstance(result["alert"], dict):
            alert_obj = result["alert"]
            sample_record["signature"] = alert_obj.get("signature")
            sample_record["signature_id"] = alert_obj.get("signature_id")
            sample_record["category"] = alert_obj.get("category")
        else:
            sample_record["signature"] = result.get("alert.signature") or result.get("signature")
            sample_record["signature_id"] = result.get("alert.signature_id")
            sample_record["category"] = result.get("alert.category")
        
        # Extract IPs/countries for traffic queries
        sample_record["src_ip"] = result.get("src_ip")
        sample_record["dest_ip"] = result.get("dest_ip")
        sample_record["country"] = result.get("geoip.country_name") or result.get("country_name")
        
        samples.append(sample_record)
    
    sample_text = json.dumps(samples, indent=2, default=str)
    
    # DEBUG: Log what samples look like
    logger.debug("[%s] Validation samples extracted: %s", SKILL_NAME, sample_text[:300])
    
    # RED FLAG: If we're searching for alert signatures but samples don't have signatures, log it
    has_any_signature = any(s.get("signature") for s in samples)
    if "alert" in question.lower() or any("alert" in str(t).lower() for t in search_terms):
        if not has_any_signature:
            logger.warning("[%s] ALERT SEARCH BUT NO SIGNATURES IN SAMPLES: question=%s samples=%s",
                          SKILL_NAME, question, sample_text[:500])
    
    prompt = f"""Validate that these search results match the user's intent.

USER QUESTION: "{question}"

SEARCH TERMS: {', '.join(search_terms)}

SAMPLED RESULTS (extracted key fields):
{sample_text}

VALIDATION TASK:
1. Do the results contain fields/values matching the search terms?
2. For signature/alert searches: Do results contain EXACT signatures? (e.g., if searching "ET POLICY", are there records with signature containing "ET POLICY"?)
3. For traffic searches: Do results contain the countries/IPs/ports requested?
4. Are results relevant to the user's intent?

RETURN STRICT JSON:
{{
  "is_valid": true/false,
  "issue": "if not valid, describe the specific problem",
  "suggestion": "how to fix the query if needed",
  "confidence": 0.0-1.0
}}

CRITICAL CHECKS:
- If searching for "ET POLICY", results must have signatures containing "ET POLICY"
- If searching for "ET EXPLOIT", results must have "ET EXPLOIT", NOT "ET POLICY" or others
- If searching for a country, results must have that country in geoip data
- Partial matches ARE acceptable for alert signatures (e.g., "ET POLICY Dropbox" contains "ET POLICY")
"""

    try:
        response = llm.complete(prompt)
        validation = None
        
        try:
            validation = json.loads(response)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                validation = json.loads(m.group())
        
        if not validation:
            return {"is_valid": True, "confidence": 0.5}  # Assume valid if we can't parse
        
        is_valid = bool(validation.get("is_valid", True))
        logger.info("[%s] Result validation: valid=%s, confidence=%.1f%%, issue='%s'",
                   SKILL_NAME, is_valid, float(validation.get("confidence", 0.5)) * 100,
                   str(validation.get("issue", ""))[:60])
        
        # Also log if this is an alert search that failed validation
        if not is_valid and ("alert" in question.lower() or any("ET" in str(t) for t in search_terms)):
            logger.warning("[%s] ALERT SEARCH VALIDATION FAILED: question=%s num_samples=%d first_sig=%s",
                          SKILL_NAME, question[:100], len(samples),
                          samples[0].get("signature", "NONE") if samples else "NO_SAMPLES")
        
        return {
            "is_valid": is_valid,
            "issue": str(validation.get("issue", "")),
            "suggestion": str(validation.get("suggestion", "")),
            "confidence": float(validation.get("confidence", 0.5)),
        }
    except Exception as exc:
        logger.warning("[%s] Result validation failed: %s", SKILL_NAME, exc)
        return {"is_valid": True, "confidence": 0.0}  # Assume valid on error


def _execute_explicit_query(context: dict, index: str) -> dict:
    """
    Execute an explicitly parameterized query (backward compatibility).
    Used when query_type, keywords, raw_query, etc. are passed directly.
    """
    from core.query_builder import (
        discover_field_mappings,
        build_keyword_query,
        build_structured_query,
        build_time_range_query,
    )
    
    db = context.get("db")
    llm = context.get("llm")
    parameters = context.get("parameters", {})
    
    query_type = parameters.get("query_type", "keyword_search")
    size = parameters.get("size", 200)
    field_mappings = discover_field_mappings(db, llm)
    
    logger.info(
        "[%s] Executing explicit %s query against index: %s",
        SKILL_NAME, query_type, index
    )
    
    try:
        query = None
        
        if query_type == "raw_query":
            query = parameters.get("raw_query")
            if not query:
                return {"status": "failed", "reason": "raw_query required"}
        
        elif query_type == "keyword_search":
            keywords = parameters.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            if not keywords:
                return {"status": "failed", "reason": "keywords required"}
            query, _ = build_keyword_query(keywords, field_mappings)
        
        elif query_type == "structured_search":
            ips = parameters.get("ips", [])
            if isinstance(ips, str):
                ips = [ips]
            domains = parameters.get("domains", [])
            if isinstance(domains, str):
                domains = [domains]
            ports = parameters.get("ports", [])
            if isinstance(ports, str):
                ports = [ports]
            time_range = parameters.get("time_range")
            query, _ = build_structured_query(ips, domains, ports, time_range, field_mappings)
        
        elif query_type == "time_range_search":
            time_range = parameters.get("time_range")
            if not time_range:
                return {"status": "failed", "reason": "time_range required"}
            query, _ = build_time_range_query(time_range, field_mappings)
        
        else:
            return {"status": "failed", "reason": f"Unknown query_type: {query_type}"}
        
        if not query:
            return {"status": "failed", "reason": "could not build query"}
        
        results = _execute_search_with_llm_repair(db, llm, index, query, size=size)
        return {
            "status": "ok",
            "results_count": len(results) if results else 0,
            "results": results[:10] if results else [],
        }
    
    except Exception as exc:
        logger.error("[%s] Explicit query failed: %s", SKILL_NAME, exc)
        return {"status": "error", "error": str(exc)}


def _plan_opensearch_query_with_llm_simplified(
    question: str,
    llm: Any,
) -> dict | None:
    """
    Attempt simplified LLM planning with a cleaner, minimal JSON prompt.
    Called as a retry when the main planning fails.
    
    Returns None if this also fails, allowing fallback to heuristic extraction.
    """
    import json
    
    prompt = f"""Analyze this question and return ONLY valid JSON, no other text:

Question: "{question}"

Return this exact JSON structure (all fields required):
{{
  "search_terms": ["term1", "term2"],
  "ports": [443, 8080],
  "search_type": "alert|traffic|ip|general",
  "matching_strategy": "phrase|token"
}}

Rules:
- search_terms: Extract key words from question (exclude: "what", "port", "associated", pronouns)
- ports: Extract numbers like "443", "8080" (must be 1-65535)
- search_type: "alert" if mentions signatures/rules, "traffic" if mentions connections, "ip" if IP addresses, else "general"
- matching_strategy: "phrase" for alerts, "token" for everything else

Output ONLY the JSON, nothing else."""

    try:
        response = llm.complete(prompt).strip()
        logger.debug("[%s] Simplified LLM response: %s", SKILL_NAME, response[:200])
        
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from response
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                return json.loads(m.group())
        
        return None
    except Exception as exc:
        logger.debug("[%s] Simplified LLM also failed: %s", SKILL_NAME, exc)
        return None


def _plan_opensearch_query_with_llm(
    question: str,
    conversation_history: list[dict],
    field_mappings: dict,
    llm: Any,
) -> dict:
    """
    Use LLM to plan OpenSearch query AND decide field-specific matching strategies.
    
    The LLM analyzes field characteristics and recommends:
    - Which fields to search in
    - What matching strategy to use (phrase vs token match) based on field content
    - Search terms and structured filters (countries, ports, protocols)
    
    This is fully data-agnostic - LLM makes all decisions based on field schema.
    """
    from pathlib import Path
    
    # Build conversation context
    conversation_summary = ""
    if conversation_history:
        relevant_msgs = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        conversation_parts = []
        for msg in relevant_msgs:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:200]
            conversation_parts.append(f"[{role}]: {content}")
        conversation_summary = "\n".join(conversation_parts)

    # ── BUILD DETAILED FIELD CONTEXT FOR LLM ──────────────────────────────────
    # Give LLM detailed field characteristics to analyze
    field_context = ""
    if field_mappings:
        field_info_parts = []
        
        # For each field category, include characteristics for LLM analysis
        field_categories = {
            "alert_fields": "ALERT/SIGNATURE FIELDS (for rule names, signatures)",
            "ip_fields": "IP ADDRESS FIELDS (for source/destination IPs)",
            "port_fields": "PORT FIELDS (for destination/source ports)",
            "country_fields": "COUNTRY FIELDS (for geoip country data)",
        }
        
        for field_type, description in field_categories.items():
            fields_list = field_mappings.get(field_type, [])
            if fields_list:
                sample_fields = fields_list[:3]
                field_info_parts.append(f"{description}: {', '.join(sample_fields)}")
        
        if field_info_parts:
            field_context = "\n\nAVAILABLE FIELDS:\n" + "\n".join(field_info_parts)

    prompt = f"""You are planning an OpenSearch query. Analyze the user's question and recommend a query strategy.

CONVERSATION CONTEXT:
{conversation_summary if conversation_summary else "(No prior context)"}{field_context}

USER QUESTION: "{question}"

TASK:
1. Extract search_terms, countries, ports, protocols from the question
2. Identify what type of data the user wants (alerts, traffic, IPs, domains, etc.)
3. For ALERT/SIGNATURE searches: Recommend PHRASE MATCHING (exact signature names - no tokenization)
4. For TRAFFIC/LOG searches: Recommend TOKEN MATCHING (standard text search)
5. Validate: if searching for "ET EXPLOIT", should match ONLY "ET EXPLOIT", not "ET INFO" or "ET POLICY"

RETURN STRICT JSON:
{{
  "reasoning": "why you chose this strategy",
  "search_type": "alert|traffic|domain|ip|general",
  "search_terms": ["term1", "term2"],
  "countries": ["country1"],
  "ports": [443, 80],
  "protocols": ["TCP"],
  "time_range": "now-90d|now-30d|now-7d|custom",
  "matching_strategy": "phrase|token|term|match",
  "field_analysis": "explanation of field choice and matching logic"
}}

CRITICAL MATCHING RULES:
- "phrase": Use match_phrase for exact phrase matching (no tokenization). Best for structured fields like rule names.
  Example: searching "ET EXPLOIT" matches ONLY "ET EXPLOIT", NOT "ET INFORMATION" or "ET POLICY"
- "token": Use standard multi_match (tokenized). For free-text fields like descriptions.
  Example: searching "malware" matches "malware", "Malware C2", "banking malware", etc.
- "term": Use term filter (exact value match, case-insensitive). For keyword fields.
  Example: country codes, exact IPs, port numbers

STRATEGY SELECTION:
- If question asks about IPs, ALWAYS recommend "term" (never "phrase" for IPs)
- If question mentions alerts/signatures/rules → use "phrase" strategy
- If question mentions traffic/flows/connections → use "token" strategy
- If question mentions owner/reputation WITH an IP → still use "term" for that IP
If question mentions IPs, ALWAYS use "term" strategy (even if asking "who is the owner of IP")
If question mentions ports, ALWAYS use "term" strategy
"""

    try:
        response = llm.complete(prompt)
        logger.debug("[%s] LLM Plan raw: %s", SKILL_NAME, response[:300])

        import json
        # Try direct parse; fallback to regex JSON extraction.
        plan = None
        try:
            plan = json.loads(response)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                plan = json.loads(m.group())

        if not plan:
            raise ValueError("No JSON in LLM response")

        # Ensure all required fields exist
        if not isinstance(plan.get("search_terms"), list):
            plan["search_terms"] = []
        if not isinstance(plan.get("countries"), list):
            plan["countries"] = []
        if not isinstance(plan.get("ports"), list):
            plan["ports"] = []
        if not isinstance(plan.get("protocols"), list):
            plan["protocols"] = []
        if not isinstance(plan.get("time_range"), str):
            plan["time_range"] = "now-90d"
        
        # Clean up matching_strategy: LLM might return "term|token" or multiple values
        # Extract the first valid strategy
        strategy = plan.get("matching_strategy", "token")
        if isinstance(strategy, str):
            # Extract first strategy if multiple are given (e.g., "term|token" -> "term")
            strategy = strategy.split("|")[0].split(",")[0].split(" ")[0].lower().strip()
        # Validate it's one of the supported strategies
        if strategy not in ("phrase", "token", "term"):
            strategy = "token"
        plan["matching_strategy"] = strategy

        logger.info(
            "[%s] LLM Plan: Type=%s | Strategy=%s | Terms=%s | Countries=%s | Reasoning=%s",
            SKILL_NAME, plan.get("search_type"), plan.get("matching_strategy"),
            plan.get("search_terms"), plan.get("countries"),
            plan.get("reasoning", "")[:60]
        )

        return plan
    except Exception as exc:
        logger.warning("[%s] LLM planning failed: %s. Attempting simplified prompt...", SKILL_NAME, exc)
        
        # Try simplified LLM planning with minimal JSON prompt
        simplified_plan = _plan_opensearch_query_with_llm_simplified(question, llm)
        if simplified_plan:
            logger.info("[%s] Simplified LLM succeeded: %s", SKILL_NAME, 
                       simplified_plan.get("search_type", "unknown"))
            # Fill in any missing fields
            simplified_plan.setdefault("reasoning", "Simplified LLM planning")
            simplified_plan.setdefault("countries", [])
            simplified_plan.setdefault("protocols", [])
            simplified_plan.setdefault("time_range", "now-90d")
            simplified_plan.setdefault("field_analysis", "Using simplified LLM planning")
            return simplified_plan
        
        # If even simplified LLM fails, fall back to heuristic extraction
        logger.warning("[%s] Simplified LLM also failed. Using fallback heuristic extraction.", SKILL_NAME)
        fallback_plan = _fallback_plan_from_question(question, None)
        logger.info("[%s] Fallback plan: %s", SKILL_NAME, fallback_plan.get("reasoning", "")[:100])
        return fallback_plan
