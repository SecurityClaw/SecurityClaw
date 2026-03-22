"""
skills/opensearch_querier/logic.py

Simplified query executor - execute LLM plans, extract aggregated ports for fingerprinting.
No heuristics, no diagnosis. Pure data access layer.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DATA-AGNOSTIC CONSTRAINT:
  
This skill MUST be completely data-agnostic. It does NOT know field names.
  
RULES (enforce in all future versions):
1. NEVER hardcode field names (dest_ip, destination.port, etc in logic.py)
2. ALWAYS require fields_querier as prerequisite to discover actual field names
3. Use ONLY field names from manifest-declared field_mappings context
4. If field names aren't in context, fail with clear error - don't guess
5. Do NOT add heuristics that try alternate field names
  
This ensures the skill works with ANY database schema without modification.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILL_NAME = "opensearch_querier"
PLANNING_PROMPT_PATH = Path(__file__).parent / "PLANNING_PROMPT.md"


def _log_excerpt(text: Any, limit: int = 400) -> str:
    """Return log-friendly preview."""
    rendered = str(text or "")[:limit]
    return rendered + " ..." if len(str(text or "")) > limit else rendered


def _load_planning_prompt() -> str:
    """Load OpenSearch planning prompt."""
    try:
        return PLANNING_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("[%s] Could not load planning prompt: %s", SKILL_NAME, exc)
        return "Extract structured OpenSearch query parameters from question. Return JSON only."


def _extract_json_object(text: str) -> dict | None:
    """
    Extract the first valid JSON object from a string, even if surrounded by
    natural language. Handles models that prefix/suffix JSON with text.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
                # Move to next '{' and retry
                start = text.find('{', i + 1)
                if start == -1:
                    return None
                depth = 0
    return None


_MINIMAL_PLANNING_PROMPT = """You are extracting OpenSearch query parameters from a network security question.

RULES:
- "aggregation_type" MUST be "fingerprint_ports" ONLY when the question explicitly asks to fingerprint a specific IP or find open ports of a specific IP. For ALL other questions (traffic, country queries, IP queries without fingerprinting), "aggregation_type" MUST be "none".
- "countries" is a list of country names when filtering by geography (e.g. "from Iran", "traffic from Russia").
- "search_terms" is a list of IP addresses to search for.
- "time_range" uses Elasticsearch date math (now-1d, now-7d, now-30d, now-2M, now-90d, now-365d).
- "ip_direction": "source" for traffic FROM a location, "destination" for traffic TO a location, "any" for all.

Output ONLY JSON with these exact fields: search_terms, countries, time_range, ip_direction, aggregation_type.

EXAMPLES:
Question: "fingerprint 1.2.3.4"
JSON: {"search_terms":["1.2.3.4"],"countries":[],"time_range":"now-30d","ip_direction":"destination","aggregation_type":"fingerprint_ports"}

Question: "traffic from Iran past 2 months"
JSON: {"search_terms":[],"countries":["Iran"],"time_range":"now-2M","ip_direction":"source","aggregation_type":"none"}

Question: "any 8.8.8.8 traffic"
JSON: {"search_terms":["8.8.8.8"],"countries":[],"time_range":"now-90d","ip_direction":"any","aggregation_type":"none"}

Question: "Russia traffic last 30 days"
JSON: {"search_terms":[],"countries":["Russia"],"time_range":"now-30d","ip_direction":"source","aggregation_type":"none"}

Question: "show me traffic from China in the last week"
JSON: {"search_terms":[],"countries":["China"],"time_range":"now-7d","ip_direction":"source","aggregation_type":"none"}
"""


def _plan_with_llm(question: str, llm: Any, grounding_context: dict | None = None, max_retries: int = 3) -> dict | None:
    """Plan query with LLM. Retries with full prompt first, falls back to minimal prompt on repeated failures."""
    if not llm:
        return None
    
    full_prompt_text = _load_planning_prompt()
    
    grounding_section = ""
    if grounding_context:
        grounding_section = f"\n\nPRE-EXTRACTED ENTITIES (treat as primary search targets):\n{json.dumps(grounding_context, indent=2)}\n"
    
    full_prompt = f"""{full_prompt_text}{grounding_section}
QUESTION: {question}

Return STRICT JSON only (no explanation or code blocks).
"""
    
    minimal_prompt = f"""{_MINIMAL_PLANNING_PROMPT}
QUESTION: {question}

Return compact JSON only.
"""
    
    for attempt in range(max_retries):
        # Attempt 1: try full planning prompt to get the best plan possible.
        # Attempts 2+: fall back to minimal prompt which is more reliable under model load.
        use_minimal = (attempt > 0)
        prompt_to_use = minimal_prompt if use_minimal else full_prompt
        if use_minimal and attempt == 1:
            logger.warning("[%s] Full prompt failed, switching to minimal prompt for attempts %d+", SKILL_NAME, attempt + 1)
        
        # DEBUG: Log what prompt we're sending (first attempt at INFO level for visibility)
        if attempt == 0:
            logger.info("[%s] [DIAGNOSTIC] Sending FULL planning prompt for question: %s", SKILL_NAME, question)
        
        try:
            response = llm.complete(prompt_to_use, format="json").strip()
            # Log at INFO level so we can diagnose failures
            logger.info("[%s] [DIAGNOSTIC] LLM response (attempt %d): %s", SKILL_NAME, attempt + 1, _log_excerpt(response, 800))
            
            # 1. Try direct JSON parse
            parsed = None
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # 2. Try markdown code block
            if parsed is None:
                matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
                for match in matches:
                    try:
                        parsed = json.loads(match.strip())
                        break
                    except json.JSONDecodeError:
                        continue
            
            # 3. Try finding any embedded JSON object (handles model preamble/postamble)
            if parsed is None:
                parsed = _extract_json_object(response)
                if parsed:
                    logger.info("[%s] Extracted JSON from within LLM response text", SKILL_NAME)
            
            # Always log what we parsed (even if empty)
            if isinstance(parsed, dict):
                logger.info(
                    "[%s] [DIAGNOSTIC] Parsed result — countries=%s | time_range=%s | search_terms=%s | agg_type=%s",
                    SKILL_NAME,
                    parsed.get("countries", []),
                    parsed.get("time_range", "unset"),
                    parsed.get("search_terms", []),
                    parsed.get("aggregation_type", "unset"),
                )
            else:
                logger.warning("[%s] [DIAGNOSTIC] Failed to parse JSON. Got type: %s, value: %s", SKILL_NAME, type(parsed), parsed)
            
            # Validate: must be a non-empty dict
            if isinstance(parsed, dict) and parsed:
                # Semantic validation: fingerprint_ports is only valid when the question
                # explicitly contains an IP address to fingerprint, and the plan's
                # search_terms consist of IPs that actually appear in the question.
                # Reject plans where the LLM hallucinated IPs not mentioned in the question.
                agg_type = parsed.get("aggregation_type", "none")
                terms = [str(t).strip() for t in (parsed.get("search_terms") or []) if t]
                if agg_type == "fingerprint_ports":
                    explicit_ips_in_question = set(re.findall(
                        r'\b(?:\d{1,3}\.){3}\d{1,3}\b', question
                    ))
                    plan_ips = set(terms)
                    if not explicit_ips_in_question or not plan_ips.intersection(explicit_ips_in_question):
                        logger.warning(
                            "[%s] Plan invalid: aggregation_type=fingerprint_ports but "
                            "search_terms %s not found in question IPs %s. Retrying.",
                            SKILL_NAME, plan_ips, explicit_ips_in_question,
                        )
                        use_minimal = True
                        continue
                return parsed
            
            if attempt < max_retries - 1:
                logger.warning("[%s] Empty or invalid plan on attempt %d, retrying...", SKILL_NAME, attempt + 1)
            else:
                logger.warning("[%s] Failed to plan after %d attempts. Last response: %s",
                               SKILL_NAME, max_retries, _log_excerpt(response, 600))
        except Exception as exc:
            logger.error("[%s] LLM planning failed (attempt %d): %s", SKILL_NAME, attempt + 1, exc)
    
    return None


def _extract_aggregated_ports(
    results: list[dict],
    target_ips: list[str],
    dest_ip_field: str,
    dest_port_field: str,
    protocol_field: str | None = None
) -> dict[int, dict]:
    """
    Extract and aggregate destination ports from results for target IPs (as servers).
    
    IMPORTANT: This function is data-agnostic. It uses ONLY the field names provided,
    which should come from field discovery. NO hardcoded field name assumptions.
    
    Args:
        results: List of records from OpenSearch
        target_ips: List of target IPs to filter for
        dest_ip_field: Field name for destination IP (from discoverd field mappings)
        dest_port_field: Field name for destination port (from discovered field mappings)
        protocol_field: Optional field name for protocol (from discovered field mappings)
    """
    try:
        if not dest_ip_field:
            raise ValueError("dest_ip_field is required (must come from field discovery)")
        if not dest_port_field:
            raise ValueError("dest_port_field is required (must come from field discovery)")
        
        target_set = {str(ip).strip() for ip in (target_ips or []) if ip}
        aggregated = {}
        
        # Debug counters
        total_records = len(results or [])
        ip_match_count = 0
        port_extract_count = 0
        invalid_port_count = 0
        
        logger.debug(
            "[%s] _extract_aggregated_ports: processing %d total records, looking for IPs: %s, using fields: ip=%s, port=%s",
            SKILL_NAME, total_records, target_set, dest_ip_field, dest_port_field
        )
        
        for i, record in enumerate(results or []):
            try:
                if not isinstance(record, dict):
                    continue
                
                # Extract destination IP using discovered field name
                dest_ip = _get_nested_field(record, dest_ip_field)
                
                # Ensure dest_ip is a string, not a dict
                if isinstance(dest_ip, dict):
                    logger.warning("[%s] Record %d has dict dest_ip at field '%s': %s", 
                                   SKILL_NAME, i, dest_ip_field, dest_ip)
                    continue
                
                if dest_ip:
                    dest_ip = str(dest_ip).strip()
                else:
                    continue
                
                # Only include if this IP is a target
                if target_set and dest_ip not in target_set:
                    continue
                
                ip_match_count += 1
                
                # Extract destination port using discovered field name
                dest_port = _get_nested_field(record, dest_port_field)
                
                # Ensure dest_port is a valid integer
                if isinstance(dest_port, dict):
                    logger.warning("[%s] Record %d has dict dest_port at field '%s': %s", 
                                   SKILL_NAME, i, dest_port_field, dest_port)
                    continue
                
                if not dest_port:
                    continue
                
                port_extract_count += 1
                
                try:
                    port = int(dest_port)
                    if not (0 < port < 65536):
                        invalid_port_count += 1
                        continue
                except (TypeError, ValueError):
                    invalid_port_count += 1
                    continue
                
                # Aggregate
                if port not in aggregated:
                    aggregated[port] = {
                        "observations": 0,
                        "protocols": set(),
                    }
                
                aggregated[port]["observations"] += 1
                
                # Track protocol if field provided
                if protocol_field:
                    proto = _get_nested_field(record, protocol_field)
                    if proto:
                        aggregated[port]["protocols"].add(str(proto).lower())
                    
            except TypeError as te:
                if "unhashable" in str(te):
                    logger.error("[%s] Unhashable type in record %d: %s", SKILL_NAME, i, te, exc_info=True)
                    continue
                raise
        
        # Convert sets to lists for JSON
        for port_data in aggregated.values():
            if isinstance(port_data.get("protocols"), set):
                port_data["protocols"] = sorted(list(port_data["protocols"]))
        
        logger.info(
            "[%s] Aggregation complete: total_records=%d, ip_matches=%d, ports_extracted=%d, "
            "invalid_ports=%d, unique_ports_aggregated=%d",
            SKILL_NAME, total_records, ip_match_count, port_extract_count, invalid_port_count, len(aggregated)
        )
        
        # Log port distribution for debugging
        if aggregated:
            port_list = sorted(aggregated.keys())
            low_ports = [p for p in port_list if p < 1024]
            registered_ports = [p for p in port_list if 1024 <= p < 32768]
            ephemeral_ports = [p for p in port_list if p >= 32768]
            logger.info(
                "[%s] Port distribution: well-known=%d, registered=%d, ephemeral=%d",
                SKILL_NAME, len(low_ports), len(registered_ports), len(ephemeral_ports)
            )
            # Show top 5 ports by obs
            top_ports = sorted(aggregated.items(), key=lambda x: -x[1].get("observations", 0))[:5]
            for port, data in top_ports:
                logger.info("[%s]   Port %d: %d observations, protocols=%s", 
                           SKILL_NAME, port, data.get("observations", 0), data.get("protocols", []))
        
        return aggregated
    
    except Exception as e:
        logger.error("[%s] EXCEPTION in _extract_aggregated_ports: %s", SKILL_NAME, e, exc_info=True)
        raise


def _get_nested_field(record: dict, field_path: str) -> Any:
    """
    Extract field value from record using dot-notation path.
    Examples: "destination.ip", "network.transport"
    """
    if not field_path or not isinstance(record, dict):
        return None
    
    parts = field_path.strip().split(".")
    value = record
    
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return None
        else:
            return None
    
    return value


def _select_ip_fields(field_mappings: dict | None) -> tuple[str, str, str | None, str | None]:
    """
    Select field names for destination IP, destination port, protocol, and source IP from discovered mappings.
    
    IMPORTANT: This is DATA-AGNOSTIC. It uses the field lists discovered by fields_querier
    and selects the FIRST/PRIMARY field from each category.
    It does NOT have fallback hardcoded field names.
    
    Returns: (dest_ip_field, dest_port_field, protocol_field or None, src_ip_field or None)
    Raises: ValueError if required fields not found in mappings
    """
    if not isinstance(field_mappings, dict):
        raise ValueError("Field mappings required for data-agnostic operation (must run fields_querier first)")
    
    # Get field lists discovered by fields_querier
    # These are lists because a schema might have multiple candidates
    dest_ip_fields = field_mappings.get("destination_ip_fields", [])
    dest_port_fields = field_mappings.get("destination_port_fields", [])
    protocol_fields = field_mappings.get("protocol_fields", [])
    src_ip_fields = field_mappings.get("source_ip_fields", [])
    
    # Select FIRST field from each list (primary/most common)
    if not dest_ip_fields or not isinstance(dest_ip_fields, list) or len(dest_ip_fields) == 0:
        raise ValueError(
            "destination_ip_fields not found or empty in field mappings. "
            "fields_querier must discover this first."
        )
    
    if not dest_port_fields or not isinstance(dest_port_fields, list) or len(dest_port_fields) == 0:
        raise ValueError(
            "destination_port_fields not found or empty in field mappings. "
            "fields_querier must discover this first."
        )
    
    # Validate they're strings, not dicts
    dest_ip_field = str(dest_ip_fields[0]).strip() if dest_ip_fields else None
    dest_port_field = str(dest_port_fields[0]).strip() if dest_port_fields else None
    protocol_field = str(protocol_fields[0]).strip() if protocol_fields else None
    src_ip_field = str(src_ip_fields[0]).strip() if src_ip_fields else None
    
    if isinstance(dest_ip_field, dict) or isinstance(dest_port_field, dict):
        raise ValueError("Field mappings contain invalid types (dict instead of field name)")
    
    if not dest_ip_field:
        raise ValueError("destination_ip_field is empty after extraction")
    if not dest_port_field:
        raise ValueError("destination_port_field is empty after extraction")
    
    return dest_ip_field, dest_port_field, protocol_field if protocol_field else None, src_ip_field if src_ip_field else None


def _build_opensearch_query(
    search_terms: list[str],
    dest_ip_field: str,
    time_range: str,
    size: int,
    src_ip_field: str | None = None,
    ip_direction: str = "any",
    countries: list[str] | None = None,
) -> dict:
    """
    Build OpenSearch query for IP/country search.
    
    IMPORTANT: This is DATA-AGNOSTIC. It uses ONLY the field names provided (from discovery),
    not hardcoded field names.
    
    ip_direction controls which IP fields to search:
    - "source":      filter on src_ip_field
    - "destination": filter on dest_ip_field
    - "any":         filter on both src_ip_field and dest_ip_field
    
    countries: if provided, adds GeoIP country filters using standardised processor output fields.
    GeoIP fields (geoip.country_name, source.geo.country_name, destination.geo.country_name) are
    standardised Logstash/Elastic processor output fields, not schema-specific field names.
    """
    try:
        if not dest_ip_field:
            raise ValueError("dest_ip_field is required (must come from field discovery)")
        
        # Sanitise: ensure all search terms are plain strings
        search_terms_clean = [
            str(t).strip() for t in (search_terms or [])
            if t and not isinstance(t, dict) and str(t).strip()
        ]
        
        dest_ip_field_str = str(dest_ip_field).strip()
        if not dest_ip_field_str:
            raise ValueError("dest_ip_field is empty after stripping")
        
        src_ip_field_str = str(src_ip_field).strip() if src_ip_field else None
        size_int = int(size) if size else 200
        time_range_str = str(time_range).strip() if time_range else "now-90d"
        direction = (ip_direction or "any").lower()
        
        # ── Time filter (always required) ──
        must_clauses: list[dict] = [
            {"range": {"@timestamp": {"gte": time_range_str}}}
        ]
        
        # ── IP term filter ──
        if search_terms_clean:
            ip_should: list[dict] = []
            
            if direction == "source" and src_ip_field_str:
                ip_should = [{"term": {src_ip_field_str: t}} for t in search_terms_clean]
            elif direction == "destination":
                ip_should = [{"term": {dest_ip_field_str: t}} for t in search_terms_clean]
            else:
                # "any" — match on either field
                ip_should = [{"term": {dest_ip_field_str: t}} for t in search_terms_clean]
                if src_ip_field_str and src_ip_field_str != dest_ip_field_str:
                    ip_should += [{"term": {src_ip_field_str: t}} for t in search_terms_clean]
            
            if ip_should:
                must_clauses.append({"bool": {"should": ip_should, "minimum_should_match": 1}})
            else:
                logger.warning("[%s] No IP clauses built despite %d search terms", SKILL_NAME, len(search_terms_clean))
        else:
            logger.warning("[%s] No search terms for query", SKILL_NAME)
        
        # ── Country filter using standardised GeoIP processor output fields ──
        # These fields (geoip.country_name, source.geo.country_name, etc.) are standardised
        # Logstash/Elastic processor output fields, not schema-specific. Not heuristics.
        if countries:
            # Select geo fields based on ip_direction
            if direction == "source":
                geo_fields = ["geoip.country_name", "source.geo.country_name"]
            elif direction == "destination":
                geo_fields = ["destination.geo.country_name"]
            else:
                geo_fields = ["geoip.country_name", "source.geo.country_name", "destination.geo.country_name"]
            
            country_should: list[dict] = []
            for country in countries:
                country_str = str(country).strip()
                if country_str:
                    for gf in geo_fields:
                        country_should.append({"match": {gf: {"query": country_str, "operator": "and"}}})
            
            if country_should:
                must_clauses.append({"bool": {"should": country_should, "minimum_should_match": 1}})
                logger.info("[%s] Added country filter for %s using %s", SKILL_NAME, countries, geo_fields)
        
        return {
            "size": size_int,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": must_clauses
                }
            }
        }
    except Exception as e:
        logger.error("[%s] EXCEPTION in _build_opensearch_query: %s", SKILL_NAME, e, exc_info=True)
        raise


def _build_aggregation_query(
    search_terms: list[str],
    ip_field: str,
    agg_field: str,
    time_range: str,
    agg_size: int = 1000
) -> dict:
    """
    Build OpenSearch aggregation query for port/field aggregation.
    
    Used when LLM instructs aggregation (e.g., "fingerprint_ports" aggregation type).
    This allows opensearch_querier to follow LLM's directive to aggregate,
    not just fetch raw records.
    
    Args:
        search_terms: Target IPs or search terms
        ip_field: Field to filter on (from discovery)
        agg_field: Field to aggregate on (typically dest_port)
        time_range: Time range string
        agg_size: Max number of aggregation buckets to return
    """
    try:
        if not ip_field:
            raise ValueError("ip_field required for aggregation query")
        if not agg_field:
            raise ValueError("agg_field required for aggregation")
        
        ip_field_str = str(ip_field).strip()
        agg_field_str = str(agg_field).strip()
        time_range_str = str(time_range).strip() if time_range else "now-90d"
        
        # Time filter
        must_clauses = [
            {"range": {"@timestamp": {"gte": time_range_str}}}
        ]
        
        # IP filter
        search_terms_clean = []
        for t in (search_terms or []):
            if t and not isinstance(t, dict):
                term_str = str(t).strip()
                if term_str and not isinstance(term_str, dict):
                    search_terms_clean.append(term_str)
        
        should_clauses = []
        if search_terms_clean:
            for term in search_terms_clean:
                term_value = str(term).strip()
                if term_value and not isinstance(term_value, dict):
                    should_clauses.append({"term": {ip_field_str: term_value}})
        
        if should_clauses:
            must_clauses.append({"bool": {"should": should_clauses, "minimum_should_match": 1}})
        
        # Build aggregation query
        return {
            "size": 0,  # Don't fetch hits, just aggregations
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "aggs": {
                "values": {
                    "terms": {
                        "field": agg_field_str,
                        "size": int(agg_size)
                    }
                }
            }
        }
    except Exception as e:
        logger.error("[%s] EXCEPTION in _build_aggregation_query: %s", SKILL_NAME, e, exc_info=True)
        raise


def run(context: dict) -> dict:
    """
    Execute OpenSearch query. Return results and aggregated ports for fingerprinting.
    
    IMPORTANT: This skill is DATA-AGNOSTIC and REQUIRES field discovery prerequisite.
    - Expects fields_querier results in context["previous_results"]["fields_querier"]
    - Uses ONLY discovered field names, NEVER hardcoded assumptions
    """
    try:
        db = context.get("db")
        llm = context.get("llm")
        cfg = context.get("config") or {}
        parameters = context.get("parameters", {})
        previous_results = context.get("previous_results", {})
        
        # Validate inputs
        if not db:
            logger.warning("[%s] No database", SKILL_NAME)
            return {"status": "skipped", "reason": "no_db"}
        
        question = parameters.get("question") or ""
        if not question:
            logger.warning("[%s] No question", SKILL_NAME)
            return {"status": "skipped", "reason": "no_question"}
        
        # Get index name safely (handle ConfigParser vs dict)
        index = parameters.get("index")
        if not index:
            try:
                # Try to get from config if it's a dict
                if isinstance(cfg, dict):
                    index = cfg.get("db", {}).get("logs_index")
                else:
                    # Try ConfigParser-style access with proper fallback handling
                    try:
                        index = cfg.get("database", "logs_index")
                        if not index:
                            index = cfg.get("db", "logs_index")
                    except (TypeError, AttributeError):
                        pass
            except Exception as e:
                logger.debug("[%s] Could not get index from config: %s", SKILL_NAME, e)
            
            # Final fallback
            if not index:
                index = "logstash*"
        
        logger.debug("[%s] Using index: %s", SKILL_NAME, index)
        
        # CRITICALLY IMPORTANT: Get field mappings from fields_querier prerequisite
        if previous_results.get("fields_querier", {}).get("status") != "ok":
            logger.error(
                "[%s] PREREQUISITE VIOLATION: fields_querier must run first. "
                "Got previous_results: %s",
                SKILL_NAME, list(previous_results.keys())
            )
            return {
                "status": "error",
                "error": "fields_querier must run first to discover actual field names (data-agnostic requirement)",
                "results": [],
                "results_count": 0,
            }
        
        field_mappings = previous_results.get("fields_querier", {}).get("field_mappings")
        if not field_mappings:
            logger.error("[%s] fields_querier did not return field_mappings", SKILL_NAME)
            return {
                "status": "error",
                "error": "fields_querier returned no field_mappings",
                "results": [],
                "results_count": 0,
            }
        
        logger.info("[%s] Using discovered field mappings: %s", SKILL_NAME, list(field_mappings.keys())[:5])
        
        # ── Plan query ──
        # Always use LLM to plan — pass any supervisor grounding context so LLM has
        # pre-extracted entities (IPs) available when forming the query plan.
        # This ensures ALL plan fields (countries, ip_direction, aggregation_type) are determined
        # by LLM reasoning against PLANNING_PROMPT.md, not by heuristics.
        question_grounding = context.get("routing_decision", {}).get("question_grounding") or {}
        
        query_plan = _plan_with_llm(question, llm, grounding_context=question_grounding or None)
        
        if not query_plan:
            logger.warning("[%s] Could not plan query", SKILL_NAME)
            return {"status": "skipped", "reason": "no_plan"}
        
        # Extract ALL planning fields from LLM response
        search_terms = [str(t).strip() for t in query_plan.get("search_terms", []) if t and not isinstance(t, dict)]
        time_range = query_plan.get("time_range") or "now-90d"
        aggregation_type = query_plan.get("aggregation_type", "none")
        countries = [str(c).strip() for c in query_plan.get("countries", []) if c and not isinstance(c, dict)]
        ip_direction = query_plan.get("ip_direction") or "any"
        size = int(parameters.get("size", 200))
        
        # If supervisor grounding contains IPs and LLM didn't extract them (e.g. for simple
        # fingerprint questions with no natural-language IP mention), merge them in.
        if question_grounding:
            raw_ips = question_grounding.get("ips", [])
            grounding_ips: list[str] = []
            if isinstance(raw_ips, list):
                for ip_value in raw_ips:
                    if not isinstance(ip_value, dict) and ip_value:
                        ip_str = str(ip_value).strip()
                        if ip_str:
                            grounding_ips.append(ip_str)
            elif raw_ips:
                grounding_ips = [str(raw_ips).strip()]
            
            if grounding_ips:
                # Union: grounding IPs take precedence; add any extra LLM-extracted terms
                merged = grounding_ips + [t for t in search_terms if t not in grounding_ips]
                search_terms = merged
                logger.info("[%s] Merged grounding IPs into search_terms: %s", SKILL_NAME, search_terms[:5])
        
        logger.info(
            "[%s] Query plan — terms: %s | time: %s | agg_type: %s | countries: %s | ip_direction: %s",
            SKILL_NAME, search_terms, time_range, aggregation_type, countries, ip_direction,
        )
        
        # Get field names from DISCOVERED mappings (data-agnostic)
        try:
            dest_ip_field, dest_port_field, protocol_field, src_ip_field = _select_ip_fields(field_mappings)
            logger.info("[%s] Using discovered fields: dest_ip=%s, src_ip=%s, dest_port=%s, protocol=%s",
                       SKILL_NAME, dest_ip_field, src_ip_field, dest_port_field, protocol_field)
        except ValueError as field_err:
            logger.error("[%s] Field selection failed (data-agnostic requirement): %s", SKILL_NAME, field_err)
            return {
                "status": "error",
                "error": f"Field selection: {str(field_err)}",
                "results": [],
                "results_count": 0,
            }
        
        # ─── RESPOND TO LLM'S AGGREGATION DIRECTIVE ───────────────────────────────
        # If LLM instructed "fingerprint_ports" aggregation, build aggregation query
        if aggregation_type == "fingerprint_ports":
            logger.info("[%s] LLM requested fingerprint_ports aggregation — building aggregation query on %s",
                       SKILL_NAME, dest_port_field)
            try:
                opensearch_query = _build_aggregation_query(
                    search_terms, dest_ip_field, dest_port_field, time_range, agg_size=1000
                )
                logger.info("[%s] Built aggregation query for port field: %s", 
                           SKILL_NAME, dest_port_field)
                logger.debug("[%s] Aggregation query JSON: %s", SKILL_NAME, _log_excerpt(json.dumps(opensearch_query)))
            except Exception as exc:
                logger.error("[%s] Aggregation query building failed: %s", SKILL_NAME, exc, exc_info=True)
                return {
                    "status": "error",
                    "error": f"Aggregation query building: {str(exc)}",
                    "results": [],
                    "results_count": 0,
                }
            
            # Execute aggregation query
            try:
                logger.info("[%s] Executing aggregation on index=%s | field=%s | search_terms=%s", 
                           SKILL_NAME, index, dest_port_field, search_terms)
                agg_response = db.aggregate(index, opensearch_query) or {}
                logger.debug("[%s] Aggregation response keys: %s", SKILL_NAME, list(agg_response.keys()))
                logger.debug("[%s] Aggregation response: %s", SKILL_NAME, _log_excerpt(json.dumps(agg_response, default=str)))
            except Exception as exc:
                logger.error("[%s] Aggregation execution failed: %s", SKILL_NAME, exc, exc_info=True)
                return {
                    "status": "error",
                    "error": f"Aggregation execution: {str(exc)}",
                    "results": [],
                    "results_count": 0,
                }
            
            # Extract ports from aggregation buckets
            try:
                aggregated_ports = {}
                aggs = agg_response.get("aggregations", {}).get("values", {}).get("buckets", [])
                
                logger.info("[%s] Aggregation returned %d port buckets", SKILL_NAME, len(aggs))
                
                for bucket in aggs:
                    port = int(bucket.get("key", 0))
                    count = int(bucket.get("doc_count", 0))
                    
                    if port and count:
                        aggregated_ports[port] = {
                            "observations": count,
                            "protocols": []  # Protocol info requires additional query
                        }
                
                logger.info("[%s] Extracted %d unique ports from aggregation buckets", 
                           SKILL_NAME, len(aggregated_ports))
                
                # Log top ports
                top_ports = sorted(aggregated_ports.items(), key=lambda x: -x[1]["observations"])[:5]
                for port, data in top_ports:
                    logger.info("[%s]   Port %d: %d observations", SKILL_NAME, port, data["observations"])
                
                return {
                    "status": "ok",
                    "results": [],  # No raw records for aggregation
                    "results_count": len(aggs),  # Total buckets
                    "search_terms": search_terms,
                    "aggregated_ports": aggregated_ports,
                    "aggregation_type": "fingerprint_ports",
                    "time_range": time_range,
                }
            except Exception as exc:
                logger.error("[%s] Failed to extract ports from aggregation: %s", SKILL_NAME, exc, exc_info=True)
                return {
                    "status": "error",
                    "error": f"Aggregation extraction: {str(exc)}",
                    "results": [],
                    "results_count": 0,
                }
        
        # ─── DEFAULT: Raw query (when LLM doesn't request aggregation) ──────────
        # Build & execute query
        try:
            opensearch_query = _build_opensearch_query(
                search_terms, dest_ip_field, time_range, size,
                src_ip_field=src_ip_field,
                ip_direction=ip_direction,
                countries=countries if countries else None,
            )
            logger.info(
                "[%s] Built raw query | dest_ip=%s | src_ip=%s | direction=%s | countries=%s",
                SKILL_NAME, dest_ip_field, src_ip_field, ip_direction, countries,
            )
        except Exception as exc:
            logger.error("[%s] Query building failed: %s", SKILL_NAME, exc, exc_info=True)
            return {
                "status": "error",
                "error": f"Query building: {str(exc)}",
                "results": [],
                "results_count": 0,
            }
        
        try:
            logger.debug("[%s] Query object: %s", SKILL_NAME, _log_excerpt(json.dumps(opensearch_query)))
        except Exception as exc:
            logger.warning("[%s] Could not JSON serialize query: %s", SKILL_NAME, exc)
        
        try:
            logger.info("[%s] Executing search on index=%s with dest_port_field=%s | size=%d", 
                       SKILL_NAME, index, dest_port_field, size)
            results = db.search(index, opensearch_query, size=size) or []
            logger.info("[%s] Query returned %d results | Will extract port field: %s", 
                       SKILL_NAME, len(results), dest_port_field)
        except Exception as exc:
            logger.error("[%s] Query execution failed: %s", SKILL_NAME, exc, exc_info=True)
            return {
                "status": "error",
                "error": f"Query execution: {str(exc)}",
                "results": [],
                "results_count": 0,
            }
        
        # Aggregate ports for fingerprinting using discovered field names
        try:
            aggregated_ports = _extract_aggregated_ports(
                results, search_terms, dest_ip_field, dest_port_field, protocol_field
            )
            logger.info("[%s] Aggregated %d unique ports", SKILL_NAME, len(aggregated_ports))
        except Exception as exc:
            logger.error("[%s] Port aggregation failed: %s", SKILL_NAME, exc, exc_info=True)
            return {
                "status": "error",
                "error": f"Port aggregation: {str(exc)}",
                "results": results,
                "results_count": len(results),
            }
        
        return {
            "status": "ok",
            "results": results,
            "results_count": len(results),
            "search_terms": search_terms,
            "aggregated_ports": aggregated_ports,
            "time_range": time_range,
            "countries": countries,
        }
    
    except Exception as e:
        logger.error("[%s] UNHANDLED EXCEPTION in run(): %s", SKILL_NAME, e, exc_info=True)
        return {
            "status": "error",
            "error": f"Unhandled: {str(e)}",
            "results": [],
            "results_count": 0,
        }
