"""
skills/baseline_querier/logic.py

Behavioral-baseline RAG querier.

Searches the OpenSearch vector index for network-behavior baseline documents
(stored by network_baseliner) and searches raw logs to answer questions about
observed traffic, IP patterns, protocols, ports, etc.

Use this skill for questions like:
  "what's normal traffic for this network?"
  "show me traffic from Iran in February"
  "any flows to 8.8.8.8?"
  "top 10 alerts from last week"

For field-schema questions ("what field holds bytes?"), use fields_querier.

Context keys consumed:
    context["db"]         -> BaseDBConnector
    context["llm"]        -> BaseLLMProvider
    context["memory"]     -> Memory instance (StateBackedMemory or CheckpointBackedMemory)
    context["config"]     -> Config
    context["parameters"] -> {"question": str, "conversation_history": list}
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "baseline_querier"
MAX_MULTI_MATCH_FIELDS = 12
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _extract_json_from_response(response: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    matches = re.findall(r'\{[\s\S]*\}', response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def _get_field_value(row: dict, *candidates: str) -> Any:
    for candidate in candidates:
        current: Any = row
        found = True
        for part in candidate.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found and current not in {None, ""}:
            return current
    return None


def _unique_fields(values: list[Any] | None) -> list[str]:
    ordered: list[str] = []
    for value in values or []:
        field = str(value or "").strip()
        if field and field not in ordered:
            ordered.append(field)
    return ordered


def _select_fields_by_tokens(
    fields: list[str],
    *,
    include_any: tuple[str, ...] = (),
    exclude_any: tuple[str, ...] = (),
) -> list[str]:
    selected: list[str] = []
    for field in _unique_fields(fields):
        lowered = field.lower()
        if include_any and not any(token in lowered for token in include_any):
            continue
        if exclude_any and any(token in lowered for token in exclude_any):
            continue
        selected.append(field)
    return selected


def _build_observation_field_candidates(field_mappings: dict | None) -> dict[str, list[str]]:
    mappings = field_mappings or {}
    all_fields = _unique_fields(mappings.get("all_fields"))
    ip_fields = _unique_fields(mappings.get("ip_fields"))
    port_fields = _unique_fields(mappings.get("port_fields"))
    timestamp_fields = _unique_fields(mappings.get("timestamp_fields"))

    source_ip_fields = _select_fields_by_tokens(
        ip_fields or all_fields,
        include_any=("source", "src", "client"),
        exclude_any=("port",),
    )
    destination_ip_fields = _select_fields_by_tokens(
        ip_fields or all_fields,
        include_any=("destination", "dest", "dst", "server"),
        exclude_any=("port",),
    )
    generic_ip_fields = [
        field for field in ip_fields
        if field not in source_ip_fields and field not in destination_ip_fields
    ]

    destination_port_fields = _select_fields_by_tokens(
        port_fields or all_fields,
        include_any=("destination", "dest", "dst"),
    )
    source_port_fields = _select_fields_by_tokens(
        port_fields or all_fields,
        include_any=("source", "src"),
    )
    generic_port_fields = [
        field for field in port_fields
        if field not in destination_port_fields and field not in source_port_fields
    ]

    if not timestamp_fields:
        timestamp_fields = _select_fields_by_tokens(
            all_fields,
            include_any=("timestamp", "time", "date", "created", "start"),
        )

    protocol_fields = _select_fields_by_tokens(
        all_fields,
        include_any=("protocol", "proto", "transport", "service"),
    )

    return {
        "source_ip": source_ip_fields + generic_ip_fields,
        "destination_ip": destination_ip_fields + [
            field for field in generic_ip_fields if field not in destination_ip_fields
        ],
        "timestamp": timestamp_fields,
        "destination_port": destination_port_fields + generic_port_fields,
        "source_port": source_port_fields + [
            field for field in generic_port_fields if field not in source_port_fields
        ],
        "protocol": protocol_fields,
    }


def _extract_focus_ips(question: str, search_terms_used: list[str]) -> list[str]:
    ips = list(dict.fromkeys(IP_PATTERN.findall(question or "")))
    if ips:
        return ips
    return [term for term in search_terms_used if IP_PATTERN.fullmatch(str(term or ""))]


def _service_labels_for_ports(ports: list[str], protocols: list[str]) -> list[str]:
    labels = [str(protocol).upper() for protocol in protocols if protocol]
    port_set = {str(port) for port in ports if port is not None}
    if "53" in port_set and "DNS" not in labels:
        labels.append("DNS")
    if "443" in port_set and "HTTPS" not in labels and "TLS" not in labels:
        labels.append("HTTPS")
    if "80" in port_set and "HTTP" not in labels:
        labels.append("HTTP")
    return labels


def _build_focus_observations(
    question: str,
    raw_logs: list[dict],
    search_terms_used: list[str],
    rag_sources: int,
    field_mappings: dict | None = None,
) -> dict:
    focus_ips = _extract_focus_ips(question, search_terms_used)
    candidates = _build_observation_field_candidates(field_mappings)
    observations: dict[str, Any] = {
        "focus_ips": focus_ips,
        "rag_sources": int(rag_sources or 0),
        "entities": {},
    }

    if not focus_ips:
        return observations

    for focus_ip in focus_ips:
        observations["entities"][focus_ip] = {
            "total_records": 0,
            "source_records": 0,
            "destination_records": 0,
            "peer_ips": set(),
            "ports": set(),
            "protocols": set(),
            "timestamps": set(),
        }

    for row in raw_logs:
        src_ip = _get_field_value(row, *candidates["source_ip"])
        dest_ip = _get_field_value(row, *candidates["destination_ip"])
        timestamp = _get_field_value(row, *candidates["timestamp"])
        candidate_ports = [
            _get_field_value(row, *candidates["destination_port"]),
            _get_field_value(row, *candidates["source_port"]),
        ]
        candidate_protocols = [
            _get_field_value(row, *candidates["protocol"]),
        ]

        for focus_ip in focus_ips:
            stats = observations["entities"][focus_ip]
            matched = False
            if str(src_ip) == focus_ip:
                stats["source_records"] += 1
                matched = True
                if dest_ip and str(dest_ip) != focus_ip:
                    stats["peer_ips"].add(str(dest_ip))
            if str(dest_ip) == focus_ip:
                stats["destination_records"] += 1
                matched = True
                if src_ip and str(src_ip) != focus_ip:
                    stats["peer_ips"].add(str(src_ip))
            if not matched:
                continue

            stats["total_records"] += 1
            if timestamp:
                stats["timestamps"].add(str(timestamp))
            for port in candidate_ports:
                if port is not None and str(port):
                    stats["ports"].add(str(port))
            for protocol in candidate_protocols:
                if protocol:
                    stats["protocols"].add(str(protocol).upper())

    for focus_ip in focus_ips:
        stats = observations["entities"][focus_ip]
        timestamps = sorted(stats["timestamps"])
        ports = sorted(stats["ports"], key=lambda value: int(value) if str(value).isdigit() else str(value))
        protocols = sorted(stats["protocols"])
        observations["entities"][focus_ip] = {
            "total_records": stats["total_records"],
            "source_records": stats["source_records"],
            "destination_records": stats["destination_records"],
            "peer_ips": sorted(stats["peer_ips"])[:10],
            "ports": ports[:10],
            "protocols": protocols[:10],
            "services": _service_labels_for_ports(ports[:10], protocols[:10])[:10],
            "earliest": timestamps[0] if timestamps else None,
            "latest": timestamps[-1] if timestamps else None,
        }

    return observations


def _build_grounded_baseline_assessment(
    question: str,
    raw_logs: list[dict],
    search_terms_used: list[str],
    rag_sources: int,
    field_mappings: dict | None = None,
) -> tuple[str, dict]:
    observations = _build_focus_observations(
        question,
        raw_logs,
        search_terms_used,
        rag_sources,
        field_mappings=field_mappings,
    )
    focus_ips = observations.get("focus_ips") or []
    if not focus_ips:
        return "", observations

    focus_ip = focus_ips[0]
    entity = (observations.get("entities") or {}).get(focus_ip) or {}
    total_records = int(entity.get("total_records", 0) or 0)
    source_records = int(entity.get("source_records", 0) or 0)
    destination_records = int(entity.get("destination_records", 0) or 0)
    services = entity.get("services") or []
    peer_ips = entity.get("peer_ips") or []
    ports = entity.get("ports") or []
    earliest = entity.get("earliest")
    latest = entity.get("latest")

    if total_records == 0:
        return (
            f"I found no sampled log records involving {focus_ip}, so I cannot determine from evidence whether it is normal behavior in this network.",
            observations,
        )

    if destination_records > 0 and source_records == 0 and total_records >= 2 and ("DNS" in services or rag_sources > 0):
        verdict = f"{focus_ip} appears to be routine destination-side DNS traffic in this network."
    elif destination_records > 0 and source_records == 0 and total_records >= 2:
        verdict = f"{focus_ip} appears to be recurring destination-side traffic in this network."
    elif source_records > 0 and destination_records == 0 and total_records >= 2 and rag_sources > 0:
        verdict = f"{focus_ip} appears to be recurring source-side traffic in this network."
    elif total_records == 1:
        verdict = f"{focus_ip} appears in the sampled data, but only once, so there is not enough evidence to call it normal behavior."
    else:
        verdict = f"{focus_ip} appears in observed traffic, but the pattern is mixed, so this is only limited evidence of baseline behavior."

    details = [
        f"It matched {total_records} log record(s) in the sampled baseline search ({source_records} as source, {destination_records} as destination)."
    ]
    if peer_ips:
        details.append(f"Peers seen: {', '.join(peer_ips[:5])}.")
    if services:
        details.append(f"Services/protocols: {', '.join(services[:5])}.")
    elif ports:
        details.append(f"Ports seen: {', '.join(ports[:5])}.")
    if earliest and latest:
        details.append(f"Earliest: {earliest}. Latest: {latest}.")
    if rag_sources:
        details.append(f"Baseline documents consulted: {rag_sources}.")

    return " ".join([verdict] + details), observations


def _heuristic_time_range(question: str, conversation_history: list[dict] | None = None) -> str:
    text = " ".join(
        [str(question or "")] + [str(m.get("content", "") or "") for m in (conversation_history or [])[-4:]]
    ).lower()
    if re.search(r"\btoday\b|\blast 24 hours\b|\bpast 24 hours\b", text):
        return "now-24h"
    if re.search(r"\blast week\b|\bpast week\b", text):
        return "now-7d"
    if re.search(r"\blast 30 days\b|\bpast 30 days\b|\blast month\b", text):
        return "now-30d"
    if re.search(r"\blast 90 days\b|\bpast 90 days\b|\bpast 3 months\b", text):
        return "now-90d"
    if "normal behavior" in text or "baseline" in text or "how often" in text:
        return "now-30d"
    return "now-90d"


def _heuristic_query_plan(question: str, conversation_history: list[dict] | None = None) -> dict:
    text = " ".join(
        [str(question or "")] + [str(m.get("content", "") or "") for m in (conversation_history or [])[-4:]]
    )
    lowered = text.lower()
    ip_matches = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)))
    port_matches = list(dict.fromkeys(re.findall(r"\bport\s+(\d{1,5})\b", lowered)))
    protocol_matches = [proto.upper() for proto in ["dns", "http", "https", "tls", "udp", "tcp"] if proto in lowered]
    country_matches = [
        country.title()
        for country in ["iran", "russia", "china", "usa", "united states", "germany", "france"]
        if country in lowered
    ]

    return {
        "reasoning": "Heuristic fallback plan derived from explicit entities in the question and recent conversation.",
        "detected_time_range": "heuristic",
        "time_range": _heuristic_time_range(question, conversation_history),
        "ports": port_matches,
        "countries": list(dict.fromkeys(country_matches)),
        "protocols": list(dict.fromkeys(protocol_matches)),
        "search_terms": ip_matches,
        "skip_search": not any([ip_matches, port_matches, country_matches, protocol_matches]),
    }


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    memory = context.get("memory")
    cfg = context.get("config")
    parameters = context.get("parameters", {})

    if db is None or llm is None:
        logger.warning("[%s] db or llm not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db/llm"}

    user_question = parameters.get("question")
    if not user_question:
        logger.warning("[%s] No question provided in parameters.", SKILL_NAME)
        return {"status": "no_question"}

    conversation_history = parameters.get("conversation_history", [])

    try:
        instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    except Exception:
        instruction = "You are a network baseline analyst. Answer questions about observed traffic patterns."

    logs_index   = cfg.get("db", "logs_index",   default="securityclaw-logs")
    vector_index = cfg.get("db", "vector_index",  default="securityclaw-vectors")

    # ── 1. Search behavioral RAG for relevant baselines ───────────────────────
    # (behavior docs: network_behavior_baseline, protocol_port_baseline,
    #  ip_communication_baseline, dns_baseline — NOT field_documentation)
    logger.info("[%s] Searching for: %s", SKILL_NAME, user_question)

    rag_docs = []
    try:
        from core.rag_engine import RAGEngine
        rag = RAGEngine(db=db, llm=llm)
        # Retrieve baseline docs; filter out field/schema documents (those live in fields_rag.json)
        all_docs = rag.retrieve(user_question, k=7)
        rag_docs = [
            doc for doc in all_docs
            if doc.get("category") not in ("field_documentation", "schema_observation")
        ]
        logger.info("[%s] Found %d behavioral baseline docs in RAG.", SKILL_NAME, len(rag_docs))
    except Exception as exc:
        logger.warning("[%s] RAG retrieval failed: %s", SKILL_NAME, exc)

    # ── 2. Search raw logs for matching records ────────────────────────────────
    raw_logs = []
    search_terms_used = []
    field_mappings = {}
    try:
        raw_logs, search_terms_used, field_mappings = _search_raw_logs(
            user_question, db, logs_index, llm, conversation_history
        )
        logger.info(
            "[%s] Found %d matching records in logs (terms: %s).",
            SKILL_NAME, len(raw_logs), search_terms_used,
        )
    except Exception as exc:
        logger.error("[%s] Raw log search failed: %s", SKILL_NAME, exc)

    # ── 3. No data shortcut ────────────────────────────────────────────────────
    if not rag_docs and not raw_logs:
        return {
            "status": "no_data",
            "findings": {
                "question": user_question,
                "answer": "No baseline data or log records found to answer this question.",
                "confidence": 0.0,
            },
        }

    # ── 4. Synthesise answer ───────────────────────────────────────────────────
    combined_context = _format_combined_context(
        rag_docs, raw_logs, user_question, search_terms_used
    )
    answer = _extract_answer_from_data(user_question, combined_context, instruction, llm)
    evidence = _extract_evidence_details(raw_logs)

    if memory and raw_logs:
        _persist_evidence_to_memory(memory, user_question, evidence)

    grounded_assessment, observations = _build_grounded_baseline_assessment(
        user_question,
        raw_logs,
        search_terms_used,
        len(rag_docs),
        field_mappings=field_mappings,
    )

    findings = {
        "question": user_question,
        "answer": answer,
        "grounded_assessment": grounded_assessment,
        "observations": observations,
        "rag_sources": len(rag_docs),
        "log_records": len(raw_logs),
        "evidence": evidence,
        "confidence": 0.85 if (rag_docs or raw_logs) else 0.0,
        "summary": {
            "baseline_insights": len(rag_docs),
            "raw_observations": len(raw_logs),
        },
    }

    logger.info(
        "[%s] Answer compiled from %d baselines + %d log records.",
        SKILL_NAME, len(rag_docs), len(raw_logs),
    )

    return {"status": "ok", "findings": findings}


# ──────────────────────────────────────────────────────────────────────────────
# Raw log search
# ──────────────────────────────────────────────────────────────────────────────

def _search_raw_logs(
    question: str,
    db: Any,
    logs_index: str,
    llm: Any = None,
    conversation_history: list[dict] = None,
) -> tuple[list[dict], list[str]]:
    from core.query_builder import discover_field_mappings, build_keyword_query

    if llm is None:
        return [], [], {}

    field_mappings = discover_field_mappings(db, llm)
    query_plan = _plan_query_with_llm(question, conversation_history, field_mappings, llm)

    if not query_plan or query_plan.get("skip_search"):
        return [], [], field_mappings

    search_terms = query_plan.get("search_terms", [])
    ports        = query_plan.get("ports", [])
    countries    = query_plan.get("countries", [])
    protocols    = query_plan.get("protocols", [])
    time_range   = query_plan.get("time_range", "now-90d")

    # Sanitize time units
    time_range = re.sub(
        r'(\d)([YMWDdy])(?![a-z])',
        lambda m: m.group(1) + m.group(2).lower(),
        time_range,
    )

    search_terms_used = (
        list(search_terms) + [str(c) for c in countries]
        + [str(p) for p in ports] + [str(p) for p in protocols]
    )

    if not (search_terms or ports or countries or protocols):
        return [], [], field_mappings

    query = _build_compact_query_with_llm(
        question=question,
        query_plan=query_plan,
        field_mappings=field_mappings,
        conversation_history=conversation_history,
        llm=llm,
    )

    if not query:
        query = _build_structured_query_from_plan(
            search_terms=search_terms,
            ports=ports,
            countries=countries,
            protocols=protocols,
            time_range=time_range,
            field_mappings=field_mappings,
        )

    if not query or query.get("query") == {"match_none": {}}:
        return [], [], field_mappings

    query["size"] = 50

    try:
        results = db.search(logs_index, query, size=50)
        if not results:
            recovery = _build_recovery_query_from_plan(
                search_terms=search_terms,
                ports=ports,
                countries=countries,
                protocols=protocols,
                time_range=time_range,
                field_mappings=field_mappings,
            )
            if recovery:
                results = db.search(logs_index, recovery, size=50)
        return results or [], search_terms_used, field_mappings
    except Exception as exc:
        from core.db_connector import QueryMalformedException
        if isinstance(exc, QueryMalformedException):
            from core.query_repair import IntelligentQueryRepair
            repair = IntelligentQueryRepair(db, llm)
            success, results, _ = repair.repair_and_retry(logs_index, exc.original_query, size=50)
            return ((results or []), search_terms_used, field_mappings) if success else ([], [], field_mappings)
        logger.error("[%s] Log search error: %s", SKILL_NAME, exc)
        return [], [], field_mappings


def _plan_query_with_llm(question, conversation_history, field_mappings, llm) -> dict:
    conversation_summary = ""
    if conversation_history:
        relevant = conversation_history[-6:]
        conversation_summary = "\n".join(
            f"[{m.get('role','?').upper()}]: {m.get('content','')[:300]}"
            for m in relevant
        )

    prompt = f"""You are a cybersecurity analyst planning a log search.

CONVERSATION HISTORY:
{conversation_summary or "(No prior context)"}

USER QUESTION: "{question}"

Extract:
1. PORTS: port numbers mentioned
2. COUNTRIES: country/region names mentioned
3. PROTOCOLS: protocols mentioned (HTTP, DNS, TLS, etc.)
4. TIME RANGE: time period referenced
5. OTHER TERMS: generic search keywords

RESPOND IN JSON:
{{
  "reasoning": "What user wants (2-3 sentences)",
  "detected_time_range": "verbatim time mention or empty",
  "time_range": "Elasticsearch range string (use lowercase units: d,w,M,y)",
  "ports": [],
  "countries": [],
  "protocols": [],
  "search_terms": [],
  "skip_search": false
}}

Time range examples:
- "past 3 months" → "now-3M"
- "February" → "2026-02-01:2026-03-01"
- "last week" → "now-1w"
- no mention → "now-90d"
"""
    try:
        resp = llm.complete(prompt)
        plan = json.loads(resp)
        for key, default in [
            ("search_terms", []), ("ports", []), ("countries", []),
            ("protocols", []), ("time_range", "now-90d"), ("reasoning", ""),
        ]:
            if not isinstance(plan.get(key), type(default)):
                plan[key] = default
        if plan.get("skip_search") or not any(
            [plan.get("search_terms"), plan.get("ports"), plan.get("countries"), plan.get("protocols")]
        ):
            heuristic = _heuristic_query_plan(question, conversation_history)
            if not heuristic.get("skip_search"):
                return heuristic
        return plan
    except Exception as exc:
        logger.warning("[%s] Query planning failed: %s", SKILL_NAME, exc)
        return _heuristic_query_plan(question, conversation_history)


def _select_compact_text_fields(field_mappings: dict, max_fields: int = MAX_MULTI_MATCH_FIELDS) -> list[str]:
    preferred = [
        "message", "event", "alert", "signature", "reason", "state", "action",
        "country", "protocol", "hostname", "url", "domain", "flow", "dns", "http", "tls",
    ]
    candidates = field_mappings.get("text_fields") or field_mappings.get("all_fields") or []
    if not candidates:
        return ["message", "event.message", "alert.signature"]
    scored = []
    for field in candidates:
        fl = str(field).lower()
        score = sum(2 for t in preferred if t in fl)
        if fl.endswith(".keyword"):
            score -= 1
        scored.append((score, field))
    scored.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    result = [f for _, f in scored[:max_fields]]
    return result or list(candidates)[:max_fields]


def _map_country_names_to_codes(country_names: list) -> list[str]:
    country_map = {
        "iran": "IR", "iraq": "IQ", "syria": "SY", "north korea": "KP",
        "china": "CN", "russia": "RU", "united states": "US", "usa": "US",
        "uk": "GB", "united kingdom": "GB", "france": "FR", "germany": "DE",
        "india": "IN", "pakistan": "PK",
    }
    return [country_map[n.lower()] for n in country_names if n.lower() in country_map]


def _parse_time_range(time_range_str: str | None) -> dict | None:
    if not time_range_str:
        return None
    if time_range_str.startswith("now"):
        return {"range": {"@timestamp": {"gte": time_range_str}}}
    if ":" in time_range_str:
        parts = time_range_str.split(":", 1)
        if len(parts) == 2:
            return {"range": {"@timestamp": {"gte": parts[0], "lte": parts[1]}}}
    return None


def _build_recovery_query_from_plan(
    search_terms, ports, countries, protocols, time_range, field_mappings
) -> dict | None:
    must_clauses = []
    all_fields    = [str(f) for f in (field_mappings.get("all_fields") or [])]
    ip_fields      = [str(f) for f in (field_mappings.get("ip_fields") or [])][:8]
    country_fields = [f for f in all_fields if "country" in f.lower()]
    port_fields    = (field_mappings.get("port_fields") or [])[:6]
    protocol_fields = [f for f in all_fields if "protocol" in f.lower() or f.lower().endswith("proto")][:6]
    compact_text   = _select_compact_text_fields(field_mappings)

    if countries:
        codes = _map_country_names_to_codes(countries)
        should = []
        for field in country_fields[:10]:
            for c in countries:
                should.append({"match_phrase": {field: c}})
            for code in codes:
                should.append({"term": {field: code.upper()}})
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    if ports:
        should = []
        for field in port_fields:
            for p in ports:
                try:
                    should.append({"term": {field: int(p)}})
                except Exception:
                    pass
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    if protocols:
        should = []
        for field in protocol_fields:
            for proto in protocols:
                should.append({"term": {field: str(proto).lower()}})
                should.append({"match": {field: str(proto)}})
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    if search_terms:
        should = []
        for term in search_terms:
            if not term:
                continue
            if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", str(term)) and ip_fields:
                should.extend({"term": {field: str(term)}} for field in ip_fields)
            else:
                should.append({"multi_match": {"query": str(term), "fields": compact_text, "operator": "OR"}})
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    if not must_clauses:
        return None
    query = {"query": {"bool": {"must": must_clauses}}}
    rf = _parse_time_range(time_range)
    if rf:
        query["query"]["bool"]["filter"] = [rf]
    return query


def _build_structured_query_from_plan(
    search_terms, ports, countries, protocols, time_range, field_mappings
) -> dict:
    must_clauses = []
    all_fields = field_mappings.get("all_fields", {})
    ip_fields = field_mappings.get("ip_fields") or []

    if ports:
        port_field = next(
            (f for f in ["dest_port", "port"] if f in all_fields), None
        )
        if port_field:
            clauses = [{"term": {port_field: p}} for p in ports]
            must_clauses.append(
                clauses[0] if len(clauses) == 1
                else {"bool": {"should": clauses, "minimum_should_match": 1}}
            )

    if countries:
        codes = _map_country_names_to_codes(countries)
        geoip_candidates = [
            "geoip.country_code2", "geoip.country_code", "geoip.country_code3",
            "country_code", "geoip.country_name",
        ]
        geoip_field = next((f for f in geoip_candidates if f in all_fields), None)
        if geoip_field and codes:
            clauses = [{"term": {geoip_field: c}} for c in codes]
            must_clauses.append(
                clauses[0] if len(clauses) == 1
                else {"bool": {"should": clauses, "minimum_should_match": 1}}
            )

    if protocols:
        proto_field = next(
            (f for f in ["protocol", "service_protocol"] if f in all_fields), None
        )
        if proto_field:
            clauses = [{"term": {proto_field: p.lower()}} for p in protocols]
            must_clauses.append(
                clauses[0] if len(clauses) == 1
                else {"bool": {"should": clauses, "minimum_should_match": 1}}
            )

    if search_terms:
        compact = _select_compact_text_fields(field_mappings)
        should = []
        for term in search_terms:
            if not term:
                continue
            if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", str(term)) and ip_fields:
                should.extend({"term": {field: str(term)}} for field in ip_fields[:8])
            else:
                should.append({"multi_match": {"query": term, "fields": compact, "operator": "OR"}})
        if should:
            must_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})

    if not must_clauses:
        return {"query": {"match_none": {}}, "size": 50}

    q = {"query": {"bool": {"must": must_clauses}}}
    rf = _parse_time_range(time_range)
    if rf:
        q["query"]["bool"]["filter"] = [rf]
    return q


def _build_compact_query_with_llm(
    question, query_plan, field_mappings, conversation_history, llm
) -> dict | None:
    compact_fields = _select_compact_text_fields(field_mappings)
    ip_fields      = (field_mappings.get("ip_fields") or [])[:6]
    port_fields    = (field_mappings.get("port_fields") or [])[:4]
    ts_fields      = (field_mappings.get("timestamp_fields") or ["@timestamp"])[:3]

    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f"- {m.get('role','?')}: {str(m.get('content',''))[:220]}"
            for m in conversation_history[-4:]
        )

    prompt = f"""You are crafting an OpenSearch query for SOC logs.

QUESTION:
{question}

EXTRACTED PLAN:
{json.dumps(query_plan, indent=2, default=str)}

RECENT CONTEXT:
{history_text or '- none'}

ALLOWED FIELD CANDIDATES:
- text_fields: {compact_fields}
- ip_fields: {ip_fields}
- port_fields: {port_fields}
- timestamp_fields: {ts_fields}

Return STRICT JSON:
{{
  "query": {{ ... valid OpenSearch DSL ... }}
}}

Rules:
- Keep compact. Never >12 fields in multi_match.
- Prefer term/range for ports, countries, protocols, IPs.
- Include time range from plan. Do not include "size".
"""
    try:
        raw = llm.complete(prompt)
        parsed = _extract_json_from_response(raw)
        if not parsed:
            return None
        query = parsed if "query" in parsed else {"query": parsed}
        return _sanitize_llm_query(query, field_mappings, query_plan.get("time_range"))
    except Exception as exc:
        logger.warning("[%s] Compact LLM query failed: %s", SKILL_NAME, exc)
        return None


def _sanitize_llm_query(query: dict, field_mappings: dict, time_range: str | None) -> dict | None:
    if not isinstance(query, dict):
        return None
    q = dict(query)
    q.pop("size", None)
    if "query" not in q:
        q = {"query": q}
    if not isinstance(q.get("query"), dict):
        return None
    bool_q = q["query"].get("bool")
    if not isinstance(bool_q, dict):
        return None

    def _norm(clause):
        if isinstance(clause, dict):
            if "should" in clause and "bool" not in clause and len(clause) == 1:
                clause = {"bool": {"should": clause["should"], "minimum_should_match": 1}}
            if "multi_match" in clause and isinstance(clause["multi_match"], dict):
                mm = dict(clause["multi_match"])
                mm["fields"] = (mm.get("fields") or _select_compact_text_fields(field_mappings))[:MAX_MULTI_MATCH_FIELDS]
                clause["multi_match"] = mm
            for key in ("bool",):
                if key in clause and isinstance(clause[key], dict):
                    b = clause[key]
                    for ak in ("must", "should", "filter", "must_not"):
                        if ak in b:
                            if not isinstance(b[ak], list):
                                b[ak] = [b[ak]]
                            b[ak] = [_norm(i) for i in b[ak] if i is not None]
                    clause[key] = b
        return clause

    for ak in ("must", "should", "filter", "must_not"):
        if ak in bool_q:
            if not isinstance(bool_q[ak], list):
                bool_q[ak] = [bool_q[ak]]
            bool_q[ak] = [_norm(i) for i in bool_q[ak] if i is not None]

    rf = _parse_time_range(time_range) if time_range else None
    if rf:
        existing = bool_q.get("filter", [])
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        if not any(isinstance(f, dict) and "range" in f for f in existing):
            existing.append(rf)
        bool_q["filter"] = existing

    q["query"]["bool"] = bool_q
    try:
        from core.query_repair import _is_valid_query_structure
        ok, _ = _is_valid_query_structure(q)
        if not ok:
            return None
    except Exception:
        pass
    return q


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction / formatting
# ──────────────────────────────────────────────────────────────────────────────

def _format_combined_context(rag_docs, raw_logs, question, search_terms=None) -> str:
    parts = [f"User Question: {question}"]
    if search_terms:
        parts.append(f"Search Terms: {', '.join(search_terms)}")
    if rag_docs:
        parts.append("=== BEHAVIORAL BASELINES ===")
        for i, doc in enumerate(rag_docs, 1):
            parts.append(
                f"[Baseline {i} | {doc.get('source','?')} | {doc.get('category','?')} "
                f"| Match: {doc.get('similarity', 0):.1%}]\n{doc.get('text','')}"
            )
    if raw_logs:
        parts.append("\n=== OBSERVED LOG RECORDS ===")
        if search_terms:
            parts.append(f"Note: matched on: {', '.join(search_terms)}")
        parts.append(_summarize_raw_logs(raw_logs, question, search_terms))
    return "\n\n".join(parts)


def _summarize_raw_logs(logs, question, search_terms=None) -> str:
    if not logs:
        return "No matching log records found."
    display = logs[:25]
    lines = [f"Found {len(logs)} matching records (showing first {len(display)}):"]
    if search_terms:
        lines.append(f"(matched on: {', '.join(search_terms)})")
    lines.append("")
    for i, log in enumerate(display, 1):
        lines.append(f"Record {i}:")
        for field in sorted(log.keys()):
            val = log[field]
            if isinstance(val, dict):
                val = str(val)[:100]
            elif isinstance(val, (list, str)):
                val = str(val)[:200]
            lines.append(f"  {field}: {val}")
        lines.append("")
    if len(logs) > len(display):
        lines.append(f"(… {len(logs) - len(display)} more records omitted)")
    return "\n".join(lines)


def _extract_answer_from_data(question, context_text, instruction, llm) -> str:
    prompt = f"""User Question: "{question}"

Context (baselines + log records):
{context_text}

Answer using exact values from the data. Summarise records in natural language.
Mention counts, timestamps, IPs, ports where relevant."""
    try:
        return llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]).strip()
    except Exception as exc:
        logger.error("[%s] Answer extraction failed: %s", SKILL_NAME, exc)
        return f"Error analysing data: {exc}"


def _extract_evidence_details(raw_logs: list[dict]) -> dict:
    if not raw_logs:
        return {"ips": [], "ports": [], "protocols": [], "record_ids": [], "timestamps": []}
    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    ips, ports, protocols, record_ids, timestamps = set(), set(), set(), [], []
    for row in raw_logs:
        if row.get("_id") and str(row["_id"]) not in record_ids:
            record_ids.append(str(row["_id"]))
        for key, value in row.items():
            kl = str(key).lower()
            if value is None:
                continue
            if isinstance(value, str):
                if "timestamp" in kl and value not in timestamps:
                    timestamps.append(value)
                if "protocol" in kl or kl in {"network.transport", "network.protocol"}:
                    protocols.add(value.lower())
                for m in ip_pattern.findall(value):
                    ips.add(m)
            if isinstance(value, (int, float)) and "port" in kl:
                if 0 < int(value) <= 65535:
                    ports.add(str(int(value)))
    return {
        "ips": sorted(ips), "ports": sorted(ports, key=int),
        "protocols": sorted(protocols),
        "record_ids": record_ids[:20], "timestamps": timestamps[:20],
    }


def _persist_evidence_to_memory(memory, question: str, evidence: dict) -> None:
    try:
        ips  = ",".join(evidence.get("ips", [])[:5]) or "none"
        ports = ",".join(evidence.get("ports", [])[:5]) or "none"
        memory.add_decision(
            f"[BaselineQuerier] q='{question[:80]}' ips={ips} ports={ports}"
        )
    except Exception as exc:
        logger.debug("[%s] Failed to persist evidence: %s", SKILL_NAME, exc)
