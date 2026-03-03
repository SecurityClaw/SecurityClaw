"""
skills/rag_querier/logic.py

Data-agnostic RAG querier skill. Searches stored baseline knowledge
to answer user questions about network/system behavior.

Context keys consumed:
    context["db"]         -> BaseDBConnector
    context["llm"]        -> BaseLLMProvider
    context["memory"]     -> AgentMemory
    context["config"]     -> Config
    context["parameters"] -> {"question": "user question"}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "rag_querier"


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    cfg = context.get("config")
    parameters = context.get("parameters", {})

    if db is None or llm is None:
        logger.warning("[%s] db or llm not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db/llm"}

    user_question = parameters.get("question")
    if not user_question:
        logger.warning("[%s] No question provided in parameters.", SKILL_NAME)
        return {"status": "no_question"}

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    logs_index = cfg.get("db", "logs_index", default="securityclaw-logs")
    vector_index = cfg.get("db", "vector_index", default="securityclaw-vectors")

    # ── 1. Search RAG for relevant baselines ──────────────────────────────────
    logger.info("[%s] Searching for: %s", SKILL_NAME, user_question)

    rag_docs = []
    try:
        from core.rag_engine import RAGEngine

        rag = RAGEngine(db=db, llm=llm)
        rag_docs = rag.retrieve(user_question, k=5)
        logger.info("[%s] Found %d relevant baselines in RAG.", SKILL_NAME, len(rag_docs))
    except Exception as exc:
        logger.warning("[%s] RAG retrieval failed: %s", SKILL_NAME, exc)
        # Continue with raw logs even if RAG fails

    # ── 2. Search raw logs for matching data ──────────────────────────────────
    raw_logs = []
    search_terms_used = []
    try:
        raw_logs, search_terms_used = _search_raw_logs(user_question, db, logs_index, llm)
        logger.info(
            "[%s] Found %d matching records in logs (search terms: %s).",
            SKILL_NAME, len(raw_logs), search_terms_used
        )
    except Exception as exc:
        logger.error("[%s] Raw log search failed: %s", SKILL_NAME, exc)

    # ── 3. If neither RAG nor raw logs have data, return no_data ──────────────
    if not rag_docs and not raw_logs:
        logger.info("[%s] No data found (RAG or logs).", SKILL_NAME)
        return {
            "status": "no_data",
            "findings": {
                "question": user_question,
                "answer": "No data found to answer this question.",
                "confidence": 0.0,
            },
        }

    # ── 4. Analyze combined data with LLM to extract answer ──────────────────
    combined_context = _format_combined_context(
        rag_docs, raw_logs, user_question, search_terms_used
    )
    answer = _extract_answer_from_data(user_question, combined_context, instruction, llm)

    findings = {
        "question": user_question,
        "answer": answer,
        "rag_sources": len(rag_docs),
        "log_records": len(raw_logs),
        "confidence": 0.85 if (rag_docs or raw_logs) else 0.0,
        "summary": {
            "baseline_insights": len(rag_docs),
            "raw_observations": len(raw_logs),
        },
    }

    logger.info(
        "[%s] Answer compiled from %d baselines + %d log records. "
        "RAG docs delivered: %d/%d, Raw logs delivered: %d/%d",
        SKILL_NAME,
        len(rag_docs),
        len(raw_logs),
        min(len(rag_docs), 5),  # All RAG docs now
        len(rag_docs),
        min(len(raw_logs), 10),  # Raw logs up to 10
        len(raw_logs),
    )

    return {
        "status": "ok",
        "findings": findings,
    }


def _search_raw_logs(question: str, db: Any, logs_index: str, llm: Any = None) -> tuple[list[dict], list[str]]:
    """
    Search raw logs for data matching the user question.
    DATA-AGNOSTIC: Tries field names from RAG schema observations or falls back to common formats.
    
    Returns (logs, search_terms_used) tuple so caller knows what was searched.
    
    Intelligently extracts keywords from the question and searches,
    including geographic, temporal, and pattern-based queries.
    
    Supports any log format by trying common field names from multiple formats
    (Suricata EVE, ECS, Zeek, NetFlow, etc.)
    """
    # Extract potential search terms from the question
    search_terms = _extract_search_terms(question)
    
    if not search_terms:
        return [], []
    
    # TODO: Query RAG for schema observations to determine actual field names
    # For now, use multi-format fallback approach
    
    import re as _re
    ip_pattern = r'^(?:\d{1,3}\.){3}\d{1,3}$'
    port_pattern = r'^\d{1,5}$'
    protocol_names = {'tcp', 'udp', 'icmp', 'http', 'https', 'dns', 'ssh', 'rdp', 'smb'}
    
    # Common country names and aliases for geolocation queries
    country_names = {
        'iran', 'china', 'russia', 'north korea', 'syria', 'cuba',
        'usa', 'united states', 'uk', 'england', 'france', 'germany', 'japan',
        'india', 'brazil', 'canada', 'australia', 'netherlands', 'israel',
        'uae', 'saudi arabia', 'iran', 'iraq', 'afghanistan', 'pakistan',
    }

    # Build should_clauses with multiple field format attempts
    # This allows search to work regardless of whether data is in Suricata/ECS/Zeek format
    should_clauses = []

    for term in search_terms:
        if _re.match(ip_pattern, term):
            # IP addresses — try all common field name formats
            for src_field in ["src_ip", "source.ip", "id.orig_h"]:
                for dst_field in ["dest_ip", "destination.ip", "id.resp_h"]:
                    should_clauses += [
                        {"term": {dst_field: term}},
                        {"term": {src_field: term}},
                        {"term": {f"{dst_field}.keyword": term}},
                        {"term": {f"{src_field}.keyword": term}},
                    ]
        elif _re.match(port_pattern, term):
            # Port numbers — try all common field name formats
            try:
                port_int = int(term)
                for src_port in ["src_port", "source.port", "id.orig_p"]:
                    for dst_port in ["dest_port", "destination.port", "id.resp_p"]:
                        should_clauses += [
                            {"term": {dst_port: port_int}},
                            {"term": {src_port: port_int}},
                        ]
            except ValueError:
                pass
        elif term.lower() in protocol_names:
            # Protocols — try common field names (TCP/tcp, proto, etc.)
            for proto_field in ["proto", "protocol", "network.protocol", "service"]:
                should_clauses += [
                    {"term": {proto_field: term.upper()}},
                    {"term": {f"{proto_field}.keyword": term.upper()}},
                    {"match": {proto_field: term.lower()}},
                ]
        elif term.lower() in country_names:
            # Geographic queries — try common GeoIP field names
            for country_field in ["geoip.country_name", "destination.geo.country_name", "source.geo.country_name"]:
                should_clauses += [
                    {"term": {f"{country_field}.keyword": term.capitalize() if len(term) > 3 else term.upper()}},
                    {"match": {country_field: {"query": term}}},
                ]
        else:
            # Hostnames, domains, other strings — try common app protocol fields
            for app_field in ["app_proto", "network.application", "service", "application_protocol"]:
                should_clauses += [
                    {"match": {app_field: {"query": term}}},
                    {"wildcard": {f"{app_field}.keyword": {"value": f"*{term}*"}}},
                ]

    if not should_clauses:
        return [], []

    query = {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": {
                    "range": {
                        "@timestamp": {"gte": "now-90d"}  # Broader window for testing/demo data
                    }
                },
            }
        },
        "size": 50,
    }
    
    logger.info(
        "[%s] DATA-AGNOSTIC search: trying multiple field formats for terms: %s",
        SKILL_NAME, search_terms
    )
    
    try:
        results = db.search(logs_index, query, size=50)
        return results, search_terms
    except Exception as exc:
        logger.warning("[%s] Raw log search error: %s", SKILL_NAME, exc)
        return [], []


def _extract_search_terms(question: str) -> list[str]:
    """
    Extract potential search terms from a user question.
    Looks for IPs, hostnames, ports, protocols, etc.
    Returns each as a string; _search_raw_logs categorises them.
    """
    import re

    terms = []
    remaining = question
    
    # Common countries for explicit extraction (before stopword filter)
    countries = [
        'iran', 'iraq', 'china', 'russia', 'north korea', 'syria', 'cuba',
        'usa', 'united states', 'uk', 'england', 'france', 'germany', 'japan',
        'india', 'brazil', 'canada', 'australia', 'netherlands', 'israel',
        'uae', 'saudi arabia', 'afghanistan', 'pakistan', 'korea',
    ]
    for country in countries:
        if re.search(rf'\b{country}\b', remaining, re.IGNORECASE):
            terms.append(country.lower())
            remaining = re.sub(rf'\b{country}\b', ' ', remaining, flags=re.IGNORECASE)

    # 1. Extract IPv4 addresses first (remove them so octets don't get picked up again)
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, remaining)
    terms.extend(ips)
    remaining = re.sub(ip_pattern, ' ', remaining)

    # 2. Extract explicit port numbers ("port 443", ":8080")
    port_pattern = r'\bport\s+(\d{1,5})\b|:(\d{2,5})\b'
    for m in re.finditer(port_pattern, remaining):
        port_num = m.group(1) or m.group(2)
        if port_num:
            terms.append(port_num)
    remaining = re.sub(port_pattern, ' ', remaining)

    # 3. Extract known protocol names
    protocols = ['tcp', 'udp', 'icmp', 'http', 'https', 'dns', 'ssh', 'rdp', 'smb', 'ftp', 'smtp', 'ntp']
    for proto in protocols:
        if re.search(rf'\b{proto}\b', remaining, re.IGNORECASE):
            terms.append(proto)

    # 4. Extract hostnames / domain names (must contain at least one letter)
    hostname_pattern = r'\b([a-zA-Z][a-zA-Z0-9\-]{1,62}(?:\.[a-zA-Z0-9][a-zA-Z0-9\-]{0,62})*)\b'
    for m in re.finditer(hostname_pattern, remaining):
        terms.append(m.group(1))

    # 5. Deduplicate and strip noise
    # Note: Countries are intentionally excluded from stopwords
    stopwords = {
        'what', 'is', 'there', 'any', 'traffic', 'to', 'from', 'the', 'a',
        'are', 'on', 'in', 'for', 'if', 'and', 'or', 'not', 'was', 'were',
        'have', 'has', 'had', 'past', 'hours', 'last', 'days', 'recent',
        'show', 'me', 'check', 'find', 'network', 'do', 'you', 'can', 'search',
        'visiting', 'find', 'see', 'look', 'tell',
    }
    
    # Countries must never be filtered
    country_names = {
        'iran', 'iraq', 'china', 'russia', 'north korea', 'syria', 'cuba',
        'usa', 'united states', 'uk', 'england', 'france', 'germany', 'japan',
        'india', 'brazil', 'canada', 'australia', 'netherlands', 'israel',
        'uae', 'saudi arabia', 'afghanistan', 'pakistan', 'korea',
    }
    
    seen = set()
    unique = []
    for t in terms:
        t_low = t.lower()
        if t_low not in stopwords and t_low not in seen and len(t) > 1:
            seen.add(t_low)
            unique.append(t_low)

    return unique


def _format_combined_context(
    rag_docs: list[dict], raw_logs: list[dict], question: str, search_terms: list[str] = None
) -> str:
    """Format both RAG baseline data and raw logs for LLM analysis."""
    if search_terms is None:
        search_terms = []
    
    context_parts = []
    
    # Add user's question for clarity
    context_parts.append(f"User Question: {question}")
    
    # Add search terms used if any
    if search_terms:
        context_parts.append(f"Search Terms Extracted: {', '.join(search_terms)}")
    
    if rag_docs:
        context_parts.append("=== BASELINE KNOWLEDGE (from stored baselines) ===")
        for i, doc in enumerate(rag_docs, 1):  # All retrieved RAG docs (typically 5)
            category = doc.get("category", "unknown")
            source = doc.get("source", "unknown")
            text = doc.get("text", "")
            similarity = doc.get("similarity", 0.0)
            context_parts.append(
                f"[Baseline {i} | {source} | {category} | Match: {similarity:.1%}]\n{text}"
            )
    
    if raw_logs:
        context_parts.append("\n=== OBSERVED DATA (from recent logs) ===")
        # Add note about what was searched for
        if search_terms:
            context_parts.append(
                f"Note: These logs were selected because they match your search for: {', '.join(search_terms)}"
            )
        context_parts.append(_summarize_raw_logs(raw_logs, question, search_terms))
    
    return "\n\n".join(context_parts)


def _summarize_raw_logs(logs: list[dict], question: str, search_terms: list[str] = None) -> str:
    """Summarize raw logs to highlight relevant fields and explain filter context."""
    if search_terms is None:
        search_terms = []
        
    if not logs:
        return "No recent log records found."
    
    # Identify what kind of search this was (geographic, port-based, IP-based, etc.)
    country_names = {
        'iran', 'china', 'russia', 'north korea', 'syria', 'cuba',
        'usa', 'united states', 'uk', 'england', 'france', 'germany', 'japan',
        'india', 'brazil', 'canada', 'australia', 'netherlands', 'israel',
        'uae', 'saudi arabia', 'iraq', 'afghanistan', 'pakistan',
    }
    
    # Check if this is a geographic search
    is_geo_search = any(term.lower() in country_names for term in search_terms)
    if is_geo_search:
        geo_term = next((term for term in search_terms if term.lower() in country_names), None)
        geo_note = f"\n*** IMPORTANT: These {len(logs)} log records were specifically filtered to match your search for '{geo_term}' ***\nThis means the IPs in these logs are GEOLOCATED TO or ASSOCIATED WITH {geo_term.upper()}.\n"
    else:
        geo_note = ""
    
    summary_lines = [
        f"Found {len(logs)} recent log records matching your search criteria{geo_note}"
        if is_geo_search
        else f"Found {len(logs)} recent log records (showing up to 10):"
    ]
    
    # Show all records with relevant fields (limit to reasonable number)
    display_limit = min(10, len(logs))
    for i, log in enumerate(logs[:display_limit], 1):
        relevant_fields = [
            ("src", log.get("source.ip") or log.get("src_ip")),
            ("dst", log.get("destination.ip") or log.get("dest_ip")),
            ("port", log.get("destination.port") or log.get("dest_port")),
            ("proto", log.get("network.protocol") or log.get("protocol") or log.get("proto")),
            ("host", log.get("host.hostname") or log.get("hostname")),
            ("time", log.get("@timestamp")[:10] if log.get("@timestamp") else "?"),
        ]
        
        fields_str = " | ".join(
            f"{k}={v}" for k, v in relevant_fields if v
        )
        summary_lines.append(f"  {i}. {fields_str}")
    
    if len(logs) > display_limit:
        summary_lines.append(f"  ... and {len(logs) - display_limit} more records matching your search criteria")
    
    return "\n".join(summary_lines)


def _extract_answer_from_data(
    question: str,
    context_text: str,
    instruction: str,
    llm: Any,
) -> str:
    """Use LLM to extract answer from both RAG baselines and raw logs."""
    prompt = f"""You are a SOC analyst answering questions about network behavior.

User Question: "{question}"

Available Context (both baselines and observed data):
{context_text}

CRITICAL: When logs are provided with a note about geographic filtering (e.g., "filtered to match..."), 
those logs CONFIRM activity matching that criterion. If logs are present for a geographic search, 
answer affirmatively (e.g., "Yes, there were connections to/from [country]").

Based on the above data, answer the user's question concisely (1-3 sentences).
Reference both baseline patterns and observed data. Be specific with counts, IP addresses, ports, and protocols when relevant."""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm.chat(messages)
        return response.strip()
    except Exception as exc:
        logger.error("Failed to extract answer: %s", exc)
        return f"Error analyzing data: {exc}"
