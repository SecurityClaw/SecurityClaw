"""
skills/forensic_examiner/logic.py

Data-agnostic forensic timeline reconstructor. Takes an incident report
and uses RAG field documentation to understand the data schema, then lets the LLM
decide what to search for and how to build a timeline.

Context keys consumed:
    context["db"]         -> BaseDBConnector
    context["llm"]        -> BaseLLMProvider
    context["memory"]     -> Memory instance (StateBackedMemory or CheckpointBackedMemory)
    context["config"]     -> Config
    context["parameters"] -> {"question": "incident description"}
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "forensic_examiner"




def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    conversation_history = context.get("conversation_history", [])

    if db is None or llm is None:
        logger.warning("[%s] db or llm not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db/llm"}

    incident_question = parameters.get("question")
    if not incident_question:
        logger.warning("[%s] No incident question provided.", SKILL_NAME)
        return {"status": "no_question"}

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    logs_index = cfg.get("db", "logs_index", default="securityclaw-logs")
    vector_index = cfg.get("db", "vector_index", default="securityclaw-vectors")

    logger.info("[%s] Analyzing incident: %s", SKILL_NAME, incident_question)

    # ── 1. Fetch field documentation from RAG ───────────────────────────────
    field_docs = _fetch_field_documentation(db, vector_index, llm)
    
    if not field_docs:
        logger.warning("[%s] No field documentation found in RAG; cannot proceed", SKILL_NAME)
        return {
            "status": "failed",
            "reason": "Field documentation not available from RAG (network baseliner not run)"
        }
    
    logger.info("[%s] Retrieved field documentation from RAG", SKILL_NAME)

    # ── 2. Extract context and run iterative plan→act→evaluate loop ──────────
    incident_context = _extract_basic_context(incident_question, conversation_history)

    investigation = _run_iterative_investigation(
        db=db,
        llm=llm,
        logs_index=logs_index,
        incident_question=incident_question,
        conversation_history=conversation_history,
        field_docs=field_docs,
        incident_context=incident_context,
    )

    search_strategy = investigation.get("initial_strategy", {})
    all_results = investigation.get("all_results", [])
    refinement_iteration = investigation.get("iterations_completed", 0)
    investigation_trace = investigation.get("trace", [])
    logger.info("[%s] Final result set: %d total results", SKILL_NAME, len(all_results))

    # ── 5. Ask LLM to build comprehensive timeline ────────────────────────────
    focused_results = _select_contextual_results(all_results, incident_context, limit=80)
    if focused_results:
        timeline_narrative = _ask_llm_for_comprehensive_timeline(
            llm, incident_question, focused_results, field_docs, instruction
        )
    else:
        timeline_narrative = _ask_llm_for_timeline_no_results(
            llm, incident_question, search_strategy, field_docs, instruction
        )

    # ── 6. Return forensic report ────────────────────────────────────────────
    forensic_report = {
        "incident_summary": incident_question,
        "initial_strategy": search_strategy,
        "refinement_rounds": refinement_iteration,
        "results_found": len(all_results),
        "focused_results_found": len(focused_results),
        "context_anchors": {
            "ips": incident_context.get("ips", []),
            "ports": incident_context.get("ports", []),
            "countries": incident_context.get("countries", []),
            "protocols": incident_context.get("protocols", []),
            "time_range_hint": incident_context.get("time_range_hint"),
        },
        "investigation_trace": investigation_trace,
        "timeline_narrative": timeline_narrative,
    }

    return {
        "status": "ok",
        "forensic_report": forensic_report,
    }


def _fetch_field_documentation(db: Any, vector_index: str, llm: Any) -> str:
    """Fetch field documentation baselines from RAG."""
    try:
        from core.rag_engine import RAGEngine
        rag = RAGEngine(db=db, llm=llm)
        
        docs = rag.retrieve("field names schema", k=5)
        field_docs = [
            doc.get("text", "")
            for doc in docs
            if doc.get("category") == "field_documentation"
        ]
        
        if field_docs:
            return "\n\n".join(field_docs[:2])
        
        return ""
    except Exception as exc:
        logger.warning("[%s] Could not fetch field documentation: %s", SKILL_NAME, exc)
        return ""




def _parse_field_mappings(field_docs: str) -> dict:
    """Parse field documentation to extract field mappings (DATA-AGNOSTIC).
    
    Discovers actual field names from field_documentation baseline instead of
    hardcoding them. This makes searches work with any data schema.
    
    CRITICAL: IP fields are NOT added to all_text_fields, only fields that are
    truly meant for text search. This prevents port numbers from being searched
    in IP-type fields (which causes OpenSearch "'1194' is not an IP" errors).
    """
    mappings = {
        "ip_fields": [],
        "port_fields": [],
        "protocol_fields": [],
        "dns_fields": [],
        "timestamp_fields": [],
        "all_text_fields": [],  # Only real text fields, NOT ip/port/proto/dns fields
    }
    
    if not field_docs:
        return mappings
    
    for line in field_docs.split("\n"):
        lower = line.lower()
        field = None
        
        if "field:" in lower:
            field = line.split(":", 1)[1].strip() if ":" in line else None
        elif "name:" in lower:
            field = line.split(":", 1)[1].strip() if ":" in line else None
        elif "- " in line:
            field = line.strip()[2:].split("(")[0].strip()
        
        if not field:
            continue
        
        # Classify field by type - EXACTLY ONE category per field
        field_lower = field.lower()
        is_ip = any(kw in lower for kw in ["ipv4", "ip address", "source.ip", "destination.ip", "src_ip", "dest_ip"])
        is_port = "port" in lower
        is_protocol = any(kw in lower for kw in ["protocol", "transport", "proto", "application.protocol"])
        is_dns = any(kw in lower for kw in ["dns", "dns.query"])
        is_timestamp = any(kw in lower for kw in ["timestamp", "@timestamp", "datetime", "time"])
        is_text = any(kw in lower for kw in ["text", "message", "description", "body", "content", "log", "event", "reason", "domain", "url", "hostname"])
        looks_non_text_by_name = any(
            token in field_lower
            for token in [
                "ip", "port", "timestamp", "time", "date", "bytes", "packets",
                "count", "size", "duration", "latency", "ttl", "asn", "geo.location"
            ]
        )
        
        if is_ip:
            if field not in mappings["ip_fields"]:
                mappings["ip_fields"].append(field)
        elif is_port:
            if field not in mappings["port_fields"]:
                mappings["port_fields"].append(field)
        elif is_protocol:
            if field not in mappings["protocol_fields"]:
                mappings["protocol_fields"].append(field)
        elif is_dns:
            if field not in mappings["dns_fields"]:
                mappings["dns_fields"].append(field)
        elif is_timestamp:
            if field not in mappings["timestamp_fields"]:
                mappings["timestamp_fields"].append(field)
        elif is_text and not looks_non_text_by_name:
            # Only add genuinely text fields to all_text_fields
            if field not in mappings["all_text_fields"]:
                mappings["all_text_fields"].append(field)
        else:
            # Unknown field types are intentionally NOT auto-added to text search fields.
            # This avoids OpenSearch parse exceptions when text queries hit numeric/ip/date fields.
            continue
    
    logger.debug(
        "[%s] Parsed field types: ip=%d port=%d text=%d proto=%d dns=%d timestamp=%d",
        SKILL_NAME,
        len(mappings["ip_fields"]),
        len(mappings["port_fields"]),
        len(mappings["all_text_fields"]),
        len(mappings["protocol_fields"]),
        len(mappings["dns_fields"]),
        len(mappings["timestamp_fields"]),
    )
    return mappings


def _extract_basic_context(question: str, conversation_history: list = None) -> dict:
    """Extract IPs, domains, keywords from incident."""
    context = {
        "ips": [],
        "domains": [],
        "ports": [],
        "countries": [],
        "protocols": [],
        "timestamps": [],
        "time_range_hint": None,
        "keywords": [],
        "has_dns_intent": False,
    }

    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    context["ips"] = list(set(re.findall(ip_pattern, question)))

    domain_pattern = r'\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b'
    context["domains"] = list(set(re.findall(domain_pattern, question.lower())))

    full_text = question

    if conversation_history:
        history_text = " ".join([
            msg.get("content", "")
            for msg in conversation_history
            if msg.get("content")
        ])
        full_text += " " + history_text
        
        ips = re.findall(ip_pattern, history_text)
        context["ips"].extend([ip for ip in ips if ip not in context["ips"]])
        
        domains = re.findall(domain_pattern, history_text.lower())
        context["domains"].extend([d for d in domains if d not in context["domains"]])

    lower_text = full_text.lower()

    # Extract ports from explicit patterns
    context["ports"] = sorted(set(re.findall(r"\bport\s*(\d{1,5})\b", lower_text)))

    # Extract explicit timestamps from question/history and build an anchored time window.
    ts_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", full_text)
    context["timestamps"] = sorted(set(ts_matches))
    if context["timestamps"]:
        parsed = []
        for ts in context["timestamps"]:
            try:
                parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except Exception:
                continue
        if parsed:
            start = min(parsed) - timedelta(days=2)
            end = max(parsed) + timedelta(days=2)
            context["time_range_hint"] = {
                "gte": start.isoformat().replace("+00:00", "Z"),
                "lte": end.isoformat().replace("+00:00", "Z"),
            }

    # Fallback relative time extraction from natural language
    if context["time_range_hint"] is None:
        rel = re.search(r"\bpast\s+(\d+)\s+(month|months|week|weeks|day|days)\b", lower_text)
        if rel:
            n = rel.group(1)
            unit = rel.group(2)
            suffix = "d"
            if "month" in unit:
                suffix = "M"
            elif "week" in unit:
                suffix = "w"
            context["time_range_hint"] = {"gte": f"now-{n}{suffix}"}

    # Extract high-signal country names used in this project
    country_candidates = [
        "iran", "iraq", "syria", "china", "russia", "north korea",
        "united states", "usa", "united kingdom", "uk", "germany", "france",
    ]
    context["countries"] = [c for c in country_candidates if c in lower_text]

    # Extract protocol hints
    protocol_candidates = ["tcp", "udp", "icmp", "dns", "http", "https", "tls"]
    context["protocols"] = [p for p in protocol_candidates if re.search(rf"\b{re.escape(p)}\b", lower_text)]
    context["has_dns_intent"] = bool(re.search(r"\bdns|domain|hostname|fqdn\b", lower_text))

    return context


def _ask_llm_for_search_strategy(
    llm: Any, incident_question: str, conversation_history: list,
    field_docs: str, incident_context: dict
) -> dict:
    """Ask LLM to design a search strategy."""
    
    history_summary = ""
    if conversation_history:
        history_summary = "\n\nConversation history:\n" + "\n".join([
            f"  {msg.get('role', '?').upper()}: {msg.get('content', '')[:200]}"
            for msg in conversation_history[-5:]
        ])

    extracted_context = ""
    if incident_context.get("ips") or incident_context.get("domains"):
        extracted_context = f"\n\nAlready identified:\n"
        if incident_context["ips"]:
            extracted_context += f"  IPs: {', '.join(incident_context['ips'])}\n"
        if incident_context["domains"]:
            extracted_context += f"  Domains: {', '.join(incident_context['domains'])}\n"

    prompt = f"""You are a forensic analyst designing searches for incident investigation.

INCIDENT: {incident_question}
{history_summary}
{extracted_context}

AVAILABLE FIELDS:
{field_docs}

CRITICAL RELEVANCE RULES:
- Keep searches anchored to incident entities already known from context/history.
- Prefer pivots on known IPs, ports, country, protocol, and tight time windows.
- Do NOT introduce DNS/domain hunting unless DNS/domain is explicitly part of the incident.
- If incident is IP+port probing (e.g., OpenVPN 1194), focus on flow/alert/protocol pivots, not generic DNS.
- Avoid broad generic keywords like "suspicious activity" unless tied to known entities.

Design a search strategy. Output JSON:
{{
  "summary": "brief summary",
  "search_queries": [
    {{"description": "what to find", "keywords": ["term1", "term2"]}}
  ],
  "time_window": "YYYY-MM-DD to YYYY-MM-DD (or leave empty for 30 days back)",
  "reasoning": "why these searches"
}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = llm.chat(messages)
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        
        return json.loads(response)
    except:
        return {
            "summary": "Search for incident context",
            "search_queries": [{"description": "Broad search", "keywords": [incident_question[:50]]}],
            "time_window": None,
            "reasoning": "Fallback search"
        }


def _ask_llm_for_investigation_plan(
    llm: Any,
    incident_question: str,
    conversation_history: list,
    field_docs: str,
    incident_context: dict,
) -> dict:
    """Generate a stepwise investigation TODO plan grounded in prior context."""
    history_summary = ""
    if conversation_history:
        history_summary = "\n".join([
            f"- {msg.get('role', '?')}: {str(msg.get('content', ''))[:220]}"
            for msg in conversation_history[-6:]
        ])

    prompt = f"""You are conducting a forensic investigation as an iterative workflow.

INCIDENT QUESTION:
{incident_question}

KNOWN CONTEXT ANCHORS:
- IPs: {incident_context.get('ips', [])}
- Ports: {incident_context.get('ports', [])}
- Countries: {incident_context.get('countries', [])}
- Protocols: {incident_context.get('protocols', [])}
- Time range hint: {incident_context.get('time_range_hint')}

RECENT CONVERSATION CONTEXT:
{history_summary or '- none'}

AVAILABLE FIELDS:
{field_docs}

Return STRICT JSON with this shape:
{{
  "summary": "one sentence plan",
  "time_window": "optional time window string",
  "todos": [
    {{
      "title": "short todo title",
      "goal": "what to prove",
      "search_queries": [
        {{"description": "action", "keywords": ["k1", "k2"]}}
      ]
    }}
  ],
  "stop_criteria": "when investigation is sufficient"
}}

Rules:
- Keep queries anchored to known context first.
- Avoid unrelated pivots.
- Include 2-4 todos maximum.
- First todo must validate known incident entities/timeframe.
"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}]).strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        parsed = json.loads(response)
        todos = parsed.get("todos") or []
        if isinstance(todos, list) and todos:
            return parsed
    except Exception as exc:
        logger.warning("[%s] Failed to produce investigation plan: %s", SKILL_NAME, exc)

    return {
        "summary": "Start from known entities, then validate nearby activity and chronology",
        "time_window": None,
        "todos": [
            {
                "title": "Validate known incident flow",
                "goal": "Confirm known source/destination/port activity in timeframe",
                "search_queries": [
                    {
                        "description": "Find traffic for known incident anchors",
                        "keywords": incident_context.get("ips", [])[:2] + incident_context.get("ports", [])[:1],
                    }
                ],
            },
            {
                "title": "Expand with constrained pivots",
                "goal": "Find adjacent behavior tied to same anchors",
                "search_queries": [
                    {
                        "description": "Find related events sharing known destination and port",
                        "keywords": incident_context.get("ips", [])[-1:] + incident_context.get("ports", [])[:1] + incident_context.get("protocols", [])[:1],
                    }
                ],
            },
        ],
        "stop_criteria": "Evidence remains anchored to prior context and supports incident narrative",
    }


def _ask_llm_to_re_evaluate_progress(
    llm: Any,
    incident_question: str,
    incident_context: dict,
    all_results: list,
    completed_todos: list,
    pending_todos: list,
) -> dict:
    """Re-evaluate whether current evidence is relevant and sufficient."""
    sample = all_results[:12]
    summary = {
        "results_count": len(all_results),
        "completed_todos": completed_todos,
        "pending_count": len(pending_todos),
    }
    prompt = f"""Re-evaluate this forensic investigation progress.

INCIDENT QUESTION:
{incident_question}

ANCHORS:
{json.dumps(incident_context, indent=2, default=str)}

INVESTIGATION STATUS:
{json.dumps(summary, indent=2, default=str)}

EVIDENCE SAMPLE:
{json.dumps(sample, indent=2, default=str)}

Respond STRICT JSON:
{{
  "is_relevant": true/false,
  "is_sufficient": true/false,
  "confidence": 0.0,
  "reasoning": "short reason",
  "gaps": ["gap1"],
  "next_action": {{
    "title": "optional next todo",
    "goal": "optional",
    "search_queries": [{{"description": "...", "keywords": ["..."]}}]
  }}
}}

Rules:
- Mark is_relevant=false if evidence drifts away from known anchors/timeframe.
- Mark is_sufficient=true only if enough relevant evidence exists to answer incident question.
- Suggest next_action only when not sufficient.
"""
    try:
        response = llm.chat([{"role": "user", "content": prompt}]).strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("[%s] Failed to re-evaluate progress: %s", SKILL_NAME, exc)

    return {
        "is_relevant": True,
        "is_sufficient": bool(all_results),
        "confidence": 0.4 if all_results else 0.1,
        "reasoning": "Fallback evaluator",
        "gaps": ["Need more anchored evidence"] if not all_results else [],
        "next_action": None,
    }


def _anchor_coverage_score(all_results: list, incident_context: dict) -> float:
    """Compute deterministic anchor coverage score in [0,1] from retrieved evidence."""
    if not all_results:
        return 0.0

    anchors = []
    anchors.extend([str(v).lower() for v in incident_context.get("ips", []) if v])
    anchors.extend([str(v).lower() for v in incident_context.get("ports", []) if v])
    anchors.extend([str(v).lower() for v in incident_context.get("countries", []) if v])
    anchors.extend([str(v).lower() for v in incident_context.get("protocols", []) if v])

    if not anchors:
        return 1.0

    haystack = "\n".join(json.dumps(item, default=str).lower() for item in all_results[:100])
    matches = sum(1 for anchor in set(anchors) if anchor in haystack)
    return matches / max(1, len(set(anchors)))


def _normalize_todo_action(action: dict, incident_context: dict) -> dict:
    """Ensure todo action has usable structure for execution."""
    title = action.get("title") or action.get("description") or "investigation step"
    goal = action.get("goal") or action.get("reason") or "Collect relevant evidence"
    search_queries = action.get("search_queries")

    if not isinstance(search_queries, list) or not search_queries:
        keywords = action.get("keywords") or []
        if not keywords:
            keywords = incident_context.get("ips", [])[:2] + incident_context.get("ports", [])[:1]
        search_queries = [{
            "description": title,
            "keywords": [str(k) for k in keywords],
        }]

    return {
        "title": title,
        "goal": goal,
        "search_queries": search_queries,
        "time_window": action.get("time_window"),
    }


def _run_iterative_investigation(
    db: Any,
    llm: Any,
    logs_index: str,
    incident_question: str,
    conversation_history: list,
    field_docs: str,
    incident_context: dict,
) -> dict:
    """Plan TODOs, execute actions, re-evaluate, and loop until context-grounded sufficiency."""
    plan = _ask_llm_for_investigation_plan(
        llm=llm,
        incident_question=incident_question,
        conversation_history=conversation_history,
        field_docs=field_docs,
        incident_context=incident_context,
    )

    initial_strategy = {
        "summary": plan.get("summary", ""),
        "search_queries": [],
        "time_window": plan.get("time_window"),
        "reasoning": plan.get("stop_criteria", ""),
    }

    pending = [_normalize_todo_action(todo, incident_context) for todo in (plan.get("todos") or [])]
    if not pending:
        pending = [_normalize_todo_action({}, incident_context)]

    all_results: list[dict] = []
    seen_ids: set[str] = set()
    completed_todos: list[dict] = []
    trace: list[dict] = []
    max_iterations = 6
    action_signatures = set()
    no_growth_rounds = 0

    for iteration in range(1, max_iterations + 1):
        if not pending:
            break

        action = pending.pop(0)
        signature = json.dumps(action, sort_keys=True, default=str)
        if signature in action_signatures:
            continue
        action_signatures.add(signature)

        strategy = {
            "summary": action.get("title", ""),
            "search_queries": action.get("search_queries", []),
            "time_window": action.get("time_window") or plan.get("time_window"),
            "reasoning": action.get("goal", ""),
        }
        if not initial_strategy["search_queries"]:
            initial_strategy["search_queries"] = list(strategy.get("search_queries") or [])

        step_results = _execute_searches(
            db=db,
            logs_index=logs_index,
            strategy=strategy,
            field_docs=field_docs,
            llm=llm,
            incident_context=incident_context,
        )

        new_count = 0
        for result in step_results:
            rid = str(result.get("_id") or json.dumps(result, sort_keys=True, default=str))
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            all_results.append(result)
            new_count += 1

        completed_todos.append({
            "title": action.get("title"),
            "goal": action.get("goal"),
            "new_results": new_count,
            "total_results": len(all_results),
        })

        if new_count == 0:
            no_growth_rounds += 1
        else:
            no_growth_rounds = 0

        evaluation = _ask_llm_to_re_evaluate_progress(
            llm=llm,
            incident_question=incident_question,
            incident_context=incident_context,
            all_results=all_results,
            completed_todos=completed_todos,
            pending_todos=pending,
        )

        anchor_score = _anchor_coverage_score(all_results, incident_context)
        relevant = bool(evaluation.get("is_relevant", False)) and anchor_score >= 0.2
        sufficient = bool(evaluation.get("is_sufficient", False)) and bool(all_results) and relevant
        if not sufficient and no_growth_rounds >= 2 and relevant and anchor_score >= 0.8 and len(all_results) >= 2:
            sufficient = True

        trace.append({
            "iteration": iteration,
            "action": action,
            "new_results": new_count,
            "total_results": len(all_results),
            "anchor_coverage": round(anchor_score, 3),
            "evaluation": {
                "is_relevant": bool(evaluation.get("is_relevant", False)),
                "is_sufficient": bool(evaluation.get("is_sufficient", False)),
                "confidence": evaluation.get("confidence"),
                "reasoning": evaluation.get("reasoning", ""),
                "gaps": evaluation.get("gaps", []),
            },
            "accepted": sufficient,
        })

        logger.info(
            "[%s] Investigation iteration %d: action='%s' new=%d total=%d relevant=%s sufficient=%s anchor=%.2f",
            SKILL_NAME,
            iteration,
            action.get("title", "step"),
            new_count,
            len(all_results),
            relevant,
            sufficient,
            anchor_score,
        )

        if sufficient:
            break

        next_action = evaluation.get("next_action")
        if isinstance(next_action, dict):
            pending.insert(0, _normalize_todo_action(next_action, incident_context))

        if not pending and not relevant:
            recovery = _normalize_todo_action(
                {
                    "title": "Re-anchor to known incident entities",
                    "goal": "Recover relevance by forcing known IP/port/protocol pivots",
                    "search_queries": [
                        {
                            "description": "Anchor search on known context",
                            "keywords": incident_context.get("ips", [])[:2]
                            + incident_context.get("ports", [])[:1]
                            + incident_context.get("protocols", [])[:1],
                        }
                    ],
                },
                incident_context,
            )
            pending.append(recovery)

    return {
        "initial_strategy": initial_strategy,
        "all_results": all_results,
        "iterations_completed": len(trace),
        "trace": trace,
    }


def _execute_searches(
    db: Any,
    logs_index: str,
    strategy: dict,
    field_docs: str,
    llm: Any = None,
    incident_context: dict | None = None,
) -> list:
    """Execute searches using DISCOVERED FIELD MAPPINGS (data-agnostic).
    
    Instead of hardcoding field names, this parses the field_documentation
    to learn which fields actually exist in the data, then uses those.
    """
    results = []
    
    # Parse field documentation to discover actual field names
    field_mappings = _parse_field_mappings(field_docs)
    ip_fields = field_mappings.get("ip_fields", [])
    port_fields = field_mappings.get("port_fields", [])
    protocol_fields = field_mappings.get("protocol_fields", [])
    text_fields = field_mappings.get("all_text_fields", [])
    
    for sq in strategy.get("search_queries", []):
        keywords = list(sq.get("keywords", []))
        description = sq.get("description", "")
        
        if not keywords:
            continue

        if incident_context and not _is_relevant_search_query(sq, incident_context):
            logger.info("[%s] Skipping irrelevant search query: %s | keywords=%s", SKILL_NAME, description, keywords)
            continue

        if incident_context:
            keywords = _augment_keywords_with_context(keywords, incident_context)
        
        logger.info("[%s] Searching: %s (using %d IP fields, %d text fields)", 
                    SKILL_NAME, description, len(ip_fields), len(text_fields))
        
        should_clauses = []
        for kw in keywords:
            # Check if keyword is an IP address
            ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
            if re.match(ip_pattern, kw):
                # IP address — search discovered IP fields
                if ip_fields:
                    for field in ip_fields:
                        should_clauses.append({"term": {field: kw}})
                        should_clauses.append({"match": {field: kw}})
                else:
                    logger.warning("[%s] No IP fields discovered; skipping IP search for %s", SKILL_NAME, kw)
            else:
                # Text keyword — search discovered text fields
                if text_fields:
                    should_clauses.append({
                        "multi_match": {
                            "query": kw,
                            "fields": text_fields,
                        }
                    })
                else:
                    logger.warning("[%s] No text fields discovered; skipping text search for %s", SKILL_NAME, kw)
        
        if not should_clauses:
            logger.warning("[%s] No search clauses built for: %s", SKILL_NAME, description)
            continue

        must_clauses = [
            {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        ]
        must_clauses.extend(
            _build_hard_anchor_constraints(
                ip_fields=ip_fields,
                port_fields=port_fields,
                protocol_fields=protocol_fields,
                incident_context=incident_context,
            )
        )

        filters = []
        time_filter = _build_time_filter_from_context(incident_context, strategy)
        if time_filter:
            filters.append(time_filter)
        
        query = {
            "query": {
                "bool": {
                    "must": must_clauses,
                }
            },
            "size": 100,
        }
        if filters:
            query["query"]["bool"]["filter"] = filters
        
        try:
            search_results = db.search(logs_index, query, size=100)
            results.extend(search_results)
            logger.info("[%s] Found %d results", SKILL_NAME, len(search_results))
        except Exception as exc:
            from core.db_connector import QueryMalformedException
            
            if isinstance(exc, QueryMalformedException):
                logger.warning("[%s] Query malformed: %s — attempting intelligent repair", SKILL_NAME, exc.error_message)
                
                from core.query_repair import IntelligentQueryRepair
                repair = IntelligentQueryRepair(db, llm)  # LLM IS available, pass it through
                success, repair_results, message = repair.repair_and_retry(logs_index, exc.original_query, size=100)

                if success:
                    repaired = repair_results or []
                    results.extend(repaired)
                    logger.info("[%s] Repair successful! Got %d results", SKILL_NAME, len(repaired))
                else:
                    logger.error("[%s] Repair failed: %s", SKILL_NAME, message)
            else:
                logger.warning("[%s] Search failed: %s", SKILL_NAME, exc)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        rid = r.get("_id") or str(r)
        if rid not in seen:
            seen.add(rid)
            unique.append(r)
    
    return unique


def _is_relevant_search_query(search_query: dict, incident_context: dict) -> bool:
    """Filter LLM-proposed searches that drift away from incident scope."""
    description = (search_query.get("description") or "").lower()
    keywords = [str(k).lower() for k in search_query.get("keywords", [])]
    text = " ".join([description] + keywords)

    known_ips = {str(ip).lower() for ip in incident_context.get("ips", [])}
    query_ips = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))
    if known_ips and query_ips and query_ips.isdisjoint(known_ips):
        return False

    # Block DNS pivots unless incident context explicitly indicates DNS intent.
    if any(token in text for token in ["dns", "domain", "hostname", "fqdn"]):
        if not incident_context.get("has_dns_intent"):
            return False

    anchors = set()
    anchors.update(str(v).lower() for v in incident_context.get("ips", []))
    anchors.update(str(v).lower() for v in incident_context.get("domains", []))
    anchors.update(str(v).lower() for v in incident_context.get("ports", []))
    anchors.update(str(v).lower() for v in incident_context.get("countries", []))
    anchors.update(str(v).lower() for v in incident_context.get("protocols", []))

    if not anchors:
        return True

    # Query is relevant if at least one anchor appears in description/keywords.
    has_anchor = any(anchor and anchor in text for anchor in anchors)
    if has_anchor:
        return True

    # If incident has concrete IP/port anchors, require at least one hard anchor mention.
    if incident_context.get("ips") or incident_context.get("ports"):
        return False

    return True


def _augment_keywords_with_context(keywords: list[str], incident_context: dict) -> list[str]:
    """Augment vague LLM keywords with concrete incident anchors."""
    merged = [str(k) for k in keywords if k is not None]
    seen = set(k.lower() for k in merged)

    for value in incident_context.get("ips", [])[:5]:
        if value.lower() not in seen:
            merged.append(value)
            seen.add(value.lower())
    for value in incident_context.get("ports", [])[:5]:
        val = str(value)
        if val.lower() not in seen:
            merged.append(val)
            seen.add(val.lower())
    for value in incident_context.get("countries", [])[:3]:
        val = str(value)
        if val.lower() not in seen:
            merged.append(val)
            seen.add(val.lower())
    for value in incident_context.get("protocols", [])[:3]:
        val = str(value)
        if val.lower() not in seen:
            merged.append(val)
            seen.add(val.lower())

    return merged


def _build_hard_anchor_constraints(
    ip_fields: list[str],
    port_fields: list[str],
    protocol_fields: list[str],
    incident_context: dict | None,
) -> list[dict]:
    """Build strict must constraints from known incident anchors to avoid drift."""
    if not incident_context:
        return []

    constraints: list[dict] = []
    ips = [str(v) for v in incident_context.get("ips", [])[:3] if v]
    ports = [str(v) for v in incident_context.get("ports", [])[:3] if str(v).isdigit()]
    protocols = [str(v).lower() for v in incident_context.get("protocols", [])[:2] if v]

    if ips and ip_fields:
        ip_terms = []
        for ip in ips:
            for field in ip_fields:
                ip_terms.append({"term": {field: ip}})
        min_match = 2 if len(ips) >= 2 else 1
        constraints.append({"bool": {"should": ip_terms, "minimum_should_match": min_match}})

    if ports and port_fields:
        port_terms = []
        for port in ports:
            for field in port_fields:
                port_terms.append({"term": {field: int(port)}})
        constraints.append({"bool": {"should": port_terms, "minimum_should_match": 1}})

    if protocols and protocol_fields:
        proto_terms = []
        for proto in protocols:
            for field in protocol_fields:
                proto_terms.append({"term": {field: proto}})
        constraints.append({"bool": {"should": proto_terms, "minimum_should_match": 1}})

    return constraints


def _result_relevance_score(record: dict, incident_context: dict) -> int:
    """Score each record for alignment with known anchors."""
    text = json.dumps(record, default=str).lower()
    score = 0

    for ip in incident_context.get("ips", [])[:5]:
        if str(ip).lower() in text:
            score += 4
    for port in incident_context.get("ports", [])[:5]:
        if str(port).lower() in text:
            score += 2
    for country in incident_context.get("countries", [])[:3]:
        if str(country).lower() in text:
            score += 1
    for proto in incident_context.get("protocols", [])[:3]:
        if str(proto).lower() in text:
            score += 1

    return score


def _select_contextual_results(all_results: list, incident_context: dict, limit: int = 80) -> list:
    """Prefer records most aligned with context anchors before timeline synthesis."""
    if not all_results:
        return []

    scored = [
        (idx, _result_relevance_score(item, incident_context), item)
        for idx, item in enumerate(all_results)
    ]
    scored.sort(key=lambda row: (row[1], -row[0]), reverse=True)

    selected = [item for _, score, item in scored if score > 0][:limit]
    if selected:
        return selected
    return all_results[:limit]


def _build_time_filter_from_context(incident_context: dict | None, strategy: dict) -> dict | None:
    """Build a range filter anchored to context timestamps/time hints when available."""
    if incident_context and incident_context.get("time_range_hint"):
        return {"range": {"@timestamp": incident_context["time_range_hint"]}}

    time_window = (strategy or {}).get("time_window")
    if isinstance(time_window, str) and time_window.strip():
        text = time_window.strip()
        if " to " in text:
            parts = text.split(" to ", 1)
            return {"range": {"@timestamp": {"gte": parts[0].strip(), "lte": parts[1].strip()}}}
        if text.startswith("now-"):
            return {"range": {"@timestamp": {"gte": text}}}

    return None


def _ask_llm_for_refined_searches(
    llm: Any, incident_question: str, initial_strategy: dict,
    current_results: list, field_docs: str
) -> dict:
    """Ask LLM for refined search angles given initial results."""
    
    results_summary = f"Found {len(current_results)} logs so far"
    if current_results:
        # Summarize what we found
        sample_results = json.dumps(current_results[:3], indent=2, default=str)
        results_summary += f":\n{sample_results}"
    
    prompt = f"""You are refining a forensic search based on initial results.

INCIDENT: {incident_question}

INITIAL SEARCH STRATEGY:
{json.dumps(initial_strategy, indent=2)}

CURRENT STATUS: {results_summary}

AVAILABLE FIELDS:
{field_docs}

Design FOLLOW-UP searches to build a more complete timeline:
- What related IPs, domains, or entities should we look for?
- What time windows or patterns should we examine?
- What additional context would help build the timeline?

Output JSON:
{{
  "summary": "brief summary of refined approach",
  "search_queries": [
    {{"description": "what to find", "keywords": ["term1", "term2"]}}
  ],
  "rationale": "why these follow-ups will help complete the timeline"
}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = llm.chat(messages)
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        
        return json.loads(response)
    except Exception as exc:
        logger.warning("[%s] Failed to get refined searches: %s", SKILL_NAME, exc)
        return {}


def _ask_llm_for_comprehensive_timeline(
    llm: Any, incident_question: str, all_results: list,
    field_docs: str, instruction: str
) -> str:
    """Ask LLM to build comprehensive forensic timeline with pattern analysis."""
    
    results_text = json.dumps(all_results[:30], indent=2, default=str)
    
    prompt = f"""Build a comprehensive forensic timeline for this incident:

INCIDENT: {incident_question}

AVAILABLE FIELDS:
{field_docs}

RAW LOGS ({len(all_results)} total):
{results_text}

ANALYZE AND GENERATE:

1. TIMELINE: Chronological sequence of events with timestamps (WHEN)
   - First occurrence: When did this start?
   - Last occurrence: Most recent activity?
   - Duration: How long has this been happening?

2. ENTITIES: Key actors and systems involved (WHO/WHERE)
   - Source IPs, destinations, ports
   - Related systems (other IPs, hostnames, domains)
   - Geographic or organizational context

3. PATTERN ANALYSIS: Behavioral characteristics (HOW)
   - Frequency: Single event, isolated incidents, or recurring?
   - Pattern: Random/sporadic or regular/periodic (human vs robot)?
   - Interval: If periodic, what's the pattern? (hours, days, etc.)
   - Consistency: Does the behavior change over time?

4. ASSESSMENT: Risk and classification
   - Is this likely human activity or automated scanning?
   - Evidence for periodic vs sporadic behavior
   - Intensity trend: Increasing, decreasing, or stable?

5. CONTEXT: What does this tell us about the incident?
   - Correlation with other activities
   - Potential threat level
   - Recommended investigation priorities

Include ALL observed timestamps and provide specific examples."""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    try:
        return llm.chat(messages).strip()
    except Exception as exc:
        logger.error("[%s] Timeline generation failed: %s", SKILL_NAME, exc)
        return "Unable to generate timeline"


def _ask_llm_for_timeline(
    llm: Any, incident_question: str, search_results: list,
    field_docs: str, instruction: str
) -> str:
    """Ask LLM to build timeline from raw results."""
    
    results_text = json.dumps(search_results[:20], indent=2, default=str)
    
    prompt = f"""Build a forensic timeline for this incident:

INCIDENT: {incident_question}

AVAILABLE FIELDS:
{field_docs}

RAW LOGS:
{results_text}

GENERATE: Detailed chronological timeline showing WHEN, WHERE, WHAT, WHO, HOW.
Include timestamps, IPs, ports, protocols, and explain the sequence of events."""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    try:
        return llm.chat(messages).strip()
    except Exception as exc:
        logger.error("[%s] Timeline generation failed: %s", SKILL_NAME, exc)
        return "Unable to generate timeline"


def _ask_llm_for_timeline_no_results(
    llm: Any, incident_question: str, strategy: dict,
    field_docs: str, instruction: str
) -> str:
    """Generate analytical response when no results found.
    
    Even with zero results, provide valuable forensic insights about:
    - Why logs might be missing
    - What the absence of logs tells us
    - Suggested next investigation steps
    """
    
    prompt = f"""No logs were found for this forensic investigation. Provide DETAILED ANALYSIS.

INCIDENT: {incident_question}

SEARCH STRATEGY ATTEMPTED:
{json.dumps(strategy, indent=2)}

AVAILABLE FIELDS:
{field_docs}

TASK: Generate comprehensive forensic analysis explaining WHY this is significant:

1. IMPLICATIONS OF ABSENCE
   - What does zero results suggest about this incident?
   - Could activity be hidden or obfuscated?
   - Are we searching the right indices/timeframes?

2. ANALYSIS GAPS
   - What search angles were tried?
   - What field combinations could be refined?
   - Are there alternative ways to detect this activity?

3. ROOT CAUSE EXPLORATION
   - Logs may have been rotated or deleted
   - Activity may predate current log collection
   - Search terms may not match data format
   - Activity may be in different network segment
   - Firewall/IDS may have blocked activity before logging
   - Logs may be in different index

4. INVESTIGATION RECOMMENDATIONS
   - Check log retention policies
   - Verify field/index names match data schema
   - Expand time Windows
   - Look for related activity (different IPs, protocols, ports)
   - Check firewall/proxy logs for blocked traffic
   - Interview network administrators about this incident

5. RISK ASSESSMENT
   - Even absence of logs can indicate compromise
   - May indicate sophisticated threat hiding tracks
   - Could show defensive actions (filtering/blocking)

Provide specific, actionable next steps."""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    try:
        return llm.chat(messages).strip()
    except Exception as exc:
        logger.error("[%s] No-results analysis failed: %s", SKILL_NAME, exc)
        return (
            "No logs found for this incident. This could indicate:\n"
            "- The activity occurred outside the log collection window\n"
            "- Traffic was blocked and not logged\n"
            "- Search terms don't match the data format\n"
            "Recommend: Verify log retention, check firewall logs, and expand search parameters."
        )
