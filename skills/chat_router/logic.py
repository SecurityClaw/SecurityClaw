"""
skills/chat_router/logic.py

Intelligent skill router for conversational SOC queries.
Routes user questions to appropriate skills, handles multi-skill workflows,
and maintains conversation context.

This is not a periodic skill—it's invoked interactively via the chat command.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
ROUTING_PROMPT_PATH = Path(__file__).parent / "ROUTING_PROMPT.md"
SUPERVISOR_NEXT_ACTION_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_NEXT_ACTION_PROMPT.md"
SUPERVISOR_EVALUATION_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_EVALUATION_PROMPT.md"
RESPONSE_THINK_PROMPT_PATH = Path(__file__).parent / "RESPONSE_THINK_PROMPT.md"
RESPONSE_REFLECTION_PROMPT_PATH = Path(__file__).parent / "RESPONSE_REFLECTION_PROMPT.md"
RESPONSE_VERIFICATION_PROMPT_PATH = Path(__file__).parent / "RESPONSE_VERIFICATION_PROMPT.md"
RESPONSE_FINAL_PROMPT_PATH = Path(__file__).parent / "RESPONSE_FINAL_PROMPT.md"
SKILL_NAME = "chat_router"


def _load_prompt_template(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("[%s] Could not load prompt template %s: %s", SKILL_NAME, path.name, exc)
        return fallback.strip()


def _render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def _question_asks_for_reputation(user_question: str) -> bool:
    question_lower = user_question.lower()
    return any(
        term in question_lower
        for term in ["reputation", "threat intel", "threat intelligence", "risk", "malicious", "verdict", "score"]
    )


def _strip_unrequested_threat_intel(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
) -> list[str]:
    """Remove threat_analyst unless the user explicitly asked for threat or reputation analysis."""
    available = {s.get("name") for s in available_skills}
    if "threat_analyst" not in available or "threat_analyst" not in selected_skills:
        return selected_skills
    if _question_asks_for_reputation(user_question):
        return selected_skills
    filtered = [skill for skill in selected_skills if skill != "threat_analyst"]
    if filtered != selected_skills:
        logger.info(
            "[%s] Removed threat_analyst because the question did not ask for reputation or threat analysis",
            SKILL_NAME,
        )
    return filtered


def _question_excludes_private_ips(user_question: str) -> bool:
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


def _question_has_explicit_entities(user_question: str) -> bool:
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question):
        return True
    return bool(re.search(r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", user_question.lower()))


def _is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def route_question(
    user_question: str,
    available_skills: list[dict],
    llm: Any,
    instruction: str,
    conversation_history: list[dict] = None,
) -> dict:
    """
    Analyze user question and decide which skill(s) to invoke.
    
    Args:
        user_question: Current user input
        available_skills: List of skill definitions
        llm: LLM provider
        instruction: System instruction
        conversation_history: Prior Q&A turns for context (optional)
    
    Returns dict with:
      - reasoning: Why this skill was chosen
      - skills: List of skill names to invoke (can be multiple for workflows)
      - parameters: Parameters to pass to skills (includes the question)
    """
    skills_description = "\n".join([
        f"- {s['name']}: {s['description']}"
        for s in available_skills
    ])

    # Build conversation context if history is provided
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                history_lines.append(f"User: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                history_lines.append(f"Agent: {msg.get('content', '')}")
        if history_lines:
            history_context = "\n\nRECENT CONVERSATION HISTORY (for context):\n" + "\n".join(history_lines)

    routing_template = _load_prompt_template(
        ROUTING_PROMPT_PATH,
        """
Analyze this security question and decide which available skills to use.
Consider the recent conversation history to maintain context.

Current Question: "{{USER_QUESTION}}"{{HISTORY_CONTEXT}}

Available skills:
{{SKILLS_DESCRIPTION}}

Return a strict JSON object with reasoning, skills, and parameters.question.
""",
    )
    prompt = _render_prompt(
        routing_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_CONTEXT": history_context,
            "SKILLS_DESCRIPTION": skills_description,
        },
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    response = llm.chat(messages)
    
    try:
        result = json.loads(response)
        # Ensure parameters has the question
        if "parameters" not in result:
            result["parameters"] = {}
        if "question" not in result["parameters"]:
            result["parameters"]["question"] = user_question

        contextual_reputation_entities = _recover_threat_followup_entities(
            user_question,
            conversation_history,
        )
        if contextual_reputation_entities and any(
            s.get("name") == "threat_analyst" for s in available_skills
        ):
            result["skills"] = ["threat_analyst"]
            result["reasoning"] = "Follow-up reputation question anchored to entities from the previous answer"
            result["parameters"]["question"] = _build_context_aware_threat_question(
                user_question,
                contextual_reputation_entities,
            )
            if conversation_history:
                result["parameters"]["conversation_history"] = conversation_history
            return result

        # PREVENTION: Remove geoip_lookup if it's the first/only skill and no specific IPs are in the question
        skills = result.get("skills", [])
        if skills and skills[0] == "geoip_lookup" and not _contains_specific_ip(user_question):
            logger.warning(
                "[%s] Removing geoip_lookup as first skill because question doesn't mention specific IPs: %s",
                SKILL_NAME,
                user_question,
            )
            skills = [s for s in skills if s != "geoip_lookup"]
            result["skills"] = skills
        result["skills"] = skills

        # Enforce explicit forensic intent routing before generic filtering.
        result["skills"] = _apply_forensic_intent_override(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
        )
        
        # Prepend field discovery when asking about specific data types (alerts, signatures, etc.)
        result["skills"] = _prepend_field_discovery_for_data_types(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
            current_results={},
        )
        result["skills"] = _prefer_field_discovery_for_natural_language_search(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
            current_results={},
        )
        result["skills"] = _enforce_evidence_then_threat_intel(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
        )
        result["skills"] = _strip_unrequested_threat_intel(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
        )
        
        # Include conversation history in parameters for skills that need context
        if conversation_history:
            result["parameters"]["conversation_history"] = conversation_history
        
        # Filter out network_baseliner if not explicitly requested
        result["skills"] = _filter_explicit_only_skills(
            result.get("skills", []),
            user_question,
        )
        return result
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse LLM routing response: %s", SKILL_NAME, response)
        # Fallback: try to extract JSON from response
        try:
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if "parameters" not in result:
                    result["parameters"] = {}
                if "question" not in result["parameters"]:
                    result["parameters"]["question"] = user_question
                return result
        except:
            pass
        
        # If all else fails, return no skills
        return {
            "reasoning": "Unable to determine relevant skill",
            "skills": [],
            "parameters": {"question": user_question},
        }


def _prepend_field_discovery_for_data_types(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
) -> list[str]:
    """Prepend fields_querier when asking about specific data types (alerts, events, signatures, etc.).

    Behavior:
    - Question mentions "alerts", "signatures", "events" → prepend fields_querier
    - This ensures we discover which fields hold that data before querying
    - Avoids malformed queries like searching "ET EXPLOIT" across wrong field types
    """
    available = {s.get("name") for s in available_skills}
    if "fields_querier" not in available:
        return selected_skills

    question_lower = user_question.lower().strip()
    current_results = current_results or {}
    
    # Keywords indicating query is about a specific data type
    data_type_keywords = [
        "alerts", "alert", "signature", "signatures", "et rules", "et exploit",
        "suricata", "snort", "rule", "rules", "events", "event type",
        "protocol mismatch", "dns queries", "tls certificates", "http requests",
    ]
    search_intent_keywords = [
        "show me", "show", "find", "list", "any", "how many", "search",
        "check", "check for", "look for", "get", "pull", "display",
        "which ip", "what ip", "their ip", "their ips", "source ip", "destination ip",
    ]
    field_schema_only_keywords = [
        "what fields", "which field", "field name", "field names", "schema",
        "available fields", "what field holds", "which field stores",
    ]
    
    asks_about_data_type = any(kw in question_lower for kw in data_type_keywords)
    asks_for_search = any(kw in question_lower for kw in search_intent_keywords)
    asks_only_for_schema = any(kw in question_lower for kw in field_schema_only_keywords)
    fields_result = current_results.get("fields_querier") or {}
    field_discovery_already_done = bool(
        isinstance(fields_result, dict)
        and (
            fields_result.get("field_mappings")
            or (fields_result.get("findings") or {}).get("field_mappings")
        )
    )

    if asks_about_data_type:
        if asks_only_for_schema:
            return ["fields_querier"] + [s for s in selected_skills if s != "fields_querier"]

        if field_discovery_already_done and "opensearch_querier" in available:
            logger.info(
                "[%s] Field discovery already available for data-type question — promoting opensearch_querier",
                SKILL_NAME,
            )
            ordered = [s for s in selected_skills if s not in {"fields_querier", "opensearch_querier"}]
            return ["opensearch_querier"] + ordered

        # First discover what fields hold this data, then search
        logger.info(
            "[%s] Question asks about specific data type — prepending fields_querier for field discovery",
            SKILL_NAME,
        )
        # Remove opensearch_querier if it was auto-selected, we'll add it after fields_querier
        filtered = [s for s in selected_skills if s != "opensearch_querier"]
        # Prepend fields_querier only once
        ordered = ["fields_querier"] + [s for s in filtered if s != "fields_querier"]
        # If we removed opensearch_querier, add it back after fields_querier
        if "opensearch_querier" in available and "opensearch_querier" not in ordered and asks_for_search:
            ordered.append("opensearch_querier")
        return ordered

    return selected_skills


def _prefer_field_discovery_for_natural_language_search(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
) -> list[str]:
    """Prepend fields_querier for natural-language searches unless explicit fields are named."""
    available = {s.get("name") for s in available_skills}
    if "fields_querier" not in available or "opensearch_querier" not in available:
        return selected_skills

    question_lower = user_question.lower().strip()
    current_results = current_results or {}

    asks_for_log_search = bool(
        re.search(r"\b(traffic|flow|flows|connection|connections|log|logs|event|events|port|ports|protocol|country|countries|ip|ips|host|hosts)\b", question_lower)
        and re.search(r"\b(show|find|search|check|list|get|what|which|when|who|where|display|pull|visited|visit|seen|look for)\b", question_lower)
    )
    explicit_field_reference = bool(
        re.search(r"(?:^|\s)(@timestamp|[a-z_][a-z0-9_]*\.[a-z0-9_.]+)(?:\s|$|=|:)", user_question)
        or re.search(
            r"\b(src_ip|dest_ip|source_ip|destination_ip|src_port|dst_port|dest_port|destination_port|source\.ip|destination\.ip|destination\.port|geoip\.[a-z0-9_.]+|alert\.[a-z0-9_.]+)\b",
            question_lower,
        )
    )

    fields_result = current_results.get("fields_querier") or {}
    field_discovery_already_done = bool(
        isinstance(fields_result, dict)
        and (
            fields_result.get("field_mappings")
            or (fields_result.get("findings") or {}).get("field_mappings")
        )
    )

    if not asks_for_log_search or explicit_field_reference:
        return selected_skills

    if field_discovery_already_done:
        ordered = [s for s in selected_skills if s != "fields_querier"]
        if "opensearch_querier" not in ordered:
            ordered.insert(0, "opensearch_querier")
        return ordered

    logger.info(
        "[%s] Natural-language log search detected — prepending fields_querier before opensearch_querier",
        SKILL_NAME,
    )
    ordered = ["fields_querier"] + [s for s in selected_skills if s not in {"fields_querier", "opensearch_querier"}]
    ordered.insert(1, "opensearch_querier")
    return ordered


def _enforce_evidence_then_threat_intel(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
) -> list[str]:
    """Force evidence gathering before threat intel for concrete alert-detail questions."""
    available = {s.get("name") for s in available_skills}
    if "opensearch_querier" not in available:
        return selected_skills

    question_lower = user_question.lower().strip()
    current_results = current_results or {}

    asks_for_threat = any(
        term in question_lower
        for term in ["threat intel", "threat intelligence", "reputation", "risk", "malicious"]
    )
    asks_for_concrete_alert_details = any(
        term in question_lower
        for term in [
            "what ip", "which ip", "source ip", "destination ip", "src ip", "dst ip",
            "when did", "timestamp", "time did", "what time", "when was",
            "this alert", "that alert", "alert happen", "alert occurred",
        ]
    )
    mentions_alert_entity = any(
        term in question_lower
        for term in ["alert", "signature", "et ", "suricata", "snort", "rule"]
    )

    if not ((asks_for_concrete_alert_details and mentions_alert_entity) or (asks_for_threat and asks_for_concrete_alert_details)):
        return selected_skills

    ordered: list[str] = []
    if "fields_querier" in selected_skills:
        ordered.append("fields_querier")

    opensearch_already_has_results = bool(
        isinstance(current_results.get("opensearch_querier"), dict)
        and (
            current_results["opensearch_querier"].get("results_count")
            or current_results["opensearch_querier"].get("results")
        )
    )
    if not opensearch_already_has_results:
        ordered.append("opensearch_querier")

    for skill in selected_skills:
        if skill not in ordered and skill != "opensearch_querier":
            ordered.append(skill)

    if asks_for_threat and "threat_analyst" in available and "threat_analyst" not in ordered:
        ordered.append("threat_analyst")

    if ordered != selected_skills:
        logger.info(
            "[%s] Enforced evidence-first routing for detailed alert/threat follow-up: %s -> %s",
            SKILL_NAME,
            selected_skills,
            ordered,
        )

    return ordered


def _apply_forensic_intent_override(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
) -> list[str]:
    """Prioritize forensic_examiner for explicit forensic/timeline intent.

    Behavior:
    - Explicit forensic wording + no concrete search filters -> forensic_examiner only
    - Explicit forensic wording + concrete traffic/log filters -> opensearch_querier then forensic_examiner
    - Otherwise keep model-selected skills unchanged
    """
    available = {s.get("name") for s in available_skills}
    if "forensic_examiner" not in available:
        return selected_skills

    question_lower = user_question.lower().strip()
    forensic_intent = any(
        phrase in question_lower
        for phrase in [
            "forensic",
            "timeline",
            "incident reconstruction",
            "reconstruct",
            "investigate incident",
            "forensic analysis",
        ]
    ) or bool(re.search(r"\binvestigat(?:e|ion)\b", question_lower))

    if not forensic_intent:
        return selected_skills

    has_search_filters = bool(
        re.search(r"\b(traffic|flow|connection|log|logs|port|protocol|from|to|country|ip|domain)\b", question_lower)
        or re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", question_lower)
    )

    if has_search_filters:
        # Direct log search with filters → opensearch_querier (not baseline_querier)
        ordered = ["opensearch_querier", "forensic_examiner"]
        return [s for s in ordered if s in available]

    return ["forensic_examiner"]


def _contains_specific_ip(user_question: str) -> bool:
    """
    Check if the question mentions a specific IP address or hostname.
    
    Returns True if the question contains:
    - IPv4 addresses (e.g., 8.8.8.8, 192.168.1.1)
    - IPv6 addresses
    - Domain names/hostnames
    
    Returns False for general location queries like "traffic from Russia".
    """
    import re
    question_lower = user_question.lower()
    
    # IP address patterns
    ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ipv6_pattern = r'(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}'
    
    # Check for IP addresses
    if re.search(ipv4_pattern, question_lower):
        return True
    if re.search(ipv6_pattern, question_lower):
        return True
    
    # Check for explicit hostname/domain mentions with context words
    # (e.g., "hostname xxx.com" or "domain google.com")
    if re.search(r'\b(?:hostname|domain|fqdn|host)\s+[\w\-\.]+', question_lower):
        return True
    
    # Check for IP-related keywords suggesting specific IPs are being discussed
    if re.search(r'\bgeolocate\s+\S+', question_lower) or re.search(r'\bwhere\s+is\s+\S+', question_lower):
        return True
    
    return False


def _filter_explicit_only_skills(skills: list[str], user_question: str) -> list[str]:
    """
    Filter out skills that are explicit-only or reserved for follow-up usage.

    network_baseliner is explicit-only: user must say:
      - "network_baseliner" / "baseliner" / "create baseline"

    fields_baseliner is explicit-only: user must say:
      - "fields_baseliner" / "scan fields" / "catalog fields"

    baseline_querier is reserved for FOLLOW-UP ONLY: only route if:
      - User explicitly asks "compare to baseline", "analyze baseline", "baseline analysis"
      - Or supervisor explicitly requests it for follow-up research
      For initial answering, use opensearch_querier instead.
    """
    filtered = []
    question_lower = user_question.lower()

    # Keywords that explicitly invoke network_baseliner
    baseliner_keywords = [
        "network_baseliner", "baseliner", "create baseline", "generate baseline",
        "build baseline", "refresh baseline", "force_refresh", "create a baseline",
        "generate a baseline", "build a baseline", "create new baseline", "generate new baseline",
    ]
    baseliner_requested = any(kw in question_lower for kw in baseliner_keywords)

    # Keywords that explicitly invoke fields_baseliner
    fields_baseliner_keywords = [
        "fields_baseliner", "scan fields", "catalog fields", "index fields",
        "refresh field schema", "update field schema", "rebuild field catalog",
    ]
    fields_baseliner_requested = any(kw in question_lower for kw in fields_baseliner_keywords)

    # Keywords for baseline_querier follow-up research
    baseline_research_keywords = [
        "compare to baseline", "baseline comparison", "baseline analysis", "analyze baseline",
        "research baseline", "baseline investigation", "what's normal", "normal behavior",
    ]
    baseline_research_requested = any(kw in question_lower for kw in baseline_research_keywords)

    for skill in skills:
        if skill == "network_baseliner" and not baseliner_requested:
            logger.info("[%s] Blocked auto-routing to network_baseliner.", SKILL_NAME)
            if "opensearch_querier" not in filtered:
                filtered.append("opensearch_querier")
        elif skill == "fields_baseliner" and not fields_baseliner_requested:
            logger.info("[%s] Blocked auto-routing to fields_baseliner.", SKILL_NAME)
        elif skill == "baseline_querier" and not baseline_research_requested:
            # baseline_querier is for follow-up research only, not initial questions
            logger.info("[%s] Blocked auto-routing to baseline_querier (follow-up only).", SKILL_NAME)
            if "opensearch_querier" not in filtered:
                filtered.append("opensearch_querier")
        else:
            filtered.append(skill)

    return filtered


def execute_skill_workflow(
    skills: list[str],
    runner: Any,
    context: dict,
    routing_decision: dict,
    conversation_history: list[dict] = None,
    aggregated_results: dict = None,
) -> dict:
    """
    Execute one or more skills in sequence, passing context between them.
    
    Args:
        skills: List of skill names to execute
        runner: Runner instance
        context: Shared context dict
        routing_decision: Dict with 'parameters' key for skill inputs
        conversation_history: Conversation history for context (optional)
        aggregated_results: Results from previous skills, for context enrichment (optional)
    
    Returns dict with results from each skill execution.
    """
    results = {}
    params = routing_decision.get("parameters", {})
    aggregated_results = aggregated_results or {}
    
    for skill_name in skills:
        logger.info("[%s] Executing skill: %s", SKILL_NAME, skill_name)
        
        try:
            # Build context with parameters
            skill_context = runner._build_context()
            skill_context["parameters"] = params.copy()
            
            # Pass conversation history for context-aware skills
            if conversation_history:
                skill_context["conversation_history"] = conversation_history
            
            # ── CONTEXT ENRICHMENT: Pass previous results to this skill ──────
            # This allows skills to see what was discovered in prior steps.
            combined_previous_results = {**aggregated_results, **results}
            if combined_previous_results:
                skill_context["previous_results"] = combined_previous_results
            
            # ── SPECIAL HANDLING: Enrich threat_analyst question with discovered entities ──
            if skill_name == "threat_analyst" and combined_previous_results:
                original_q = skill_context["parameters"].get("question", "")
                entities = _recover_threat_followup_entities(
                    original_q,
                    conversation_history,
                    combined_previous_results,
                )
                if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
                    enriched_q = _build_context_aware_threat_question(original_q, entities)
                    skill_context["parameters"]["question"] = enriched_q
                    logger.info("[%s] Enriched threat_analyst question with discovered entities", SKILL_NAME)
            elif skill_name == "threat_analyst" and conversation_history:
                original_q = skill_context["parameters"].get("question", "")
                entities = _recover_threat_followup_entities(
                    original_q,
                    conversation_history,
                )
                if entities:
                    enriched_q = _build_context_aware_threat_question(original_q, entities)
                    skill_context["parameters"]["question"] = enriched_q
                    logger.info("[%s] Anchored threat_analyst to conversation-history entities", SKILL_NAME)
            
            # Dispatch skill with context
            result = runner.dispatch(skill_name, context=skill_context)
            results[skill_name] = result
            logger.info("[%s] Skill %s completed with status: %s", 
                       SKILL_NAME, skill_name, result.get("status"))

            # Auto-chain threat_analyst after forensic results to include reputation context.
            if (
                skill_name == "forensic_examiner"
                and "threat_analyst" not in skills
                and "threat_analyst" not in results
                and result.get("status") == "ok"
            ):
                threat_question = _build_threat_followup_question(result)
                if threat_question:
                    try:
                        logger.info("[%s] Auto-chaining skill: threat_analyst", SKILL_NAME)
                        threat_context = runner._build_context()
                        threat_params = dict(params)
                        threat_params["question"] = threat_question
                        threat_context["parameters"] = threat_params
                        if conversation_history:
                            threat_context["conversation_history"] = conversation_history

                        threat_result = runner.dispatch("threat_analyst", context=threat_context)
                        results["threat_analyst"] = threat_result
                        logger.info(
                            "[%s] Skill threat_analyst completed with status: %s",
                            SKILL_NAME,
                            threat_result.get("status"),
                        )
                    except Exception as threat_exc:
                        logger.error("[%s] Auto-chained threat_analyst failed: %s", SKILL_NAME, threat_exc)
                        results["threat_analyst"] = {
                            "status": "error",
                            "error": str(threat_exc),
                        }
        except Exception as e:
            logger.error("[%s] Skill %s failed: %s", SKILL_NAME, skill_name, e)
            results[skill_name] = {
                "status": "error",
                "error": str(e),
            }
    
    return results


def _apply_result_aware_recovery(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
) -> list[str]:
    """Promote recovery skills when prior results show unmet needs."""
    current_results = current_results or {}
    available = {s.get("name") for s in available_skills}
    ordered = list(selected_skills)
    question_lower = user_question.lower()

    os_result = current_results.get("opensearch_querier") or {}
    os_issue = " ".join(
        part for part in [
            str(os_result.get("validation_issue", "")),
            str((os_result.get("reasoning_chain") or {}).get("validation_issue", "")),
            str((os_result.get("reasoning_chain") or {}).get("validation_reflection", "")),
        ] if part
    ).lower()
    os_validation_failed = bool(os_result.get("validation_failed"))

    asks_for_country = any(
        token in question_lower
        for token in ["country", "countries", "origin", "origins", "where from", "geolocation", "geoip"]
    )
    asks_for_reputation = any(
        token in question_lower
        for token in ["reputation", "threat intel", "threat intelligence", "risk", "malicious", "verdict", "score"]
    )
    missing_fields_or_schema = any(
        token in os_issue
        for token in [
            "missing field",
            "required fields",
            "field information",
            "country information",
            "port information",
            "do not contain the required fields",
        ]
    )

    entities = _extract_entities_from_previous_results(current_results) if current_results else {}
    has_ips = bool((entities or {}).get("ips"))

    if os_validation_failed and missing_fields_or_schema:
        if "fields_querier" in available and "fields_querier" not in current_results:
            ordered = ["fields_querier"] + [s for s in ordered if s != "fields_querier"]
            if "opensearch_querier" in available and "opensearch_querier" not in ordered:
                ordered.append("opensearch_querier")
            logger.info(
                "[%s] Added fields_querier recovery after opensearch validation reported missing fields/schema",
                SKILL_NAME,
            )

    if asks_for_country and has_ips:
        if "geoip_lookup" in available and "geoip_lookup" not in ordered and "geoip_lookup" not in current_results:
            ordered.append("geoip_lookup")
            logger.info(
                "[%s] Added geoip_lookup recovery because countries were requested and IPs are available",
                SKILL_NAME,
            )

    if asks_for_reputation:
        threat_result = current_results.get("threat_analyst") or {}
        has_threat_intel = threat_result.get("status") == "ok"
        if "threat_analyst" in available and not has_threat_intel and "threat_analyst" not in ordered:
            ordered.append("threat_analyst")

    deduped: list[str] = []
    for skill in ordered:
        if skill and skill not in deduped:
            deduped.append(skill)
    return deduped


def orchestrate_with_supervisor(
    user_question: str,
    available_skills: list[dict],
    runner: Any,
    llm: Any,
    instruction: str,
    cfg: Any = None,
    conversation_history: list[dict] | None = None,
    step_callback: Any = None,
) -> dict:
    """Iterative LLM supervisor loop: decide skills, run, evaluate, repeat until satisfied.

    Args:
        step_callback: Optional callable(event, data, step, max_steps) for real-time display.
            event is one of: "deciding", "running", "evaluated"
    """
    max_steps = 4
    if cfg:
        max_steps = int(cfg.get("chat", "supervisor_max_steps", default=4) or 4)
    max_steps = max(1, min(max_steps, 8))

    conversation_history = conversation_history or []
    trace: list[dict] = []
    aggregated_results: dict = {}
    last_eval = {
        "satisfied": False,
        "reasoning": "No evaluation yet",
        "confidence": 0.0,
        "missing": ["No skills executed yet"],
    }
    # Track which skills were run in previous steps to detect unhelpful repetition.
    previously_run_skills: list[list[str]] = []

    for step in range(1, max_steps + 1):
        decision = _supervisor_next_action(
            user_question=user_question,
            available_skills=available_skills,
            llm=llm,
            instruction=instruction,
            conversation_history=conversation_history,
            previous_trace=trace,
            current_results=aggregated_results,
            previous_eval=last_eval,
        )

        # Normalize skills with existing router guards.
        selected = _apply_forensic_intent_override(
            user_question=user_question,
            selected_skills=decision.get("skills", []),
            available_skills=available_skills,
        )
        selected = _filter_explicit_only_skills(selected, user_question)
        selected = _enforce_evidence_then_threat_intel(
            user_question=user_question,
            selected_skills=selected,
            available_skills=available_skills,
            current_results=aggregated_results,
        )
        selected = _apply_result_aware_recovery(
            user_question=user_question,
            selected_skills=selected,
            available_skills=available_skills,
            current_results=aggregated_results,
        )
        decision["skills"] = selected

        # ── Real-time callback: supervisor has decided ─────────────────────
        if step_callback:
            step_callback("deciding", decision, step, max_steps)

        # Anti-repeat guard: if same skills chosen again and we already have
        # results from them, try a recovery plan and otherwise refuse duplicates.
        if selected and selected in previously_run_skills and aggregated_results:
            improved = _enforce_evidence_then_threat_intel(
                user_question=user_question,
                selected_skills=selected,
                available_skills=available_skills,
                current_results=aggregated_results,
            )
            improved = _apply_result_aware_recovery(
                user_question=user_question,
                selected_skills=improved,
                available_skills=available_skills,
                current_results=aggregated_results,
            )
            if improved != selected and improved not in previously_run_skills:
                logger.info(
                    "[%s] Supervisor repeated bad plan %s on step %d — upgrading to %s instead of finalizing",
                    SKILL_NAME, selected, step, improved,
                )
                selected = improved
                decision["skills"] = selected
            else:
                logger.info(
                    "[%s] Supervisor repeated identical skill set %s on step %d — blocking duplicate execution",
                    SKILL_NAME, selected, step,
                )
                last_eval = {
                    "satisfied": False,
                    "confidence": 0.4,
                    "reasoning": "Supervisor proposed the same unsuccessful skills again; duplicate execution was blocked.",
                    "missing": last_eval.get("missing", []) if isinstance(last_eval, dict) else [],
                }
                trace.append({
                    "step": step,
                    "decision": decision,
                    "selected_skills": selected,
                    "step_result_keys": [],
                    "evaluation": last_eval,
                })
                if step_callback:
                    step_callback("evaluated", last_eval, step, max_steps)
                break

        step_results = {}
        if selected:
            if step_callback:
                step_callback("running", {"skills": selected}, step, max_steps)
            step_results = execute_skill_workflow(
                selected,
                runner,
                {},
                decision,
                conversation_history=conversation_history,
                aggregated_results=aggregated_results,
            )
            aggregated_results.update(step_results)
            previously_run_skills.append(selected)

        last_eval = _supervisor_evaluate_satisfaction(
            user_question=user_question,
            llm=llm,
            instruction=instruction,
            conversation_history=conversation_history,
            skill_results=aggregated_results,
            step=step,
            max_steps=max_steps,
        )

        trace.append(
            {
                "step": step,
                "decision": decision,
                "selected_skills": selected,
                "step_result_keys": list(step_results.keys()),
                "evaluation": last_eval,
            }
        )

        # ── Real-time callback: evaluation complete ────────────────────────
        if step_callback:
            step_callback("evaluated", last_eval, step, max_steps)

        logger.info(
            "[%s] Supervisor step %d/%d | skills=%s | satisfied=%s (%.2f)",
            SKILL_NAME,
            step,
            max_steps,
            selected,
            bool(last_eval.get("satisfied", False)),
            float(last_eval.get("confidence", 0.0) or 0.0),
        )

        if last_eval.get("satisfied", False):
            break

        if not selected and step > 1:
            # No actionable next step from supervisor.
            break

    if not trace:
        # Absolute fallback to single-pass routing.
        routing = route_question(
            user_question,
            available_skills,
            llm,
            instruction,
            conversation_history,
        )
        if routing.get("skills"):
            aggregated_results = execute_skill_workflow(
                routing["skills"],
                runner,
                {},
                routing,
                conversation_history=conversation_history,
                aggregated_results=aggregated_results,
            )
        trace = [{"step": 1, "decision": routing, "selected_skills": routing.get("skills", []), "evaluation": last_eval}]

    final_routing = {
        "reasoning": last_eval.get("reasoning", "Supervisor completed orchestration."),
        "skills": list(aggregated_results.keys()),
        "parameters": {"question": user_question},
    }
    response = format_response(user_question, final_routing, aggregated_results, llm, cfg, 
                             available_skills=available_skills)

    return {
        "response": response,
        "routing": final_routing,
        "skill_results": aggregated_results,
        "trace": trace,
        "evaluation": last_eval,
    }


def _supervisor_next_action(
    user_question: str,
    available_skills: list[dict],
    llm: Any,
    instruction: str,
    conversation_history: list[dict],
    previous_trace: list[dict],
    current_results: dict,
    previous_eval: dict,
) -> dict:
    """Ask LLM supervisor what skill(s) to run next.
    
    Uses skill manifests for intelligent routing when available,
    enabling modular skill discovery and auto-adaptation.
    """
    # Try to load skill manifests for structured capability awareness
    manifest_context = ""
    try:
        from core.skill_manifest import SkillManifestLoader
        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        if manifests:
            manifest_context = "\n" + loader.build_supervisor_context(manifests)
            logger.debug("[%s] Loaded %d skill manifests for intelligent routing", SKILL_NAME, len(manifests))
    except Exception as e:
        logger.debug("[%s] Manifest loading failed, falling back to skill descriptions: %s", SKILL_NAME, e)
    
    skills_description = "\n".join(
        f"- {s.get('name')}: {s.get('description', '')}"
        for s in available_skills
    )
    history_text = "\n".join(
        f"- {m.get('role', '?')}: {str(m.get('content', ''))[:220]}"
        for m in conversation_history[-6:]
    )
    prior_steps = json.dumps(previous_trace[-3:], indent=2, default=str) if previous_trace else "[]"
    result_keys = list(current_results.keys())

    # Summarize what each result returned so the supervisor can make intelligent choices.
    result_summary_lines = []
    for skill_name, result in current_results.items():
        count = result.get("results_count") or result.get("log_records") or (
            len(result.get("results", [])) if isinstance(result.get("results"), list) else 0
        )
        status = result.get("status", "?")
        result_summary_lines.append(f"  {skill_name}: status={status}, records_found={count}")
    result_summary = "\n".join(result_summary_lines) or "  (no skills have run yet)"

    next_action_template = _load_prompt_template(
        SUPERVISOR_NEXT_ACTION_PROMPT_PATH,
        """
You are the SOC supervisor orchestrator.

QUESTION:
{{USER_QUESTION}}

AVAILABLE SKILLS:
{{SKILLS_DESCRIPTION}}{{MANIFEST_CONTEXT}}

PRIOR EXECUTION TRACE:
{{PRIOR_STEPS}}

RESULTS ALREADY GATHERED:
{{RESULT_SUMMARY}}

PREVIOUS EVALUATION:
{{PREVIOUS_EVALUATION}}

Return strict JSON with reasoning, skills, and parameters.question.
""",
    )
    prompt = _render_prompt(
        next_action_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_TEXT": history_text or "- none",
            "SKILLS_DESCRIPTION": skills_description,
            "MANIFEST_CONTEXT": manifest_context,
            "PRIOR_STEPS": prior_steps,
            "RESULT_SUMMARY": result_summary,
            "PREVIOUS_EVALUATION": json.dumps(previous_eval, indent=2, default=str),
        },
    )

    try:
        response = llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ])
        parsed = _parse_json_object(response) or {}
        if "parameters" not in parsed:
            parsed["parameters"] = {}
        parsed["parameters"]["question"] = user_question
        if conversation_history:
            parsed["parameters"]["conversation_history"] = conversation_history
        if not isinstance(parsed.get("skills"), list):
            parsed["skills"] = []
        if not isinstance(parsed.get("reasoning"), str):
            parsed["reasoning"] = "Supervisor selected next action"

        contextual_reputation_entities = _recover_threat_followup_entities(
            user_question,
            conversation_history,
            current_results,
        )
        if contextual_reputation_entities and any(
            s.get("name") == "threat_analyst" for s in available_skills
        ):
            parsed["skills"] = ["threat_analyst"]
            parsed["reasoning"] = "Follow-up reputation question anchored to entities from the previous answer"
            parsed["parameters"]["question"] = _build_context_aware_threat_question(
                user_question,
                contextual_reputation_entities,
            )
            if conversation_history:
                parsed["parameters"]["conversation_history"] = conversation_history
            return parsed
        
        # Apply field discovery prepending for data type queries (same as route_question)
        parsed["skills"] = _prepend_field_discovery_for_data_types(
            user_question=user_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
            current_results=current_results,
        )
        parsed["skills"] = _prefer_field_discovery_for_natural_language_search(
            user_question=user_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
            current_results=current_results,
        )
        parsed["skills"] = _enforce_evidence_then_threat_intel(
            user_question=user_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
            current_results=current_results,
        )
        parsed["skills"] = _apply_result_aware_recovery(
            user_question=user_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
            current_results=current_results,
        )
        parsed["skills"] = _strip_unrequested_threat_intel(
            user_question=user_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
        )
        
        # Additional rule: If opensearch_querier found results and question asks for reputation,
        # auto-queue threat_analyst for next step
        question_lower = user_question.lower()
        asks_for_reputation = any(
            term in question_lower
            for term in ["reputation", "threat intel", "threat intelligence", "malicious", "risk", "dangerous", "score", "verdict"]
        )
        opensearch_has_results = bool(
            current_results.get("opensearch_querier") and
            current_results["opensearch_querier"].get("results_count", 0) > 0
        )
        has_threat_intel = bool(
            current_results.get("threat_analyst") and
            current_results["threat_analyst"].get("status") == "ok"
        )
        
        if asks_for_reputation and opensearch_has_results and not has_threat_intel:
            if "threat_analyst" in {s.get("name") for s in available_skills}:
                if "threat_analyst" not in parsed.get("skills", []):
                    parsed["skills"].append("threat_analyst")
                    logger.info(
                        "[%s] Auto-queueing threat_analyst: question asks for reputation and opensearch found results",
                        SKILL_NAME
                    )
        
        # ── AUTO-QUEUE opensearch_querier AFTER fields_querier ──────────────────────
        # CRITICAL: If fields_querier just returned field_mappings and the supervisor
        # tries to run fields_querier AGAIN (duplicate), automatically queue opensearch_querier
        # to actually USE those discovered fields instead of re-discovering them.
        fields_just_ran = "fields_querier" in parsed.get("skills", [])
        fields_has_results = bool(
            current_results.get("fields_querier") and (
                current_results["fields_querier"].get("field_mappings") or
                (current_results["fields_querier"].get("findings") or {}).get("field_mappings")
            )
        )
        # Pattern: "about traffic from Russia" or "logs from Iran" or "show connections on port 443"
        asks_for_log_search = bool(
            re.search(
                r"\b(traffic|flow|flows|connection|connections|log|logs|event|events|port|ports|protocol|country|countries).*(from|in|on|for|to|at|during|between)\b",
                question_lower
            )
        )
        
        if fields_just_ran and fields_has_results and asks_for_log_search:
            if "opensearch_querier" in {s.get("name") for s in available_skills}:
                if "opensearch_querier" not in parsed.get("skills", []):
                    parsed["skills"].append("opensearch_querier")
                    logger.info(
                        "[%s] Auto-queueing opensearch_querier: fields_querier discovered fields, now search logs with them",
                        SKILL_NAME
                    )
        
        return parsed
    except Exception as exc:
        logger.warning("[%s] Supervisor next action failed: %s", SKILL_NAME, exc)
        return {
            "reasoning": "Fallback routing due to supervisor parse failure",
            "skills": [],
            "parameters": {"question": user_question},
        }


def _supervisor_evaluate_satisfaction(
    user_question: str,
    llm: Any,
    instruction: str,
    conversation_history: list[dict],
    skill_results: dict,
    step: int,
    max_steps: int,
) -> dict:
    """Evaluate whether current aggregated results sufficiently answer the question."""
    # ── SMART FAST PATH: if records found, verify they answer the question ──
    # Don't auto-satisfy if reputation/threat intel was asked but threat_analyst wasn't run
    total_records_found = 0
    for skill_name, result in skill_results.items():
        if result.get("validation_failed"):
            continue
        count = result.get("results_count") or result.get("log_records") or (
            len(result.get("results", [])) if isinstance(result.get("results"), list) else 0
        )
        total_records_found += int(count or 0)

    # Check if question asks for reputation/threat intelligence
    question_lower = user_question.lower()
    asks_for_reputation = any(
        term in question_lower
        for term in ["reputation", "threat", "malicious", "risk", "dangerous", "score", "verdict"]
    )
    has_threat_intel = bool(
        skill_results.get("threat_analyst") and 
        skill_results["threat_analyst"].get("status") == "ok"
    )
    threat_verdicts = skill_results.get("threat_analyst", {}).get("verdicts") or []
    asks_for_additional_evidence = any(
        term in question_lower
        for term in ["traffic", "country", "countries", "log", "logs", "when", "where", "port", "protocol", "connection", "connections", "flow", "flows"]
    )

    if asks_for_reputation and has_threat_intel and threat_verdicts and not asks_for_additional_evidence:
        logger.info(
            "[%s] Evaluation: threat_analyst returned %d verdict(s) for a reputation question — marking satisfied",
            SKILL_NAME,
            len(threat_verdicts),
        )
        return {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": f"Threat intelligence verdicts were produced for {len(threat_verdicts)} entity(s).",
            "missing": [],
        }
    
    if total_records_found > 0:
        # If reputation was asked but we don't have threat intel yet, don't finalize
        if asks_for_reputation and not has_threat_intel:
            logger.info(
                "[%s] Found %d records but reputation/threat was requested and threat_analyst not yet run — continuing",
                SKILL_NAME, total_records_found
            )
            return {
                "satisfied": False,
                "confidence": 0.6,
                "reasoning": f"Found {total_records_found} records but need threat intelligence enrichment.",
                "missing": ["threat reputation and risk assessment"],
            }
        
        # Records found and no actionable missing piece → satisfied
        logger.info(
            "[%s] Evaluation: %d records found across skills — marking satisfied",
            SKILL_NAME, total_records_found,
        )
        return {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": f"Found {total_records_found} matching records across executed skills.",
            "missing": [],
        }

    os_result = skill_results.get("opensearch_querier") or {}
    directional_alternative = os_result.get("directional_alternative") or {}
    if int(directional_alternative.get("results_count", 0) or 0) > 0:
        alternative_direction = directional_alternative.get("direction", "opposite")
        alternative_count = int(directional_alternative.get("results_count", 0) or 0)
        logger.info(
            "[%s] Evaluation: primary direction had zero results, but %d opposite-direction records were found — marking satisfied",
            SKILL_NAME,
            alternative_count,
        )
        return {
            "satisfied": True,
            "confidence": 0.85,
            "reasoning": f"No records matched the requested direction, but {alternative_count} {alternative_direction}-direction records were found for the same IP.",
            "missing": [],
        }

    history_text = "\n".join(
        f"- {m.get('role', '?')}: {str(m.get('content', ''))[:220]}"
        for m in conversation_history[-6:]
    )
    result_summary = json.dumps(skill_results, indent=2, default=str)[:6000]

    evaluation_template = _load_prompt_template(
        SUPERVISOR_EVALUATION_PROMPT_PATH,
        """
Evaluate whether the current skill outputs are sufficient.

QUESTION:
{{USER_QUESTION}}

SKILL RESULTS:
{{RESULT_SUMMARY}}

TOTAL RECORDS FOUND ACROSS ALL SKILLS: {{TOTAL_RECORDS_FOUND}}

STEP:
{{STEP}}/{{MAX_STEPS}}

Return strict JSON with satisfied, confidence, reasoning, and missing.
""",
    )
    prompt = _render_prompt(
        evaluation_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_TEXT": history_text or "- none",
            "RESULT_SUMMARY": result_summary,
            "TOTAL_RECORDS_FOUND": total_records_found,
            "STEP": step,
            "MAX_STEPS": max_steps,
        },
    )

    try:
        response = llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ])
        parsed = _parse_json_object(response) or {}
        return {
            "satisfied": bool(parsed.get("satisfied", False)),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided")),
            "missing": parsed.get("missing", []) if isinstance(parsed.get("missing", []), list) else [],
        }
    except Exception as exc:
        logger.warning("[%s] Supervisor evaluation failed: %s", SKILL_NAME, exc)
        # Conservative fallback: continue until max steps.
        return {
            "satisfied": step >= max_steps,
            "confidence": 0.0,
            "reasoning": "Evaluation unavailable; using step limit fallback",
            "missing": ["evaluation parsing failed"],
        }


def _parse_json_object(response: str) -> dict | None:
    """Best-effort JSON parsing from raw or fenced model output."""
    try:
        return json.loads(response)
    except Exception:
        pass

    try:
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response)
        for block in fenced:
            try:
                return json.loads(block.strip())
            except Exception:
                continue
    except Exception:
        pass

    try:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return None


def _build_threat_followup_question(forensic_result: dict) -> str:
    """Build a compact prompt for threat_analyst from forensic output."""
    report = forensic_result.get("forensic_report", {}) if forensic_result else {}
    incident = report.get("incident_summary", "")
    timeline = (report.get("timeline_narrative", "") or "")[:800]
    anchors = report.get("context_anchors", {}) or {}
    ips = anchors.get("ips", [])[:5]
    ports = anchors.get("ports", [])[:3]
    countries = anchors.get("countries", [])[:3]
    protocols = anchors.get("protocols", [])[:3]
    anchor_text = (
        f"Anchors: IPs={ips}, Ports={ports}, Countries={countries}, Protocols={protocols}."
        if (ips or ports or countries or protocols)
        else ""
    )

    if not incident and not timeline:
        return ""

    return (
        "Perform threat reputation analysis for entities in this forensic report. "
        "Prioritize the provided anchor entities and do not pivot to unrelated IPs unless strongly justified by evidence. "
        "Focus on maliciousness signals, confidence, and actionable response. "
        f"Incident: {incident}\n"
        f"{anchor_text}\n"
        f"Timeline excerpt: {timeline}"
    )


def _extract_entities_from_previous_results(aggregated_results: dict) -> dict:
    """
    Extract IPs, domains, countries, and ports from previous skill results.
    
    Returns dict with keys:
      - ips: list of unique IP addresses found
      - domains: list of unique domains found
      - countries: list of unique countries found
      - ports: list of unique ports found
      - sources: which skills found these entities
    """
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
    
    # Extract from baseline_querier / fields_querier results (legacy rag_querier support deprecated)
    for rag_skill in ("baseline_querier", "fields_querier"):
        if rag_skill in aggregated_results:
            result = aggregated_results[rag_skill]
            entities["sources"].append(rag_skill)

            # Extract IPs and ports from RAG findings
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

    for msg in reversed(conversation_history[-8:]):
        text = str(msg.get("content", "") or "")
        if not text:
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


def _followup_reputation_entities(user_question: str, conversation_history: list[dict] | None) -> dict:
    """Recover prior entities for follow-up reputation questions like 'what about the others?'"""
    if not _question_asks_for_reputation(user_question):
        return {}
    if _question_has_explicit_entities(user_question):
        return {}

    question_lower = user_question.lower()
    referential_cues = [
        "those",
        "them",
        "above",
        "above ip",
        "above ips",
        "the others",
        "others",
        "these",
        "this ip",
        "that ip",
        "mentioned",
        "just mentioned",
        "you just mentioned",
        "you've just mentioned",
        "previously mentioned",
        "prior ip",
        "prior ips",
        "previous ip",
        "previous ips",
        "aside from",
        "excluding",
        "exclude",
        "except",
        "other than",
    ]
    asks_for_new_evidence = any(
        token in question_lower
        for token in [
            "traffic",
            "country",
            "countries",
            "log",
            "logs",
            "port",
            "protocol",
            "connection",
            "connections",
            "flow",
            "flows",
            "when",
            "where",
            "search",
            "show",
            "find",
            "list",
        ]
    )
    if not any(cue in question_lower for cue in referential_cues):
        return {}
    if asks_for_new_evidence and not _question_excludes_private_ips(user_question):
        return {}

    history_entities = _extract_entities_from_conversation_history(conversation_history)
    history_entities = _filter_entities_for_question(history_entities, user_question)
    if history_entities.get("ips") or history_entities.get("domains"):
        return history_entities
    return {}


def _recover_threat_followup_entities(
    user_question: str,
    conversation_history: list[dict] | None,
    aggregated_results: dict | None = None,
) -> dict:
    """Recover concrete entities for threat follow-ups from current results or prior conversation."""
    entities = _followup_reputation_entities(user_question, conversation_history)
    if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
        return entities

    if aggregated_results:
        entities = _extract_entities_from_previous_results(aggregated_results)
        entities = _filter_entities_for_question(entities, user_question)
        if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
            return entities

    return {}


def _build_context_aware_threat_question(original_question: str, entities: dict) -> str:
    """
    Build a context-aware question for threat_analyst when prior results found entities.
    
    This enriches the generic question with actual IPs/domains/countries discovered,
    so threat_analyst analyzes SPECIFIC entities rather than doing a generic lookup.
    """
    entities = _filter_entities_for_question(entities, original_question)
    if not entities or not any([entities.get("ips"), entities.get("domains"), entities.get("countries")]):
        # If no entities extracted, use original question
        return original_question
    
    ips = entities.get("ips", [])
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
        logger.info("[%s] Enriched threat_analyst question with discovered entities: %s", SKILL_NAME, context_str)
    
    return enriched


def format_response(
    user_question: str,
    routing_decision: dict,
    skill_results: dict,
    llm: Any,
    cfg: Any = None,  # Pass config for anti-hallucination setting
    available_skills: list[dict] | None = None,
) -> str:
    """
    Format skill results into a natural language response with thinking-action-reflection loop.
    
    Implements:
      1. THINK: Analyze what the question is asking for
      2. ACTION: Execute skills (already done)
      3. REFLECTION: Check if results answer the question
      4. ANTI-HALLUCINATION: Recheck before presenting
    """
    if not routing_decision.get("skills"):
        # Generate dynamic list of available skills instead of hardcoded fallback
        if available_skills:
            skill_names = [s.get("name") for s in available_skills if s.get("name")]
            skills_str = ", ".join(sorted(skill_names))
        else:
            skills_str = "network_baseliner, anomaly_triage, threat_analyst"
        return f"I couldn't determine which skills would help with that question. Available skills are: {skills_str}."
    
    # ── FORENSIC-FIRST RENDERING ────────────────────────────────────────────
    forensic_result = skill_results.get("forensic_examiner", {})
    if forensic_result and forensic_result.get("status") == "ok":
        return _format_forensic_response(user_question, forensic_result, skill_results.get("threat_analyst", {}))

    geoip_result = skill_results.get("geoip_lookup", {})
    geoip_has_lookup = bool(geoip_result.get("ip") or geoip_result.get("status") == "not_found")
    if geoip_result and geoip_result.get("status") in {"ok", "not_found"} and geoip_has_lookup:
        return _format_geoip_response(geoip_result)

    # ── PRIORITIZE OPENSEARCH/RAG BY DATA AVAILABILITY ──────────────────────
    # Check which has actual results (log records, not just findings)
    os_result = skill_results.get("opensearch_querier", {})
    os_has_data = (
        os_result
        and os_result.get("status") == "ok"
        and os_result.get("results_count", 0) > 0
        and not os_result.get("validation_failed")
    )
    
    # Check baseline_querier / fields_querier results
    rag_result = skill_results.get("baseline_querier") or skill_results.get("fields_querier") or {}
    rag_has_data = rag_result and rag_result.get("status") == "ok" and rag_result.get("log_records", 0) > 0
    
    # If opensearch has records, prioritize it (likely more precise results)
    if os_has_data:
        response = _format_opensearch_response(user_question, os_result)
        threat_result = skill_results.get("threat_analyst", {})
        if threat_result and threat_result.get("status") == "ok":
            response = _append_threat_intel_summary(response, threat_result)
        return response

    if geoip_result and geoip_result.get("status") in {"ok", "not_found"}:
        return _format_geoip_response(geoip_result)
    
    # Only return RAG if it has actual log records (not just schema/findings)
    if rag_has_data:
        return _format_rag_response(user_question, rag_result)

    # ── PHASE 1: THINK ──────────────────────────────────────────────────────
    think_template = _load_prompt_template(
        RESPONSE_THINK_PROMPT_PATH,
        """
Analyze what the user is asking for.

Question: "{{USER_QUESTION}}"

Extract the main intent, key entities, and success criteria.
Be specific and concise.
""",
    )
    think_prompt = _render_prompt(think_template, {"USER_QUESTION": user_question})
    
    think_response = llm.chat([
        {"role": "system", "content": "You are a security analyst. Extract structured intent."},
        {"role": "user", "content": think_prompt},
    ])
    
    # ── PHASE 2: ACTION (already done above) ──────────────────────────────
    # skill_results already contains results from executed skills
    
    # ── PHASE 3: REFLECTION ─────────────────────────────────────────────────
    results_text = "\n".join([
        f"\n[{skill_name}]\n{json.dumps(result, indent=2)}"
        for skill_name, result in skill_results.items()
    ])
    
    reflection_template = _load_prompt_template(
        RESPONSE_REFLECTION_PROMPT_PATH,
        """
You extracted the user's intent as:
{{THINK_RESPONSE}}

Now you received these skill results:
{{RESULTS_TEXT}}

Briefly assess whether the results cover the intent, entities, success criteria, and any gaps.
""",
    )
    reflection_prompt = _render_prompt(
        reflection_template,
        {
            "THINK_RESPONSE": think_response,
            "RESULTS_TEXT": results_text,
        },
    )
    
    reflection_response = llm.chat([
        {"role": "system", "content": "You are a critical analyst. Assess if results are sufficient."},
        {"role": "user", "content": reflection_prompt},
    ])
    
    # ── PHASE 4: ANTI-HALLUCINATION CHECK ───────────────────────────────────
    # Check if anti-hallucination is enabled in config
    anti_hallucination_enabled = True  # Default to enabled
    if cfg:
        anti_hallucination_enabled = cfg.get("llm", "anti_hallucination_check", default=True)
    
    final_response = ""
    if anti_hallucination_enabled:
        verification_template = _load_prompt_template(
            RESPONSE_VERIFICATION_PROMPT_PATH,
            """
Internally verify your answer against these facts.

User question: "{{USER_QUESTION}}"
Skill results:
{{RESULTS_TEXT}}

Provide only the direct answer in 2-4 sentences with no verification text or preamble.
""",
        )
        verification_prompt = _render_prompt(
            verification_template,
            {
                "USER_QUESTION": user_question,
                "RESULTS_TEXT": results_text,
            },
        )
        
        final_response = llm.chat([
            {"role": "system", "content": "You are a rigorous security analyst. Verify internally but output only clean answers without preamble."},
            {"role": "user", "content": verification_prompt},
        ])
    else:
        # Standard response without extra verification
        final_template = _load_prompt_template(
            RESPONSE_FINAL_PROMPT_PATH,
            """
Based on these skill execution results, provide a concise response to the user.

User question: "{{USER_QUESTION}}"

Skill results:
{{RESULTS_TEXT}}

Provide a clear, actionable answer in 2-4 sentences.
""",
        )
        final_prompt = _render_prompt(
            final_template,
            {
                "USER_QUESTION": user_question,
                "RESULTS_TEXT": results_text,
            },
        )
        
        final_response = llm.chat([
            {"role": "system", "content": "You are a helpful SOC analyst. Provide clear, actionable insights."},
            {"role": "user", "content": final_prompt},
        ])
    
    # ── APPEND THREAT INTEL APIs INFO if threat_analyst was used ──────────────
    threat_analyst_result = skill_results.get("threat_analyst", {})
    if threat_analyst_result and threat_analyst_result.get("status") == "ok":
        # Extract API query information from verdicts
        all_apis = set()
        if threat_analyst_result.get("verdicts"):
            for verdict in threat_analyst_result["verdicts"]:
                apis = verdict.get("_queried_apis", [])
                if apis:
                    all_apis.update(apis)
        
        if all_apis:
            apis_str = ", ".join(sorted(all_apis))
            final_response += f"\n\n_[Threat Intelligence Sources Queried: {apis_str}]_"
    
    return final_response


def _append_threat_intel_summary(base_response: str, threat_result: dict) -> str:
    """Append concise threat-intel verdicts to a data-backed response."""
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


def _shorten_naturally(text: str, max_len: int = 180) -> str:
    """Shorten text at a sentence or word boundary instead of mid-token."""
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


def _format_forensic_response(user_question: str, forensic_result: dict, threat_result: dict | None = None) -> str:
    """Render a detailed forensic report (timeline + pattern + entities + reputation)."""
    report = forensic_result.get("forensic_report", {})
    incident = report.get("incident_summary") or user_question
    results_found = report.get("results_found", 0)
    refinements = report.get("refinement_rounds", 0)
    narrative = report.get("timeline_narrative", "") or ""

    timeline_lines = []
    for line in narrative.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}\b|\bUTC\b", line_stripped, re.IGNORECASE):
            timeline_lines.append(line_stripped)
        if len(timeline_lines) >= 6:
            break

    if not timeline_lines and narrative:
        timeline_lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative) if s.strip()][:4]

    entities = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", incident + "\n" + narrative)))
    ports = sorted(set(re.findall(r"\bport\s*(\d{1,5})\b|\b(\d{2,5})/tcp\b|\b(\d{2,5})/udp\b", (incident + "\n" + narrative).lower())))
    flat_ports = sorted({p for tup in ports for p in tup if p})

    paragraph1 = (
        f"Forensic report for '{incident}': analyzed {results_found} matching events "
        f"with {refinements} refinement rounds. The objective was incident reconstruction "
        "with timeline, behavior frequency, key entities, and risk implications."
    )

    timeline_text = "\n".join([f"- {line}" for line in timeline_lines]) if timeline_lines else "- No concrete timestamped events were returned by the data source."
    paragraph2 = f"Timeline:\n{timeline_text}"

    entity_text = ", ".join(entities[:10]) if entities else "No IP entities extracted"
    port_text = ", ".join(flat_ports[:10]) if flat_ports else "No explicit ports extracted"
    pattern_hint = ""
    for sentence in re.split(r"(?<=[.!?])\s+", narrative):
        if re.search(r"pattern|periodic|sporadic|frequency|interval|automated|bot|risk|threat", sentence, re.IGNORECASE):
            pattern_hint = sentence.strip()
            break
    if not pattern_hint:
        pattern_hint = "Pattern/risk signal was not explicit in the model output and should be treated as low confidence."
    paragraph3 = (
        f"Entities and behavior: IPs involved: {entity_text}. Ports involved: {port_text}. "
        f"Frequency/pattern assessment: {pattern_hint}"
    )

    if threat_result and threat_result.get("status") == "ok" and threat_result.get("verdicts"):
        verdict_lines = []
        for v in threat_result.get("verdicts", [])[:3]:
            verdict = v.get("verdict", "UNKNOWN")
            confidence = v.get("confidence", 0)
            reason = (v.get("reasoning") or "").strip().replace("\n", " ")
            verdict_lines.append(f"- {verdict} ({confidence}% confidence): {reason}")
        paragraph4 = "Reputation and threat intel:\n" + "\n".join(verdict_lines)
    else:
        paragraph4 = (
            "Reputation and threat intel: no explicit reputation verdict was returned. "
            "If API keys are configured, rerun with threat_analyst enabled to include AbuseIPDB/VirusTotal/OTX/Talos signals."
        )

    return "\n\n".join([paragraph1, paragraph2, paragraph3, paragraph4])


def _format_opensearch_response(user_question: str, os_result: dict) -> str:
    """Render opensearch_querier results with evidence detail."""
    results = os_result.get("results") or []
    results_count = os_result.get("results_count", len(results))
    countries = os_result.get("countries", [])
    ports = os_result.get("ports", [])
    protocols = os_result.get("protocols", [])
    time_range = os_result.get("time_range_label") or os_result.get("time_range", "")
    search_terms = os_result.get("search_terms", [])
    directional_alternative = os_result.get("directional_alternative") or {}

    if not results:
        if directional_alternative:
            requested_direction = os_result.get("ip_direction") or "requested"
            alternative_direction = directional_alternative.get("direction") or "opposite"
            alternative_count = int(directional_alternative.get("results_count", 0) or 0)
            alt_time_range = directional_alternative.get("time_range_label") or time_range
            queried_ip = "/".join(str(term) for term in search_terms[:3]) or "the requested IP"
            detail_parts = [
                f"No traffic {requested_direction} {queried_ip} was found in the {time_range} window.",
                f"However, {alternative_count} record(s) were found in the {alternative_direction} direction for the same IP in the {alt_time_range} window.",
            ]
            sample_peers = directional_alternative.get("sample_peers") or []
            if sample_peers:
                detail_parts.append(f"Peers seen: {', '.join(sample_peers[:10])}.")
            earliest = directional_alternative.get("earliest")
            latest = directional_alternative.get("latest")
            if earliest and latest:
                detail_parts.append(f"Earliest: {earliest}. Latest: {latest}.")
            return " ".join(detail_parts)

        filter_parts = []
        if countries:
            filter_parts.append(f"country={'/' .join(countries)}")
        if ports:
            filter_parts.append(f"port={'/'.join(str(p) for p in ports)}")
        if protocols:
            filter_parts.append(f"protocol={'/'.join(protocols)}")
        filter_desc = ", ".join(filter_parts) or "the specified criteria"
        return f"No matching records found for {filter_desc} in the {time_range} window."

    # ── SPECIAL HANDLING FOR ALERT QUERIES ──────────────────────────────────
    # If the question is about alerts/signatures, show alert-specific information
    question_lower = user_question.lower()
    is_alert_query = any(kw in question_lower for kw in [
        "alert", "signature", "et exploit", "et rule", "et drop", "et policy", "suricata", "snort", "rule"
    ])
    
    if is_alert_query:
        # For alert queries, extract alert/signature information
        alert_signatures: set = set()
        alert_types: set = set()
        alert_count_by_sig: dict = {}
        alert_ips: set = set()
        alert_countries: set = set()
        alert_timestamps: list[str] = []
        
        for row in results:
            # Extract alert signatures
            sig = row.get("alert.signature") or row.get("signature") or row.get("alert", {}).get("signature")
            if sig:
                sig_str = str(sig)
                alert_signatures.add(sig_str)
                alert_count_by_sig[sig_str] = alert_count_by_sig.get(sig_str, 0) + 1
            
            # Extract alert types/categories
            alert_type = row.get("alert.category") or row.get("event.category")
            if alert_type:
                alert_types.add(str(alert_type))

            ts = row.get("@timestamp") or row.get("timestamp")
            if ts:
                alert_timestamps.append(str(ts))

            for value in (
                row.get("src_ip"),
                row.get("dest_ip"),
                row.get("source.ip"),
                row.get("destination.ip"),
                row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
                row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
            ):
                if value:
                    alert_ips.add(str(value))

            geo = row.get("geoip") or {}
            if isinstance(geo, dict):
                for cn in (geo.get("country_name"), geo.get("country")):
                    if cn:
                        alert_countries.add(str(cn))
            for cn in (
                row.get("geoip.country_name"),
                row.get("country_name"),
                row.get("source.geo.country_name"),
                row.get("destination.geo.country_name"),
            ):
                if cn:
                    alert_countries.add(str(cn))
        
        # Build alert-focused summary
        summary = f"Found {results_count} alert record(s) matching {' / '.join(search_terms)} in the {time_range} window."
        
        detail_parts = []
        if alert_signatures:
            top_sigs = sorted(alert_signatures)[:5]
            detail_parts.append(f"Alert signatures: {', '.join(top_sigs)}.")
        if alert_types:
            detail_parts.append(f"Alert categories: {', '.join(sorted(alert_types))}.")
        asks_for_alert_details = any(
            term in question_lower
            for term in [
                "what ip", "which ip", "their ip", "their ips", "source ip", "destination ip",
                "what countr", "which countr", "what countries", "what country", "where are they from",
                "when did", "when was", "timestamp", "what time", "alert happen",
            ]
        )
        if asks_for_alert_details and alert_ips:
            detail_parts.append(f"IPs seen in matching alerts: {', '.join(sorted(alert_ips)[:12])}.")
        if asks_for_alert_details and alert_countries:
            detail_parts.append(f"Countries seen in matching alerts: {', '.join(sorted(alert_countries)[:12])}.")
        if asks_for_alert_details and alert_timestamps:
            ts_sorted = sorted(alert_timestamps)
            detail_parts.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")
        
        if detail_parts:
            return summary + " " + " ".join(detail_parts)
        return summary

    # ── STANDARD TRAFFIC QUERY FORMATTING ──────────────────────────────────
    # Extract key evidence
    import re as _re
    ips: set = set()
    source_ips: set = set()
    ts_list: list = []
    countries_seen: set = set()

    for row in results:
        # Timestamps
        ts = row.get("@timestamp") or row.get("timestamp")
        if ts:
            ts_list.append(str(ts))

        # IPs
        for v in (
            row.get("src_ip"),
            row.get("source_ip"),
            row.get("source.ip"),
            row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
        ):
            if v:
                ips.add(str(v))
                source_ips.add(str(v))
        for v in (
            row.get("dest_ip"),
            row.get("destination_ip"),
            row.get("destination.ip"),
            row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
        ):
            if v:
                ips.add(str(v))

        # Countries from geoip
        geo = row.get("geoip") or {}
        if isinstance(geo, dict):
            cn = geo.get("country_name")
            if cn:
                countries_seen.add(str(cn))
        for cn in (
            row.get("geoip.country_name"),
            row.get("country_name"),
            row.get("source.geo.country_name"),
            row.get("destination.geo.country_name"),
        ):
            if cn:
                countries_seen.add(str(cn))

    # Build summary
    filter_parts = []
    if countries:
        filter_parts.append("/".join(countries))
    if ports:
        filter_parts.append("port " + "/".join(str(p) for p in ports))
    if protocols:
        filter_parts.append("/".join(protocols))
    if search_terms and not filter_parts:
        shown_terms = "/".join(str(term) for term in search_terms[:3])
        if len(search_terms) > 3:
            shown_terms += "/…"
        filter_parts.append(shown_terms)
    filter_desc = ", ".join(filter_parts) or "the query criteria"

    summary = f"Found {results_count} record(s) matching {filter_desc} in the {time_range} window."

    # For port-specific queries, extract discovered port values from results (not just restate filter)
    extracted_ports: set = set()
    if ports:  # Only extract if a port was specifically queried
        for row in results:
            # Try both nested and flat field names
            port_candidates = [
                row.get("destination.port"),
                row.get("destination", {}).get("port") if isinstance(row.get("destination"), dict) else None,
                row.get("destination_port"),
                row.get("dst_port"),
                row.get("dest_port"),
                row.get("dport"),
                row.get("port"),
            ]
            for p in port_candidates:
                if p is not None:
                    try:
                        extracted_ports.add(int(p))
                    except (ValueError, TypeError):
                        pass

    detail_parts = []
    if countries_seen:
        detail_parts.append(f"Countries seen: {', '.join(sorted(countries_seen))}.")
    if ips:
        if ports and source_ips:
            detail_parts.append(f"Remote peers: {', '.join(sorted(source_ips)[:10])}.")
        else:
            detail_parts.append(f"Source/destination IPs: {', '.join(sorted(ips)[:10])}.")
    if ts_list:
        ts_sorted = sorted(ts_list)
        detail_parts.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")
    matched_ports = extracted_ports.intersection({int(p) for p in ports if str(p).isdigit()}) if ports else extracted_ports
    if matched_ports:
        detail_parts.append(f"Destination port(s): {', '.join(str(p) for p in sorted(matched_ports))}.")

    if detail_parts:
        return summary + " " + " ".join(detail_parts)
    return summary


def _format_rag_response(user_question: str, rag_result: dict) -> str:
    """Render rag_querier responses with explicit evidence details."""
    findings = rag_result.get("findings", {})
    base_answer = _strip_json_like_content((findings.get("answer") or "").strip())
    evidence = findings.get("evidence", {}) or {}

    ips = evidence.get("ips", [])
    ports = evidence.get("ports", [])
    protocols = evidence.get("protocols", [])
    timestamps = evidence.get("timestamps", [])

    details = (
        "Evidence details: "
        f"IPs involved: {', '.join(ips[:10]) if ips else 'none extracted'}. "
        f"Ports: {', '.join(ports[:10]) if ports else 'none extracted'}. "
        f"Protocols: {', '.join(protocols[:10]) if protocols else 'none extracted'}. "
        f"Timestamps: {', '.join(timestamps[:6]) if timestamps else 'not available'}."
    )

    if base_answer:
        return f"{base_answer}\n\n{details}"
    return details


def _format_geoip_response(geoip_result: dict) -> str:
    """Render direct GeoIP lookup or maintenance results without LLM synthesis."""
    action = geoip_result.get("action", "ready")
    db_path = geoip_result.get("db_path")
    warning = geoip_result.get("warning")

    lookups = geoip_result.get("lookups") or []
    if lookups:
        rendered: list[str] = []
        for lookup in lookups[:15]:
            ip = lookup.get("ip", "unknown")
            if lookup.get("status") == "not_found":
                rendered.append(f"{ip}: not found in the MaxMind database")
                continue
            if lookup.get("status") == "error":
                rendered.append(f"{ip}: lookup error ({lookup.get('error', 'unknown error')})")
                continue

            geo = lookup.get("geo") or {}
            location_parts = []
            for field in ("city", "subdivision", "country"):
                value = geo.get(field)
                if value and value not in location_parts:
                    location_parts.append(value)
            location = ", ".join(location_parts) if location_parts else "an unknown location"
            rendered.append(f"{ip}: {location}")

        response = "Resolved GeoIP for the referenced IPs: " + "; ".join(rendered) + "."
        if db_path:
            response += f" Database: {db_path}."
        if warning:
            response += f" Warning: {warning}."
        return response

    if geoip_result.get("status") == "not_found":
        response = f"No MaxMind geolocation record was found for IP {geoip_result.get('ip', 'unknown')}."
        if db_path:
            response += f" Database: {db_path}."
        return response

    ip = geoip_result.get("ip")
    geo = geoip_result.get("geo") or {}
    if not ip:
        response = f"GeoIP database check complete. Status: {action}."
        if db_path:
            response += f" Database: {db_path}."
        if warning:
            response += f" Warning: {warning}."
        return response

    location_parts = []
    for field in ("city", "subdivision", "country"):
        value = geo.get(field)
        if value and value not in location_parts:
            location_parts.append(value)
    location = ", ".join(location_parts) if location_parts else "an unknown location"

    response = f"IP {ip} resolves to {location}."
    extra = []
    if geo.get("country_iso_code"):
        extra.append(f"country code {geo['country_iso_code']}")
    if geo.get("timezone"):
        extra.append(f"timezone {geo['timezone']}")
    if geo.get("postal_code"):
        extra.append(f"postal code {geo['postal_code']}")
    if geo.get("latitude") is not None and geo.get("longitude") is not None:
        extra.append(f"coordinates {geo['latitude']}, {geo['longitude']}")
    if extra:
        response += " " + "; ".join(extra) + "."

    response += f" GeoIP DB status: {action}."
    if warning:
        response += f" Warning: {warning}."
    return response


def _strip_json_like_content(text: str) -> str:
    """Remove raw JSON/code-block style dumps from model answers."""
    if not text:
        return text

    # Remove fenced blocks first.
    cleaned = re.sub(r"```[\s\S]*?```", "", text)

    # Remove obvious JSON object dumps that start on their own line.
    cleaned = re.sub(r"\n\s*\{[\s\S]*?\}\s*", "\n", cleaned)

    # Collapse excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Memory Management
# ──────────────────────────────────────────────────────────────────────────────

CONVERSATIONS_DIR = Path(__file__).parent.parent.parent / "conversations"


def _ensure_conversations_dir():
    """Create conversations directory if it doesn't exist."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_conversation_history(conversation_id: str) -> list[dict]:
    """Load conversation history from disk."""
    _ensure_conversations_dir()
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not conv_file.exists():
        return []
    
    try:
        with open(conv_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load conversation %s: %s", conversation_id, e)
        return []


def save_conversation_history(conversation_id: str, history: list[dict]) -> None:
    """Save conversation history to disk."""
    _ensure_conversations_dir()
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    try:
        with open(conv_file, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error("Failed to save conversation %s: %s", conversation_id, e)


def list_conversations() -> list[dict]:
    """List all saved conversations with metadata."""
    _ensure_conversations_dir()
    conversations = []
    
    for conv_file in sorted(CONVERSATIONS_DIR.glob("*.json")):
        try:
            with open(conv_file, "r") as f:
                history = json.load(f)
            
            if history:
                first_user = next((entry for entry in history if entry.get("role") == "user"), {})
                last_entry = history[-1] if history else {}
                timestamp = last_entry.get("timestamp", "Unknown")
                conversations.append({
                    "id": conv_file.stem,
                    "messages": len(history),
                    "first_question": first_user.get("content", "Unknown"),
                    "last_update": timestamp,
                    "timestamp": timestamp,
                })
        except Exception as e:
            logger.warning("Failed to read conversation file %s: %s", conv_file, e)
    
    return conversations


def add_to_history(conversation_id: str, question: str, answer: str, 
                  routing: dict, skill_results: dict) -> None:
    """Add a Q&A exchange to conversation history."""
    from datetime import datetime, timezone
    
    history = load_conversation_history(conversation_id)
    
    # Save user question
    user_entry = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history.append(user_entry)
    
    # Save assistant answer
    assistant_entry = {
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "routing_skills": routing.get("skills", []),
        "routing_reasoning": routing.get("reasoning", ""),
        "skill_results": skill_results,
    }
    history.append(assistant_entry)
    
    save_conversation_history(conversation_id, history)


def get_context_summary(conversation_id: str, last_n: int = 3) -> str:
    """Get summary of recent conversation for context injection."""
    history = load_conversation_history(conversation_id)
    
    if not history:
        return ""
    
    recent = history[-(last_n * 2):]
    summary_lines = []

    pending_question = ""
    for entry in recent:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            pending_question = content
        elif role == "assistant":
            summary_lines.append(f"Q: {pending_question}")
            answer = content
            if len(answer) > 200:
                answer = answer[:200] + "..."
            summary_lines.append(f"A: {answer}")
            pending_question = ""
    
    return "\n".join(summary_lines)
