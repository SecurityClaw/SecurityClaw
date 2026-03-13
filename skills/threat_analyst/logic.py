"""
skills/threat_analyst/logic.py

RAG-powered reasoning loop that reviews HIGH/CRITICAL findings queued
by AnomalyWatcher, retrieves behavioral baseline context, enriches findings
with external reputation intelligence (AbuseIPDB, AlienVault, VirusTotal, Talos),
and issues a verdict (FALSE_POSITIVE | TRUE_THREAT).

Context keys consumed:
    context["db"]     -> BaseDBConnector
    context["llm"]    -> BaseLLMProvider
    context["memory"] -> Memory instance (StateBackedMemory or CheckpointBackedMemory)
    context["config"] -> Config
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "threat_analyst"


def _is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def _question_excludes_private_ips(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
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


def _extract_public_ipv4s(text: str) -> list[str]:
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    return list(dict.fromkeys(ip for ip in re.findall(ip_pattern, text or "") if not _is_private_ip(ip)))


def _finding_requires_rag_context(finding_desc: str) -> bool:
    """Return True only for findings that need behavioral baseline context."""
    lowered = finding_desc.lower()
    is_reputation_question = any(
        term in lowered
        for term in ["reputation", "threat intel", "threat intelligence", "malicious", "risk", "verdict", "score"]
    )
    return not is_reputation_question


def _build_grounded_reputation_reasoning(
    requested_ips: list[str],
    verdict_label: str,
    reputation_context: str,
) -> str:
    """Build a grounded explanation that only references the requested IPs."""
    subject = ", ".join(requested_ips) if requested_ips else "the requested entities"
    detail_lines = []
    for line in (reputation_context or "").splitlines():
        normalized = line.strip().lstrip("•").strip()
        if not normalized:
            continue
        if requested_ips and not any(ip in normalized for ip in requested_ips):
            continue
        detail_lines.append(normalized)
        if len(detail_lines) >= 3:
            break

    if detail_lines:
        return f"Reputation analysis for {subject}: {verdict_label}. " + " ".join(detail_lines)
    return f"Reputation analysis for {subject}: {verdict_label}. No additional IPs were considered."


def _sanitize_verdict_entities(parsed: dict, finding_desc: str, reputation_context: str) -> dict:
    """Ensure threat verdict reasoning does not introduce IPs outside the requested set."""
    requested_ips = _extract_public_ipv4s(finding_desc)
    reasoning_ips = set(_extract_public_ipv4s(str(parsed.get("reasoning", "") or "")))

    if requested_ips and reasoning_ips and not reasoning_ips.issubset(set(requested_ips)):
        logger.warning(
            "[%s] Threat verdict mentioned unexpected IPs %s; constraining reasoning to %s",
            SKILL_NAME,
            sorted(reasoning_ips),
            requested_ips,
        )
        parsed["reasoning"] = _build_grounded_reputation_reasoning(
            requested_ips,
            str(parsed.get("verdict", "UNKNOWN") or "UNKNOWN"),
            reputation_context,
        )

    parsed["_requested_ips"] = requested_ips
    return parsed


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    memory = context.get("memory")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    conversation_history = context.get("conversation_history", [])

    if llm is None:
        logger.warning("[%s] llm not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no llm"}

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")

    # ── 1. Check for direct chat question first ──────────────────────────────
    # In chat mode, the question comes via parameters["question"]
    chat_question = parameters.get("question")
    
    # ── 2. Read escalation queue from memory ──────────────────────────────────
    escalations = _parse_escalations(memory)
    
    # If no escalations but we have a chat question, use that instead
    if not escalations and not chat_question:
        logger.debug("[%s] No escalations or question pending.", SKILL_NAME)
        return {"status": "ok", "analyzed": 0}
    
    if not escalations and chat_question:
        escalations = [chat_question]
        logger.info("[%s] Analyzing question: %s", SKILL_NAME, chat_question[:80])
    elif escalations:
        logger.info("[%s] Analyzing %d escalation(s)…", SKILL_NAME, len(escalations))

    rag = None
    needs_rag_context = any(_finding_requires_rag_context(item) for item in escalations)
    if needs_rag_context:
        if db is None:
            logger.warning("[%s] db not available for baseline-backed threat analysis — skipping.", SKILL_NAME)
            return {"status": "skipped", "reason": "no db"}
        from core.rag_engine import RAGEngine
        rag = RAGEngine(db=db, llm=llm)

    verdicts = []

    for item in escalations:
        verdict = _analyze_finding(item, instruction, rag, llm, conversation_history)
        verdicts.append(verdict)

        # ── 2. Write verdict back to memory ───────────────────────────────────
        if memory:
            v = verdict.get("verdict", "UNKNOWN")
            conf = verdict.get("confidence", 0)
            rec = verdict.get("recommended_action", "")
            memory.add_decision(
                f"[{v}] confidence={conf}% | {item[:80]} | action: {rec}"
            )
            if v == "TRUE_THREAT":
                memory.set_focus(f"Active threat investigation: {item[:120]}")

    # ── 3. Clear processed escalations ────────────────────────────────────────
    if memory and verdicts:
        memory.set_section("Escalation Queue", "None")

    return {
        "status": "ok",
        "analyzed": len(verdicts),
        "verdicts": verdicts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core reasoning loop (one finding)
# ──────────────────────────────────────────────────────────────────────────────

def _analyze_finding(finding_desc: str, instruction: str, rag, llm, 
                     conversation_history: list[dict] = None) -> dict:
    """
    Retrieve RAG context, fetch reputation intelligence, and ask the LLM for a verdict.
    Supports extracting context from conversation history for follow-up questions.
    
    Returns dict with analysis verdict and API query information.
    """
    # Extract IPs/domains from the finding first to determine question type
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ips_in_finding = set(re.findall(ip_pattern, finding_desc))
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"
    domains_in_finding = set(re.findall(domain_pattern, finding_desc.lower()))
    
    # For reputation/threat analysis, skip RAG baseline context.
    # Reputation checks should analyze ONLY the explicitly-queried entities, not browse
    # baseline behavior which introduces noise and causes the LLM to analyze wrong IPs.
    # Baseline context is for behavioral anomaly analysis, not threat reputation.
    is_reputation_question = not _finding_requires_rag_context(finding_desc)
    
    if is_reputation_question or rag is None:
        # Reputation questions don't need baseline context
        baseline_section = ""
    else:
        # For other finding types, use RAG context
        rag_context = rag.build_context_string(
            query=finding_desc,
            category="network_baseline",
        )
        has_relevant_baseline = "_No relevant context found._" not in rag_context
        baseline_section = f"**Baseline Context:**\n{rag_context}\n\n" if has_relevant_baseline else ""

    # Extract and enrich with external reputation intelligence
    # Pass conversation history to help extract IPs/domains from context
    reputation_context, queried_apis = _enrich_with_reputation(finding_desc, conversation_history)

    # Extract the specific IPs from the finding so the LLM stays anchored to them.
    _ip_pattern_str = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    _finding_ips = re.findall(_ip_pattern_str, finding_desc)
    _public_finding_ips = [ip for ip in _finding_ips if not _is_private_ip(ip)]
    _anchor_note = (
        f"\nIMPORTANT: Only reference these specific IPs in your verdict: {', '.join(_public_finding_ips)}. "
        "Do not introduce IPs or domains from the baseline context that are not part of this finding."
        if _public_finding_ips else ""
    )

    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                f"**Anomaly Finding:**\n{finding_desc}\n\n"
                f"{baseline_section}"
                f"**Reputation Intelligence:**\n{reputation_context}\n\n"
                f"Based on the above context and reputation data, provide your verdict.{_anchor_note}"
            ),
        },
    ]

    try:
        response = llm.chat(messages)
        parsed = _parse_json(response)
        if parsed:
            parsed = _sanitize_verdict_entities(parsed, finding_desc, reputation_context)
            parsed["_finding"] = finding_desc[:200]
            parsed["_queried_apis"] = queried_apis  # Include which APIs were queried
            return parsed
        return {
            "verdict": "UNKNOWN",
            "confidence": 0,
            "reasoning": response[:500],
            "_finding": finding_desc[:200],
            "_queried_apis": queried_apis,
        }
    except Exception as exc:
        logger.error("[%s] LLM analysis failed: %s", SKILL_NAME, exc)
        return {
            "verdict": "ERROR",
            "confidence": 0,
            "reasoning": str(exc),
            "_finding": finding_desc[:200],
            "_queried_apis": queried_apis,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_escalations(memory) -> list[str]:
    """Extract non-empty escalation items from agent memory."""
    if memory is None:
        return []
    raw = memory.get_section("Escalation Queue")
    if not raw or raw.strip() == "None":
        return []
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("- ["):
            # Strip bullet and timestamp prefix
            # Format: - [2026-03-02 12:00:00 UTC] [HIGH] Needs ThreatAnalyst…
            match = re.match(r"- \[.*?\]\s+(.*)", line)
            items.append(match.group(1) if match else line[2:])
    return [i for i in items if i]


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _enrich_with_reputation(finding_desc: str, conversation_history: list[dict] = None) -> tuple[str, list[str]]:
    """
    Extract IPs and domains from finding (and conversation history),
    fetch reputation intelligence, and format for LLM consumption.
    
    Returns:
        tuple of (formatted_reputation_string, list_of_queried_apis)
    """
    try:
        from skills.threat_analyst.reputation_intel import get_ip_reputation, get_domain_reputation
    except ImportError:
        logger.warning("[%s] reputation_intel module not available", SKILL_NAME)
        return "No external reputation data available.", []

    # Extract IPs from finding
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ips = set(re.findall(ip_pattern, finding_desc))

    # Extract domains from finding
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"
    domains = set(re.findall(domain_pattern, finding_desc.lower()))

    # If no IPs/domains found in the question, check conversation history
    if not ips and not domains and conversation_history:
        logger.debug("[%s] No IPs/domains in question, searching conversation history", SKILL_NAME)
        # Look through recent messages for IPs and domains
        for msg in conversation_history[-5:]:  # Check recent 5 messages
            text = msg.get("content", "")
            found_ips = set(re.findall(ip_pattern, text))
            found_domains = set(re.findall(domain_pattern, text.lower()))
            ips.update(found_ips)
            domains.update(found_domains)
            if ips or domains:
                logger.debug("[%s] Found in history: IPs=%s, domains=%s", SKILL_NAME, ips, domains)
                break

    # Always exclude private/RFC-1918 IPs: they have no external threat reputation
    # and including them confuses the LLM (they may match baseline entries for other reasons).
    ips = {ip for ip in ips if not _is_private_ip(ip)}

    if not ips and not domains:
        if _question_excludes_private_ips(finding_desc):
            return "No external reputation data needed after excluding private/internal IPs.", []
        return "No external reputation data needed (no IPs or domains in question or history).", []

    reputation_lines = []
    all_queries = set()  # Track all APIs queried

    # Fetch IP reputation
    for ip in sorted(ips)[:5]:  # Limit to 5 IPs for performance
        try:
            intel = get_ip_reputation(ip)
            if intel.get("queries"):
                # Track which APIs were queried
                all_queries.update(intel.get("queries", []))
                
                risk = intel.get("combined_risk", "UNKNOWN")
                reputation_lines.append(f"  • IP {ip}: Risk={risk}")
                
                # Add details from available sources
                if "abuseipdb" in intel:
                    score = intel["abuseipdb"].get("abuse_score", 0)
                    reports = intel["abuseipdb"].get("reports", 0)
                    reputation_lines.append(f"    - AbuseIPDB: {score}% suspicious ({reports} reports)")
                
                if "alienvault" in intel:
                    reputation = intel["alienvault"].get("reputation", "unknown")
                    pulses = intel["alienvault"].get("pulses", 0)
                    reputation_lines.append(f"    - AlienVault: {reputation} reputation ({pulses} threat pulses)")
                
                if "virustotal" in intel:
                    malicious = intel["virustotal"].get("malicious", 0)
                    if malicious > 0:
                        reputation_lines.append(f"    - VirusTotal: {malicious} vendors flagged as malicious")
        except Exception as e:
            logger.debug(f"[{SKILL_NAME}] Reputation lookup failed for IP {ip}: {e}")

    # Fetch domain reputation
    for domain in sorted(domains)[:5]:  # Limit to 5 domains for performance
        try:
            intel = get_domain_reputation(domain)
            if intel.get("queries"):
                # Track which APIs were queried
                all_queries.update(intel.get("queries", []))
                
                risk = intel.get("combined_risk", "UNKNOWN")
                reputation_lines.append(f"  • Domain {domain}: Risk={risk}")
                
                # Add details from available sources
                if "alienvault" in intel:
                    reputation = intel["alienvault"].get("reputation", "unknown")
                    pulses = intel["alienvault"].get("pulses", 0)
                    reputation_lines.append(f"    - AlienVault: {reputation} reputation ({pulses} threat pulses)")
                
                if "virustotal" in intel:
                    malicious = intel["virustotal"].get("malicious", 0)
                    if malicious > 0:
                        reputation_lines.append(f"    - VirusTotal: {malicious} vendors flagged as malicious")
        except Exception as e:
            logger.debug(f"[{SKILL_NAME}] Reputation lookup failed for domain {domain}: {e}")

    # Format output with queries info
    result = ""
    if all_queries:
        queries_str = ", ".join(sorted(all_queries))
        result = f"**External Reputation Intelligence** (Queried: {queries_str}):\n" + "\n".join(reputation_lines)
    elif reputation_lines:
        result = "**External Reputation Intelligence:**\n" + "\n".join(reputation_lines)
    else:
        result = "No external reputation data available (API keys not configured)."
    
    return result, list(all_queries)
