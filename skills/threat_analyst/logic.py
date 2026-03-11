"""
skills/threat_analyst/logic.py

RAG-powered reasoning loop that reviews HIGH/CRITICAL findings queued
by AnomalyWatcher, retrieves behavioral baseline context, enriches findings
with external reputation intelligence (AbuseIPDB, AlienVault, VirusTotal, Talos),
and issues a verdict (FALSE_POSITIVE | TRUE_THREAT).

Context keys consumed:
    context["db"]     -> BaseDBConnector
    context["llm"]    -> BaseLLMProvider
    context["memory"] -> AgentMemory
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


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    memory = context.get("memory")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    conversation_history = context.get("conversation_history", [])

    if db is None or llm is None:
        logger.warning("[%s] db or llm not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db/llm"}

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
    # Retrieve relevant baseline context
    rag_context = rag.build_context_string(
        query=finding_desc,
        category="network_baseline",
    )

    # Extract and enrich with external reputation intelligence
    # Pass conversation history to help extract IPs/domains from context
    reputation_context, queried_apis = _enrich_with_reputation(finding_desc, conversation_history)

    # Only include baseline context if relevant information was actually found
    # Avoid including "No relevant context found" which misleads the LLM
    has_relevant_baseline = "_No relevant context found._" not in rag_context
    
    if has_relevant_baseline:
        baseline_section = f"**Baseline Context:**\n{rag_context}\n\n"
    else:
        baseline_section = ""

    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": (
                f"**Anomaly Finding:**\n{finding_desc}\n\n"
                f"{baseline_section}"
                f"**Reputation Intelligence:**\n{reputation_context}\n\n"
                "Based on the above context and reputation data, provide your verdict."
            ),
        },
    ]

    try:
        response = llm.chat(messages)
        parsed = _parse_json(response)
        if parsed:
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

    if _question_excludes_private_ips(finding_desc):
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
