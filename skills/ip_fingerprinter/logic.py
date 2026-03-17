"""
skills/ip_fingerprinter/logic.py

Data-agnostic IP fingerprinting skill.

This skill is completely data-agnostic:
- Receives aggregated port data from opensearch_querier (via LLM orchestration)
- Performs port analysis (service name lookup, role inference, OS family scoring)
- Does NOT perform field discovery or query building

All data access and schema knowledge is delegated to opensearch_querier and fields_querier.
This skill focuses on ANALYSIS, not DATA ACCESS.
"""
from __future__ import annotations

import ipaddress
import logging
import re
from typing import Any

from skills.ip_fingerprinter.port_registry import load_port_registry

logger = logging.getLogger(__name__)

SKILL_NAME = "ip_fingerprinter"
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# OS-specific service ports for role inference
WINDOWS_SERVICE_PORTS = {88, 135, 137, 138, 139, 389, 445, 464, 593, 636, 3268, 3269, 3389, 5985, 5986, 9389}
LINUX_SERVICE_PORTS = {22, 111, 2049, 2375, 2376, 4242, 6443, 8080, 9100}
MACOS_SERVICE_PORTS = {548, 3283, 5009, 5353, 62078}


def _is_valid_ip(value: str) -> bool:
    """Check if a value is a valid IP address."""
    try:
        ipaddress.ip_address(str(value).strip())
        return True
    except ValueError:
        return False


def _extract_ip(parameters: dict[str, Any], previous_results: dict[str, Any]) -> str | None:
    """
    Extract target IP from parameters or question text.
    
    This function is data-agnostic: it doesn't search through records,
    only extracts from explicit parameters or query text.
    """
    # Check explicit IP parameter
    direct_ip = parameters.get("ip")
    if isinstance(direct_ip, str) and _is_valid_ip(direct_ip):
        return direct_ip

    # Check ips list parameter
    ips = parameters.get("ips")
    if isinstance(ips, list):
        for value in ips:
            if isinstance(value, str) and _is_valid_ip(value):
                return value

    # Extract from question/query text
    question = str(parameters.get("question") or parameters.get("query") or "")
    for candidate in IP_PATTERN.findall(question):
        if _is_valid_ip(candidate):
            return candidate

    return None


def _analyze_ports(aggregated_ports: dict[int, dict[str, Any]], registry: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Analyze destination ports using pre-aggregated port counts.
    
    This function is completely data-agnostic:
    - Receives aggregated port data (already counts computed by oversearch_querier)
    - Performs enrichment: service name lookup, registration status, ephemeral classification
    - Does NOT perform any data access, field discovery, or query building
    
    Args:
        aggregated_ports: Dict mapping port number to {observations: count, protocols: list, ...}
        registry: Port registry for service classification
    
    Returns:
        (enriched_ports, evidence) - List of enriched port details and summary statistics
    """
    enriched_ports: list[dict[str, Any]] = []
    
    for port, aggregate in sorted(aggregated_ports.items(), key=lambda item: (-item[1].get("observations", 0), item[0])):
        # Skip ephemeral ports (kernel-allocated temporary ports)
        # On Linux: >= 32768 (default ephemeral range)
        # We filter here to focus on listening services, not client connections
        if port >= 32768:
            logger.debug("[%s] Skipping ephemeral port %d (>= 32768)", SKILL_NAME, port)
            continue
            
        # Get protocol(s) associated with this port
        protocols = sorted(aggregate.get("protocols", []) or []) or [None]
        
        # Classify port using registry (service name, registered status, etc.)
        classification = registry.classify(port, protocols[0].lower() if protocols[0] else None)
        
        enriched_ports.append({
            "port": port,
            "protocols": [value for value in protocols if value],
            "service_name": classification["service_name"],
            "description": classification["description"],
            "registered": classification["registered"],
            "range_class": classification["range_class"],
            "ephemeral_likelihood": classification["ephemeral_likelihood"],
            "ephemeral_reason": classification["ephemeral_reason"],
            "observations": aggregate.get("observations", 0),
            "peer_count": len(aggregate.get("peers", set())),
            "peers": sorted(aggregate.get("peers", set()))[:10] if aggregate.get("peers") else [],
            "first_seen": aggregate.get("first_seen"),
            "last_seen": aggregate.get("last_seen"),
        })

    evidence = {
        "port_count": len(enriched_ports),
    }
    return enriched_ports, evidence


def _infer_role(ports: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Infer host role (server vs client) based on listening (destination) ports.
    
    This analysis is data-agnostic: it only looks at port numbers and
    their registered status in the port registry, not field values.
    """
    server_score = 0.0
    reasons: list[str] = []

    for entry in ports:
        port = int(entry["port"])
        
        # Score based on registered service ports with stable lifecycle
        # (not ephemeral client ports)
        if entry["registered"] and entry["ephemeral_likelihood"] == "unlikely":
            increment = 2.5
            if entry["observations"] >= 3:
                increment += 0.5
            server_score += increment
            if len(reasons) < 4:
                label = entry.get("service_name") or str(port)
                reasons.append(f"Listening on service port {port} ({label})")

    # Classify based on score
    if server_score >= 2.5:
        classification = "likely_server"
    elif server_score > 0:
        classification = "server"
    else:
        classification = "inconclusive"

    confidence = int(min(95, 50 + server_score * 10)) if server_score > 0 else 20
    
    return {
        "classification": classification,
        "confidence": confidence,
        "listening_score": round(server_score, 2),
        "reasons": reasons[:4],
    }


def _score_os_families(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score OS families based on listening ports.
    
    This analysis is data-agnostic: it only uses the port numbers themselves,
    which are hardware/protocol constants, not data schema-dependent.
    
    Important: Applies sample-size penalties when OS evidence is limited.
    With only 1-2 matching ports, confidence is reduced significantly because:
    - A single Windows port (137, 445, 389, etc.) doesn't prove Windows
    - A single Linux port (22, 3268, etc.) doesn't prove Linux
    - Need multiple diverse ports for stronger OS identification
    """
    scores = {"Windows": 0.0, "Linux": 0.0, "macOS": 0.0}
    reasons = {"Windows": [], "Linux": [], "macOS": []}

    # Count how many OS-specific ports we actually found
    os_port_counts = {"Windows": 0, "Linux": 0, "macOS": 0}

    for entry in ports:
        port = int(entry["port"])
        
        # Score based on OS-specific service ports (these are standards, not schema-dependent)
        if port in WINDOWS_SERVICE_PORTS:
            scores["Windows"] += 2.0
            os_port_counts["Windows"] += 1
            reasons["Windows"].append(f"Listening on Windows-associated port {port}")
        if port in LINUX_SERVICE_PORTS:
            scores["Linux"] += 1.5
            os_port_counts["Linux"] += 1
            reasons["Linux"].append(f"Listening on Linux/common Unix-associated port {port}")
        if port in MACOS_SERVICE_PORTS:
            scores["macOS"] += 2.0
            os_port_counts["macOS"] += 1
            reasons["macOS"].append(f"Listening on macOS/Apple-associated port {port}")

    total = sum(scores.values())
    if total <= 0:
        return []

    # Apply sample-size penalty: small samples (< 3 matching ports) reduce confidence
    total_ports_matching = sum(os_port_counts.values())
    sample_size_penalty = 1.0
    if total_ports_matching <= 2:
        # With only 1-2 ports, reduce confidence significantly
        sample_size_penalty = 0.5
        logger.info(
            "[%s] Applied sample-size penalty (only %d matching ports) — confidence reduced by 50%%",
            SKILL_NAME,
            total_ports_matching,
        )
    elif total_ports_matching <= 4:
        # With 3-4 ports, reduce confidence moderately
        sample_size_penalty = 0.75
        logger.info(
            "[%s] Applied sample-size penalty (only %d matching ports) — confidence reduced by 25%%",
            SKILL_NAME,
            total_ports_matching,
        )

    likelihoods: list[dict[str, Any]] = []
    for family, score in scores.items():
        if score <= 0:
            continue
        normalized = round(score / total, 2)
        
        # Determine confidence BEFORE applying penalty
        if normalized >= 0.7 and score >= 3.0:
            base_confidence = "high"
        elif normalized >= 0.5 and score >= 2.0:
            base_confidence = "medium"
        else:
            base_confidence = "low"
        
        # Apply sample-size penalty to confidence level
        if sample_size_penalty < 1.0:
            if base_confidence == "high":
                confidence = "medium"  # Reduce high → medium
            elif base_confidence == "medium":
                confidence = "low"  # Reduce medium → low
            else:
                confidence = "low"  # Keep low as is
        else:
            confidence = base_confidence
        
        likelihoods.append({
            "family": family,
            "score": normalized,
            "confidence": confidence,
            "reasons": reasons[family][:3],
        })

    likelihoods.sort(key=lambda entry: entry["score"], reverse=True)
    return likelihoods


def run(context: dict) -> dict:
    """
    Run IP fingerprinting analysis on pre-aggregated port data.
    
    ARCHITECTURE:
    This skill receives pre-aggregated port counts from opensearch_querier
    (orchestrated by LLM) and performs purely analytical tasks:
    - Port enrichment (service name, registration status)
    - Role classification (server vs client)
    - OS family likelihood scoring
    
    This skill is COMPLETELY DATA-AGNOSTIC:
    - Does NOT know about field names
    - Does NOT perform queries
    - Does NOT discover schema
    - Only performs analysis on received data
    
    Expected inputs:
    - parameters.ip: Target IP address to fingerprint
    - parameters.aggregated_ports: Pre-computed port counts {port: {observations: int, ...}, ...}
    - config: Configuration for port registry
    
    Typical flow (LLM-orchestrated):
    1. LLM sees user request: "Fingerprint 192.168.0.17"
    2. LLM invokes ip_fingerprinter with target IP
    3. LLM calls fields_querier → gets field mappings (destination_ip, destination_port, etc.)
    4. LLM calls opensearch_querier with:
           target_ip="192.168.0.17"
           query_type="ip_fingerprinting" 
           (LLM instructs it to aggregate destination ports where dest_ip=target)
    5. opensearch_querier returns aggregated_ports
    6. ip_fingerprinter (this function) receives aggregated_ports in parameters
    7. Returns fingerprint analysis
    """
    parameters = context.get("parameters", {}) or {}
    previous_results = context.get("previous_results", {}) or {}
    cfg = context.get("config")
    force_update = bool(parameters.get("force_update", False))

    # Load port registry for service classification
    registry = load_port_registry(cfg, force_update=force_update)
    
    # Extract target IP (from parameters or question text)
    target_ip = _extract_ip(parameters, previous_results)
    if not target_ip:
        return {
            "status": "error",
            "error": "No target IP was provided. LLM should extract and pass parameters.ip.",
            "registry_status": {
                "action": registry.action,
                "source": registry.source,
                "cache_path": registry.cache_path,
                "warning": registry.warning,
            },
        }

    # Get pre-aggregated ports from parameters (provided by LLM orchestration via opensearch_querier)
    aggregated_ports = parameters.get("aggregated_ports") or {}
    if not isinstance(aggregated_ports, dict):
        aggregated_ports = {}
    
    # FALLBACK: If aggregated_ports not in parameters, try to extract from previous opensearch_querier results
    if not aggregated_ports and previous_results:
        os_result = previous_results.get("opensearch_querier", {})
        if os_result.get("status") == "ok":
            prior_aggregated_ports = os_result.get("aggregated_ports") or {}
            if isinstance(prior_aggregated_ports, dict) and prior_aggregated_ports:
                logger.info(
                    "[%s] Found %d aggregated ports from opensearch_querier in previous_results",
                    SKILL_NAME,
                    len(prior_aggregated_ports),
                )
                for port_value, aggregate in prior_aggregated_ports.items():
                    try:
                        port_num = int(port_value)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(aggregate, dict):
                        aggregate = {}
                    aggregated_ports[port_num] = {
                        "observations": int(aggregate.get("observations", 0) or 0),
                        "protocols": list(aggregate.get("protocols") or []),
                        "is_known": bool(aggregate.get("is_known", True)),
                    }

    if not aggregated_ports:
        logger.warning(
            "[%s] No aggregated ports provided in parameters or previous_results. "
            "LLM should orchestrate opensearch_querier to compute aggregated_ports and pass to ip_fingerprinter.",
            SKILL_NAME
        )
    
    # Analyze ports: enrichment, role inference, OS scoring
    ports, evidence = _analyze_ports(aggregated_ports, registry)

    result = {
        "status": "ok" if ports else "no_data",
        "ip": target_ip,
        "ports": ports,
        "port_summary": {
            "listening_ports": [entry["port"] for entry in ports],
            "registered_ports": [entry["port"] for entry in ports if entry["registered"]],
            "unregistered_ports": [entry["port"] for entry in ports if not entry["registered"]],
        },
        "likely_role": _infer_role(ports),
        "os_family_likelihoods": _score_os_families(ports),
        "registry_status": {
            "action": registry.action,
            "source": registry.source,
            "cache_path": registry.cache_path,
            "warning": registry.warning,
        },
    }

    if not ports:
        result["reason"] = "No listening ports found in aggregated data for this IP."
    
    return result
