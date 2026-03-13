"""
skills/network_baseliner/logic.py

Behavioral network baseliner.  Analyzes recent network logs and stores
behavioral baselines in the RAG vector index: traffic patterns, IP/port
relationships, protocol distributions, DNS activity, and GeoIP data.

Field-schema documentation (discovering what fields exist and their types)
is handled by fields_baseliner, which writes data/fields_rag.json.

Features:
  - Stores up to MAX_BASELINE_DOCS (100) docs; evicts oldest on overflow.
  - force_refresh=True (via parameters) deletes all existing baseline docs
    before rebuilding the index from scratch.

Context keys consumed:
    context["db"]         -> BaseDBConnector
    context["llm"]        -> BaseLLMProvider
    context["memory"]     -> Memory instance (StateBackedMemory or CheckpointBackedMemory)
    context["config"]     -> Config
    context["parameters"] -> {"force_refresh": bool}  # optional
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.query_builder import discover_field_mappings

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
SKILL_NAME = "network_baseliner"
MAX_BASELINE_DOCS = 100   # cap: evict oldest docs when index exceeds this


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


def _execute_search_with_llm_repair(db: Any, llm: Any, index: str, query: dict, size: int = 100) -> list[dict]:
    """
    Execute search with intelligent repair on malformed queries.
    
    Uses QueryRepairMemory to remember successful fixes and avoid redundant LLM calls.
    Retries up to 3 times with progressively detailed prompts.
    """
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
            logger.error("[%s] Unexpected search error (type: %s): %s", SKILL_NAME, type(exc).__name__, exc)
            return []


def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db = context.get("db")
    llm = context.get("llm")
    memory = context.get("memory")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    force_refresh = parameters.get("force_refresh", False)

    if db is None or llm is None:
        msg = "[%s] db or llm not available — cannot proceed."
        if force_refresh:
            # Startup requires database and LLM
            logger.error(msg, SKILL_NAME)
            return {"status": "error", "reason": "database or LLM unavailable on startup"}
        else:
            logger.warning(msg, SKILL_NAME)
            return {"status": "skipped", "reason": "no db/llm"}

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    logs_index   = cfg.get("db", "logs_index",   default="securityclaw-logs")
    vector_index = cfg.get("db", "vector_index",  default="securityclaw-vectors")

    # ── Test database connection on startup ────────────────────────────────────
    if force_refresh:
        try:
            # Simple test to verify database is reachable
            if hasattr(db, "_client"):
                db._client.info()
        except Exception as exc:
            logger.error("[%s] Database connection failed on startup: %s", SKILL_NAME, exc)
            return {"status": "error", "reason": f"database connection failed: {str(exc)[:100]}"}

    # ── 0a. Force-refresh: wipe all existing behavioural baseline docs ────────
    if force_refresh:
        logger.info("[%s] force_refresh=True — deleting all existing baseline docs.", SKILL_NAME)
        _delete_all_baseline_docs(db, llm, vector_index)

    # ── 0. Discover field mappings from RAG ────────────────────────────────────
    field_mappings = discover_field_mappings(db, llm)
    logger.info("[%s] Discovered field mappings: %s", SKILL_NAME, field_mappings.keys())

    # ── 1. Fetch recent logs (last 6 hours) ──────────────────────────────────
    since = _epoch_ms_ago(hours=6)
    query = {
        "query": {
            "range": {"@timestamp": {"gte": since, "format": "epoch_millis"}}
        },
    }
    raw_logs = _execute_search_with_llm_repair(db, llm, logs_index, query, size=10000)

    if not raw_logs:
        logger.info("[%s] No logs found in the last 6 hours.", SKILL_NAME)
        return {"status": "no_data"}

    # ── 2. Detect network/sensor identifier and group logs ────────────────────
    identifier_field = _detect_identifier_field(raw_logs)
    grouped_logs = _group_logs_by_identifier(raw_logs, identifier_field)
    
    logger.info(
        "[%s] Detected identifier field: %s. Found %d networks/sensors.",
        SKILL_NAME,
        identifier_field,
        len(grouped_logs),
    )

    # ── 3. Check embedding dimension vs existing index dimension ─────────────
    # Use the LLM's actual embedding dimension
    current_embed_dim = llm.embedding_dimension if llm is not None else None
    index_dim = _get_index_dim(db, vector_index)
    fresh_start = False

    if current_embed_dim and index_dim and current_embed_dim != index_dim:
        logger.warning(
            "[%s] Embedding dimension mismatch: index has %d dims, embed model produces %d dims. "
            "Deleting and recreating vector index with new dimensions…",
            SKILL_NAME, index_dim, current_embed_dim,
        )
        # Delete the incompatible index so RAGEngine creates a fresh one
        try:
            if hasattr(db, '_client'):
                client = db._client
                if client.indices.exists(index=vector_index):
                    client.indices.delete(index=vector_index)
                    logger.info("[%s] Deleted vector index '%s' due to dimension mismatch.", SKILL_NAME, vector_index)
        except Exception as exc:
            logger.warning("[%s] Could not delete vector index: %s", SKILL_NAME, exc)
        fresh_start = True
    elif index_dim is None:
        fresh_start = True  # No existing index — starting fresh

    # ── 4. Read existing baselines before RAGEngine init ─
    existing_baselines_by_id: dict[str, dict[str, str]] = {}
    if not fresh_start:
        for ident in grouped_logs:
            prior = _fetch_existing_baselines(db, llm, vector_index, ident)
            if prior:
                existing_baselines_by_id[ident] = prior
                logger.info(
                    "[%s] Loaded %d existing baseline docs for '%s' — will update them.",
                    SKILL_NAME, len(prior), ident,
                )

    # ── 5. Init RAGEngine (creates fresh index with correct dimensions) ────────
    from core.rag_engine import RAGEngine

    rag = RAGEngine(db=db, llm=llm)

    all_stored_docs = []
    for identifier, logs_group in grouped_logs.items():
        if not logs_group:
            continue
        
        logger.info(
            "[%s] Processing %s (%d logs)…",
            SKILL_NAME,
            identifier,
            len(logs_group),
        )
        
        # Analyze this network/sensor's logs
        analytics = _analyze_network_logs(logs_group, field_mappings)
        analytics_text = _format_analytics(analytics)

        # Include any existing baselines for this identifier as prior context
        prior_baselines = existing_baselines_by_id.get(identifier, {})

        # Generate baselines specific to this network/sensor
        baselines = _generate_baseline_documents(
            analytics,
            analytics_text,
            llm,
            instruction,
            existing_baselines=prior_baselines,
        )

        if not baselines:
            logger.warning(
                "[%s] Failed to generate baselines for %s",
                SKILL_NAME,
                identifier,
            )
            continue

        # Check which baselines have actually changed before storing
        stored_count = 0
        for baseline in baselines:
            category = baseline["category"]
            summary = baseline["summary"]
            
            # Extract current metrics
            metrics = _extract_analytics_metrics(analytics, category)
            
            # Check if this baseline has changed
            prior_baseline_text = prior_baselines.get(category)
            has_changed, change_summary = _has_baseline_changed(metrics, prior_baseline_text)
            
            if not has_changed:
                logger.info(
                    "[%s] Skipping %s for %s — %s",
                    SKILL_NAME,
                    category,
                    identifier,
                    change_summary,
                )
                continue
            
            # Store baseline since it changed
            doc_id = rag.store(
                text=summary,
                category=category,
                source=SKILL_NAME,
                metadata={
                    "identifier_field": identifier_field,
                    "identifier_value": identifier,
                    "dimension": category.replace("network_baseline_", ""),
                    "change_summary": change_summary,
                },
            )
            all_stored_docs.append(
                {
                    "category": category,
                    "identifier": identifier,
                    "doc_id": doc_id,
                    "change_summary": change_summary,
                }
            )
            stored_count += 1
            logger.info(
                "[%s] Stored %s for %s (id=%s) — %s",
                SKILL_NAME,
                category,
                identifier,
                doc_id[:8],
                change_summary,
            )
        
    # ── 4b. Evict docs beyond the 100-doc cap ──────────────────────────────────
    _evict_old_baseline_docs(db, llm, vector_index)

    # ── 4. Update agent memory ────────────────────────────────────────────────
    if memory:
        memory.add_decision(
            f"NetworkBaseliner analyzed {len(grouped_logs)} networks/sensors across "
            f"{len(raw_logs)} logs. Stored {len(all_stored_docs)} baseline documents "
            f"with context: {identifier_field}={', '.join(grouped_logs.keys())}"
        )

    return {
        "status": "ok",
        "records_processed": len(raw_logs),
        "networks_analyzed": len(grouped_logs),
        "documents_stored": len(all_stored_docs),
        "identifier_field": identifier_field,
        "identifiers": list(grouped_logs.keys()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Network/Sensor Detection and Grouping
# ──────────────────────────────────────────────────────────────────────────────

def _detect_identifier_field(logs: list[dict]) -> str:
    """
    Detect which field is the likely network/sensor identifier.
    
    Candidates (in order of preference):
      1. agent_id, sensor_id, client_id, source_id - explicit identifiers
      2. host.hostname, hostname - machine identity
      3. host.ip, source.ip (first octet or /24) - network segment
      4. event.source - log source
      5. None - treat all logs as single network
    """
    if not logs:
        return "sensor_id"  # Default fallback
    
    # Sample first 100 logs to check for identifier fields
    sample = logs[:100]
    
    # Candidate fields to check
    candidates = [
        ("agent_id", "exact"),
        ("sensor_id", "exact"),
        ("client_id", "exact"),
        ("source_id", "exact"),
        ("host.hostname", "exact"),
        ("hostname", "exact"),
        ("event.source", "exact"),
        ("source.ip", "subnet"),  # Group by /24 subnet
        ("host.ip", "subnet"),
    ]
    
    for field_path, mode in candidates:
        found_values = set()
        populated_count = 0
        
        for log in sample:
            value = _extract_value(log, [field_path])
            if value is not None:
                populated_count += 1
                if mode == "subnet":
                    # Extract /24 subnet from IP
                    if isinstance(value, str):
                        parts = value.split(".")
                        if len(parts) == 4:
                            value = ".".join(parts[:3]) + ".0"
                found_values.add(value)
        
        # If this field is populated in >80% of samples and has multiple distinct values
        if populated_count >= len(sample) * 0.8 and len(found_values) > 1:
            logger.info(
                "[%s] Auto-detected identifier field: %s (found %d distinct values)",
                SKILL_NAME,
                field_path,
                len(found_values),
            )
            return field_path
    
    # No good identifier field found - treat all as one network
    logger.info("[%s] No multi-network identifier detected; treating all logs as single network", SKILL_NAME)
    return "sensor_id"


def _group_logs_by_identifier(logs: list[dict], identifier_field: str) -> dict[str, list[dict]]:
    """
    Group logs by the identified field to separate network/sensor baselines.
    
    Returns dict mapping identifier value → list of logs for that network/sensor.
    """
    groups = defaultdict(list)
    
    for log in logs:
        value = _extract_value(log, [identifier_field])
        
        if value is None:
            value = "unknown"
        
        # For subnet grouping, extract /24
        if "." in str(value):
            parts = str(value).split(".")
            if len(parts) == 4 and identifier_field in ("source.ip", "host.ip"):
                try:
                    int(parts[0])  # Verify it's an IP
                    value = ".".join(parts[:3]) + ".0"
                except ValueError:
                    pass
        
        groups[str(value)].append(log)
    
    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding dimension helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_embed_dim(llm) -> int | None:
    """Return the current embedding dimension produced by the LLM provider, or None."""
    try:
        return len(llm.embed("test"))
    except Exception as exc:
        logger.warning("[%s] Could not detect embedding dimension: %s", SKILL_NAME, exc)
        return None


def _get_index_dim(db, vector_index: str) -> int | None:
    """Return the embedding dimension stored in the index mapping, or None if unavailable."""
    try:
        if not hasattr(db, "_client"):
            return None
        client = db._client
        if not client.indices.exists(index=vector_index):
            return None
        mapping = client.indices.get_mapping(index=vector_index)
        return (
            mapping.get(vector_index, {})
            .get("mappings", {})
            .get("properties", {})
            .get("embedding", {})
            .get("dimension")
        )
    except Exception as exc:
        logger.warning("[%s] Could not read index dimension: %s", SKILL_NAME, exc)
        return None


def _fetch_existing_baselines(db, llm, vector_index: str, identifier_value: str) -> dict[str, str]:
    """Return existing baseline docs for this network/sensor as {category: text}."""
    try:
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"source": SKILL_NAME}},
                        {"term": {"identifier_value": identifier_value}},
                    ]
                }
            },
            "_source": ["text", "category"],
        }
        docs = _execute_search_with_llm_repair(db, llm, vector_index, query, size=50)
        result: dict[str, str] = {}
        for doc in docs:
            cat = doc.get("category", "")
            text = doc.get("text", "")
            if cat and text:
                result[cat] = text
        return result
    except Exception as exc:
        logger.warning(
            "[%s] Could not fetch existing baselines for '%s': %s",
            SKILL_NAME, identifier_value, exc,
        )
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Baseline doc cap helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_baseline_docs(db: Any, vector_index: str) -> int:
    """Return the number of docs stored by network_baseliner in the vector index."""
    try:
        query = {
            "query": {"bool": {"filter": [{"term": {"source": SKILL_NAME}}]}},
            "size": 0,
        }
        if hasattr(db, "_client"):
            resp = db._client.count(index=vector_index, body={"query": query["query"]})
            return int(resp.get("count", 0))
        # Fallback: search and count
        results = db.search(vector_index, query, size=1000) or []
        return len(results)
    except Exception as exc:
        logger.debug("[%s] Could not count baseline docs: %s", SKILL_NAME, exc)
        return 0


def _delete_all_baseline_docs(db: Any, llm: Any, vector_index: str) -> None:
    """Delete ALL behavioral baseline docs stored by network_baseliner."""
    try:
        query = {
            "query": {"bool": {"filter": [{"term": {"source": SKILL_NAME}}]}},
            "_source": False,
            "size": 1000,
        }
        docs = db.search(vector_index, query, size=1000) or []
        if not docs:
            return
        if hasattr(db, "_client"):
            db._client.delete_by_query(
                index=vector_index,
                body={"query": query["query"]},
                conflicts="proceed",
                refresh=True,
            )
            logger.info("[%s] Deleted all %d baseline docs (force_refresh).", SKILL_NAME, len(docs))
        else:
            for doc in docs:
                doc_id = doc.get("_id")
                if doc_id:
                    try:
                        db._client.delete(index=vector_index, id=doc_id, refresh=True)
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("[%s] Could not delete baseline docs: %s", SKILL_NAME, exc)


def _evict_old_baseline_docs(db: Any, llm: Any, vector_index: str) -> None:
    """
    If the number of baseline docs exceeds MAX_BASELINE_DOCS, delete the
    oldest ones (by timestamp ASC) to keep the index within the cap.
    """
    try:
        count = _count_baseline_docs(db, vector_index)
        if count <= MAX_BASELINE_DOCS:
            return

        overage = count - MAX_BASELINE_DOCS
        logger.info(
            "[%s] Index has %d baseline docs (cap=%d); evicting %d oldest.",
            SKILL_NAME, count, MAX_BASELINE_DOCS, overage,
        )

        # Fetch oldest docs sorted by timestamp ascending
        query = {
            "query": {"bool": {"filter": [{"term": {"source": SKILL_NAME}}]}},
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": False,
            "size": overage + 10,   # small buffer
        }
        oldest_docs = db.search(vector_index, query, size=overage + 10) or []
        to_delete = oldest_docs[:overage]

        if hasattr(db, "_client"):
            ids = [d["_id"] for d in to_delete if d.get("_id")]
            if ids:
                db._client.delete_by_query(
                    index=vector_index,
                    body={"query": {"ids": {"values": ids}}},
                    conflicts="proceed",
                    refresh=True,
                )
                logger.info("[%s] Evicted %d old baseline docs.", SKILL_NAME, len(ids))
        else:
            for doc in to_delete:
                doc_id = doc.get("_id")
                if doc_id:
                    try:
                        db._client.delete(index=vector_index, id=doc_id, refresh=True)
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning("[%s] Eviction failed (non-critical): %s", SKILL_NAME, exc)


def _with_prior(prompt: str, existing_baselines: dict, category: str) -> str:
    """Append existing baseline text to the prompt so the LLM can update it."""
    prior = existing_baselines.get(category, "")
    if not prior:
        return prompt
    return (
        prompt
        + "\n\nPRIOR BASELINE (incorporate into your updated summary;"
        " note any changes if network behaviour has evolved):\n"
        + prior
    )


def _extract_analytics_metrics(analytics: dict, category: str) -> dict:
    """Extract key metrics from analytics for comparison against existing baselines."""
    metrics = {
        "category": category,
        "unique_src_ips": len(analytics.get("source_ips", {})),
        "unique_dst_ips": len(analytics.get("dest_ips", {})),
        "unique_src_ports": len(analytics.get("source_ports", {})),
        "unique_dst_ports": len(analytics.get("dest_ports", {})),
        "unique_protocols": len(analytics.get("protocols", {})),
        "unique_dns_domains": len(analytics.get("dns_queries", {})),
        "total_flows": analytics.get("flow_stats", {}).get("total_flows", 0),
        "total_bytes": analytics.get("flow_stats", {}).get("total_bytes", 0),
        "total_packets": analytics.get("flow_stats", {}).get("total_packets", 0),
        "unique_fields": len(analytics.get("discovered_fields", {})),
    }
    
    # Extract top items for comparison
    src_ips = analytics.get("source_ips", {})
    metrics["top_5_src_ips"] = sorted(src_ips.items(), key=lambda x: x[1], reverse=True)[:5]
    
    dst_ips = analytics.get("dest_ips", {})
    metrics["top_5_dst_ips"] = sorted(dst_ips.items(), key=lambda x: x[1], reverse=True)[:5]
    
    dst_ports = analytics.get("dest_ports", {})
    metrics["top_5_dst_ports"] = sorted(dst_ports.items(), key=lambda x: x[1], reverse=True)[:5]
    
    protocols = analytics.get("protocols", {})
    metrics["top_protocols"] = sorted(protocols.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return metrics


def _has_baseline_changed(new_metrics: dict, baseline_text: str | None) -> tuple[bool, str]:
    """
    Compare new metrics against existing baseline text.
    Returns (changed: bool, change_summary: str).
    
    Detects meaningful changes like new IPs, port distribution changes, traffic growth, etc.
    """
    if not baseline_text:
        return True, "New baseline (no prior baseline exists)"
    
    changes = []
    
    # Try to parse metrics from baseline text
    # Look for patterns like "Total flows: 10000" or "unique IPs: 42"
    
    # Check for field count changes (indicates schema evolution)
    if "unique_fields" in baseline_text:
        # Extract old field count using regex
        match = re.search(r"(\d+) fields", baseline_text)
        if match:
            old_field_count = int(match.group(1))
            new_field_count = new_metrics["unique_fields"]
            if abs(new_field_count - old_field_count) > 2:  # Allow small variance
                changes.append(f"Field count: {old_field_count} → {new_field_count}")
    
    # Check for flow count changes (significant growth/shrinkage)
    if "Total flows:" in baseline_text or "total flows:" in baseline_text.lower():
        match = re.search(r"[Tt]otal flows[:\s]+(\d+)", baseline_text)
        if match:
            old_flows = int(match.group(1))
            new_flows = new_metrics["total_flows"]
            change_pct = abs(new_flows - old_flows) / max(old_flows, 1) * 100
            if change_pct > 10:  # >10% change
                changes.append(f"Total flows: {old_flows:,} → {new_flows:,} ({change_pct:.0f}% change)")
    
    # Check for traffic volume changes
    if "Total bytes:" in baseline_text:
        match = re.search(r"[Tt]otal bytes[:\s]+(\d+)", baseline_text)
        if match:
            old_bytes = int(match.group(1))
            new_bytes = new_metrics["total_bytes"]
            change_pct = abs(new_bytes - old_bytes) / max(old_bytes, 1) * 100
            if change_pct > 15:  # >15% change
                changes.append(f"Traffic volume: {old_bytes:,} → {new_bytes:,} bytes")
    
    # Check for new IPs in top talkers
    if "TOP SOURCE IPs" in baseline_text or "TOP DESTINATION IPs" in baseline_text:
        new_src_ips = set(ip for ip, _ in new_metrics.get("top_5_src_ips", []))
        new_dst_ips = set(ip for ip, _ in new_metrics.get("top_5_dst_ips", []))
        
        # Simple heuristic: if top 5 IPs have changed, something changed
        baseline_mentions_ips = bool(
            re.search(r"\d+\.\d+\.\d+\.\d+", baseline_text)
        )
        
        if baseline_mentions_ips and (new_src_ips or new_dst_ips):
            # Extract IPs from baseline text
            old_ips = set(re.findall(r"\d+\.\d+\.\d+\.\d+", baseline_text))
            if old_ips and new_src_ips | new_dst_ips:
                new_ips = (new_src_ips | new_dst_ips) - old_ips
                if new_ips:
                    changes.append(f"New IPs detected: {', '.join(sorted(new_ips)[:3])}")
    
    # Check for port usage changes
    if "TOP DESTINATION PORTS" in baseline_text:
        match = re.findall(r"(\d+)/[^:]+:\s+(\d+) flows", baseline_text)
        if match:
            old_top_ports = {int(m[0]): int(m[1]) for m in match[:3]}
            new_top_ports = {port: count for port, count in new_metrics.get("top_5_dst_ports", [])[:3]}
            
            # Check if port distribution significantly changed
            old_port_set = set(old_top_ports.keys())
            new_port_set = set(new_top_ports.keys())
            
            if old_port_set != new_port_set:
                added = new_port_set - old_port_set
                removed = old_port_set - new_port_set
                change_note = []
                if added:
                    change_note.append(f"new ports {added}")
                if removed:
                    change_note.append(f"removed ports {removed}")
                if change_note:
                    changes.append(f"Port distribution changed: {', '.join(change_note)}")
    
    if changes:
        return True, "; ".join(changes)
    else:
        return False, "No significant changes detected"


# ──────────────────────────────────────────────────────────────────────────────

def _generate_baseline_documents(
    analytics: dict,
    analytics_text: str,
    llm,
    instruction: str,
    existing_baselines: dict | None = None,
) -> list[dict]:
    """
    Generate behavioral baseline documents for the RAG vector index.

    Covers: traffic-flow patterns, IP-to-IP communication, port/protocol
    distribution, DNS activity, and GeoIP geographic data.

    Field-schema documentation is handled exclusively by fields_baseliner
    (data/fields_rag.json) and is NOT stored in the vector index here.

    If existing_baselines ({category: prior_text}) is supplied, each LLM
    prompt includes the prior summary so the model can update rather than
    replace it.

    Returns list of dicts with "summary" and "category" for each baseline.
    """
    existing_baselines = existing_baselines or {}
    baselines = []

    # Pull analytics fields used across all baseline types
    flow_stats = analytics.get("flow_stats", {})
    ip_pairs   = analytics.get("ip_pairs", {})

    # ── NETWORK BEHAVIOR BASELINE ──────────────────────────────────────────────

    if ip_pairs or flow_stats:
        behavior_lines = [
            "NETWORK BASELINE BEHAVIOR PATTERNS",
            "=" * 60,
            "",
            "FLOW STATISTICS:",
            f"  Total flows: {flow_stats.get('total_flows', 0):,}",
            f"  Total bytes: {flow_stats.get('total_bytes', 0):,}",
            f"  Total packets: {flow_stats.get('total_packets', 0):,}",
            f"  Average bytes per flow: {flow_stats.get('avg_bytes_per_flow', 0):.1f}",
            f"  Average duration: {flow_stats.get('avg_duration_us', 0):.0f} microseconds",
            "",
            "COMMON IP-TO-IP CONNECTIONS (Source → Destination):",
        ]
        
        for (src, dst), count in list(ip_pairs.items())[:20]:
            pct = (count / max(flow_stats.get("total_flows", 1), 1)) * 100
            behavior_lines.append(f"  {src} → {dst}: {count} flows ({pct:.1f}%)")
        
        behavior_lines.extend([
            "",
            "=" * 60,
            "These patterns represent the established baseline for this network.",
            "Use these to identify anomalies or unexpected communication patterns.",
        ])
        
        behavior_text = "\n".join(behavior_lines)
        response = llm.chat([
            {"role": "system", "content": "You are documenting network baseline behavior. Be precise about traffic patterns."},
            {"role": "user", "content": _with_prior(
                f"Document this network baseline:\n\n{behavior_text}",
                existing_baselines,
                "network_behavior_baseline"
            )},
        ])
        baselines.append({
            "summary": response,
            "category": "network_behavior_baseline",
        })
    
    # ── PROTOCOL & PORT ANALYSIS BASELINE ──────────────────────────────────────
    # Document which protocols and ports are used in normal baseline traffic
    protocols = analytics.get("protocols", {})
    dest_ports = analytics.get("dest_ports", {})
    source_ports = analytics.get("source_ports", {})
    services = analytics.get("services", {})
    
    if protocols or dest_ports:
        protocol_lines = [
            "PROTOCOL AND PORT USAGE BASELINE",
            "=" * 60,
            "",
        ]
        
        if protocols:
            protocol_lines.append("PROTOCOL DISTRIBUTION:")
            total_flows = max(flow_stats.get("total_flows", 1), 1)
            for proto, count in list(protocols.items())[:15]:
                pct = (count / total_flows) * 100
                protocol_lines.append(f"  {proto}: {count} flows ({pct:.1f}%)")
            protocol_lines.append("")
        
        if dest_ports:
            protocol_lines.append("TOP DESTINATION PORTS (Server Ports):")
            for port, count in list(dest_ports.items())[:20]:
                service = services.get(port, "unknown")
                pct = (count / total_flows) * 100
                protocol_lines.append(f"  {port}/{service}: {count} flows ({pct:.1f}%)")
            protocol_lines.append("")
        
        if source_ports:
            protocol_lines.append("TOP SOURCE PORTS (Client Ports):")
            for port, count in list(source_ports.items())[:10]:
                pct = (count / total_flows) * 100
                protocol_lines.append(f"  {port}: {count} flows ({pct:.1f}%)")
        
        protocol_lines.extend([
            "",
            "=" * 60,
            "These ports represent normal baseline communication.",
            "Unused ports or unexpected protocols may indicate threats.",
        ])
        
        protocol_text = "\n".join(protocol_lines)
        response = llm.chat([
            {"role": "system", "content": "You are documenting protocol and port usage. Be specific about what is normal."},
            {"role": "user", "content": _with_prior(
                f"Document this protocol baseline:\n\n{protocol_text}",
                existing_baselines,
                "protocol_port_baseline"
            )},
        ])
        baselines.append({
            "summary": response,
            "category": "protocol_port_baseline",
        })
    
    # ── IP RELATIONSHIPS BASELINE ──────────────────────────────────────────────
    # Document which IPs communicate internally, which are external, geographic patterns
    src_ips = analytics.get("source_ips", {})
    dst_ips = analytics.get("dest_ips", {})
    geoip = analytics.get("geoip_data", {})
    
    if src_ips or dst_ips or geoip:
        ip_lines = [
            "IP COMMUNICATION BASELINE",
            "=" * 60,
            "",
        ]
        
        if src_ips:
            ip_lines.append("TOP SOURCE IPs (Internal Hosts):")
            for ip, count in list(src_ips.items())[:15]:
                pct = (count / max(flow_stats.get("total_flows", 1), 1)) * 100
                is_private = _is_private_ip(ip)
                ip_type = "Internal" if is_private else "External"
                ip_lines.append(f"  {ip} ({ip_type}): {count} flows ({pct:.1f}%)")
            ip_lines.append("")
        
        if dst_ips:
            ip_lines.append("TOP DESTINATION IPs (Targets):")
            for ip, count in list(dst_ips.items())[:15]:
                pct = (count / max(flow_stats.get("total_flows", 1), 1)) * 100
                is_private = _is_private_ip(ip)
                ip_type = "Internal" if is_private else "External"
                location = geoip.get(ip, "Unknown")
                ip_lines.append(f"  {ip} ({ip_type}) from {location}: {count} flows ({pct:.1f}%)")
        
        ip_lines.extend([
            "",
            "=" * 60,
            "These IPs represent established communication patterns.",
            "New or unexpected IPs may warrant investigation.",
        ])
        
        ip_text = "\n".join(ip_lines)
        response = llm.chat([
            {"role": "system", "content": "You are documenting IP communication patterns. Distinguish internal vs external."},
            {"role": "user", "content": _with_prior(
                f"Document this IP baseline:\n\n{ip_text}",
                existing_baselines,
                "ip_communication_baseline"
            )},
        ])
        baselines.append({
            "summary": response,
            "category": "ip_communication_baseline",
        })
    
    # ── DNS BASELINE ────────────────────────────────────────────────────────────
    # Document which domains are queried in normal baseline traffic
    dns = analytics.get("dns_queries", {})
    
    if dns:
        dns_lines = [
            "DNS QUERY BASELINE",
            "=" * 60,
            "",
            "COMMON DNS QUERIES:",
        ]
        
        for domain, count in list(dns.items())[:20]:
            dns_lines.append(f"  {domain}: {count} queries")
        
        dns_lines.extend([
            "",
            "=" * 60,
            "These DNS queries represent normal baseline domain lookups.",
            "Unexpected DNS queries to suspicious domains may indicate threats.",
        ])
        
        dns_text = "\n".join(dns_lines)
        response = llm.chat([
            {"role": "system", "content": "You are documenting DNS query patterns. List the domains accessed."},
            {"role": "user", "content": _with_prior(
                f"Document this DNS baseline:\n\n{dns_text}",
                existing_baselines,
                "dns_baseline"
            )},
        ])
        baselines.append({
            "summary": response,
            "category": "dns_baseline",
        })
    
    return baselines


def _is_private_ip(ip: str) -> bool:
    """Check if IP is in private ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)."""
    if not isinstance(ip, str):
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        first = int(parts[0])
        second = int(parts[1]) if first == 172 else 0
        return (first == 10 or 
                (first == 172 and 16 <= second <= 31) or 
                first == 192)
    except (ValueError, IndexError):
        return False


# ──────────────────────────────────────────────────────────────────────────────

def _epoch_ms_ago(hours: int = 6) -> int:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return int(dt.timestamp() * 1000)


def _extract_value(obj: Any, paths: list[str]) -> Any:
    """Recursively extract value from nested dict using multiple potential paths."""
    for path in paths:
        current = obj
        for key in path.split("."):
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = None
                break
        if current is not None:
            return current
    return None


def _analyze_network_logs(logs: list[dict], field_mappings: dict) -> dict:
    """
    Perform comprehensive field-agnostic network analytics.

    Args:
        logs: List of raw log dicts
        field_mappings: Field mappings discovered from RAG (keys:ip_fields, port_fields, text_fields, etc.)
    
    Returns dict with:
      - source_ips: {ip: count}
      - dest_ips: {ip: count}
      - source_ports: {port: count}
      - dest_ports: {port: count}
      - protocols: {protocol: count}
      - directions: {direction: count}
      - ip_pairs: [(src, dst): count]
      - ip_port_pairs: {ip: {port: count}}
      - services: {port: name}
      - geoip_data: {ip: location}
      - dns_queries: {domain: count}
      - flow_stats: {metric: value}
      - discovered_fields: {field_name: count} — schema observation
    """
    source_ips = Counter()
    dest_ips = Counter()
    source_ports = Counter()
    dest_ports = Counter()
    protocols = Counter()
    directions = Counter()
    ip_pairs = Counter()
    ip_port_connections = defaultdict(Counter)  # src_ip -> {dest_ip: count}
    ip_port_usage = defaultdict(Counter)  # ip -> {port: count}
    services = {}
    geoip_data = {}
    dns_queries = Counter()
    discovered_fields = Counter()  # Track which fields we actually find
    
    total_bytes = 0
    total_packets = 0
    durations = []

    for log in logs:
        # Track all top-level fields we discover
        for field in log.keys():
            discovered_fields[field] += 1
        
        # ── Extract source/dest IP and port using discovered fields ───────────
        src_ip_fields = field_mappings.get("source_ip_fields", [])
        dst_ip_fields = field_mappings.get("destination_ip_fields", [])
        src_port_fields = field_mappings.get("source_port_fields", [])
        dst_port_fields = field_mappings.get("destination_port_fields", [])
        
        src_ip = _extract_value(log, src_ip_fields) if src_ip_fields else None
        src_port = _extract_value(log, src_port_fields) if src_port_fields else None
        dst_ip = _extract_value(log, dst_ip_fields) if dst_ip_fields else None
        dst_port = _extract_value(log, dst_port_fields) if dst_port_fields else None
        
        # ── Extract protocol/service info using discovered fields ────────────
        protocol_fields = field_mappings.get("protocol_fields", [])
        service_fields = field_mappings.get("service_fields", [])
        protocol = _extract_value(log, protocol_fields) if protocol_fields else None
        service = _extract_value(log, service_fields) if service_fields else None
        
        # ── Extract direction using discovered fields ──────────────────────────
        direction_fields = field_mappings.get("direction_fields", [])
        direction = _extract_value(log, direction_fields) if direction_fields else None
        
        # ── Extract volume metrics using discovered fields ───────────────────
        src_bytes_fields = field_mappings.get("source_bytes_fields", [])
        dst_bytes_fields = field_mappings.get("destination_bytes_fields", [])
        bytes_fields = field_mappings.get("bytes_fields", [])
        packets_fields = field_mappings.get("packets_fields", [])
        duration_fields = field_mappings.get("duration_fields", [])
        
        src_bytes = _extract_value(log, src_bytes_fields) if src_bytes_fields else None
        dst_bytes = _extract_value(log, dst_bytes_fields) if dst_bytes_fields else None
        total_net_bytes = _extract_value(log, bytes_fields) if bytes_fields else None
        packets = _extract_value(log, packets_fields) if packets_fields else None
        duration = _extract_value(log, duration_fields) if duration_fields else None
        
        # ── Extract GeoIP info using discovered fields ──────────────────────
        geoip_fields = field_mappings.get("geoip_fields", [
            "destination.geo",           # ECS format
            "destination.geoip",
            "geoip",                     # Suricata format (flat)
            "dest_geoip",
        ])
        dst_geo = _extract_value(log, geoip_fields) if geoip_fields else None
        
        if dst_ip and dst_geo:
            # Handle both dict and string formats
            geo_info = dst_geo
            if isinstance(dst_geo, dict):
                country = dst_geo.get("country_name") or dst_geo.get("country") or dst_geo.get("iso_code")
                city = dst_geo.get("city_name") or dst_geo.get("city")
                geo_info = f"{country or '?'} {city or ''}".strip()
            geoip_data[dst_ip] = geo_info
        
        # ── Extract DNS info using discovered fields ─────────────────────────
        dns_fields = field_mappings.get("dns_query_fields", [
            "dns.question.name",
            "dns.query",
            "query"
        ])
        dns_question = _extract_value(log, dns_fields) if dns_fields else None
        if dns_question:
            dns_queries[dns_question] += 1
        
        # ── Aggregate counters ─────────────────────────────────────────────────
        if src_ip:
            source_ips[src_ip] += 1
        if dst_ip:
            dest_ips[dst_ip] += 1
        if src_port:
            source_ports[src_port] += 1
        if dst_port:
            dest_ports[dst_port] += 1
        if protocol:
            protocols[protocol] += 1
        if direction:
            directions[direction] += 1
        
        # ── Track IP-to-IP relationships ───────────────────────────────────────
        if src_ip and dst_ip:
            ip_pairs[(src_ip, dst_ip)] += 1
            ip_port_connections[src_ip][dst_ip] += 1
        
        # ── Track port usage per IP ────────────────────────────────────────────
        if src_ip and src_port:
            ip_port_usage[src_ip][src_port] += 1
        if dst_ip and dst_port:
            ip_port_usage[dst_ip][dst_port] += 1
        
        # ── Map service names to ports ────────────────────────────────────────
        if dst_port and service:
            services[dst_port] = service
        
        # ── Accumulate volume stats ────────────────────────────────────────────
        if src_bytes and isinstance(src_bytes, (int, float)):
            total_bytes += src_bytes
        if dst_bytes and isinstance(dst_bytes, (int, float)):
            total_bytes += dst_bytes
        if total_net_bytes and isinstance(total_net_bytes, (int, float)):
            total_bytes += total_net_bytes
        if packets and isinstance(packets, (int, float)):
            total_packets += packets
        if duration and isinstance(duration, (int, float)):
            durations.append(duration)

    # ── Compute flow statistics ────────────────────────────────────────────────
    avg_duration = sum(durations) / len(durations) if durations else 0
    flow_stats = {
        "total_flows": len(logs),
        "total_bytes": total_bytes,
        "total_packets": total_packets,
        "avg_bytes_per_flow": total_bytes / max(len(logs), 1),
        "avg_duration_us": avg_duration,
    }

    return {
        "source_ips": dict(source_ips.most_common(50)),
        "dest_ips": dict(dest_ips.most_common(50)),
        "source_ports": dict(source_ports.most_common(30)),
        "dest_ports": dict(dest_ports.most_common(30)),
        "protocols": dict(protocols.most_common(10)),
        "directions": dict(directions.most_common(5)),
        "ip_pairs": dict(ip_pairs.most_common(50)),
        "ip_port_connections": {k: dict(v.most_common(20)) for k, v in ip_port_connections.items()},
        "ip_port_usage": {k: dict(v.most_common(20)) for k, v in ip_port_usage.items()},
        "services": services,
        "geoip_data": dict(list(geoip_data.items())[:30]),
        "dns_queries": dict(dns_queries.most_common(30)),
        "flow_stats": flow_stats,
        "discovered_fields": dict(discovered_fields.most_common(50)),  # New: field discovery
    }


def _format_analytics(analytics: dict) -> str:
    """Format comprehensive analytics into readable text for LLM."""
    lines = []
    
    # ── Discovered Field Mapping (Schema Observation) ───────────────────────────
    discovered = analytics.get("discovered_fields", {})
    if discovered:
        lines.append("═ DETECTED FIELDS IN DATA ═")
        lines.append("(This schema information is stored for future queries)")
        for field, count in list(discovered.items())[:20]:
            pct = (count / max(sum(discovered.values()), 1)) * 100
            lines.append(f"  {field}: {pct:.1f}%")
        lines.append("")
    
    # Flow statistics
    stats = analytics.get("flow_stats", {})
    lines.append("═ FLOW STATISTICS ═")
    lines.append(f"  Total flows: {stats.get('total_flows', 0)}")
    lines.append(f"  Total bytes: {stats.get('total_bytes', 0):,}")
    lines.append(f"  Total packets: {stats.get('total_packets', 0):,}")
    lines.append(f"  Avg bytes/flow: {stats.get('avg_bytes_per_flow', 0):.1f}")
    lines.append(f"  Avg duration: {stats.get('avg_duration_us', 0):.0f} µs")
    lines.append("")

    # Protocols
    protocols = analytics.get("protocols", {})
    if protocols:
        lines.append("═ PROTOCOLS ═")
        for proto, count in list(protocols.items())[:10]:
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {proto}: {count} flows ({pct:.1f}%)")
        lines.append("")

    # Directions
    directions = analytics.get("directions", {})
    if directions:
        lines.append("═ TRAFFIC DIRECTION ═")
        for direction, count in directions.items():
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {direction}: {count} flows ({pct:.1f}%)")
        lines.append("")

    # Destination ports
    dest_ports = analytics.get("dest_ports", {})
    services = analytics.get("services", {})
    if dest_ports:
        lines.append("═ TOP DESTINATION PORTS ═")
        for port, count in list(dest_ports.items())[:15]:
            service = services.get(port, "unknown")
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {port}/{service}: {count} flows ({pct:.1f}%)")
        lines.append("")

    # Source ports
    source_ports = analytics.get("source_ports", {})
    if source_ports:
        lines.append("═ TOP SOURCE PORTS ═")
        for port, count in list(source_ports.items())[:10]:
            lines.append(f"  {port}: {count} flows")
        lines.append("")

    # Source and destination IPs
    src_ips = analytics.get("source_ips", {})
    dst_ips = analytics.get("dest_ips", {})
    if src_ips:
        lines.append("═ TOP SOURCE IPs ═")
        for ip, count in list(src_ips.items())[:10]:
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {ip}: {count} flows ({pct:.1f}%)")
        lines.append("")

    if dst_ips:
        lines.append("═ TOP DESTINATION IPs ═")
        for ip, count in list(dst_ips.items())[:10]:
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {ip}: {count} flows ({pct:.1f}%)")
        lines.append("")

    # IP-to-IP relationships
    ip_pairs = analytics.get("ip_pairs", {})
    if ip_pairs:
        lines.append("═ COMMON IP PAIRS (Source → Destination) ═")
        for (src, dst), count in list(ip_pairs.items())[:15]:
            pct = (count / stats.get("total_flows", 1)) * 100
            lines.append(f"  {src} → {dst}: {count} flows ({pct:.1f}%)")
        lines.append("")

    # IP-Port usage (which IPs use which ports)
    ip_port_usage = analytics.get("ip_port_usage", {})
    if ip_port_usage:
        lines.append("═ IP-PORT USAGE (Most Active) ═")
        for ip, ports in list(ip_port_usage.items())[:10]:
            port_list = ", ".join(str(p) for p, _ in list(ports.items())[:5])
            lines.append(f"  {ip}: ports {port_list}")
        lines.append("")

    # GeoIP data
    geoip = analytics.get("geoip_data", {})
    if geoip:
        lines.append("═ GEOLOCATION DATA ═")
        for ip, location in list(geoip.items())[:10]:
            lines.append(f"  {ip}: {location}")
        lines.append("")

    # DNS queries
    dns = analytics.get("dns_queries", {})
    if dns:
        lines.append("═ DNS QUERIES ═")
        for domain, count in list(dns.items())[:15]:
            lines.append(f"  {domain}: {count} queries")
        lines.append("")

    return "\n".join(lines)


def _parse_json_response(text: str) -> dict | None:
    """Extract and parse a JSON block from LLM output."""
    # Try the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first JSON block from markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Heuristic: find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


