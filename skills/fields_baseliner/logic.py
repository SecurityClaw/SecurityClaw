"""
skills/fields_baseliner/logic.py

Comprehensive field-schema documentation builder.

Runs when RECORDS_THRESHOLD (10 000) new log records have appeared since the
last run.  Samples up to SAMPLE_SIZE records, builds a complete field catalog
(name · type · frequency · examples) and writes it to data/fields_rag.json.

fields_querier reads that local file at query time — no OpenSearch vector
index is involved.

Context keys consumed:
    context["db"]         -> BaseDBConnector
    context["llm"]        -> BaseLLMProvider   (optional — used only for richer descriptions)
    context["config"]     -> Config
    context["parameters"] -> {"force_refresh": bool}   # optional
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILL_NAME        = "fields_baseliner"
RECORDS_THRESHOLD = 10_000          # re-run every 10 k new records
SAMPLE_SIZE       = 5_000           # logs to sample each run
MAX_EXAMPLES      = 5               # distinct example values to keep per field
MAX_FIELDS        = 200             # cap on catalogued fields

DATA_DIR     = Path(__file__).parents[2] / "data"
STATE_FILE   = DATA_DIR / "fields_baseliner_state.json"
OUTPUT_FILE  = DATA_DIR / "fields_rag.json"

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"


# ──────────────────────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"last_record_count": 0, "last_run": None}


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Data access
# ──────────────────────────────────────────────────────────────────────────────

def _count_logs(db: Any, logs_index: str) -> int:
    """Return the approximate total number of documents in the logs index."""
    try:
        if hasattr(db, "_client"):
            resp = db._client.count(index=logs_index)
            return int(resp.get("count", 0))
        # Fallback: issue a size=0 search and check hits.total
        result = db.search(logs_index, {"query": {"match_all": {}}, "size": 0})
        if isinstance(result, list):
            return len(result)
        # Some connectors return a raw response dict
        return int((result or {}).get("hits", {}).get("total", {}).get("value", 0))
    except Exception as exc:
        logger.warning("[%s] Could not count docs: %s", SKILL_NAME, exc)
        return 0


def _sample_logs(db: Any, logs_index: str, size: int = SAMPLE_SIZE) -> list[dict]:
    """Return a random sample of log records."""
    # Random-score sampling
    try:
        query = {
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {},
                }
            },
            "size": size,
        }
        records = db.search(logs_index, query, size=size) or []
        if records:
            return records
    except Exception as exc:
        logger.debug("[%s] Random-score sample failed: %s", SKILL_NAME, exc)

    # Fallback: latest N records
    try:
        records = db.search(
            logs_index,
            {
                "query": {"match_all": {}},
                "sort": [{"@timestamp": {"order": "desc"}}],
                "size": size,
            },
            size=size,
        )
        return records or []
    except Exception as exc:
        logger.warning("[%s] Sample fallback failed: %s", SKILL_NAME, exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Field analysis
# ──────────────────────────────────────────────────────────────────────────────

def _walk_log(obj: Any, prefix: str, counts: Counter, examples: dict) -> None:
    """Recursively walk a log dict and record field names + example values."""
    if not isinstance(obj, dict):
        return
    for key, val in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        counts[full_key] += 1
        exs = examples.setdefault(full_key, set())
        if len(exs) < MAX_EXAMPLES and val is not None:
            ex_str = str(val)
            if len(ex_str) < 120:
                exs.add(ex_str)
        if isinstance(val, dict):
            _walk_log(val, full_key, counts, examples)


def _infer_type(field: str, examples: list[str]) -> str:
    """Guess data type from field name and example values."""
    fl = field.lower()
    if any(k in fl for k in ("timestamp", "created", "occurred", "date", "time")):
        return "datetime"
    if any(k in fl for k in ("src_ip", "dest_ip", "source.ip", "destination.ip",
                              "id.orig_h", "id.resp_h", "host.ip", "client.ip", "server.ip")):
        return "IPv4"
    if "ip" in fl or "address" in fl:
        # Confirm with examples
        if examples and re.match(r"\d{1,3}\.\d{1,3}", examples[0]):
            return "IPv4"
    if "port" in fl:
        return "integer (port 1-65535)"
    if any(k in fl for k in ("bytes", "size", "length", "packets", "count", "duration")):
        return "integer"
    if any(k in fl for k in ("proto", "protocol", "transport")):
        return "keyword"
    if any(k in fl for k in ("geo", "country", "city", "region")):
        return "geo/string"
    if any(k in fl for k in ("domain", "hostname", "fqdn", "dns")):
        return "domain string"
    # Check example values
    if examples:
        first = examples[0]
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", first):
            return "IPv4"
        if first.lstrip("-").isdigit():
            return "integer"
    return "string"


def _infer_description(field: str) -> str:
    """Return a short human-readable description for recognised field patterns."""
    fl = field.lower()
    known = {
        "src_ip": "Source IP address",
        "source.ip": "Source IP address",
        "id.orig_h": "Zeek source (originator) IP",
        "dest_ip": "Destination IP address",
        "destination.ip": "Destination IP address",
        "id.resp_h": "Zeek destination (responder) IP",
        "src_port": "Source port number",
        "source.port": "Source port number",
        "id.orig_p": "Zeek source port",
        "dest_port": "Destination port number",
        "destination.port": "Destination port number",
        "id.resp_p": "Zeek destination port",
        "protocol": "Network protocol (tcp/udp/icmp/…)",
        "proto": "Network protocol abbreviation",
        "network.transport": "Transport layer protocol",
        "app_proto": "Application protocol detected",
        "@timestamp": "Primary event timestamp (ISO 8601)",
        "timestamp": "Event timestamp",
        "event.created": "Event creation time",
        "bytes": "Total bytes transferred",
        "network.bytes": "Total network bytes",
        "bytes_sent": "Bytes sent by source",
        "bytes_received": "Bytes received by destination",
        "packets": "Total packet count",
        "network.packets": "Total network packet count",
        "dns.question.name": "DNS query domain name",
        "dns.query": "DNS query string",
        "alert.signature": "IDS/IPS alert rule signature",
        "alert.category": "IDS/IPS alert category",
        "event.type": "Event type classification",
        "destination.geo": "Destination geographic data",
        "geoip": "GeoIP location data",
        "geoip.country_name": "Country of the destination IP",
        "geoip.country_code2": "ISO 2-letter country code",
        "geoip.city_name": "City of the destination IP",
    }
    for k, v in known.items():
        if k in fl:
            return v
    return "Network log field"


def _analyze_fields(logs: list[dict]) -> dict[str, dict]:
    """
    Walk every log record and build a comprehensive field catalog.

    Returns mapping: field_name → {count, pct, examples, inferred_type, description}
    """
    total = max(len(logs), 1)
    counts: Counter = Counter()
    examples: dict[str, set] = {}

    for log in logs:
        _walk_log(log, "", counts, examples)

    catalog: dict[str, dict] = {}
    for field, count in counts.most_common(MAX_FIELDS):
        exs = sorted(examples.get(field, set()))[:MAX_EXAMPLES]
        catalog[field] = {
            "count": count,
            "pct": round((count / total) * 100, 1),
            "examples": exs,
            "inferred_type": _infer_type(field, exs),
            "description": _infer_description(field),
        }

    return catalog


# ──────────────────────────────────────────────────────────────────────────────
# Document builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_field_documents(catalog: dict[str, dict], total_logs: int) -> list[dict]:
    """Convert the field catalog into storable JSON documents."""
    now = datetime.now(timezone.utc).isoformat()
    sorted_fields = sorted(catalog.items(), key=lambda x: -x[1]["count"])

    # ── Document 1: Schema overview ───────────────────────────────────────────
    def _cat_list(keywords: list[str]) -> str:
        return ", ".join(f for f, _ in sorted_fields if any(k in f.lower() for k in keywords))[:300] or "(none detected)"

    schema_lines = [
        "SCHEMA OBSERVATION — All Available Fields",
        "=" * 60,
        f"Total records sampled: {total_logs:,}",
        "",
        "FIELDS (sorted by frequency):",
    ]
    for field, info in sorted_fields[:80]:
        schema_lines.append(
            f"  {field}: {info['pct']}% freq — {info['inferred_type']}"
        )
    schema_lines.extend([
        "",
        "FIELD CATEGORIES:",
        f"  IP-related:   {_cat_list(['ip', 'address', 'host.ip', 'orig_h', 'resp_h'])}",
        f"  Port-related: {_cat_list(['port', 'orig_p', 'resp_p'])}",
        f"  Protocol:     {_cat_list(['proto', 'protocol', 'transport'])}",
        f"  Geographic:   {_cat_list(['geo', 'country', 'city', 'region'])}",
        f"  Timing:       {_cat_list(['timestamp', 'time', 'date', 'created'])}",
        f"  Volume:       {_cat_list(['byte', 'packet', 'size', 'length'])}",
        f"  DNS:          {_cat_list(['dns', 'query', 'domain', 'fqdn'])}",
        f"  Alert:        {_cat_list(['alert', 'signature', 'event.type'])}",
        "",
        "Use field names exactly as listed above in OpenSearch queries.",
    ])

    schema_doc = {
        "category": "schema_observation",
        "text": "\n".join(schema_lines),
        "generated_at": now,
        "records_processed": total_logs,
    }

    # ── Document 2: Per-field detail ──────────────────────────────────────────
    detail_lines = [
        "COMPREHENSIVE FIELD DOCUMENTATION",
        "=" * 60,
        f"Sampled from {total_logs:,} log records.",
        "",
    ]
    for field, info in sorted_fields[:100]:
        detail_lines.extend([
            f"FIELD: {field}",
            f"  Type:        {info['inferred_type']}",
            f"  Description: {info['description']}",
            f"  Frequency:   {info['pct']}% ({info['count']:,} of {total_logs:,} records)",
            f"  Examples:    {', '.join(info['examples'][:3]) or '(none captured)'}",
            "",
        ])

    detail_doc = {
        "category": "field_documentation",
        "text": "\n".join(detail_lines),
        "generated_at": now,
        "records_processed": total_logs,
        # Structured data for fields_querier to build query_builder-style mappings
        "fields": {field: info for field, info in sorted_fields[:100]},
    }

    return [schema_doc, detail_doc]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    db         = context.get("db")
    cfg        = context.get("config")
    parameters = context.get("parameters", {})
    force      = parameters.get("force_refresh", False)

    if db is None:
        msg = "[%s] db not available — cannot proceed."
        if force:
            # First-startup requires database access
            logger.error(msg, SKILL_NAME)
            return {"status": "error", "reason": "database unavailable on first startup"}
        else:
            logger.warning(msg, SKILL_NAME)
            return {"status": "skipped", "reason": "no db"}

    logs_index = cfg.get("db", "logs_index", default="securityclaw-logs")

    # ── Threshold guard ────────────────────────────────────────────────────────
    state         = _load_state()
    last_count    = state.get("last_record_count", 0)
    current_count = _count_logs(db, logs_index)
    delta         = current_count - last_count

    # On first startup (force=True), we need to detect if DB is actually unavailable
    # vs just empty. If current_count is still 0 due to connection failure, that's an error.
    if force and current_count == 0:
        try:
            # Try a simple connection test
            if hasattr(db, "_client"):
                db._client.info()
        except Exception as exc:
            logger.error("[%s] Database connection failed on first startup: %s", SKILL_NAME, exc)
            return {"status": "error", "reason": f"database connection failed: {str(exc)[:100]}"}

    if not force and delta < RECORDS_THRESHOLD:
        logger.info(
            "[%s] Only %d new records since last run (threshold=%d) — skipping.",
            SKILL_NAME, delta, RECORDS_THRESHOLD,
        )
        return {
            "status": "skipped",
            "reason": f"threshold not met ({delta} new records < {RECORDS_THRESHOLD})",
            "new_records": delta,
        }

    if force:
        logger.info("[%s] force_refresh=True — running unconditionally.", SKILL_NAME)
    else:
        logger.info("[%s] %d new records — running field documentation refresh.", SKILL_NAME, delta)

    # ── Sample logs ────────────────────────────────────────────────────────────
    logs = _sample_logs(db, logs_index, size=SAMPLE_SIZE)
    if not logs:
        logger.info("[%s] No logs found in index '%s'.", SKILL_NAME, logs_index)
        return {"status": "no_data"}

    # ── Analyse fields ─────────────────────────────────────────────────────────
    catalog = _analyze_fields(logs)
    logger.info(
        "[%s] Catalogued %d distinct fields from %d sampled records.",
        SKILL_NAME, len(catalog), len(logs),
    )

    # ── Build RAG documents ────────────────────────────────────────────────────
    docs = _build_field_documents(catalog, len(logs))

    # ── Persist to local file ──────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[%s] Wrote %d field documents → %s", SKILL_NAME, len(docs), OUTPUT_FILE)

    # ── Update state ───────────────────────────────────────────────────────────
    _save_state({
        "last_record_count": current_count,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "fields_documented": len(catalog),
        "records_sampled": len(logs),
    })

    return {
        "status": "ok",
        "fields_documented": len(catalog),
        "records_sampled": len(logs),
        "documents_written": len(docs),
        "output_file": str(OUTPUT_FILE),
    }
