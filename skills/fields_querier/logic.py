"""
skills/fields_querier/logic.py

Reads the local field-schema catalog (data/fields_rag.json) produced by
fields_baseliner and answers field-schema questions.

Use this skill BEFORE opensearch_querier when you need to know the exact
field names to use in a query — e.g. "what field holds the source IP?",
"which field contains byte counts?", "what are the alert fields?".

Because the catalog is a local JSON file there is NO OpenSearch dependency
and responses are instant.

Context keys consumed:
    context["llm"]        -> BaseLLMProvider
    context["parameters"] -> {"question": str}
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILL_NAME       = "fields_querier"
DATA_DIR         = Path(__file__).parents[2] / "data"
FIELDS_FILE      = DATA_DIR / "fields_rag.json"
INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"


# ──────────────────────────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────────────────────────

def _load_fields_rag() -> list[dict]:
    """Load field documents from data/fields_rag.json."""
    try:
        return json.loads(FIELDS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "[%s] fields_rag.json not found — run fields_baseliner first.",
            SKILL_NAME,
        )
        return []
    except Exception as exc:
        logger.error("[%s] Failed to load fields_rag.json: %s", SKILL_NAME, exc)
        return []


def _extract_field_text(docs: list[dict]) -> str:
    """Combine all field documents into one context string for the LLM."""
    parts = []
    for doc in docs:
        category = doc.get("category", "unknown").upper()
        text = doc.get("text", "")
        if text:
            parts.append(f"[{category}]\n{text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Structured field-mappings extractor
# ──────────────────────────────────────────────────────────────────────────────

def _extract_field_mappings(docs: list[dict]) -> dict:
    """
    Build a structured field_mappings dict compatible with query_builder format.

    Keys returned:
        ip_fields, source_ip_fields, destination_ip_fields,
        port_fields, source_port_fields, destination_port_fields,
        text_fields, timestamp_fields, domain_fields, geoip_fields,
        country_fields,
        protocol_fields, bytes_fields, all_fields
    """
    mappings: dict[str, list] = {
        "ip_fields": [],
        "source_ip_fields": [],
        "destination_ip_fields": [],
        "port_fields": [],
        "source_port_fields": [],
        "destination_port_fields": [],
        "text_fields": [],
        "timestamp_fields": [],
        "domain_fields": [],
        "geoip_fields": [],
        "country_fields": [],
        "protocol_fields": [],
        "bytes_fields": [],
        "all_fields": [],
    }

    for doc in docs:
        if doc.get("category") != "field_documentation":
            continue
        fields: dict[str, dict] = doc.get("fields", {})
        for field, info in fields.items():
            inferred = info.get("inferred_type", "string").lower()
            fl = field.lower()

            # Track everything
            if field not in mappings["all_fields"]:
                mappings["all_fields"].append(field)

            # --- IP ---
            if "ipv4" in inferred or ("ip" in inferred and "field" not in inferred):
                if field not in mappings["ip_fields"]:
                    mappings["ip_fields"].append(field)
                if any(k in fl for k in ("src", "source", "orig", "client", "from")):
                    if field not in mappings["source_ip_fields"]:
                        mappings["source_ip_fields"].append(field)
                elif any(k in fl for k in ("dst", "dest", "destination", "resp", "server", "to")):
                    if field not in mappings["destination_ip_fields"]:
                        mappings["destination_ip_fields"].append(field)

            # --- Port ---
            if "port" in inferred or "port" in fl:
                if field not in mappings["port_fields"]:
                    mappings["port_fields"].append(field)
                if any(k in fl for k in ("src", "source", "orig", "client")):
                    if field not in mappings["source_port_fields"]:
                        mappings["source_port_fields"].append(field)
                elif any(k in fl for k in ("dst", "dest", "destination", "resp", "server")):
                    if field not in mappings["destination_port_fields"]:
                        mappings["destination_port_fields"].append(field)

            # --- Timestamp ---
            if "datetime" in inferred or any(k in fl for k in ("timestamp", "time", "date", "created")):
                if field not in mappings["timestamp_fields"]:
                    mappings["timestamp_fields"].append(field)

            # --- GeoIP ---
            if any(k in fl for k in ("geo", "country", "city", "region")):
                if field not in mappings["geoip_fields"]:
                    mappings["geoip_fields"].append(field)
            if "country" in fl and field not in mappings["country_fields"]:
                mappings["country_fields"].append(field)

            # --- Protocol ---
            if any(k in fl for k in ("proto", "protocol", "transport")):
                if field not in mappings["protocol_fields"]:
                    mappings["protocol_fields"].append(field)

            # --- Bytes/Volume ---
            if any(k in fl for k in ("byte", "packet", "size", "length")):
                if field not in mappings["bytes_fields"]:
                    mappings["bytes_fields"].append(field)

            # --- Domain ---
            if any(k in fl for k in ("domain", "hostname", "fqdn", "dns")):
                if field not in mappings["domain_fields"]:
                    mappings["domain_fields"].append(field)

            # --- Text (keyword/string fields good for multi_match) ---
            if "string" in inferred or "keyword" in inferred:
                if field not in mappings["text_fields"]:
                    mappings["text_fields"].append(field)

    return mappings


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """Entry point called by the Runner."""
    llm        = context.get("llm")
    parameters = context.get("parameters", {})
    user_question = parameters.get("question", "").strip()

    if not user_question:
        return {"status": "no_question"}

    # ── Load local field catalog ───────────────────────────────────────────────
    field_docs = _load_fields_rag()
    if not field_docs:
        return {
            "status": "no_data",
            "findings": {
                "question": user_question,
                "answer": (
                    "No field schema documentation is available yet. "
                    "Run fields_baseliner first to build the schema catalog."
                ),
                "field_mappings": {},
                "confidence": 0.0,
            },
        }

    field_context = _extract_field_text(field_docs)
    field_mappings = _extract_field_mappings(field_docs)

    # ── LLM answers the schema question ───────────────────────────────────────
    if llm is None:
        logger.warning("[%s] LLM not available — returning raw field catalog.", SKILL_NAME)
        return {
            "status": "ok",
            "findings": {
                "question": user_question,
                "answer": field_context[:3000],
                "field_mappings": field_mappings,
                "confidence": 0.5,
            },
            "field_mappings": field_mappings,
        }

    try:
        instruction = INSTRUCTION_PATH.read_text(encoding="utf-8")
    except Exception:
        instruction = "You are a field-schema expert. Answer field questions from the provided catalog."

    prompt = f"""User Question: "{user_question}"

AVAILABLE FIELD SCHEMA CATALOG:
{field_context}

Based ONLY on the schema catalog above:
1. Identify every field relevant to the user's question.
2. List the EXACT field name(s) to use in OpenSearch queries.
3. State the type and show 1-2 example values for each field.
4. If multiple candidate fields exist, list all of them.

Be concise and actionable — the output will be used to build a search query."""

    try:
        answer = llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ])
    except Exception as exc:
        logger.error("[%s] LLM call failed: %s", SKILL_NAME, exc)
        answer = field_context[:2000]

    return {
        "status": "ok",
        "findings": {
            "question": user_question,
            "answer": answer.strip() if isinstance(answer, str) else str(answer),
            "field_mappings": field_mappings,
            "confidence": 0.9,
        },
        "field_mappings": field_mappings,
    }
