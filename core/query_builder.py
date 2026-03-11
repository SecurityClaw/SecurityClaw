"""
core/query_builder.py

Centralized OpenSearch query building utilities.

Shared by all skills to ensure consistent, data-agnostic query construction.
This module discovers available fields from RAG and builds intelligent queries.

All field-aware query building happens here. No hardcoded field names.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _classify_directional_ip_field(field_name: str, mappings: dict) -> None:
    """Add an IP field to directional buckets using generic field-name hints."""
    lower_name = field_name.lower()

    if field_name not in mappings["ip_fields"]:
        mappings["ip_fields"].append(field_name)

    if any(token in lower_name for token in ("src", "source", "client", "orig", "remote", "local")):
        if field_name not in mappings["source_ip_fields"]:
            mappings["source_ip_fields"].append(field_name)
    if any(token in lower_name for token in ("dst", "dest", "destination", "server", "resp", "peer")):
        if field_name not in mappings["destination_ip_fields"]:
            mappings["destination_ip_fields"].append(field_name)


def discover_field_mappings(db: Any, llm: Any) -> dict:
    """Discover available fields from OpenSearch mappings and RAG documentation.
    
    This ensures queries use actual field names from the data schema,
    making all skills data-agnostic.
    
    Returns:
        Dict mapping field types to lists of field names:
        {
            "ip_fields": ["source_ip", "dest_ip"],
            "text_fields": ["message", "description"],
            ...
        }
    """
    mappings = {
        "ip_fields": [],
        "source_ip_fields": [],
        "destination_ip_fields": [],
        "country_fields": [],
        "text_fields": [],
        "port_fields": [],
        "domain_fields": [],
        "geo_fields": [],
        "timestamp_fields": [],
        "all_fields": [],  # Fallback for multi_match
    }

    # Try to get field mappings from OpenSearch first
    try:
        logs_index = "logstash*"  # Default, may be overridden
        if hasattr(db, '_client'):
            # Query OpenSearch for actual field mappings
            mapping_resp = db._client.indices.get_mapping(index=logs_index)
            seen_fields = set()  # Track fields to avoid duplicates
            for index_name, index_mapping in mapping_resp.items():
                properties = index_mapping.get("mappings", {}).get("properties", {})
                
                # Process top-level fields
                for field_name, field_info in properties.items():
                    if field_name in seen_fields:
                        continue  # Skip duplicates
                    seen_fields.add(field_name)
                    
                    field_type = field_info.get("type", "")
                    mappings["all_fields"].append(field_name)
                    
                    # Classify by type
                    if field_type == "ip":
                        _classify_directional_ip_field(field_name, mappings)
                    elif field_type == "geo_point":
                        if field_name not in mappings["geo_fields"]:
                            mappings["geo_fields"].append(field_name)
                    elif field_type == "keyword":
                        if "country" in field_name.lower():
                            if field_name not in mappings["country_fields"]:
                                mappings["country_fields"].append(field_name)
                            if field_name not in mappings["text_fields"]:
                                mappings["text_fields"].append(field_name)
                        elif any(kw in field_name.lower() for kw in ["port", "destination.port"]):
                            if field_name not in mappings["port_fields"]:
                                mappings["port_fields"].append(field_name)
                        elif any(kw in field_name.lower() for kw in ["domain", "hostname", "fqdn"]):
                            if field_name not in mappings["domain_fields"]:
                                mappings["domain_fields"].append(field_name)
                        else:
                            if field_name not in mappings["text_fields"]:
                                mappings["text_fields"].append(field_name)
                    elif field_type in ("text", "wildcard"):
                        if field_name not in mappings["text_fields"]:
                            mappings["text_fields"].append(field_name)
                    elif field_type == "date":
                        if field_name not in mappings["timestamp_fields"]:
                            mappings["timestamp_fields"].append(field_name)
                    
                    # Handle nested/object fields (e.g., geoip with geoip.country_code2)
                    if field_type == "object" or "properties" in field_info:
                        nested_props = field_info.get("properties", {})
                        for nested_name, nested_info in nested_props.items():
                            full_field_name = f"{field_name}.{nested_name}"
                            if full_field_name not in seen_fields:
                                seen_fields.add(full_field_name)
                                mappings["all_fields"].append(full_field_name)
                                nested_type = nested_info.get("type", "")
                                
                                # Classify nested fields
                                if nested_type == "keyword":
                                    if "country" in full_field_name.lower():
                                        # Country fields for geoIP filtering
                                        if full_field_name not in mappings["country_fields"]:
                                            mappings["country_fields"].append(full_field_name)
                                        if full_field_name not in mappings["text_fields"]:
                                            mappings["text_fields"].append(full_field_name)
                                        logger.debug("Found country field: %s", full_field_name)
                                    elif any(kw in full_field_name.lower() for kw in ["port"]):
                                        if full_field_name not in mappings["port_fields"]:
                                            mappings["port_fields"].append(full_field_name)
                                    else:
                                        if full_field_name not in mappings["text_fields"]:
                                            mappings["text_fields"].append(full_field_name)
                                elif nested_type in ("text", "wildcard"):
                                    if full_field_name not in mappings["text_fields"]:
                                        mappings["text_fields"].append(full_field_name)
                                elif nested_type == "ip":
                                    _classify_directional_ip_field(full_field_name, mappings)
                                elif nested_type == "geo_point":
                                    if full_field_name not in mappings["geo_fields"]:
                                        mappings["geo_fields"].append(full_field_name)
            
            if mappings["all_fields"]:
                logger.debug(
                    "Discovered fields from OpenSearch: %d IP, %d text, %d total",
                    len(mappings["ip_fields"]),
                    len(mappings["text_fields"]),
                    len(mappings["all_fields"]),
                )
                return mappings
    except Exception as exc:
        logger.debug("Could not get mappings from OpenSearch: %s", exc)

    # Fallback 1: Read fields_rag.json written by fields_baseliner (fastest path)
    if not mappings["all_fields"]:
        try:
            import json as _json
            from pathlib import Path as _Path
            _fields_rag = _Path(__file__).parents[1] / "data" / "fields_rag.json"
            if _fields_rag.exists():
                _rag_docs = _json.loads(_fields_rag.read_text(encoding="utf-8"))
                for _doc in _rag_docs:
                    if _doc.get("category") == "field_documentation" and "fields" in _doc:
                        for _fname, _finfo in _doc["fields"].items():
                            _ftype = _finfo.get("inferred_type", "text")
                            if _fname not in mappings["all_fields"]:
                                mappings["all_fields"].append(_fname)
                            if "country" in _fname.lower() and _fname not in mappings["country_fields"]:
                                mappings["country_fields"].append(_fname)
                            if _ftype == "ip" and _fname not in mappings["ip_fields"]:
                                mappings["ip_fields"].append(_fname)
                            elif _ftype == "port" and _fname not in mappings["port_fields"]:
                                mappings["port_fields"].append(_fname)
                            elif _ftype in ("domain", "fqdn") and _fname not in mappings["domain_fields"]:
                                mappings["domain_fields"].append(_fname)
                            elif _ftype in ("timestamp", "date") and _fname not in mappings["timestamp_fields"]:
                                mappings["timestamp_fields"].append(_fname)
                            elif _fname not in mappings["text_fields"]:
                                mappings["text_fields"].append(_fname)
                if mappings["all_fields"]:
                    logger.debug(
                        "Discovered fields from fields_rag.json: %d total",
                        len(mappings["all_fields"]),
                    )
        except Exception as exc:
            logger.debug("fields_rag.json field discovery failed: %s", exc)

    # Fallback 2: RAG vector index (legacy — field_documentation category)
    if llm and not mappings["all_fields"]:
        try:
            from core.rag_engine import RAGEngine
            rag = RAGEngine(db=db, llm=llm)

            # Query RAG for field documentation
            docs = rag.retrieve("field names schema types", k=3)
            field_docs = [
                doc.get("text", "")
                for doc in docs
                if doc.get("category") == "field_documentation"
            ]

            if field_docs:
                for field_doc in field_docs:
                    _parse_field_documentation(field_doc, mappings)
        except Exception as exc:
            logger.debug("RAG field discovery failed: %s", exc)
    
    # If still no fields discovered, use generic fallbacks
    if not mappings["all_fields"]:
        logger.debug("No fields discovered; using generic fallback fields")
        mappings["all_fields"] = ["message", "description", "payload", "data", "content", "@message"]
        mappings["text_fields"] = ["message", "description", "payload", "data", "content", "@message"]
        mappings["timestamp_fields"] = ["@timestamp", "timestamp"]
    
    logger.debug(
        "Final mappings: %d IP, %d text, %d port, %d timestamp, %d total",
        len(mappings["ip_fields"]),
        len(mappings["text_fields"]),
        len(mappings["port_fields"]),
        len(mappings["timestamp_fields"]),
        len(mappings["all_fields"]),
    )
    
    return mappings


def _parse_field_documentation(field_doc: str, mappings: dict) -> None:
    """Parse field_documentation text to classify field names by type."""
    for line in field_doc.split("\n"):
        lower = line.lower()
        field = None

        # Extract field name from various documentation formats
        if "field:" in lower:
            parts = line.split(":", 1)
            field = parts[1].strip() if len(parts) > 1 else None
        elif "name:" in lower:
            parts = line.split(":", 1)
            field = parts[1].strip() if len(parts) > 1 else None
        elif line.strip().startswith("- "):
            field = line.strip()[2:].split("(")[0].strip()

        if not field:
            continue

        # Classify by keywords in documentation
        if any(kw in lower for kw in ["ipv4", "ip address", "src_ip", "dest_ip", "source ip", "destination ip"]):
            if field not in mappings["ip_fields"]:
                mappings["ip_fields"].append(field)
        elif "country" in lower and field not in mappings["country_fields"]:
            mappings["country_fields"].append(field)
        elif any(kw in lower for kw in ["port", "destination.port"]):
            if field not in mappings["port_fields"]:
                mappings["port_fields"].append(field)
        elif any(kw in lower for kw in ["domain", "hostname", "fqdn"]):
            if field not in mappings["domain_fields"]:
                mappings["domain_fields"].append(field)
        elif any(kw in lower for kw in ["timestamp", "@timestamp", "datetime", "time"]):
            if field not in mappings["timestamp_fields"]:
                mappings["timestamp_fields"].append(field)

        # All fields can be text fields (fallback for multi_match)
        if field not in mappings["all_fields"]:
            mappings["all_fields"].append(field)


def build_keyword_query(keywords: list[str], field_mappings: dict) -> tuple[dict, dict]:
    """Build intelligent keyword search query using discovered fields.
    
    Args:
        keywords: Terms to search for
        field_mappings: Result from discover_field_mappings()
    
    Returns:
        (query_dict, metadata_dict) where query_dict is ready for db.search()
    """
    should_clauses = []
    metadata = {
        "fields_used": [],
        "keywords_searched": keywords,
    }

    text_fields = field_mappings.get("text_fields", [])
    all_fields = field_mappings.get("all_fields", [])
    ip_fields = field_mappings.get("ip_fields", [])

    # Use text fields if available, fall back to all_fields
    search_fields = text_fields if text_fields else all_fields
    
    if not search_fields:
        logger.warning("No search fields available in mappings")
        return {"query": {"match_none": {}}}, metadata

    for kw in keywords:
        # Check if keyword is an IP address
        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        if re.match(ip_pattern, kw):
            # Search IP fields
            if ip_fields:
                for field in ip_fields:
                    should_clauses.append({"term": {field: kw}})
                metadata["fields_used"] = list(set(metadata["fields_used"] + ip_fields))
        else:
            # Search text fields
            if search_fields:
                should_clauses.append({
                    "multi_match": {
                        "query": kw,
                        "fields": search_fields,
                        "operator": "OR",
                        "fuzziness": "AUTO",  # Allow fuzzy matching for typos
                    }
                })
                metadata["fields_used"] = list(set(metadata["fields_used"] + search_fields))

    if not should_clauses:
        logger.warning("No search clauses built for keywords: %s", keywords)
        return {"query": {"match_none": {}}}, metadata

    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        }
    }, metadata


def build_structured_query(
    ips: list[str],
    domains: list[str],
    ports: list[int],
    time_range: Optional[dict],
    field_mappings: dict,
) -> tuple[dict, dict]:
    """Build structured query for IPs, domains, ports with optional time filter.
    
    Args:
        ips: IP addresses to search for
        domains: Domain names to search for
        ports: Ports to search for
        time_range: Dict with "start" and "end" ISO timestamps
        field_mappings: Result from discover_field_mappings()
    
    Returns:
        (query_dict, metadata_dict)
    """
    must_clauses = []
    should_clauses = []
    metadata = {
        "fields_used": [],
        "keywords_searched": [],
    }

    ip_fields = field_mappings.get("ip_fields", [])
    port_fields = field_mappings.get("port_fields", [])
    domain_fields = field_mappings.get("domain_fields", [])
    timestamp_fields = field_mappings.get("timestamp_fields", [])

    # Add IP searches
    for ip in ips:
        if ip_fields:
            for field in ip_fields:
                should_clauses.append({"term": {field: ip}})
            metadata["fields_used"].extend(ip_fields)
        metadata["keywords_searched"].append(ip)

    # Add port searches
    for port in ports:
        if port_fields:
            for field in port_fields:
                should_clauses.append({"term": {field: port}})
            metadata["fields_used"].extend(port_fields)
        metadata["keywords_searched"].append(str(port))

    # Add domain searches
    for domain in domains:
        if domain_fields:
            for field in domain_fields:
                should_clauses.append({"match": {field: domain}})
            metadata["fields_used"].extend(domain_fields)
        metadata["keywords_searched"].append(domain)

    # Add time range filter if provided
    if time_range and timestamp_fields:
        for ts_field in timestamp_fields:
            if "start" in time_range:
                must_clauses.append({
                    "range": {
                        ts_field: {
                            "gte": time_range["start"],
                            "lte": time_range.get("end", "now"),
                        }
                    }
                })
        metadata["time_window"] = f"{time_range.get('start')} to {time_range.get('end', 'now')}"

    # If no specific searches but we have results, use match_all
    if not should_clauses and not must_clauses:
        should_clauses = [{"match_all": {}}]

    # Build bool query
    bool_query: dict[str, Any] = {}
    if must_clauses:
        bool_query["must"] = must_clauses
    if should_clauses:
        bool_query["should"] = should_clauses
        bool_query["minimum_should_match"] = 1

    metadata["fields_used"] = list(set(metadata["fields_used"]))
    return {"query": {"bool": bool_query}}, metadata


def build_time_range_query(
    time_range: dict, field_mappings: dict
) -> tuple[dict, dict]:
    """Build time-range only query.
    
    Args:
        time_range: Dict with "start" and "end" ISO timestamps
        field_mappings: Result from discover_field_mappings()
    
    Returns:
        (query_dict, metadata_dict)
    """
    must_clauses = []
    metadata = {
        "fields_used": [],
        "keywords_searched": [],
        "time_window": f"{time_range.get('start')} to {time_range.get('end', 'now')}",
    }

    timestamp_fields = field_mappings.get("timestamp_fields", [])

    if not timestamp_fields:
        logger.warning("No timestamp fields discovered; cannot filter by time")
        return {"query": {"match_all": {}}}, metadata

    for ts_field in timestamp_fields:
        must_clauses.append({
            "range": {
                ts_field: {
                    "gte": time_range.get("start"),
                    "lte": time_range.get("end", "now"),
                }
            }
        })
    metadata["fields_used"] = timestamp_fields

    return {"query": {"bool": {"must": must_clauses}}}, metadata
