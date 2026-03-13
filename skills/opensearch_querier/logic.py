"""
skills/opensearch_querier/logic.py

Skill wrapper around core.query_builder utilities.

This skill provides:
1. A direct interface for user queries via chat
2. Shared query_builder utilities that other skills import

All query logic is in core.query_builder (DRY principle).
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
PLANNING_PROMPT_PATH = Path(__file__).parent / "PLANNING_PROMPT.md"
SKILL_NAME = "opensearch_querier"
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_NON_IP_FIELD_TERMS = {
    "latitude",
    "longitude",
    "country",
    "city",
    "geohash",
    "timezone",
    "region",
    "postal",
    "continent",
    "location",
    "asn",
}
_COUNTRY_CODE_REVERSE_MAP = {
    "IR": "Iran",
    "IQ": "Iraq",
    "SY": "Syria",
    "KP": "North Korea",
    "CN": "China",
    "RU": "Russia",
    "US": "United States",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "IN": "India",
    "PK": "Pakistan",
}


def _parse_time_expression(value: str, *, now: datetime | None = None) -> datetime | None:
    """Parse a limited subset of OpenSearch date math and ISO timestamps."""
    if not isinstance(value, str) or not value.strip():
        return None

    current = now or datetime.now(timezone.utc)
    text = value.strip()
    normalized = text.lower()

    if normalized == "now":
        return current
    if normalized == "now/d":
        return current.replace(hour=0, minute=0, second=0, microsecond=0)
    if normalized == "now/w":
        start_of_day = current.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day - timedelta(days=start_of_day.weekday())
    if normalized == "now/m":
        return current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    relative_match = re.fullmatch(r"now-(\d+)([hdwm])", normalized)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        if unit == "h":
            return current - timedelta(hours=amount)
        if unit == "d":
            return current - timedelta(days=amount)
        if unit == "w":
            return current - timedelta(weeks=amount)
        if unit == "m":
            return current - timedelta(days=30 * amount)

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _format_time_range_label(raw_time_range: Any, resolved_time_range: Any) -> str:
    """Return a human-friendly label when the resolved time window differs from the raw plan."""
    if raw_time_range == "custom":
        if resolved_time_range == "now/d":
            return "today"
        if resolved_time_range == "now-24h":
            return "past 24 hours"
        if resolved_time_range == "now/w":
            return "this week"
        if resolved_time_range == "now/M":
            return "this month"
    return str(raw_time_range) if raw_time_range is not None else "now-90d"


def _resolve_time_range_for_question(question: str, raw_time_range: Any) -> tuple[str | dict[str, str], str]:
    """Map vague/custom LLM time windows to concrete date math for common question forms."""
    question_text = str(question or "")
    q_lower = question_text.lower()

    if re.search(r"\btoday\b|\bsince midnight\b", q_lower):
        return "now/d", "today"
    if re.search(r"\b(this\s+week)\b", q_lower):
        return "now/w", "this week"
    if re.search(r"\b(this\s+month)\b", q_lower):
        return "now/M", "this month"
    if re.search(r"\b(?:past|last)\s+month\b", q_lower):
        return "now-30d", "past month"
    if re.search(r"\b(?:past|last)\s+year\b", q_lower):
        return "now-1y", "past year"

    relative_match = re.search(r"\b(?:past|last)\s+(\d+)\s+(hour|hours|day|days|week|weeks|month|months)\b", q_lower)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        suffix = "d"
        if "hour" in unit:
            suffix = "h"
        elif "week" in unit:
            suffix = "w"
        elif "month" in unit:
            suffix = "M"
        label = relative_match.group(0)
        return f"now-{amount}{suffix}", label

    if isinstance(raw_time_range, str) and raw_time_range.strip():
        normalized = raw_time_range.strip()
        if normalized.lower() == "custom":
            logger.warning("[%s] Unresolved custom time range from planner for question: %s", SKILL_NAME, question_text[:200])
            return "now-90d", "now-90d"
        return normalized, _format_time_range_label(normalized, normalized)

    return "now-90d", "now-90d"


def _build_time_filter(time_range: str | dict[str, str]) -> dict:
    """Build the @timestamp range filter from a normalized time range spec."""
    if isinstance(time_range, dict):
        bounds = {
            key: value
            for key, value in time_range.items()
            if key in {"gte", "lte", "gt", "lt"} and isinstance(value, str) and value.strip()
        }
        if bounds:
            return {"range": {"@timestamp": bounds}}
    return {"range": {"@timestamp": {"gte": str(time_range)}}}


def _filter_results_for_time_range(results: list[dict], time_range: str | dict[str, str]) -> list[dict]:
    """Enforce supported relative time windows client-side so repaired queries cannot widen scope."""
    if not results:
        return results

    now = datetime.now(timezone.utc)
    lower_bound: datetime | None = None
    upper_bound: datetime | None = now
    lower_inclusive = True
    upper_inclusive = True

    if isinstance(time_range, dict):
        if "gte" in time_range:
            lower_bound = _parse_time_expression(time_range["gte"], now=now)
        elif "gt" in time_range:
            lower_bound = _parse_time_expression(time_range["gt"], now=now)
            lower_inclusive = False
        if "lte" in time_range:
            upper_bound = _parse_time_expression(time_range["lte"], now=now)
        elif "lt" in time_range:
            upper_bound = _parse_time_expression(time_range["lt"], now=now)
            upper_inclusive = False
    else:
        lower_bound = _parse_time_expression(str(time_range), now=now)

    if lower_bound is None and upper_bound is None:
        return results

    filtered: list[dict] = []
    for row in results:
        timestamp_value = row.get("@timestamp") or row.get("timestamp")
        parsed_timestamp = _parse_time_expression(str(timestamp_value), now=now) if timestamp_value else None
        if parsed_timestamp is None:
            continue
        if lower_bound is not None:
            if lower_inclusive and parsed_timestamp < lower_bound:
                continue
            if not lower_inclusive and parsed_timestamp <= lower_bound:
                continue
        if upper_bound is not None:
            if upper_inclusive and parsed_timestamp > upper_bound:
                continue
            if not upper_inclusive and parsed_timestamp >= upper_bound:
                continue
        filtered.append(row)
    return filtered


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


def _extract_ips_from_text(text: str) -> list[str]:
    """Extract unique IPv4 addresses from free text while preserving order."""
    if not text:
        return []

    seen: set[str] = set()
    ips: list[str] = []
    for ip in _IP_PATTERN.findall(str(text)):
        if ip not in seen:
            seen.add(ip)
            ips.append(ip)
    return ips


def _extract_ports_from_text(text: str) -> list[int]:
    """Extract port numbers from text (patterns like 'port 443' or 'port:443')."""
    if not text:
        return []
    
    ports = set()
    
    # Handle compound patterns like "ports 80 and 443" or "ports: 80, 443"
    compound_match = re.search(r'\bports?\s*:?\s*([\d\s,and]+)', text, re.IGNORECASE)
    if compound_match:
        port_list = compound_match.group(1)
        # Extract all numbers from the port list
        for num in re.findall(r'\d+', port_list):
            port = int(num)
            if 0 < port < 65536:
                ports.add(port)
    
    # Also match individual "port NNN" patterns
    for match in re.finditer(r'\bport\s*:?\s*(\d+)', text, re.IGNORECASE):
        port = int(match.group(1))
        if 0 < port < 65536:
            ports.add(port)
    
    # Also match colons like ":443"
    for match in re.finditer(r':(\d{4,5})\b', text):
        port = int(match.group(1))
        if 0 < port < 65536:
            ports.add(port)
    
    return sorted(list(ports))


def _load_planning_prompt() -> str:
    """Load the static OpenSearch planning prompt from markdown."""
    try:
        return PLANNING_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("[%s] Could not load planning prompt markdown: %s", SKILL_NAME, exc)
        return (
            "Analyze the user's question and extract structured OpenSearch query parameters. "
            "Return strict JSON with reasoning, search_type, search_terms, countries, ports, protocols, "
            "time_range, matching_strategy, and field_analysis."
        )


def _log_excerpt(text: Any, limit: int = 600) -> str:
    """Return a log-friendly preview with an explicit ellipsis instead of abrupt slicing."""
    rendered = str(text or "")
    if len(rendered) <= limit:
        return rendered
    trimmed = rendered[:limit].rstrip()
    sentence_end = max(trimmed.rfind(". "), trimmed.rfind("\n"))
    if sentence_end >= max(0, limit - 120):
        trimmed = trimmed[:sentence_end + 1].rstrip()
    return trimmed + " ..."


def _extract_countries_from_text(text: str) -> list[str]:
    """Extract country names from text."""
    if not text:
        return []
    
    # Common country names that might appear
    countries_map = {
        'united states': 'United States',
        'us': 'United States',
        'usa': 'United States',
        'china': 'China',
        'russia': 'Russia',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        'germany': 'Germany',
        'india': 'India',
        'france': 'France',
        'japan': 'Japan',
        'brazil': 'Brazil',
        'canada': 'Canada',
        'mexico': 'Mexico',
        'australia': 'Australia',
    }
    
    countries = set()
    text_lower = text.lower()
    
    for country_key, country_name in countries_map.items():
        if country_key in text_lower:
            countries.add(country_name)
    
    return sorted(list(countries))


def _question_asks_for_country_aggregation(question: str) -> bool:
    q = str(question or "").lower()
    asks_for_country_list = any(
        phrase in q
        for phrase in (
            "what countries",
            "which countries",
            "countries other than",
            "countries besides",
            "countries except",
            "countries excluding",
            "non-us countries",
            "countries outside",
        )
    )
    asks_about_traffic = any(
        token in q
        for token in ("traffic", "connection", "connections", "flow", "flows", "activity", "activities", "do we get")
    )
    return asks_for_country_list and asks_about_traffic


def _extract_excluded_countries_from_text(text: str) -> list[str]:
    if not re.search(r"\b(other than|except|excluding|exclude|besides|outside)\b", str(text or "").lower()):
        return []
    return _extract_countries_from_text(text)


def _extract_result_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\btop\s+(\d+)\b", str(question or "").lower())
    if not match:
        return default
    try:
        return max(1, min(50, int(match.group(1))))
    except Exception:
        return default


def _maybe_attach_country_aggregation_plan(question: str, plan: dict | None) -> dict:
    normalized = dict(plan or {})
    if not _question_asks_for_country_aggregation(question):
        return normalized

    excluded_countries = normalized.get("exclude_countries")
    if not isinstance(excluded_countries, list):
        excluded_countries = []

    question_exclusions = _extract_excluded_countries_from_text(question)
    if question_exclusions:
        excluded_countries = question_exclusions

    countries = normalized.get("countries")
    if not isinstance(countries, list):
        countries = []
    if not excluded_countries and countries and re.search(r"\b(other than|except|excluding|exclude|besides|outside)\b", str(question or "").lower()):
        excluded_countries = countries
        countries = []

    normalized["countries"] = countries
    normalized["exclude_countries"] = excluded_countries
    normalized["aggregation_type"] = "country_terms"
    normalized["aggregation_field"] = "country"
    normalized["result_limit"] = _extract_result_limit(question)
    normalized["search_type"] = normalized.get("search_type") or "traffic"
    normalized["matching_strategy"] = normalized.get("matching_strategy") or "term"
    normalized["skip_search"] = False
    normalized["reasoning"] = (
        (str(normalized.get("reasoning", "")).strip() + " ") if normalized.get("reasoning") else ""
    ) + "Use a country aggregation over traffic logs instead of a document hit query."
    return normalized


def _rank_country_aggregation_fields(field_mappings: dict) -> list[str]:
    candidates = [str(field) for field in _candidate_country_fields(field_mappings)]
    if not candidates:
        return []

    def _score(field_name: str) -> tuple[int, int, str]:
        lower_name = field_name.lower()
        rank = 0
        if "country_name" in lower_name:
            rank += 30
        elif "country" in lower_name and "code" not in lower_name:
            rank += 20
        elif "country_code" in lower_name:
            rank += 10
        if ".keyword" in lower_name or lower_name.endswith("keyword"):
            rank += 5
        return (-rank, len(field_name), field_name)

    ranked = sorted(dict.fromkeys(candidates), key=_score)
    expanded: list[str] = []
    for field_name in ranked:
        lower_name = field_name.lower()
        if ".keyword" not in lower_name and not lower_name.endswith("keyword"):
            expanded.append(f"{field_name}.keyword")
        expanded.append(field_name)
    return list(dict.fromkeys(expanded))


def _country_terms_for_field(country_name: str, field_name: str) -> list[str]:
    lower_country = str(country_name or "").lower()
    lower_field = str(field_name or "").lower()

    if "country_code" in lower_field:
        code = _COUNTRY_CODE_MAP.get(lower_country)
        return [code] if code else []

    if lower_country == "united states":
        return ["United States", "USA", "US"]
    if lower_country == "united kingdom":
        return ["United Kingdom", "UK", "GB"]
    return [str(country_name)]


def _normalize_country_bucket_label(field_name: str, raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return value
    if "country_code" in str(field_name or "").lower():
        return _COUNTRY_CODE_REVERSE_MAP.get(value.upper(), value.upper())
    return value


def _build_country_aggregation_query(
    field_name: str,
    time_range: str | dict[str, str],
    exclude_countries: list[str],
    result_limit: int,
) -> dict:
    bool_query: dict[str, Any] = {"filter": [_build_time_filter(time_range)]}
    exclusion_values = []
    for country in exclude_countries:
        exclusion_values.extend(_country_terms_for_field(country, field_name))
    if exclusion_values:
        bool_query["must_not"] = [{"terms": {field_name: list(dict.fromkeys(exclusion_values))}}]

    return {
        "size": 0,
        "query": {"bool": bool_query},
        "aggs": {
            "country_counts": {
                "terms": {
                    "field": field_name,
                    "size": int(result_limit or 10),
                    "order": {"_count": "desc"},
                }
            }
        },
    }


def _aggregate_country_buckets_from_hits(
    hits: list[dict],
    field_name: str,
    exclude_countries: list[str],
    result_limit: int,
) -> list[dict]:
    excluded_labels = {
        label.lower()
        for country in exclude_countries
        for label in _country_terms_for_field(country, field_name)
    }
    counts: dict[str, int] = {}
    for row in hits:
        raw_value = _get_nested_value(row, field_name)
        if raw_value in (None, "", [], {}):
            continue
        normalized_raw = str(raw_value).strip()
        if normalized_raw.lower() in excluded_labels:
            continue
        label = _normalize_country_bucket_label(field_name, normalized_raw)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: int(result_limit or 10)]
    return [{"country": country, "count": count} for country, count in ordered]


def _execute_country_aggregation_query(
    db: Any,
    index: str,
    field_mappings: dict,
    time_range: str | dict[str, str],
    exclude_countries: list[str],
    result_limit: int,
) -> dict:
    country_fields = _rank_country_aggregation_fields(field_mappings)
    fallback_candidates: list[str] = []
    for field_name in country_fields:
        query = _build_country_aggregation_query(field_name, time_range, exclude_countries, result_limit)
        buckets: list[dict] = []
        excluded_labels = {
            label.lower()
            for country in exclude_countries
            for label in _country_terms_for_field(country, field_name)
        }

        if hasattr(db, "_client"):
            try:
                raw_response = db._client.search(index=index, body=query, size=0)
                raw_buckets = (((raw_response or {}).get("aggregations") or {}).get("country_counts") or {}).get("buckets") or []
                buckets = [
                    {"country": _normalize_country_bucket_label(field_name, bucket.get("key")), "count": int(bucket.get("doc_count", 0) or 0)}
                    for bucket in raw_buckets
                    if bucket.get("key") not in (None, "")
                ]
                buckets = [
                    bucket for bucket in buckets
                    if bucket["country"] and str(bucket["country"]).lower() not in excluded_labels
                ]
            except Exception as exc:
                logger.warning("[%s] Country aggregation failed on field %s: %s", SKILL_NAME, field_name, exc)

        if buckets:
            total_count = sum(bucket["count"] for bucket in buckets)
            return {
                "status": "ok",
                "results_count": total_count,
                "results": [],
                "country_buckets": buckets,
                "aggregation_type": "country_terms",
                "aggregation_field": field_name,
                "excluded_countries": exclude_countries,
            }

        fallback_candidates.append(field_name)

    for field_name in fallback_candidates:
        query = _build_country_aggregation_query(field_name, time_range, exclude_countries, result_limit)
        if not buckets:
            try:
                fallback_hits = db.search(index, {"query": query["query"]}, size=max(int(result_limit or 10) * 100, 500))
                buckets = _aggregate_country_buckets_from_hits(fallback_hits, field_name, exclude_countries, result_limit)
            except Exception as exc:
                logger.warning("[%s] Country aggregation fallback failed on field %s: %s", SKILL_NAME, field_name, exc)

        if buckets:
            total_count = sum(bucket["count"] for bucket in buckets)
            return {
                "status": "ok",
                "results_count": total_count,
                "results": [],
                "country_buckets": buckets,
                "aggregation_type": "country_terms",
                "aggregation_field": field_name,
                "excluded_countries": exclude_countries,
            }

    return {
        "status": "ok",
        "results_count": 0,
        "results": [],
        "country_buckets": [],
        "aggregation_type": "country_terms",
        "excluded_countries": exclude_countries,
    }


def _score_ip_query_field(field_name: str) -> int:
    """Score candidate fields for exact IP queries while excluding geo metadata."""
    lower_name = str(field_name).lower()
    leaf_name = lower_name.split(".")[-1]
    tokens = {token for token in re.split(r"[^a-z0-9]+", lower_name) if token}

    if not lower_name or any(term in lower_name for term in _NON_IP_FIELD_TERMS):
        return -100
    if lower_name in {"geoip", "source", "destination", "client", "server"}:
        return -100

    score = 0
    if leaf_name in {"ip", "ip_address", "ipaddress", "ipv4", "ipv6"}:
        score += 90
    elif lower_name.endswith(("_ip", ".ip", ".ip_address", ".ipv4", ".ipv6")):
        score += 80
    elif tokens.intersection({"ip", "ipv4", "ipv6"}):
        score += 70
    elif leaf_name == "address" and tokens.intersection({
        "src",
        "source",
        "dst",
        "dest",
        "destination",
        "client",
        "server",
        "remote",
        "local",
        "orig",
        "resp",
    }):
        score += 55

    if tokens.intersection({"src", "source", "client", "orig", "remote", "local"}):
        score += 10
    if tokens.intersection({"dst", "dest", "destination", "server", "resp", "peer"}):
        score += 10
    if lower_name.startswith("geoip.") and leaf_name == "ip":
        score -= 5

    return score


def _select_ip_query_fields(field_mappings: dict, ip_direction: str = "any") -> list[str]:
    """Choose a small, high-confidence set of actual IP fields for IP searches."""
    ranked_fields: dict[str, int] = {}
    candidate_lists: list[list[str]] = []
    preferred_fields = (
        field_mappings.get("source_ip_fields") if ip_direction == "source" else
        field_mappings.get("destination_ip_fields") if ip_direction == "destination" else
        None
    ) or []

    if preferred_fields:
        candidate_lists.append(preferred_fields)

    if ip_direction == "any" or not preferred_fields:
        candidate_lists.extend([
            field_mappings.get("source_ip_fields") or [],
            field_mappings.get("destination_ip_fields") or [],
            field_mappings.get("ip_fields") or [],
            field_mappings.get("all_fields") or [],
        ])

    for list_index, candidate_list in enumerate(candidate_lists):
        priority_bonus = 20 if list_index == 0 and ip_direction in {"source", "destination"} else 0
        for field_name in candidate_list:
            score = _score_ip_query_field(field_name)
            if score <= 0:
                continue
            ranked_fields[field_name] = max(score + priority_bonus, ranked_fields.get(field_name, -1000))

    ordered_fields = sorted(
        ranked_fields,
        key=lambda field_name: (-ranked_fields[field_name], len(str(field_name)), str(field_name)),
    )
    return ordered_fields[:6]


def _fallback_plan_from_question(
    question: str,
    previous_results: dict | None = None
) -> dict:
    """
    Fallback query planning when LLM fails: extract search parameters from question text.
    
    This allows the system to continue even when LLM planning fails with JSON parse errors.
    Uses regex and heuristics to extract:
    - Ports (from "port 443" or ":443" patterns)
    - IPs (from IP addresses in text)
    - Countries (from country names)
    - Search terms (words that aren't structural)
    - Search type (inferred from keywords)
    
    For follow-up questions, extracts IPs from previous results and adds them to search_terms.
    """
    question_lower = question.lower()
    
    # Extract structured data
    ports = _extract_ports_from_text(question)
    ips = _extract_ips_from_text(question)
    countries = _extract_countries_from_text(question)
    
    # Also extract IPs from previous results if available
    # For follow-up questions, add these directly to search_terms
    if previous_results:
        previous_ips = _extract_ips_from_previous_results(previous_results)
        for ip in previous_ips:
            if ip not in ips:
                ips.append(ip)
    
    # Infer search type from keywords
    search_type = "general"
    if any(kw in question_lower for kw in ["alert", "signature", "et policy", "et exploit", "et info", "et drop"]):
        search_type = "alert"
    elif any(kw in question_lower for kw in ["traffic", "connection", "flow", "packet", "network", "port", "protocol", "happening", "activity"]):
        search_type = "traffic"
    elif ips or "ip" in question_lower or "country" in question_lower:
        search_type = "ip"
    elif any(kw in question_lower for kw in ["domain", "host", "dns"]):
        search_type = "domain"
    
    # Extract key terms (remove common words and structural elements)
    stop_words = {
        "what", "where", "when", "why", "how", "is", "are", "was", "were", "the", "a", "an",
        "on", "in", "at", "to", "from", "with", "by", "was", "port", "ports", "associated",
        "traffic", "alert", "signature", "and", "or", "this", "that", "these", "those",
        "for", "of", "this", "country", "countries", "ip", "ips", "address", "addresses",
        "happening", "activity", "protocol", "protocols", "connection", "flow", "packet"
    }
    
    search_terms = []
    for word in question.split():
        clean_word = word.strip('.,!?;:"\'-').lower()
        if clean_word and clean_word not in stop_words and len(clean_word) > 2:
            # Exclude IPs and ports
            if not re.match(r'^\d+$', clean_word) and not _IP_PATTERN.match(clean_word):
                search_terms.append(clean_word)
    
    # Remove duplicates while preserving order
    seen = set()
    search_terms = [t for t in search_terms if not (t in seen or seen.add(t))]
    
    # For IP-type searches (or when IPs are extracted), add them to search_terms
    if (search_type == "ip" or ips) and ips:
        # IPs become search terms for IP-specific searches
        search_terms = ips + search_terms
    
    return {
        "reasoning": f"Fallback plan (LLM planning failed): Extracted ports={ports}, countries={countries}, ips={ips}, search_type={search_type}",
        "search_type": search_type,
        "search_terms": search_terms,
        "countries": countries,
        "ports": ports,
        "protocols": [],
        "time_range": "now-90d",
        "matching_strategy": "token",
        "field_analysis": "Using fallback heuristic extraction from question text",
        "skip_search": False,  # Important: don't skip the search
    }


def _extract_ips_from_previous_results(previous_results: dict) -> list[str]:
    """Extract IPs from previous skill results for follow-up questions."""
    if not previous_results:
        return []

    seen: set[str] = set()
    ips: list[str] = []
    field_candidates = (
        "src_ip",
        "dest_ip",
        "source.ip",
        "destination.ip",
        "source_ip",
        "destination_ip",
        "ip",
    )

    for skill_result in previous_results.values():
        if not isinstance(skill_result, dict):
            continue

        for record in skill_result.get("results", []) or []:
            if not isinstance(record, dict):
                continue

            for field in field_candidates:
                value = record.get(field)
                if isinstance(value, str) and value not in seen and _IP_PATTERN.fullmatch(value):
                    seen.add(value)
                    ips.append(value)

            for value in (
                record.get("source", {}).get("ip") if isinstance(record.get("source"), dict) else None,
                record.get("destination", {}).get("ip") if isinstance(record.get("destination"), dict) else None,
            ):
                if isinstance(value, str) and value not in seen and _IP_PATTERN.fullmatch(value):
                    seen.add(value)
                    ips.append(value)

    return ips


def _infer_ip_direction(question: str, reasoning: str = "") -> str:
    """Infer whether the user means source, destination, or any IP direction."""
    combined = f"{question} {reasoning}".lower()

    if any(re.search(pattern, combined) for pattern in (
        r"\bfrom\s+",
        r"\boriginating\s+from\b",
        r"\bsource\s+ip\b",
        r"\bclient\s+ip\b",
        r"\bremote\s+ip\b",
    )):
        return "source"
    if any(re.search(pattern, combined) for pattern in (
        r"\bto\s+",
        r"\bdestination\s+ip\b",
        r"\bdest\s+ip\b",
        r"\bserver\s+ip\b",
        r"\btarget\s+ip\b",
    )):
        return "destination"
    return "any"


def _opposite_ip_direction(ip_direction: str) -> str:
    if ip_direction == "source":
        return "destination"
    if ip_direction == "destination":
        return "source"
    return "any"


def _get_nested_value(record: dict, field_name: str) -> Any:
    """Return a dotted-path value from a record if present."""
    if field_name in record:
        return record.get(field_name)

    current: Any = record
    for part in field_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current.get(part)
    return current


def _first_present_value(record: dict, field_names: list[str]) -> Any:
    """Return the first non-empty value for the provided candidate fields."""
    for field_name in field_names:
        value = _get_nested_value(record, field_name)
        if value not in (None, "", [], {}):
            return value
    return None


def _candidate_country_fields(field_mappings: dict | None) -> list[str]:
    """Return country-related fields discovered from RAG/schema metadata."""
    if not isinstance(field_mappings, dict):
        return []

    preferred = list(field_mappings.get("country_fields") or [])
    if preferred:
        return preferred

    all_fields = field_mappings.get("all_fields") or []
    return [field for field in all_fields if "country" in str(field).lower()][:12]


def _extract_validation_samples(results: list[dict], field_mappings: dict | None = None) -> list[dict]:
    """Extract compact, schema-aware validation samples from search results."""
    country_fields = _candidate_country_fields(field_mappings)
    country_fallbacks = [
        "geoip.country_name",
        "country_name",
        "country",
        "geo.country_name",
    ]
    if country_fields:
        country_candidates = country_fields + [field for field in country_fallbacks if field not in country_fields]
    else:
        country_candidates = country_fallbacks

    samples: list[dict] = []
    for result in results[:3]:
        sample_record: dict[str, Any] = {}

        if "alert" in result and isinstance(result["alert"], dict):
            alert_obj = result["alert"]
            sample_record["signature"] = alert_obj.get("signature")
            sample_record["signature_id"] = alert_obj.get("signature_id")
            sample_record["category"] = alert_obj.get("category")
        else:
            sample_record["signature"] = _first_present_value(result, ["alert.signature", "signature"])
            sample_record["signature_id"] = _first_present_value(result, ["alert.signature_id", "signature_id"])
            sample_record["category"] = _first_present_value(result, ["alert.category", "category"])

        sample_record["timestamp"] = _first_present_value(result, ["@timestamp", "timestamp", "flow.start", "flow.end"])
        sample_record["src_ip"] = _first_present_value(result, ["src_ip", "source.ip", "flow.src_ip", "alert.source.ip"])
        sample_record["dest_ip"] = _first_present_value(result, ["dest_ip", "destination.ip", "flow.dest_ip", "alert.target.ip"])
        sample_record["country"] = _first_present_value(result, country_candidates)
        sample_record["country_field"] = next(
            (field_name for field_name in country_candidates if _get_nested_value(result, field_name) not in (None, "", [], {})),
            None,
        )

        samples.append(sample_record)

    return samples


def _filter_results_for_exact_ip_match(
    results: list[dict],
    search_terms: list,
    field_mappings: dict,
    ip_direction: str,
) -> list[dict]:
    """Keep only records that actually contain the requested IP in the intended fields."""
    requested_ips = {str(term).strip().lower() for term in search_terms if _IP_PATTERN.fullmatch(str(term).strip())}
    if not requested_ips or not results:
        return results

    candidate_fields = _select_ip_query_fields(field_mappings, ip_direction)
    if not candidate_fields:
        return results

    filtered_results: list[dict] = []
    for row in results:
        for field_name in candidate_fields:
            field_value = _get_nested_value(row, field_name)
            if isinstance(field_value, str) and field_value.strip().lower() in requested_ips:
                filtered_results.append(row)
                break
    return filtered_results


def _build_directional_alternative_hint(
    results: list[dict],
    alternative_direction: str,
    time_range_label: str,
    total_results_count: int | None = None,
) -> dict | None:
    """Summarize opposite-direction IP hits when the requested direction has no matches."""
    if not results or alternative_direction not in {"source", "destination"}:
        return None

    timestamps: list[str] = []
    peers: set[str] = set()
    for row in results[:25]:
        ts = row.get("@timestamp") or row.get("timestamp")
        if ts:
            timestamps.append(str(ts))

        if alternative_direction == "destination":
            peer_candidates = [
                row.get("src_ip"),
                row.get("source_ip"),
                row.get("source.ip"),
                row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
            ]
        else:
            peer_candidates = [
                row.get("dest_ip"),
                row.get("destination_ip"),
                row.get("destination.ip"),
                row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
            ]

        for candidate in peer_candidates:
            if candidate:
                peers.add(str(candidate))

    timestamps = sorted(timestamps)
    return {
        "direction": alternative_direction,
        "results_count": int(total_results_count or len(results)),
        "time_range_label": time_range_label,
        "sample_peers": sorted(peers)[:10],
        "earliest": timestamps[0] if timestamps else None,
        "latest": timestamps[-1] if timestamps else None,
        "results": results[:10],
    }


def _build_sorted_sample_query(query: dict, *, size: int, order: str, track_total_hits: bool) -> dict:
    sample_query = copy.deepcopy(query)
    sample_query["size"] = size
    sample_query["sort"] = [{"@timestamp": {"order": order, "unmapped_type": "date"}}]
    sample_query["track_total_hits"] = track_total_hits
    return sample_query


def _dedupe_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []

    for row in rows:
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        destination = row.get("destination") if isinstance(row.get("destination"), dict) else {}
        key = (
            row.get("_id"),
            row.get("@timestamp") or row.get("timestamp"),
            row.get("src_ip") or row.get("source_ip") or source.get("ip"),
            row.get("dest_ip") or row.get("destination_ip") or destination.get("ip"),
            row.get("destination.port") or row.get("destination_port") or row.get("dest_port") or row.get("dport"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def _question_asks_for_ip_geolocation(question: str) -> bool:
    """Return True for follow-ups asking where referenced IPs are from."""
    q = str(question or "").lower()
    has_geo_intent = any(token in q for token in ("country", "countries", "origin", "where are", "where is", "from which country"))
    refers_to_prior_ips = "these ip" in q or "those ip" in q or "the ip" in q or "their country" in q
    return has_geo_intent and refers_to_prior_ips


def _question_asks_for_followup_details(question: str) -> bool:
    """Return True for follow-ups asking for details about previously mentioned traffic/IPs."""
    q = str(question or "").lower()
    # Check for follow-up patterns asking about traffic/connection details
    asks_for_details = any(token in q for token in (
        "what port", "which port", "what ports", 
        "what protocol", "which protocol",
        "traffic", "connection",
        "associated with", "associated with this",
        "from that", "from those", "from the"
    ))
    # More flexible pattern matching for previous context references
    refers_to_prior_context = any(phrase in q for phrase in (
        "this traffic", "these ips", "that ip", "the traffic",
        "that connection", "that traffic", "these connections"
    ))
    return asks_for_details and refers_to_prior_context


def _recover_followup_plan_from_context(
    question: str,
    query_plan: dict,
    previous_results: dict,
    conversation_history: list[dict],
) -> dict:
    """
    Recover concrete IP/traffic criteria for follow-up questions from context.

    Handles:
    - Geographic follow-ups: "What countries are these IPs from?"
    - Port/Protocol follow-ups: "What port was associated with this traffic?"
    - Traffic detail follow-ups: "What protocols were used?"
    
    Example:
      - Prior answer listed IPs from an alert search (147.185.132.112, 192.168.0.16)
      - User asks: "What countries are these IPs from?" → Recovered IPs from context
      - User asks: "What port was associated with this traffic?" → Search for ports on recovered IPs
    """
    plan = dict(query_plan or {})
    if plan.get("search_terms") or plan.get("countries") or plan.get("ports") or plan.get("protocols"):
        return plan

    # Check if this is a follow-up asking for details about previously mentioned traffic
    if _question_asks_for_ip_geolocation(question):
        # Existing geolocation recovery logic
        candidate_ips = _extract_ips_from_previous_results(previous_results)
        if not candidate_ips:
            for message in reversed(conversation_history or []):
                content = message.get("content", "") if isinstance(message, dict) else ""
                extracted = _extract_ips_from_text(content)
                for ip in extracted:
                    if ip not in candidate_ips:
                        candidate_ips.append(ip)
                if candidate_ips:
                    break

        if not candidate_ips:
            return plan

        recovered_ips = candidate_ips[:12]
        reasoning_prefix = (plan.get("reasoning") or "").strip()
        recovery_reason = f"Recovered {len(recovered_ips)} IP(s) from prior context for geographic follow-up lookup."

        plan["search_type"] = "ip"
        plan["search_terms"] = recovered_ips
        plan["countries"] = []
        plan["ports"] = plan.get("ports", []) if isinstance(plan.get("ports"), list) else []
        plan["protocols"] = plan.get("protocols", []) if isinstance(plan.get("protocols"), list) else []
        plan["matching_strategy"] = "term"
        plan["time_range"] = plan.get("time_range") or "now-90d"
        plan["field_analysis"] = "Using contextual IP addresses from previous results/history because the current question is a referential follow-up."
        plan["reasoning"] = f"{reasoning_prefix} {recovery_reason}".strip()
        return plan
    
    # Handle follow-ups asking about traffic details (ports, protocols, etc.)
    if _question_asks_for_followup_details(question):
        candidate_ips = _extract_ips_from_previous_results(previous_results)
        if not candidate_ips:
            for message in reversed(conversation_history or []):
                content = message.get("content", "") if isinstance(message, dict) else ""
                extracted = _extract_ips_from_text(content)
                for ip in extracted:
                    if ip not in candidate_ips:
                        candidate_ips.append(ip)
                if candidate_ips:
                    break

        if not candidate_ips:
            return plan

        recovered_ips = candidate_ips[:12]
        reasoning_prefix = (plan.get("reasoning") or "").strip()
        recovery_reason = f"Recovered {len(recovered_ips)} IP(s) from prior context for traffic detail follow-up."
        
        # Infer what details are being asked for from the question
        q_lower = question.lower()
        is_port_query = "port" in q_lower
        is_protocol_query = "protocol" in q_lower
        
        # Build search focused on the recovered IPs and the requested details
        plan["search_type"] = "traffic"
        plan["search_terms"] = recovered_ips
        if is_port_query:
            plan["ports"] = []  # Empty ports means "search for any port on these IPs"
        if is_protocol_query:
            plan["protocols"] = []  # Empty protocols means "search for any protocol on these IPs"
        plan["countries"] = plan.get("countries", []) if isinstance(plan.get("countries"), list) else []
        plan["matching_strategy"] = "term"
        plan["time_range"] = plan.get("time_range") or "now-90d"
        plan["field_analysis"] = f"Searching traffic details on recovered IPs: {', '.join(recovered_ips[:3])}"
        plan["reasoning"] = f"{reasoning_prefix} {recovery_reason}".strip()

    return plan


def _execute_search_with_llm_repair(db: Any, llm: Any, index: str, query: dict, size: int = None) -> list[dict]:
    """
    Execute search with intelligent repair on malformed queries.
    
    Uses QueryRepairMemory to remember successful fixes and avoid redundant LLM calls.
    Retries up to 3 times with progressively detailed prompts.
    """
    if size is None:
        size = query.get("size", 200)
    
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
            # Non-malformed errors
            logger.error("[%s] Unexpected search error (type: %s): %s", SKILL_NAME, type(exc).__name__, exc)
            return []


def _execute_search_with_metadata_repair(db: Any, llm: Any, index: str, query: dict, size: int = None) -> dict[str, Any]:
    """Execute a search and preserve total-hit metadata when supported by the backend."""
    if size is None:
        size = query.get("size", 200)

    try:
        logger.debug("[%s] Executing metadata search query on index: %s", SKILL_NAME, index)
        if hasattr(db, "search_with_metadata"):
            response = db.search_with_metadata(index, query, size=size)
            results = response.get("results", [])
            return {
                "results": results,
                "total": int(response.get("total", len(results)) or 0),
            }

        results = db.search(index, query, size=size)
        return {"results": results, "total": len(results)}
    except Exception as exc:
        from core.db_connector import QueryMalformedException

        if isinstance(exc, QueryMalformedException):
            logger.warning("[%s] Query malformed: %s — attempting intelligent repair", SKILL_NAME, exc.error_message)

            from core.query_repair import IntelligentQueryRepair
            repair = IntelligentQueryRepair(db, llm)
            success, results, message = repair.repair_and_retry(index, exc.original_query, size=size)

            if success:
                logger.info("[%s] Repair successful! Got %d results", SKILL_NAME, len(results or []))
                repaired_results = results or []
                return {"results": repaired_results, "total": len(repaired_results)}

            logger.error("[%s] Repair failed: %s", SKILL_NAME, message)
            return {"results": [], "total": 0}

        logger.error("[%s] Unexpected search error (type: %s): %s", SKILL_NAME, type(exc).__name__, exc)
        return {"results": [], "total": 0}


def _sample_results_across_time_range(
    db: Any,
    llm: Any,
    index: str,
    query: dict,
    *,
    sample_size: int,
    search_terms: list[str],
    field_mappings: dict[str, Any],
    ip_direction: str,
    resolved_time_range: str,
) -> dict[str, Any]:
    """Collect bounded samples from the start and end of the requested time range."""
    newest_query = _build_sorted_sample_query(query, size=sample_size, order="desc", track_total_hits=True)
    newest_results = _execute_search_with_llm_repair(db, llm, index, newest_query, size=sample_size)
    newest_results = _filter_results_for_exact_ip_match(newest_results, search_terms, field_mappings, ip_direction)
    newest_results = _filter_results_for_time_range(newest_results, resolved_time_range)
    total_hits = len(newest_results)

    total_query = copy.deepcopy(query)
    total_query["size"] = 0
    total_query["track_total_hits"] = True
    total_response = _execute_search_with_metadata_repair(db, llm, index, total_query, size=0)
    total_hits = max(total_hits, int(total_response.get("total", 0) or 0))

    if total_hits <= sample_size:
        ordered_results = sorted(
            newest_results,
            key=lambda row: str(row.get("@timestamp") or row.get("timestamp") or ""),
        )
        return {
            "display_results": newest_results,
            "summary_results": ordered_results,
            "results_count": total_hits,
            "page_results_count": len(newest_results),
            "sampled_results_count": len(ordered_results),
            "sample_strategy": "page",
            "oldest_sample_count": len(ordered_results),
            "newest_sample_count": len(ordered_results),
        }

    oldest_query = _build_sorted_sample_query(query, size=sample_size, order="asc", track_total_hits=False)
    oldest_results = _execute_search_with_llm_repair(db, llm, index, oldest_query, size=sample_size)
    oldest_results = _filter_results_for_exact_ip_match(oldest_results, search_terms, field_mappings, ip_direction)
    oldest_results = _filter_results_for_time_range(oldest_results, resolved_time_range)

    summary_results = _dedupe_results(oldest_results + list(reversed(newest_results)))
    summary_results = sorted(
        summary_results,
        key=lambda row: str(row.get("@timestamp") or row.get("timestamp") or ""),
    )
    return {
        "display_results": newest_results,
        "summary_results": summary_results,
        "results_count": total_hits,
        "page_results_count": len(newest_results),
        "sampled_results_count": len(summary_results),
        "sample_strategy": "edge_windows",
        "oldest_sample_count": len(oldest_results),
        "newest_sample_count": len(newest_results),
    }

SKILL_NAME = "opensearch_querier"


def run(context: dict) -> dict:
    """Entry point for opensearch_querier skill."""
    from core.query_builder import (
        discover_field_mappings,
        build_keyword_query,
    )

    db = context.get("db")
    llm = context.get("llm")
    cfg = context.get("config")
    parameters = context.get("parameters", {})
    previous_results = context.get("previous_results", {})

    # Defensive check: ensure db is actually a database connector
    if db is None:
        logger.warning("[%s] db not available — skipping.", SKILL_NAME)
        return {"status": "skipped", "reason": "no db"}
    
    from core.db_connector import BaseDBConnector
    if not isinstance(db, BaseDBConnector):
        logger.error("[%s] db is not a BaseDBConnector! Got type=%s. Context keys: %s", 
                    SKILL_NAME, type(db).__name__, list(context.keys()))
        return {"status": "error", "error": f"db is corrupted: {type(db).__name__}"}

    # Get query parameters - if provided explicitly, use them
    # Otherwise, use defaults and let LLM determine search strategy
    index = parameters.get("index", cfg.get("db", "logs_index", default="securityclaw-logs"))
    question = parameters.get("question", parameters.get("query"))
    
    # If neither question/query provided, this was likely a direct dispatch with
    # explicit parameters like keywords, query_type, etc.
    if not question and (parameters.get("keywords") or parameters.get("raw_query")):
        return _execute_explicit_query(context, index)
    
    if not question:
        logger.warning("[%s] No question provided in parameters", SKILL_NAME)
        return {"status": "skipped", "reason": "no question"}
    
    # ── LLM PLANNING PHASE (like rag_querier) ────────────────────────────────
    # Use LLM to understand what to search for
    if llm is None:
        logger.warning("[%s] LLM not available for query planning.", SKILL_NAME)
        return {"status": "skipped", "reason": "no llm"}
    
    conversation_history = parameters.get("conversation_history", [])
    
    # ── CHECK FOR PREVIOUSLY DISCOVERED FIELDS (from fields_querier) ──────────
    # If fields_querier ran before, use its discovered field mappings
    field_mappings = None
    if previous_results.get("fields_querier"):
        fields_result = previous_results["fields_querier"]
        if fields_result.get("status") == "ok":
            field_mappings = fields_result.get("field_mappings") or fields_result.get("findings", {}).get("field_mappings")
            if field_mappings:
                logger.info("[%s] Using field mappings discovered by fields_querier", SKILL_NAME)
    
    # Always ensure we have complete field mappings from DB schema
    # This handles cases where fields_querier returns partial mappings
    full_field_mappings = discover_field_mappings(db, llm)
    
    # Merge field_mappings: use fields_querier as primary, but supplement with DB discovery
    # This ensures we have both the curated fields AND any fields fields_querier might have missed
    if field_mappings:
        logger.debug("[%s] run: Before merge - text_fields count: %d, has_alert.signature: %s", 
                    SKILL_NAME, len(field_mappings.get("text_fields", [])), 
                    "alert.signature" in field_mappings.get("text_fields", []))
        # Merge: add any missing categories and fields from discovery
        for category, fields in full_field_mappings.items():
            if category not in field_mappings:
                field_mappings[category] = fields
            elif isinstance(fields, list) and isinstance(field_mappings.get(category), list):
                # Merge lists: fields_querier first, then add any from discovery that aren't there
                existing = set(field_mappings[category])
                for field in fields:
                    if field not in existing:
                        field_mappings[category].append(field)
        logger.info("[%s] Supplemented fields_querier mappings with full DB discovery", SKILL_NAME)
        logger.debug("[%s] run: After merge - text_fields count: %d, has_alert.signature: %s", 
                    SKILL_NAME, len(field_mappings.get("text_fields", [])), 
                    "alert.signature" in field_mappings.get("text_fields", []))
    else:
        field_mappings = full_field_mappings
        logger.info("[%s] Using full field mappings from DB discovery", SKILL_NAME)
    
    query_plan = _plan_opensearch_query_with_llm(
        question, conversation_history, field_mappings, llm
    )
    query_plan = _recover_followup_plan_from_context(
        question,
        query_plan,
        previous_results,
        conversation_history,
    )
    query_plan = _maybe_attach_country_aggregation_plan(question, query_plan)
    
    if not query_plan or query_plan.get("skip_search"):
        logger.info("[%s] LLM determined no search needed.", SKILL_NAME)
        return {"status": "no_action", "reason": "query not needed for raw logs"}
    
    # ── LOG REASONING STEP 1: Intent Analysis ──────────────────────────
    logger.info("[%s] REASONING CHAIN - Step 1: Intent Analysis", SKILL_NAME)
    logger.info("[%s]   Search Type: %s", SKILL_NAME, query_plan.get("search_type"))
    logger.info("[%s]   Matching Strategy: %s", SKILL_NAME, query_plan.get("matching_strategy"))
    logger.info("[%s]   Reasoning: %s", SKILL_NAME, _log_excerpt(query_plan.get("reasoning", ""), limit=500))
    
    search_terms = query_plan.get("search_terms", [])
    countries = query_plan.get("countries", [])
    ports = query_plan.get("ports", [])
    protocols = query_plan.get("protocols", [])
    raw_time_range = query_plan.get("time_range", "now-90d")
    resolved_time_range, time_range_label = _resolve_time_range_for_question(question, raw_time_range)
    matching_strategy = query_plan.get("matching_strategy", "token")
    ip_direction = _infer_ip_direction(question, query_plan.get("reasoning", ""))
    aggregation_type = query_plan.get("aggregation_type")
    exclude_countries = query_plan.get("exclude_countries", []) if isinstance(query_plan.get("exclude_countries"), list) else []
    result_limit = int(query_plan.get("result_limit", 10) or 10)
    requested_filters = {
        "countries": countries,
        "ports": ports,
        "protocols": protocols,
        "time_range": resolved_time_range,
        "time_range_label": time_range_label,
    }

    has_criteria = bool(search_terms or countries or ports or protocols or aggregation_type)
    if not has_criteria:
        logger.info("[%s] LLM planning: no search criteria extracted.", SKILL_NAME)
        return {"status": "no_action"}

    if aggregation_type == "country_terms":
        logger.info("[%s] REASONING CHAIN - Step 2: Country Aggregation", SKILL_NAME)
        logger.info(
            "[%s]   Executing country aggregation | Time(raw=%s,resolved=%s) | Excluding=%s | Limit=%d",
            SKILL_NAME,
            raw_time_range,
            resolved_time_range,
            exclude_countries,
            result_limit,
        )
        aggregation_result = _execute_country_aggregation_query(
            db=db,
            index=index,
            field_mappings=field_mappings,
            time_range=resolved_time_range,
            exclude_countries=exclude_countries,
            result_limit=result_limit,
        )
        aggregation_result.update(
            {
                "search_terms": search_terms,
                "countries": countries,
                "ports": ports,
                "protocols": protocols,
                "time_range": raw_time_range,
                "time_range_label": time_range_label,
                "time_range_resolved": resolved_time_range,
                "reasoning": query_plan.get("reasoning", ""),
                "ip_direction": ip_direction,
                "directional_alternative": None,
                "validation_failed": False,
                "validation_issue": "",
                "validation_reflection": "",
                "reasoning_chain": {
                    "planning": query_plan.get("reasoning"),
                    "strategy_used": matching_strategy,
                    "validation_issue": "",
                    "validation_reflection": "",
                    "recovery_performed": False,
                },
            }
        )
        logger.info(
            "[%s]   Aggregated countries found: %d",
            SKILL_NAME,
            len(aggregation_result.get("country_buckets") or []),
        )
        return aggregation_result

    # ── BUILD QUERY using LLM-determined strategy ──────────────────────────────
    # Let LLM decide all aspects including field selection and matching strategy
    query = _build_opensearch_query(
        search_terms=search_terms,
        countries=countries,
        ports=ports,
        protocols=protocols,
        time_range=resolved_time_range,
        field_mappings=field_mappings,
        matching_strategy=matching_strategy,
        ip_direction=ip_direction,
    )
    query["size"] = parameters.get("size", 200)

    logger.debug("[%s] Built query: %s", SKILL_NAME, _log_excerpt(query, limit=700))
    logger.info(
        "[%s] Querying '%s': %s | Strategy=%s | Time(raw=%s,resolved=%s) | Countries: %s | Ports: %s | Terms: %s | Field_mappings_type=%s",
        SKILL_NAME, index, query_plan.get("reasoning", ""), matching_strategy, raw_time_range, resolved_time_range, countries, ports, search_terms,
        type(field_mappings).__name__
    )

    try:
        validation: dict[str, Any] = {"is_valid": True, "issue": "", "reflection": ""}
        sampled_search = _sample_results_across_time_range(
            db,
            llm,
            index,
            query,
            sample_size=int(parameters.get("size", 200) or 200),
            search_terms=search_terms,
            field_mappings=field_mappings,
            ip_direction=ip_direction,
            resolved_time_range=resolved_time_range,
        )
        results = sampled_search.get("display_results", [])
        summary_results = sampled_search.get("summary_results", results)
        total_results_count = int(sampled_search.get("results_count", len(results)) or 0)
        page_results_count = int(sampled_search.get("page_results_count", len(results)) or 0)
        sampled_results_count = int(sampled_search.get("sampled_results_count", len(summary_results)) or 0)
        logger.info(
            "[%s] Raw results from opensearch: fetched=%d sampled=%d total_hits=%d",
            SKILL_NAME,
            page_results_count,
            sampled_results_count,
            total_results_count,
        )
        
        # ── LOG REASONING STEP 2: Query Execution ──────────────────────────
        logger.info("[%s] REASONING CHAIN - Step 2: Query Execution", SKILL_NAME)
        logger.info(
            "[%s]   Results Found: fetched=%d | total_hits=%d | sampled=%d",
            SKILL_NAME,
            page_results_count,
            total_results_count,
            sampled_results_count,
        )
        if not results:
            logger.info("[%s]   Status: ZERO RESULTS - will attempt multi-turn diagnosis", SKILL_NAME)
        
        # DEBUG: Log the actual query and first few results for investigation
        import json
        query_must = query.get("query", {}).get("bool", {}).get("must", [])
        if query_must and isinstance(query_must[0], dict) and "bool" in query_must[0]:
            should_clause = query_must[0]["bool"].get("should", [])
            field_names = set()
            for clause in should_clause:
                for key in clause:
                    if isinstance(clause[key], dict):
                        field_names.update(clause[key].keys())
            logger.debug("[%s] Query fields in should clause: %s", SKILL_NAME, field_names)
        
        if results:
            logger.debug("[%s] First result signature: %s", SKILL_NAME, 
                        results[0].get("alert", {}).get("signature", "N/A")[:80] if results[0].get("alert", {}).get("signature") else "N/A")

        # ── RESULT VALIDATION WITH REFLECTION ──────────────────────────
        # If we got results, validate that they actually match the intent
        if results:
            validation = _llm_validate_results_reflective(
                question=question,
                search_terms=search_terms,
                requested_filters=requested_filters,
                results=results,
                field_mappings=field_mappings,
                previous_validation_failed=False,
                llm=llm,
            )
            
            # ── LOG REASONING STEP 3: Validation & Reflection ──────────────────────────
            logger.info("[%s] REASONING CHAIN - Step 3: Validation & Reflection", SKILL_NAME)
            logger.info("[%s]   Valid: %s | Confidence: %.1f%%", SKILL_NAME, 
                       validation.get("is_valid"), validation.get("confidence", 0))
            if not validation.get("is_valid"):
                logger.warning("[%s]   Issue: %s", SKILL_NAME, validation.get("issue"))
                logger.info("[%s]   LLM Reflection: %s", SKILL_NAME, _log_excerpt(validation.get("reflection", "none"), limit=700))
            
            if not validation.get("is_valid"):
                logger.warning(
                    "[%s] LLM validation failed: %s | Reflection: %s",
                    SKILL_NAME, validation.get("issue"), _log_excerpt(validation.get("reflection", "none"), limit=1200)
                )
                # Try recovery: switch matching strategy
                recovery_strategy = "token" if matching_strategy == "phrase" else "phrase"
                logger.info("[%s] Trying recovery with alternate strategy: %s", SKILL_NAME, recovery_strategy)
                
                recovery_query = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=resolved_time_range,
                    field_mappings=field_mappings,
                    matching_strategy=recovery_strategy,
                    ip_direction=ip_direction,
                )
                recovery_query["size"] = parameters.get("size", 200)
                
                recovery_results = _execute_search_with_llm_repair(db, llm, index, recovery_query)
                recovery_results = _filter_results_for_exact_ip_match(recovery_results, search_terms, field_mappings, ip_direction)
                recovery_results = _filter_results_for_time_range(recovery_results, resolved_time_range)
                
                if recovery_results:
                    recovery_validation = _llm_validate_results_reflective(
                        question=question,
                        search_terms=search_terms,
                        requested_filters=requested_filters,
                        results=recovery_results,
                        field_mappings=field_mappings,
                        previous_validation_failed=True,
                        llm=llm,
                    )
                    if recovery_validation.get("is_valid"):
                        logger.info("[%s] Recovery strategy succeeded after reflection", SKILL_NAME)
                        sampled_search = _sample_results_across_time_range(
                            db,
                            llm,
                            index,
                            recovery_query,
                            sample_size=int(parameters.get("size", 200) or 200),
                            search_terms=search_terms,
                            field_mappings=field_mappings,
                            ip_direction=ip_direction,
                            resolved_time_range=resolved_time_range,
                        )
                        results = sampled_search.get("display_results", recovery_results)
                        summary_results = sampled_search.get("summary_results", recovery_results)
                        total_results_count = int(sampled_search.get("results_count", len(recovery_results)) or 0)
                        page_results_count = int(sampled_search.get("page_results_count", len(results)) or 0)
                        sampled_results_count = int(sampled_search.get("sampled_results_count", len(summary_results)) or 0)
                        validation = recovery_validation
                    else:
                        logger.warning("[%s] Recovery failed, keeping original results", SKILL_NAME)

        directional_alternative = None

        if not results and search_terms and ip_direction in {"source", "destination"}:
            alternative_direction = _opposite_ip_direction(ip_direction)
            logger.info(
                "[%s] No %s-direction IP matches found; probing %s direction before LLM diagnosis",
                SKILL_NAME,
                ip_direction,
                alternative_direction,
            )
            alternative_query = _build_opensearch_query(
                search_terms=search_terms,
                countries=countries,
                ports=ports,
                protocols=protocols,
                time_range=resolved_time_range,
                field_mappings=field_mappings,
                matching_strategy=matching_strategy,
                ip_direction=alternative_direction,
            )
            alternative_query["size"] = parameters.get("size", 200)
            alternative_sample = _sample_results_across_time_range(
                db,
                llm,
                index,
                alternative_query,
                sample_size=int(parameters.get("size", 200) or 200),
                search_terms=search_terms,
                field_mappings=field_mappings,
                ip_direction=alternative_direction,
                resolved_time_range=resolved_time_range,
            )
            alternative_results = alternative_sample.get("summary_results", [])
            if alternative_results:
                directional_alternative = _build_directional_alternative_hint(
                    alternative_results,
                    alternative_direction,
                    time_range_label,
                    total_results_count=int(alternative_sample.get("results_count", len(alternative_results)) or 0),
                )
                logger.info(
                    "[%s] Found %d opposite-direction IP matches (%s) in %s window",
                    SKILL_NAME,
                    int(alternative_sample.get("results_count", len(alternative_results)) or 0),
                    alternative_direction,
                    time_range_label,
                )

        # Recovery: if primary query returns 0 results, diagnose why and suggest recovery
        if not results:
            # Fast check: pattern-based diagnosis (keep for speed)
            logger.warning("[%s] Primary query returned 0 results - performing multi-turn diagnosis", SKILL_NAME)
            
            # ── LOG REASONING STEP 3: Multi-Turn Diagnosis ──────────────────────────
            logger.info("[%s] REASONING CHAIN - Step 3: Zero Results Diagnosis", SKILL_NAME)
            
            diagnosis = _diagnose_query_failure(
                question=question,
                search_terms=search_terms,
                field_mappings=field_mappings,
                last_strategy=matching_strategy,
                llm=llm,
            )
            
            logger.info("[%s]   Suggested Recovery: %s", SKILL_NAME, _log_excerpt(diagnosis.get("suggested_recovery", "none"), limit=700))
            
            # Try LLM-suggested recovery if it looks promising
            if diagnosis.get("should_try_recovery"):
                # First try: phrase → token recovery (most common fix)
                recovery_strategy = "token" if matching_strategy == "phrase" else "phrase"
                logger.info("[%s] Attempting strategy switch recovery: %s → %s", 
                           SKILL_NAME, matching_strategy, recovery_strategy)
                
                recovery_query = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=resolved_time_range,
                    field_mappings=field_mappings,
                    matching_strategy=recovery_strategy,
                    ip_direction=ip_direction,
                )
                if recovery_query:
                    recovery_query["size"] = parameters.get("size", 200)
                    results = _execute_search_with_llm_repair(db, llm, index, recovery_query)
                    results = _filter_results_for_exact_ip_match(results, search_terms, field_mappings, ip_direction)
                    if results:
                        sampled_search = _sample_results_across_time_range(
                            db,
                            llm,
                            index,
                            recovery_query,
                            sample_size=int(parameters.get("size", 200) or 200),
                            search_terms=search_terms,
                            field_mappings=field_mappings,
                            ip_direction=ip_direction,
                            resolved_time_range=resolved_time_range,
                        )
                        results = sampled_search.get("display_results", results)
                        summary_results = sampled_search.get("summary_results", results)
                        total_results_count = int(sampled_search.get("results_count", len(results)) or 0)
                        page_results_count = int(sampled_search.get("page_results_count", len(results)) or 0)
                        sampled_results_count = int(sampled_search.get("sampled_results_count", len(summary_results)) or 0)
                        logger.info("[%s] Recovery successful: got %d results after strategy switch", 
                                   SKILL_NAME, len(results))
            
            # If still nothing and we have countries, try relaxed matching
            if not results and countries:
                logger.info("[%s] Still no results with countries filter - trying relaxed matching", SKILL_NAME)
                recovery = _build_opensearch_query(
                    search_terms=search_terms,
                    countries=countries,
                    ports=ports,
                    protocols=protocols,
                    time_range=resolved_time_range,
                    field_mappings=field_mappings,
                    ip_direction=ip_direction,
                    relaxed=True,
                )
                if recovery:
                    recovery["size"] = parameters.get("size", 200)
                    results = _execute_search_with_llm_repair(db, llm, index, recovery)
                    results = _filter_results_for_exact_ip_match(results, search_terms, field_mappings, ip_direction)
                    results = _filter_results_for_time_range(results, resolved_time_range)
                    if results:
                        sampled_search = _sample_results_across_time_range(
                            db,
                            llm,
                            index,
                            recovery,
                            sample_size=int(parameters.get("size", 200) or 200),
                            search_terms=search_terms,
                            field_mappings=field_mappings,
                            ip_direction=ip_direction,
                            resolved_time_range=resolved_time_range,
                        )
                        results = sampled_search.get("display_results", results)
                        summary_results = sampled_search.get("summary_results", results)
                        total_results_count = int(sampled_search.get("results_count", len(results)) or 0)
                        page_results_count = int(sampled_search.get("page_results_count", len(results)) or 0)
                        sampled_results_count = int(sampled_search.get("sampled_results_count", len(summary_results)) or 0)
                        logger.info("[%s] Relaxed recovery succeeded: got %d results", SKILL_NAME, len(results))

        return {
            "status": "ok",
            "results_count": total_results_count,
            "page_results_count": page_results_count,
            "sampled_results_count": sampled_results_count,
            "sample_strategy": sampled_search.get("sample_strategy", "page"),
            "oldest_sample_count": int(sampled_search.get("oldest_sample_count", sampled_results_count) or 0),
            "newest_sample_count": int(sampled_search.get("newest_sample_count", sampled_results_count) or 0),
            "results": results[:25],  # Return top 25 for display
            "summary_results": summary_results,
            "search_terms": search_terms,
            "countries": countries,
            "ports": ports,
            "protocols": protocols,
            "time_range": raw_time_range,
            "time_range_label": time_range_label,
            "time_range_resolved": resolved_time_range,
            "reasoning": query_plan.get("reasoning", ""),
            "ip_direction": ip_direction,
            "directional_alternative": directional_alternative,
            "validation_failed": bool(results and not validation.get("is_valid", True)),
            "validation_issue": str(validation.get("issue", "")),
            "validation_reflection": str(validation.get("reflection", "")),
            "reasoning_chain": {
                "planning": query_plan.get("reasoning"),
                "strategy_used": matching_strategy,
                "validation_issue": str(validation.get("issue", "")),
                "validation_reflection": str(validation.get("reflection", "")),
                "recovery_performed": not results if not results else False,
            }
        }
    except Exception as exc:
        logger.error("[%s] Search failed: %s", SKILL_NAME, exc)
        return {"status": "error", "error": str(exc)}


# Country code mapping (same as rag_querier)
_COUNTRY_CODE_MAP = {
    "iran": "IR", "iraq": "IQ", "syria": "SY", "north korea": "KP",
    "china": "CN", "russia": "RU", "united states": "US", "usa": "US",
    "uk": "GB", "united kingdom": "GB", "france": "FR", "germany": "DE",
    "india": "IN", "pakistan": "PK",
}


def _build_opensearch_query(
    search_terms: list,
    countries: list,
    ports: list,
    protocols: list,
    time_range: str | dict[str, str],
    field_mappings: dict,
    matching_strategy: str = "token",
    ip_direction: str = "any",
    relaxed: bool = False,
) -> dict:
    """
    Build a robust OpenSearch query using LLM-recommended matching strategy.
    
    Args:
        matching_strategy: "phrase" (exact phrase), "token" (tokenized), "term" (exact term)
                          - Chosen by LLM based on field analysis
    """
    # CRITICAL: Override matching strategy for IP addresses
    # IP addresses must use term matching (exact match), never phrase matching
    if matching_strategy == "phrase" and search_terms:
        import re
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$|^[a-f0-9:]{3,}$'  # IPv4 or IPv6-like
        if any(re.match(ip_pattern, str(t)) for t in search_terms):
            logger.info("[%s] _build_opensearch_query: Detected IP address in search terms — overriding strategy from 'phrase' to 'term'", SKILL_NAME)
            matching_strategy = "term"
    
    must_clauses = []
    all_fields = field_mappings.get("all_fields") or []
    country_fields = _candidate_country_fields(field_mappings)[:10]
    port_fields = (field_mappings.get("port_fields") or [])[:6]
    
    # ── PRIORITIZE SPECIFIC FIELD TYPES ──────────────────────────────────
    # If specific text_fields are provided (e.g., for alert queries), use them
    # Otherwise, infer common fields that are likely to have the data
    # Skip timestamp/numeric fields that shouldn't be used for text matching
    if field_mappings.get("text_fields"):
        available_text = field_mappings["text_fields"]
    else:
        available_text = []
    
    logger.debug("[%s] _build_opensearch_query: available_text before injection: %s (length=%d, has_alert.signature=%s)", 
                SKILL_NAME, available_text[:15] if len(available_text) > 15 else available_text, 
                len(available_text), "alert.signature" in available_text)
    
    # Make sure alert fields are included if available
    if not available_text or "alert.signature" not in available_text:
        all_text_with_alerts = list(available_text) if available_text else []
        # Add alert-related fields from all_fields if they're not already there
        alert_fields_to_add = [f for f in all_fields if "alert" in str(f).lower()]
        logger.debug("[%s] _build_opensearch_query: Adding %d alert fields", SKILL_NAME, len(alert_fields_to_add))
        for af in alert_fields_to_add:
            if af not in all_text_with_alerts:
                all_text_with_alerts.append(af)
        available_text = all_text_with_alerts
        logger.debug("[%s] _build_opensearch_query: After injection: length=%d, has_alert.signature=%s", 
                    SKILL_NAME, len(available_text), "alert.signature" in available_text)
    
    # Now filter out timestamp/numeric fields
    filter_keywords = ("timestamp", "time", "date", "epoch", "_ms", "port", "bytes", "count", "length", "size")
    filtered_text = [
        f for f in available_text
        if not any(k in str(f).lower() for k in filter_keywords)
    ]
    
    # CRITICAL: Prioritize alert.* fields to ensure they're included in top 12
    # Separate alert fields from others
    alert_fields_list = [f for f in filtered_text if f.startswith("alert.")]
    non_alert_fields = [f for f in filtered_text if not f.startswith("alert.")]
    
    # Put alert fields first, then other fields, take first 12
    # This ensures alert.signature, alert.category, etc. are included
    prioritized = alert_fields_list + non_alert_fields
    text_fields = prioritized[:12]
    
    # CRITICAL FIX: If we have alert fields AND search terms that look like alert signatures,
    # use ONLY alert fields to avoid OpenSearch field mismatch issues  
    # (mixing alert.signature with flow.state causes "no results" in some cases)
    has_alert_fields = len(alert_fields_list) > 0
    search_looks_like_alert = any(
        term.upper() in ("EXPLOIT", "MALWARE", "POLICY", "CVE", "BACKDOOR", "TROJAN", "WORM")
        or "ET " in term.upper() 
        for term in search_terms
    )
    
    logger.debug("[%s] _build_opensearch_query: has_alert_fields=%s, search_looks_like_alert=%s, search_terms=%s",
                SKILL_NAME, has_alert_fields, search_looks_like_alert, search_terms)
    
    if has_alert_fields and search_looks_like_alert:
        # Use only alert fields for alert signature searches
        text_fields = alert_fields_list[:12] if alert_fields_list else text_fields
        logger.info("[%s] _build_opensearch_query: Using alert-only fields for alert signature search. Fields: %s", SKILL_NAME, text_fields)
    
    logger.debug("[%s] _build_opensearch_query: Prioritized fields with alerts first: %s (has_alert.signature=%s)", 
                SKILL_NAME, text_fields, "alert.signature" in text_fields)

    # Country matching — use match_phrase for full name AND term for ISO code
    # IMPORTANT: Skip country filters for alert signature searches - the alert itself is what matters
    # Alert records may not have geoip data, and filtering by country would lose alert results
    if countries and country_fields and not (search_looks_like_alert and has_alert_fields):
        country_should = []
        for field in country_fields:
            for country in countries:
                country_should.append({"match_phrase": {field: country}})
                code = _COUNTRY_CODE_MAP.get(country.lower())
                if code:
                    country_should.append({"term": {field: code}})
                    country_should.append({"term": {field: code.lower()}})
        if country_should:
            must_clauses.append({"bool": {"should": country_should, "minimum_should_match": 1}})
    elif countries and not country_fields:
        # No discovered country fields — fall back to multi_match on text fields
        for country in countries:
            must_clauses.append({"multi_match": {"query": country, "fields": text_fields or ["*"]}})

    # Port matching
    if ports and port_fields:
        port_should = []
        for field in port_fields:
            for p in ports:
                try:
                    port_should.append({"term": {field: int(p)}})
                except Exception:
                    pass
        if port_should:
            must_clauses.append({"bool": {"should": port_should, "minimum_should_match": 1}})

    # Protocol matching
    proto_fields = [f for f in all_fields if "proto" in str(f).lower()][:6]
    if protocols and proto_fields:
        proto_should = []
        for field in proto_fields:
            for proto in protocols:
                proto_should.append({"term": {field: str(proto).lower()}})
                proto_should.append({"match": {field: str(proto)}})
        if proto_should:
            must_clauses.append({"bool": {"should": proto_should, "minimum_should_match": 1}})

    # Keyword search - use LLM-recommended matching strategy
    if search_terms and (relaxed or not must_clauses):
        # CRITICAL: For IP addresses, search in IP fields, not text fields
        import re
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$|^[a-f0-9:]{3,}$'  # IPv4 or IPv6-like
        is_ip_search = any(re.match(ip_pattern, str(t)) for t in search_terms)
        
        if is_ip_search:
            # Use only high-confidence IP fields to avoid malformed geo/metadata queries.
            fields = _select_ip_query_fields(field_mappings, ip_direction)
            logger.info("[%s] _build_opensearch_query: Detected IP search - using IP fields: %s", SKILL_NAME, fields[:5] if len(fields) > 5 else fields)
        else:
            # Select text fields for keyword search
            # Priority 1: Use provided text_fields if available  (already prioritized with alerts first)
            # Priority 2: Fall back to fields containing "message", "alert", "signature", etc.
            
            # NOTE: text_fields was already carefully selected and prioritized above
            # to ensure alert fields come first. Don't re-extract or re-filter here!
            if not text_fields:
                # Only fall back if text_fields is empty (which shouldn't happen after prioritization above)
                if field_mappings.get("text_fields"):
                    text_fields = field_mappings["text_fields"][:12]
                    text_fields = [
                        f for f in text_fields
                        if not any(k in str(f).lower() for k in ("timestamp", "time", "date", "epoch", "_ms", "port", "id", "bytes", "count", "length", "size", "timeout"))
                    ]
                else:
                    text_fields = [
                        f for f in all_fields
                        if any(k in str(f).lower() for k in ("message", "alert", "signature", "event", "hostname", "domain"))
                        and not any(k in str(f).lower() for k in ("timestamp", "time", "date", "epoch", "_ms", "port", "id", "bytes", "count", "length", "size", "timeout"))
                    ][:12]

            fields = text_fields or ["*"]
        logger.debug("[%s] _build_opensearch_query: Building keyword search with fields: %s (first 5 shown)", 
                    SKILL_NAME, fields[:5] if len(fields) > 5 else fields)
        kw_should = []
        
        if matching_strategy == "phrase":
            # Exact phrase matching (no tokenization)
            # Use for structured fields like rule names
            for field in fields:
                for t in search_terms:
                    if t:
                        kw_should.append({"match_phrase": {field: str(t)}})
            logger.debug("[%s] _build_opensearch_query: Created %d phrase match clauses", SKILL_NAME, len(kw_should))
        
        elif matching_strategy == "term":
            # Exact term matching (case-insensitive, no analysis)
            # Use for keyword fields
            for field in fields:
                for t in search_terms:
                    if t:
                        kw_should.append({"term": {field: str(t).lower()}})
        
        else:  # "token" or default
            # Standard multi_match (tokenized text search)
            # Use for free-text fields
            kw_should = [{"multi_match": {"query": str(t), "fields": fields}} for t in search_terms if t]
        
        if kw_should:
            must_clauses.append({"bool": {"should": kw_should, "minimum_should_match": 1}})

    if not must_clauses:
        return {"query": {"match_none": {}}}

    time_filter = _build_time_filter(time_range)
    return {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": [time_filter],
            }
        }
    }


def _diagnose_query_failure(
    question: str,
    search_terms: list,
    field_mappings: dict,
    last_strategy: str,
    llm: Any,
) -> dict:
    """
    When initial query returns 0 results, ask LLM to reason about why.
    Multi-turn reflection to understand the real issue.
    """
    logger.info("[%s] _diagnose_query_failure: Analyzing why query with strategy '%s' returned 0 results",
                SKILL_NAME, last_strategy)
    
    # First pass: initial diagnosis
    diagnosis_prompt = f"""The user asked: "{question}"

We searched for: {search_terms}
Using strategy: {last_strategy}
Available fields: {list(field_mappings.keys())[:10]}...

We got 0 results. Why might that be? Consider:
1. Is the search term likely to exist in this database?
2. Is the strategy (phrase/token/term) wrong for this data type?
3. Are we searching in the right kind of fields?
4. Is the term too specific or too general?
5. Could this be a data structure issue (nested vs flat)?

Diagnose the most likely root cause."""

    try:
        diagnosis = llm.complete(diagnosis_prompt)
        logger.info("[%s] _diagnose_query_failure DIAGNOSIS:\n%s", SKILL_NAME, _log_excerpt(diagnosis, limit=1200))
        
        # Second pass: suggest recovery
        recovery_prompt = f"""Based on this diagnosis: {diagnosis}

What is ONE specific thing we should try next?
- Change matching strategy? (phrase→token or vice versa)
- Search in different fields?
- Relax field selection?
- Ask user for clarification?
- Accept that this data isn't in the database?

Return actionable next step with explanation."""
        
        recovery = llm.complete(recovery_prompt)
        logger.info("[%s] _diagnose_query_failure RECOVERY SUGGESTION:\n%s", SKILL_NAME, _log_excerpt(recovery, limit=1200))
        
        return {
            "diagnosis": diagnosis,
            "suggested_recovery": recovery,
            "should_try_recovery": "change" in recovery.lower() or "search" in recovery.lower(),
        }
    except Exception as exc:
        logger.warning("[%s] Diagnosis failed: %s", SKILL_NAME, exc)
        return {
            "diagnosis": f"Diagnosis failed: {exc}",
            "suggested_recovery": "fallback to token matching",
            "should_try_recovery": True,
        }


def _llm_validate_results_reflective(
    question: str,
    search_terms: list,
    requested_filters: dict,
    results: list,
    field_mappings: dict | None,
    previous_validation_failed: bool,
    llm: Any,
) -> dict:
    """
    Validate results with reflection. If validation fails, reason about why before giving up.
    """
    # First pass: quick validation
    validation = _llm_validate_results(question, search_terms, requested_filters, results, field_mappings, llm)
    
    if validation.get("is_valid"):
        logger.info("[%s] Results validated on first pass", SKILL_NAME)
        return {**validation, "reflection": "Valid on first pass", "iterations": 1}
    
    # Validation failed - reflect on why
    logger.warning("[%s] Results failed validation: %s. Reflecting...", SKILL_NAME, validation.get("issue"))
    
    reflection_prompt = f"""We got results, but validation failed.

Failure reason: {validation.get("issue")}

Looking at the results more carefully:
- Are they the wrong type of data? (e.g., got traffic logs when looking for alerts?)
- Wrong values but right structure? (e.g., alerts for different signatures?)
- Could this be a misunderstanding of the question?
- Is the validation criterion too strict?

Why did validation fail?"""
    
    try:
        reflection = llm.complete(reflection_prompt)
        logger.info("[%s] REFLECTION ON VALIDATION FAILURE:\n%s", SKILL_NAME, _log_excerpt(reflection, limit=1200))
        
        return {
            **validation,
            "reflection": reflection,
            "iterations": 2 if previous_validation_failed else 1,
            "should_retry": "strict" in reflection.lower() or "misunderstanding" in reflection.lower(),
        }
    except Exception as exc:
        logger.warning("[%s] Reflection failed: %s", SKILL_NAME, exc)
        return {**validation, "reflection": "Reflection attempt failed", "iterations": 1}


def _llm_validate_results(
    question: str,
    search_terms: list,
    requested_filters: dict,
    results: list,
    field_mappings: dict | None,
    llm: Any,
) -> dict:
    """
    Ask LLM to validate that returned results are actually relevant to the question.
    
    This catches cases like:
    - Searching for "ET EXPLOIT" but getting "ET INFO" or "ET POLICY"
    - Results that don't match the intent even if they match some criteria
    
    Returns dict with:
      - is_valid: bool (true if results match intent)
      - issue: str (description of what's wrong, if any)
      - suggestion: str (how to fix it)
    """
    if not results:
        return {"is_valid": False, "issue": "No results returned", "suggestion": "Try relaxed search or different terms"}
    
    structured_filters = requested_filters if isinstance(requested_filters, dict) else {}
    samples = _extract_validation_samples(results, field_mappings)
    
    sample_text = json.dumps(samples, indent=2, default=str)
    
    # DEBUG: Log what samples look like
    logger.debug("[%s] Validation samples extracted: %s", SKILL_NAME, sample_text[:300])
    
    # RED FLAG: If we're searching for alert signatures but samples don't have signatures, log it
    has_any_signature = any(s.get("signature") for s in samples)
    if "alert" in question.lower() or any("alert" in str(t).lower() for t in search_terms):
        if not has_any_signature:
            logger.warning("[%s] ALERT SEARCH BUT NO SIGNATURES IN SAMPLES: question=%s samples=%s",
                          SKILL_NAME, question, sample_text[:500])
    
    prompt = f"""Validate that these search results match the user's intent.

USER QUESTION: "{question}"

SEARCH TERMS: {', '.join(search_terms)}
REQUESTED FILTERS: {json.dumps(structured_filters, default=str)}

SAMPLED RESULTS (extracted key fields):
{sample_text}

VALIDATION TASK:
1. Do the results contain fields/values matching the search terms?
2. For signature/alert searches: Do results contain EXACT signatures? (e.g., if searching "ET POLICY", are there records with signature containing "ET POLICY"?)
3. For traffic searches: Do results contain the countries/IPs/ports/protocols/time window requested?
4. Are results relevant to the user's intent?

RETURN STRICT JSON:
{{
  "is_valid": true/false,
  "issue": "if not valid, describe the specific problem",
  "suggestion": "how to fix the query if needed",
  "confidence": 0.0-1.0
}}

CRITICAL CHECKS:
- If searching for "ET POLICY", results must have signatures containing "ET POLICY"
- If searching for "ET EXPLOIT", results must have "ET EXPLOIT", NOT "ET POLICY" or others
- If searching for a country, results must have that country in geoip data
- If a sample includes `country_field`, trust that as evidence that the record carries country metadata
- If a sample includes timestamps inside the requested time range, do not fail validation only because the prompt did not restate the time window verbatim
- Partial matches ARE acceptable for alert signatures (e.g., "ET POLICY Dropbox" contains "ET POLICY")
"""

    try:
        response = llm.complete(prompt)
        validation = None
        
        try:
            validation = json.loads(response)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                validation = json.loads(m.group())
        
        if not validation:
            return {"is_valid": True, "confidence": 0.5}  # Assume valid if we can't parse
        
        is_valid = bool(validation.get("is_valid", True))
        logger.info("[%s] Result validation: valid=%s, confidence=%.1f%%, issue='%s'",
                   SKILL_NAME, is_valid, float(validation.get("confidence", 0.5)) * 100,
                   str(validation.get("issue", ""))[:60])
        
        # Also log if this is an alert search that failed validation
        if not is_valid and ("alert" in question.lower() or any("ET" in str(t) for t in search_terms)):
            logger.warning("[%s] ALERT SEARCH VALIDATION FAILED: question=%s num_samples=%d first_sig=%s",
                          SKILL_NAME, question[:100], len(samples),
                          samples[0].get("signature", "NONE") if samples else "NO_SAMPLES")
        
        return {
            "is_valid": is_valid,
            "issue": str(validation.get("issue", "")),
            "suggestion": str(validation.get("suggestion", "")),
            "confidence": float(validation.get("confidence", 0.5)),
        }
    except Exception as exc:
        logger.warning("[%s] Result validation failed: %s", SKILL_NAME, exc)
        return {"is_valid": True, "confidence": 0.0}  # Assume valid on error


def _execute_explicit_query(context: dict, index: str) -> dict:
    """
    Execute an explicitly parameterized query (backward compatibility).
    Used when query_type, keywords, raw_query, etc. are passed directly.
    """
    from core.query_builder import (
        discover_field_mappings,
        build_keyword_query,
        build_structured_query,
        build_time_range_query,
    )
    
    db = context.get("db")
    llm = context.get("llm")
    parameters = context.get("parameters", {})
    
    query_type = parameters.get("query_type", "keyword_search")
    size = parameters.get("size", 200)
    field_mappings = discover_field_mappings(db, llm)
    
    logger.info(
        "[%s] Executing explicit %s query against index: %s",
        SKILL_NAME, query_type, index
    )
    
    try:
        query = None
        
        if query_type == "raw_query":
            query = parameters.get("raw_query")
            if not query:
                return {"status": "failed", "reason": "raw_query required"}
        
        elif query_type == "keyword_search":
            keywords = parameters.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            if not keywords:
                return {"status": "failed", "reason": "keywords required"}
            query, _ = build_keyword_query(keywords, field_mappings)
        
        elif query_type == "structured_search":
            ips = parameters.get("ips", [])
            if isinstance(ips, str):
                ips = [ips]
            domains = parameters.get("domains", [])
            if isinstance(domains, str):
                domains = [domains]
            ports = parameters.get("ports", [])
            if isinstance(ports, str):
                ports = [ports]
            time_range = parameters.get("time_range")
            query, _ = build_structured_query(ips, domains, ports, time_range, field_mappings)
        
        elif query_type == "time_range_search":
            time_range = parameters.get("time_range")
            if not time_range:
                return {"status": "failed", "reason": "time_range required"}
            query, _ = build_time_range_query(time_range, field_mappings)
        
        else:
            return {"status": "failed", "reason": f"Unknown query_type: {query_type}"}
        
        if not query:
            return {"status": "failed", "reason": "could not build query"}
        
        results = _execute_search_with_llm_repair(db, llm, index, query, size=size)
        return {
            "status": "ok",
            "results_count": len(results) if results else 0,
            "results": results[:10] if results else [],
        }
    
    except Exception as exc:
        logger.error("[%s] Explicit query failed: %s", SKILL_NAME, exc)
        return {"status": "error", "error": str(exc)}


def _plan_opensearch_query_with_llm_simplified(
    question: str,
    llm: Any,
) -> dict | None:
    """
    Attempt simplified LLM planning with a cleaner, minimal JSON prompt.
    Called as a retry when the main planning fails.
    
    Returns None if this also fails, allowing fallback to heuristic extraction.
    """
    import json
    
    prompt = f"""Analyze this question and return ONLY valid JSON, no other text:

Question: "{question}"

Return this exact JSON structure (all fields required):
{{
  "search_terms": ["term1", "term2"],
  "ports": [443, 8080],
  "search_type": "alert|traffic|ip|general",
  "matching_strategy": "phrase|token"
}}

Rules:
- search_terms: Extract key words from question (exclude: "what", "port", "associated", pronouns)
- ports: Extract numbers like "443", "8080" (must be 1-65535)
- search_type: "alert" if mentions signatures/rules, "traffic" if mentions connections, "ip" if IP addresses, else "general"
- matching_strategy: "phrase" for alerts, "token" for everything else

Output ONLY the JSON, nothing else."""

    try:
        response = llm.complete(prompt).strip()
        logger.debug("[%s] Simplified LLM response: %s", SKILL_NAME, response[:200])
        
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from response
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                return json.loads(m.group())
        
        return None
    except Exception as exc:
        logger.debug("[%s] Simplified LLM also failed: %s", SKILL_NAME, exc)
        return None


def _plan_opensearch_query_with_llm(
    question: str,
    conversation_history: list[dict],
    field_mappings: dict,
    llm: Any,
) -> dict:
    """
    Use LLM to plan OpenSearch query AND decide field-specific matching strategies.
    
    The LLM analyzes field characteristics and recommends:
    - Which fields to search in
    - What matching strategy to use (phrase vs token match) based on field content
    - Search terms and structured filters (countries, ports, protocols)
    
    This is fully data-agnostic - LLM makes all decisions based on field schema.
    """
    from pathlib import Path
    
    # Build conversation context
    conversation_summary = ""
    if conversation_history:
        relevant_msgs = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        conversation_parts = []
        for msg in relevant_msgs:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:200]
            conversation_parts.append(f"[{role}]: {content}")
        conversation_summary = "\n".join(conversation_parts)

    # ── BUILD DETAILED FIELD CONTEXT FOR LLM ──────────────────────────────────
    # Give LLM detailed field characteristics to analyze
    field_context = ""
    if field_mappings:
        field_info_parts = []
        
        # For each field category, include characteristics for LLM analysis
        field_categories = {
            "alert_fields": "ALERT/SIGNATURE FIELDS (for rule names, signatures)",
            "ip_fields": "IP ADDRESS FIELDS (for source/destination IPs)",
            "port_fields": "PORT FIELDS (for destination/source ports)",
            "country_fields": "COUNTRY FIELDS (for geoip country data)",
        }
        
        for field_type, description in field_categories.items():
            fields_list = field_mappings.get(field_type, [])
            if fields_list:
                sample_fields = fields_list[:3]
                field_info_parts.append(f"{description}: {', '.join(sample_fields)}")
        
        if field_info_parts:
            field_context = "\n\nAVAILABLE FIELDS:\n" + "\n".join(field_info_parts)

        planning_prompt = _load_planning_prompt()
        prompt = f"""{planning_prompt}

CONVERSATION CONTEXT:
{conversation_summary if conversation_summary else "(No prior context)"}{field_context}

USER QUESTION: "{question}"

Additional runtime requirements:
- Include search_type as one of: alert|traffic|domain|ip|general
- Include matching_strategy as one of: phrase|token|term|match
- Include field_analysis explaining which discovered field categories matter most
- Return STRICT JSON only
"""

    try:
        response = llm.complete(prompt)
        logger.debug("[%s] LLM Plan raw: %s", SKILL_NAME, _log_excerpt(response, limit=700))

        import json
        # Try direct parse; fallback to regex JSON extraction.
        plan = None
        try:
            plan = json.loads(response)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                plan = json.loads(m.group())

        if not plan:
            raise ValueError("No JSON in LLM response")

        # Ensure all required fields exist
        if not isinstance(plan.get("search_terms"), list):
            plan["search_terms"] = []
        if not isinstance(plan.get("countries"), list):
            plan["countries"] = []
        if not isinstance(plan.get("ports"), list):
            plan["ports"] = []
        if not isinstance(plan.get("protocols"), list):
            plan["protocols"] = []
        if not isinstance(plan.get("time_range"), str):
            plan["time_range"] = "now-90d"
        
        # Clean up matching_strategy: LLM might return "term|token" or multiple values
        # Extract the first valid strategy
        strategy = plan.get("matching_strategy", "token")
        if isinstance(strategy, str):
            # Extract first strategy if multiple are given (e.g., "term|token" -> "term")
            strategy = strategy.split("|")[0].split(",")[0].split(" ")[0].lower().strip()
        # Validate it's one of the supported strategies
        if strategy not in ("phrase", "token", "term"):
            strategy = "token"
        plan["matching_strategy"] = strategy

        logger.info(
            "[%s] LLM Plan: Type=%s | Strategy=%s | Terms=%s | Countries=%s | Reasoning=%s",
            SKILL_NAME, plan.get("search_type"), plan.get("matching_strategy"),
            plan.get("search_terms"), plan.get("countries"),
            plan.get("reasoning", "")[:60]
        )

        return _maybe_attach_country_aggregation_plan(question, plan)
    except Exception as exc:
        logger.warning("[%s] LLM planning failed: %s. Attempting simplified prompt...", SKILL_NAME, exc)
        
        # Try simplified LLM planning with minimal JSON prompt
        simplified_plan = _plan_opensearch_query_with_llm_simplified(question, llm)
        if simplified_plan:
            logger.info("[%s] Simplified LLM succeeded: %s", SKILL_NAME, 
                       simplified_plan.get("search_type", "unknown"))
            # Fill in any missing fields
            simplified_plan.setdefault("reasoning", "Simplified LLM planning")
            simplified_plan.setdefault("countries", [])
            simplified_plan.setdefault("protocols", [])
            simplified_plan.setdefault("time_range", "now-90d")
            simplified_plan.setdefault("field_analysis", "Using simplified LLM planning")
            return _maybe_attach_country_aggregation_plan(question, simplified_plan)
        
        # If even simplified LLM fails, fall back to heuristic extraction
        logger.warning("[%s] Simplified LLM also failed. Using fallback heuristic extraction.", SKILL_NAME)
        fallback_plan = _fallback_plan_from_question(question, None)
        logger.info("[%s] Fallback plan: %s", SKILL_NAME, fallback_plan.get("reasoning", "")[:100])
        return _maybe_attach_country_aggregation_plan(question, fallback_plan)
