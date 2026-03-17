from __future__ import annotations

import ipaddress

from skills.threat_analyst.hooks import append_summary


def _is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def _format_threat_only_response(verdict_result: dict, user_question: str = "") -> str:
    """Format threat verdict results only."""
    if not verdict_result or verdict_result.get("status") != "ok":
        return "No threat intelligence verdict was produced."

    verdicts = verdict_result.get("verdicts") or []
    if not verdicts:
        return "No threat intelligence verdict was produced."

    primary = verdicts[0]
    verdict_label = str(primary.get("verdict", "UNKNOWN") or "UNKNOWN")
    confidence = int(primary.get("confidence", 0) or 0)
    from skills.threat_analyst.hooks import _shorten_naturally
    reasoning = _shorten_naturally(" ".join(str(primary.get("reasoning", "") or "").split()), 320)

    response = f"Threat verdict: {verdict_label} ({confidence}% confidence)."
    if reasoning:
        response += f" {reasoning}"

    return response


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    # If validation failed, don't format opensearch results
    if result.get("validation_failed"):
        # Fall back to threat intel if available
        threat_result = (skill_results or {}).get("threat_analyst") or {}
        if threat_result.get("status") == "ok" and threat_result.get("verdicts"):
            return _format_threat_only_response(threat_result, user_question)
        return "The query validation failed, so no reliable results are available."
    
    results = result.get("results") or []
    summary_results = result.get("summary_results") or results
    results_count = result.get("results_count", len(results))
    sampled_results_count = int(result.get("sampled_results_count", len(summary_results)) or 0)
    sample_strategy = result.get("sample_strategy") or "page"
    oldest_sample_count = int(result.get("oldest_sample_count", sampled_results_count) or 0)
    newest_sample_count = int(result.get("newest_sample_count", sampled_results_count) or 0)
    countries = result.get("countries", [])
    ports = result.get("ports", [])
    protocols = result.get("protocols", [])
    time_range = result.get("time_range_label") or result.get("time_range", "")
    search_terms = result.get("search_terms", [])
    directional_alternative = result.get("directional_alternative") or {}
    country_buckets = result.get("country_buckets") or []

    if result.get("aggregation_type") == "country_terms":
        excluded_countries = result.get("excluded_countries") or []
        if not country_buckets:
            exclusion_text = f" excluding {', '.join(excluded_countries)}" if excluded_countries else ""
            return f"No matching country aggregates were found in the {time_range} window{exclusion_text}."
        bucket_summary = ", ".join(
            f"{bucket.get('country')} ({int(bucket.get('count', 0) or 0)})"
            for bucket in country_buckets[:10]
            if bucket.get("country")
        )
        exclusion_text = f" excluding {', '.join(excluded_countries)}" if excluded_countries else ""
        base = (
            f"Observed traffic from {len(country_buckets)} country(s) in the {time_range} window{exclusion_text}: "
            f"{bucket_summary}."
        )
        return append_summary(base, (skill_results or {}).get("threat_analyst") or {})

    if result.get("aggregation_type") == "fingerprint_ports":
        observed_ports = result.get("observed_ports") or []
        remote_destination_ports = result.get("remote_destination_ports") or {}
        fingerprint_summary = str(result.get("fingerprint_summary") or "").strip()
        likely_role = str(result.get("fingerprint_likely_role") or "").strip()
        evidence = result.get("fingerprint_evidence") or []
        field_plan = result.get("fingerprint_field_plan") or {}
        total_hits = int(result.get("results_count", 0) or 0)
        target_ip = next((str(term) for term in (search_terms or []) if term), "the requested IP")

        if not observed_ports:
            return f"No observed port profile was found for {target_ip} in the {time_range} window."

        port_summary = ", ".join(str(port) for port in observed_ports[:12])
        base = f"Observed {len(observed_ports)} target-owned port(s) for {target_ip} across {total_hits} matching record(s) in the {time_range} window: {port_summary}."
        if fingerprint_summary:
            base += f" {fingerprint_summary}"
        elif likely_role:
            base += f" Likely role: {likely_role}."

        if evidence:
            base += f" Evidence: {'; '.join(str(item) for item in evidence[:3])}."

        if remote_destination_ports:
            remote_summary = ", ".join(str(port) for port in sorted(int(port) for port in remote_destination_ports.keys())[:12])
            base += f" Remote destination ports contacted by {target_ip} were also observed: {remote_summary}."

        selected_port_fields = field_plan.get("port_fields") or []
        if selected_port_fields:
            base += f" Aggregated via fields: {', '.join(str(field) for field in selected_port_fields)}."

        return append_summary(base, (skill_results or {}).get("threat_analyst") or {})

    if result.get("status") == "no_action":
        time_range = time_range or "requested time range"
        return (
            f"I couldn't produce a grounded OpenSearch query for that request, so I do not have log evidence to answer it for the {time_range} window. "
            "The previous step only identified schema/field information, not matching traffic records."
        )

    if not results:
        if directional_alternative:
            requested_direction = result.get("ip_direction") or "requested"
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
            filter_parts.append(f"country={'/'.join(countries)}")
        if ports:
            filter_parts.append(f"port={'/'.join(str(port) for port in ports)}")
        if protocols:
            filter_parts.append(f"protocol={'/'.join(protocols)}")
        filter_desc = ", ".join(filter_parts) or "the specified criteria"
        return f"No matching records found for {filter_desc} in the {time_range} window."

    question_lower = str(user_question or "").lower()
    is_alert_query = any(keyword in question_lower for keyword in ["alert", "signature", "et exploit", "et rule", "et drop", "et policy", "suricata", "snort", "rule"])
    if is_alert_query:
        alert_signatures: set[str] = set()
        alert_types: set[str] = set()
        alert_ips: set[str] = set()
        alert_countries: set[str] = set()
        alert_timestamps: list[str] = []
        for row in summary_results:
            signature = row.get("alert.signature") or row.get("signature") or row.get("alert", {}).get("signature")
            if signature:
                alert_signatures.add(str(signature))
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
                for country in (geo.get("country_name"), geo.get("country")):
                    if country:
                        alert_countries.add(str(country))
            for country in (
                row.get("geoip.country_name"),
                row.get("country_name"),
                row.get("source.geo.country_name"),
                row.get("destination.geo.country_name"),
            ):
                if country:
                    alert_countries.add(str(country))

        summary = f"Found {results_count} total alert record(s) matching {' / '.join(search_terms)} in the {time_range} window."
        if sampled_results_count and sampled_results_count < int(results_count or 0):
            if sample_strategy == "edge_windows":
                summary += f" Details below are sampled from up to {oldest_sample_count} earliest and {newest_sample_count} latest matching records."
            else:
                summary += f" Details below are sampled from {sampled_results_count} matching records."

        detail_parts = []
        if alert_signatures:
            detail_parts.append(f"Alert signatures: {', '.join(sorted(alert_signatures)[:5])}.")
        if alert_types:
            detail_parts.append(f"Alert categories: {', '.join(sorted(alert_types))}.")
        asks_for_alert_details = any(
            term in question_lower
            for term in ["what ip", "which ip", "their ip", "their ips", "source ip", "destination ip", "what countr", "which countr", "what countries", "what country", "where are they from", "when did", "when was", "timestamp", "what time", "alert happen"]
        )
        if asks_for_alert_details and alert_ips:
            detail_parts.append(f"IPs seen in matching alerts: {', '.join(sorted(alert_ips)[:12])}.")
        if asks_for_alert_details and alert_countries:
            detail_parts.append(f"Countries seen in matching alerts: {', '.join(sorted(alert_countries)[:12])}.")
        if asks_for_alert_details and alert_timestamps:
            timestamps = sorted(alert_timestamps)
            detail_parts.append(f"Earliest: {timestamps[0]}. Latest: {timestamps[-1]}.")
        base = summary + (" " + " ".join(detail_parts) if detail_parts else "")
        return append_summary(base, (skill_results or {}).get("threat_analyst") or {})

    ips: set[str] = set()
    source_ips: set[str] = set()
    timestamps: list[str] = []
    countries_seen: set[str] = set()
    for row in summary_results:
        ts = row.get("@timestamp") or row.get("timestamp")
        if ts:
            timestamps.append(str(ts))
        for value in (
            row.get("src_ip"),
            row.get("source_ip"),
            row.get("source.ip"),
            row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
        ):
            if value:
                ips.add(str(value))
                source_ips.add(str(value))
        for value in (
            row.get("dest_ip"),
            row.get("destination_ip"),
            row.get("destination.ip"),
            row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
        ):
            if value:
                ips.add(str(value))
        geo = row.get("geoip") or {}
        if isinstance(geo, dict) and geo.get("country_name"):
            countries_seen.add(str(geo.get("country_name")))
        for country in (
            row.get("geoip.country_name"),
            row.get("country_name"),
            row.get("source.geo.country_name"),
            row.get("destination.geo.country_name"),
        ):
            if country:
                countries_seen.add(str(country))

    filter_parts = []
    if countries:
        filter_parts.append("/".join(countries))
    if ports:
        filter_parts.append("port " + "/".join(str(port) for port in ports))
    if protocols:
        filter_parts.append("/".join(protocols))
    if search_terms and not filter_parts:
        shown_terms = "/".join(str(term) for term in search_terms[:3])
        if len(search_terms) > 3:
            shown_terms += "/..."
        filter_parts.append(shown_terms)
    filter_desc = ", ".join(filter_parts) or "the query criteria"

    summary = f"Found {results_count} total record(s) matching {filter_desc} in the {time_range} window."
    if sampled_results_count and sampled_results_count < int(results_count or 0):
        if sample_strategy == "edge_windows":
            summary += f" Details below are sampled from up to {oldest_sample_count} earliest and {newest_sample_count} latest matching records."
        else:
            summary += f" Details below are sampled from {sampled_results_count} matching records."

    extracted_ports: set[int] = set()
    # IMPORTANT: Extract ports from results ALWAYS, not just when ports filter was specified.
    # This handles "What ports are associated with..." follow-up questions where we search
    # with ports=[] (meaning "find all ports on these IPs") but need to display the actual ports found.
    for row in summary_results:
        for candidate in [
            row.get("destination.port"),
            row.get("destination", {}).get("port") if isinstance(row.get("destination"), dict) else None,
            row.get("destination_port"),
            row.get("dst_port"),
            row.get("dest_port"),
            row.get("dport"),
            row.get("port"),
        ]:
            if candidate is None:
                continue
            try:
                extracted_ports.add(int(candidate))
            except (TypeError, ValueError):
                continue

    detail_parts = []
    if countries_seen:
        detail_parts.append(f"Countries seen: {', '.join(sorted(countries_seen))}.")
    if ips:
        if countries:
            public_ips = {ip for ip in ips if not _is_private_ip(ip)}
            display_ips = sorted(public_ips)[:10] if public_ips else sorted(source_ips)[:10]
            if display_ips:
                detail_parts.append(f"Source IPs: {', '.join(display_ips)}.")
        elif extracted_ports and source_ips:
            detail_parts.append(f"Remote peers: {', '.join(sorted(source_ips)[:10])}.")
        else:
            detail_parts.append(f"Source/destination IPs: {', '.join(sorted(ips)[:10])}.")
    if timestamps:
        ts_sorted = sorted(timestamps)
        detail_parts.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")
    # Display extracted ports from results (all ports found, not filtered by search query)
    if extracted_ports:
        detail_parts.append(f"Destination port(s): {', '.join(str(port) for port in sorted(extracted_ports))}.")

    base = summary + (" " + " ".join(detail_parts) if detail_parts else "")
    return append_summary(base, (skill_results or {}).get("threat_analyst") or {})