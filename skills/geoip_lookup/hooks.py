from __future__ import annotations


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    action = result.get("action", "ready")
    db_path = result.get("db_path")
    warning = result.get("warning")

    lookups = result.get("lookups") or []
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

    if result.get("status") == "not_found":
        response = f"No MaxMind geolocation record was found for IP {result.get('ip', 'unknown')}."
        if db_path:
            response += f" Database: {db_path}."
        return response

    ip = result.get("ip")
    geo = result.get("geo") or {}
    if not ip:
        # This is just a maintenance check with no actual geoip data
        # Return None to let other formatters take precedence
        return None

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