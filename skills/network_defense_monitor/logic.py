"""Passive ARP/NDP and local network integrity monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.endpoint_security import collect_network_defense

STATE_PATH = Path("data/network_defense_state.json")


def _identity(neighbor: dict) -> tuple[str, str]:
    ip = str(neighbor.get("dst") or neighbor.get("IPAddress") or "").strip().lower()
    mac = str(neighbor.get("lladdr") or neighbor.get("LinkLayerAddress") or "").strip().lower().replace("-", ":")
    return ip, mac


def _load() -> dict:
    try: return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {}


def _save(value: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(STATE_PATH)


def run(context: dict) -> dict:
    telemetry = collect_network_defense(limit=2000)
    timestamp = datetime.now(timezone.utc).isoformat()
    current = {ip: mac for ip, mac in map(_identity, telemetry.get("neighbors") or []) if ip and mac and mac not in {"00:00:00:00:00:00", ""}}
    previous = _load()
    previous_bindings = previous.get("bindings") or {}
    gateway_ips = {
        str(route.get("gateway") or route.get("NextHop") or "").lower()
        for route in telemetry.get("gateways") or []
        if route.get("gateway") or route.get("NextHop")
    }
    findings = []
    for ip, mac in current.items():
        old_mac = previous_bindings.get(ip)
        if old_mac and old_mac != mac:
            findings.append({"type": "gateway_mac_changed" if ip in gateway_ips else "neighbor_mac_changed", "severity": "high" if ip in gateway_ips else "medium", "ip": ip, "previous_mac": old_mac, "current_mac": mac, "description": "The observed IP-to-MAC binding changed since the previous trusted observation."})
    ips_by_mac: dict[str, list[str]] = {}
    for ip, mac in current.items(): ips_by_mac.setdefault(mac, []).append(ip)
    for mac, ips in ips_by_mac.items():
        if len(ips) >= 5:
            findings.append({"type": "mac_address_concentration", "severity": "medium", "mac": mac, "ips": sorted(ips), "description": "One link-layer address currently represents multiple IP neighbors. This may be legitimate proxy ARP or a spoofing indicator and requires validation."})
    _save({"timestamp": timestamp, "bindings": current, "gateway_ips": sorted(gateway_ips)})
    return {
        **telemetry, "timestamp": timestamp, "findings": findings,
        "finding_count": len(findings), "baseline_initialized": not bool(previous_bindings),
        "analysis_note": "ARP/NDP changes are detection leads, not proof of spoofing. Validate DHCP, proxy ARP, failover, virtualization, and gateway changes before containment.",
    }
