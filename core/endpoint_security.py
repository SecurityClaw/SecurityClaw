"""Read-only endpoint telemetry collectors for Linux and Windows."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import re
import math
from pathlib import Path
from typing import Any

import requests

try:
    import pwd
except ImportError:  # Windows
    pwd = None


def _run(command: list[str], timeout: int = 15) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "command failed").strip())
    return completed.stdout


def platform_name() -> str:
    system = platform.system().lower()
    return "windows" if system == "windows" else "linux" if system == "linux" else system


def collect_inventory() -> dict[str, Any]:
    users: list[str] = []
    if platform_name() == "linux":
        users = [entry.pw_name for entry in pwd.getpwall() if entry.pw_uid >= 1000 or entry.pw_uid == 0] if pwd else []
    else:
        try:
            users = [line.strip() for line in _run(["whoami", "/user"]).splitlines() if line.strip()]
        except Exception:
            users = [os.environ.get("USERNAME", "unknown")]
    return {
        "status": "ok",
        "platform": platform_name(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "users": users,
        "python": platform.python_version(),
    }


def collect_processes(limit: int = 500) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if platform_name() == "linux":
        for proc_dir in sorted(Path("/proc").glob("[0-9]*"), key=lambda p: int(p.name))[:limit]:
            try:
                stat = (proc_dir / "stat").read_text(encoding="utf-8", errors="replace")
                close = stat.rfind(")")
                fields = stat[close + 2 :].split()
                try:
                    executable = os.readlink(proc_dir / "exe")
                except OSError:
                    executable = None
                try:
                    command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                except OSError:
                    command = ""
                records.append({
                    "pid": int(proc_dir.name),
                    "name": stat[stat.find("(") + 1 : close],
                    "ppid": int(fields[1]),
                    "executable": executable,
                    "command": command,
                    "uid": (proc_dir / "status").stat().st_uid,
                })
            except (OSError, ValueError, IndexError) as exc:
                errors.append(f"{proc_dir.name}: {exc}")
    else:
        output = _run(["tasklist", "/FO", "CSV", "/NH"])
        for row in list(csv.reader(io.StringIO(output)))[:limit]:
            if len(row) >= 2:
                records.append({"name": row[0], "pid": int(row[1]), "session": row[2] if len(row) > 2 else None})
    return {"status": "ok", "platform": platform_name(), "count": len(records), "processes": records, "errors": errors[:20]}


def collect_network_connections(limit: int = 500) -> dict[str, Any]:
    if platform_name() == "linux":
        output = _run(["ss", "-H", "-tunap"])
        connections = []
        for line in output.splitlines()[:limit]:
            parts = line.split()
            if len(parts) >= 6:
                connections.append({
                    "protocol": parts[0], "state": parts[1],
                    "local": parts[4], "remote": parts[5],
                    "process": " ".join(parts[6:]) if len(parts) > 6 else None,
                })
    else:
        script = (
            "Get-NetTCPConnection | Select-Object -First %d State,LocalAddress,LocalPort,"
            "RemoteAddress,RemotePort,OwningProcess | ConvertTo-Json -Compress" % limit
        )
        raw = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
        parsed = json.loads(raw or "[]")
        connections = parsed if isinstance(parsed, list) else [parsed]
    return {"status": "ok", "platform": platform_name(), "count": len(connections), "connections": connections}


def collect_network_defense(limit: int = 1000) -> dict[str, Any]:
    """Collect ARP/NDP neighbors, routes, interfaces, and default gateways."""
    neighbors: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    interfaces: list[dict[str, Any]] = []
    errors: list[str] = []
    if platform_name() == "linux":
        commands = {
            "neighbors": ["ip", "-j", "neigh", "show"],
            "routes": ["ip", "-j", "route", "show"],
            "interfaces": ["ip", "-j", "address", "show"],
        }
        for name, command in commands.items():
            try:
                value = json.loads(_run(command, timeout=30) or "[]")
                if name == "neighbors": neighbors = value[:limit]
                elif name == "routes": routes = value[:limit]
                else: interfaces = value[:limit]
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    else:
        scripts = {
            "neighbors": "Get-NetNeighbor | Select-Object -First %d IPAddress,LinkLayerAddress,State,InterfaceAlias,InterfaceIndex | ConvertTo-Json -Compress" % limit,
            "routes": "Get-NetRoute | Select-Object -First %d DestinationPrefix,NextHop,RouteMetric,InterfaceAlias,InterfaceIndex | ConvertTo-Json -Compress" % limit,
            "interfaces": "Get-NetAdapter | Select-Object -First %d Name,InterfaceDescription,Status,MacAddress,LinkSpeed,ifIndex | ConvertTo-Json -Compress" % limit,
        }
        for name, script in scripts.items():
            try:
                value = json.loads(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=45) or "[]")
                value = value if isinstance(value, list) else [value]
                if name == "neighbors": neighbors = value
                elif name == "routes": routes = value
                else: interfaces = value
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    gateways = []
    for route in routes:
        destination = route.get("dst") or route.get("DestinationPrefix")
        if destination in {"default", "0.0.0.0/0", "::/0"}:
            gateways.append(route)
    return {
        "status": "ok" if neighbors or routes or interfaces else "partial",
        "platform": platform_name(), "neighbors": neighbors, "neighbor_count": len(neighbors),
        "routes": routes, "route_count": len(routes), "interfaces": interfaces,
        "interface_count": len(interfaces), "gateways": gateways, "errors": errors,
    }


def collect_persistence(limit: int = 500) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if platform_name() == "linux":
        locations = [Path("/etc/cron.d"), Path("/etc/cron.daily"), Path("/etc/systemd/system")]
        for location in locations:
            if not location.exists():
                continue
            for item in sorted(location.iterdir())[:limit]:
                try:
                    entries.append({
                        "type": "startup_file", "path": str(item),
                        "target": os.readlink(item) if item.is_symlink() else None,
                        "modified": item.stat().st_mtime,
                    })
                except OSError:
                    continue
    else:
        output = _run(["schtasks", "/Query", "/FO", "CSV", "/V"])
        for row in list(csv.DictReader(io.StringIO(output)))[:limit]:
            entries.append({"type": "scheduled_task", **row})
    return {"status": "ok", "platform": platform_name(), "count": len(entries), "persistence": entries}


def collect_file_integrity(paths: list[str] | None = None, max_files: int = 1000) -> dict[str, Any]:
    if not paths:
        paths = [r"C:\Windows\System32\drivers\etc\hosts"] if platform_name() == "windows" else ["/etc/hosts", "/etc/passwd", "/etc/ssh/sshd_config"]
    files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if len(files) >= max_files:
                break
            try:
                if not candidate.is_file() or candidate.stat().st_size > 50 * 1024 * 1024:
                    continue
                digest = hashlib.sha256()
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                stat = candidate.stat()
                files.append({
                    "path": str(candidate), "sha256": digest.hexdigest(),
                    "size": stat.st_size, "modified": stat.st_mtime,
                })
            except OSError:
                continue
    return {"status": "ok", "platform": platform_name(), "count": len(files), "files": files, "requested_paths": paths}


def collect_software_inventory(limit: int = 2000) -> dict[str, Any]:
    """Collect versioned operating-system and Python packages without modifying the host."""
    packages: list[dict[str, Any]] = []
    errors: list[str] = []
    if platform_name() == "linux":
        try:
            output = _run(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"], timeout=30)
            for line in output.splitlines()[:limit]:
                fields = line.split("\t")
                if len(fields) >= 2:
                    packages.append({"name": fields[0], "version": fields[1], "architecture": fields[2] if len(fields) > 2 else None, "ecosystem": "Debian", "source": "dpkg"})
        except Exception as exc:
            errors.append(f"dpkg: {exc}")
            try:
                output = _run(["rpm", "-qa", "--qf", "%{NAME}\t%{VERSION}-%{RELEASE}\t%{ARCH}\n"], timeout=30)
                for line in output.splitlines()[:limit]:
                    fields = line.split("\t")
                    if len(fields) >= 2:
                        packages.append({"name": fields[0], "version": fields[1], "architecture": fields[2] if len(fields) > 2 else None, "ecosystem": "Rocky Linux", "source": "rpm"})
            except Exception as rpm_exc:
                errors.append(f"rpm: {rpm_exc}")
    else:
        script = (
            "$paths=@('HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',"
            "'HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*');"
            "Get-ItemProperty $paths -ErrorAction SilentlyContinue | Where-Object DisplayName | "
            "Select-Object -First %d DisplayName,DisplayVersion,Publisher,InstallLocation | ConvertTo-Json -Compress" % limit
        )
        try:
            parsed = json.loads(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=45) or "[]")
            for item in parsed if isinstance(parsed, list) else [parsed]:
                packages.append({"name": item.get("DisplayName"), "version": item.get("DisplayVersion"), "publisher": item.get("Publisher"), "location": item.get("InstallLocation"), "source": "registry"})
        except Exception as exc:
            errors.append(f"registry: {exc}")
    return {"status": "ok" if packages else "partial", "platform": platform_name(), "count": len(packages), "packages": packages, "errors": errors}


def collect_security_posture(limit: int = 500) -> dict[str, Any]:
    """Inspect services, firewall state and high-value defensive configuration."""
    services: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    errors: list[str] = []
    if platform_name() == "linux":
        commands = {
            "firewall": ["sh", "-c", "command -v ufw >/dev/null && ufw status || (command -v firewall-cmd >/dev/null && firewall-cmd --state) || echo unavailable"],
            "failed_units": ["systemctl", "--failed", "--no-legend", "--plain"],
            "security_updates": ["sh", "-c", "command -v apt-get >/dev/null && apt-get -s upgrade 2>/dev/null | grep -c '^Inst' || echo unknown"],
        }
        for name, command in commands.items():
            try:
                checks[name] = _run(command, timeout=30).strip()[:12000]
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        try:
            output = _run(["systemctl", "list-units", "--type=service", "--all", "--no-legend", "--plain"], timeout=30)
            for line in output.splitlines()[:limit]:
                fields = line.split(None, 4)
                if fields:
                    services.append({"name": fields[0], "load": fields[1] if len(fields) > 1 else None, "active": fields[2] if len(fields) > 2 else None, "sub": fields[3] if len(fields) > 3 else None, "description": fields[4] if len(fields) > 4 else None})
        except Exception as exc:
            errors.append(f"services: {exc}")
    else:
        scripts = {
            "firewall": "Get-NetFirewallProfile | Select Name,Enabled,DefaultInboundAction,DefaultOutboundAction | ConvertTo-Json -Compress",
            "defender": "Get-MpComputerStatus | Select AntivirusEnabled,RealTimeProtectionEnabled,BehaviorMonitorEnabled,AntivirusSignatureLastUpdated | ConvertTo-Json -Compress",
            "services": "Get-CimInstance Win32_Service | Select-Object -First %d Name,State,StartMode,PathName,DisplayName | ConvertTo-Json -Compress" % limit,
        }
        for name, script in scripts.items():
            try:
                value = json.loads(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=45) or "[]")
                if name == "services":
                    services = value if isinstance(value, list) else [value]
                else:
                    checks[name] = value
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    return {"status": "ok", "platform": platform_name(), "checks": checks, "service_count": len(services), "services": services, "errors": errors}


def scan_vulnerabilities(packages: list[dict[str, Any]], max_packages: int = 300, timeout: int = 60) -> dict[str, Any]:
    """Correlate versioned packages with OSV and return grounded advisory identifiers."""
    candidates = [p for p in packages if p.get("name") and p.get("version")][:max_packages]
    queries = []
    for package in candidates:
        query_package = {"name": str(package["name"])}
        if package.get("ecosystem"):
            query_package["ecosystem"] = str(package["ecosystem"])
        queries.append({"package": query_package, "version": str(package["version"])})
    if not queries:
        return {"status": "partial", "vulnerabilities": [], "count": 0, "coverage": {"queried": 0, "available": len(packages)}, "errors": ["No versioned packages were available for correlation."]}
    try:
        response = requests.post("https://api.osv.dev/v1/querybatch", json={"queries": queries}, timeout=timeout)
        response.raise_for_status()
        results = response.json().get("results") or []
    except Exception as exc:
        return {"status": "partial", "vulnerabilities": [], "count": 0, "coverage": {"queried": len(queries), "available": len(packages)}, "errors": [f"OSV correlation unavailable: {exc}"]}
    findings: list[dict[str, Any]] = []
    for package, result in zip(candidates, results):
        for vuln in result.get("vulns") or []:
            aliases = list(vuln.get("aliases") or [])
            cves = [alias for alias in aliases if re.fullmatch(r"CVE-\d{4}-\d+", alias, re.IGNORECASE)]
            severity = list(vuln.get("severity") or [])
            score, severity_label, severity_source = classify_vulnerability_severity(vuln)
            findings.append({
                "id": vuln.get("id"), "cves": cves, "aliases": aliases,
                "package": package.get("name"), "installed_version": package.get("version"),
                "summary": vuln.get("summary") or vuln.get("details"), "severity": severity,
                "severity_score": score, "severity_label": severity_label,
                "severity_source": severity_source,
                "modified": vuln.get("modified"), "references": list(vuln.get("references") or [])[:10],
                "source": "OSV",
            })
    return {"status": "ok", "vulnerabilities": findings, "count": len(findings), "coverage": {"queried": len(queries), "available": len(packages), "source": "OSV"}, "errors": []}


def _cvss_v3_score(vector: str) -> float | None:
    """Calculate a CVSS v3.0/v3.1 base score from an official vector."""
    if not str(vector).startswith(("CVSS:3.0/", "CVSS:3.1/")):
        return None
    metrics = {}
    for component in str(vector).split("/")[1:]:
        if ":" in component:
            key, value = component.split(":", 1)
            metrics[key] = value
    required = {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    if not required.issubset(metrics):
        return None
    weights = {
        "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
        "AC": {"L": 0.77, "H": 0.44}, "UI": {"N": 0.85, "R": 0.62},
        "C": {"N": 0.0, "L": 0.22, "H": 0.56}, "I": {"N": 0.0, "L": 0.22, "H": 0.56},
        "A": {"N": 0.0, "L": 0.22, "H": 0.56},
    }
    try:
        scope_changed = metrics["S"] == "C"
        pr_weights = {"N": 0.85, "L": 0.68 if scope_changed else 0.62, "H": 0.5 if scope_changed else 0.27}
        exploitability = 8.22 * weights["AV"][metrics["AV"]] * weights["AC"][metrics["AC"]] * pr_weights[metrics["PR"]] * weights["UI"][metrics["UI"]]
        iss = 1 - (1 - weights["C"][metrics["C"]]) * (1 - weights["I"][metrics["I"]]) * (1 - weights["A"][metrics["A"]])
        impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15) if scope_changed else 6.42 * iss
        if impact <= 0: return 0.0
        base = min(1.08 * (impact + exploitability), 10) if scope_changed else min(impact + exploitability, 10)
        return math.ceil((base - 1e-10) * 10) / 10
    except (KeyError, ValueError, TypeError):
        return None


def classify_vulnerability_severity(vulnerability: dict[str, Any]) -> tuple[float | None, str, str]:
    """Normalize published advisory severity without inventing missing scores."""
    candidates = []
    for entry in vulnerability.get("severity") or []:
        raw = str(entry.get("score") or "").strip()
        try: candidates.append((float(raw), "published_score"))
        except ValueError:
            calculated = _cvss_v3_score(raw)
            if calculated is not None: candidates.append((calculated, "cvss_v3_vector"))
    if candidates:
        score, source = max(candidates, key=lambda item: item[0])
        label = "critical" if score >= 9 else "high" if score >= 7 else "medium" if score >= 4 else "low" if score > 0 else "none"
        return score, label, source
    for container in (vulnerability.get("database_specific") or {}, vulnerability.get("ecosystem_specific") or {}):
        label = str(container.get("severity") or "").strip().lower()
        if label in {"critical", "high", "medium", "moderate", "low"}:
            return None, "medium" if label == "moderate" else label, "published_severity"
    return None, "unknown", "not_provided"
