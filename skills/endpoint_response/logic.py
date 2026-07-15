from __future__ import annotations

import ipaddress
import os
import shutil
import signal
import subprocess
import uuid
import re
from pathlib import Path

from core.action_authorization import consume_authorization, request_authorization
from core.endpoint_security import platform_name


def _arguments(action: str, parameters: dict) -> dict:
    if action == "terminate_process":
        pid = int(parameters.get("pid", 0))
        if pid <= 1 or pid == os.getpid():
            raise ValueError("A valid non-critical process ID is required")
        return {"pid": pid}
    if action == "block_ip":
        return {"ip": str(ipaddress.ip_address(str(parameters.get("ip", ""))))}
    if action == "delete_neighbor_entry":
        ip = str(ipaddress.ip_address(str(parameters.get("ip", ""))))
        interface = str(parameters.get("interface") or "").strip()
        if platform_name() != "windows" and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", interface):
            raise ValueError("A valid network interface is required on this platform")
        return {"ip": ip, "interface": interface}
    if action == "quarantine_file":
        path = Path(str(parameters.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise ValueError("An existing file path is required")
        return {"path": str(path)}
    raise ValueError("Unsupported endpoint response action")


def _execute(action: str, arguments: dict) -> dict:
    if action == "terminate_process":
        pid = arguments["pid"]
        if platform_name() == "windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/T"], check=True, capture_output=True, text=True)
        else:
            os.kill(pid, signal.SIGTERM)
        return {"status": "ok", "action": action, "pid": pid}

    if action == "block_ip":
        ip = arguments["ip"]
        rule_name = f"SecurityClaw-{ip.replace(':', '-') }"
        if platform_name() == "windows":
            subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "New-NetFirewallRule", "-DisplayName", rule_name,
                    "-Direction", "Inbound", "-Action", "Block", "-RemoteAddress", ip,
                ],
                check=True, capture_output=True, text=True,
            )
        elif shutil.which("ufw"):
            subprocess.run(["ufw", "deny", "from", ip], check=True, capture_output=True, text=True)
        else:
            raise RuntimeError("No supported firewall manager was found; install UFW or configure a platform adapter")
        return {"status": "ok", "action": action, "ip": ip, "rule_name": rule_name}

    if action == "delete_neighbor_entry":
        ip = arguments["ip"]
        interface = arguments["interface"]
        if platform_name() == "windows":
            script = f"Get-NetNeighbor -IPAddress '{ip}' -ErrorAction Stop | Remove-NetNeighbor -Confirm:$false"
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True, capture_output=True, text=True)
        else:
            subprocess.run(["ip", "neigh", "del", ip, "dev", interface], check=True, capture_output=True, text=True)
        return {"status": "ok", "action": action, "ip": ip, "interface": interface}

    source = Path(arguments["path"])
    quarantine_dir = Path("data") / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / f"{uuid.uuid4().hex}-{source.name}"
    shutil.move(str(source), destination)
    destination.chmod(0o600)
    return {
        "status": "ok", "action": action,
        "source": str(source), "quarantine_path": str(destination),
    }


def run(context: dict) -> dict:
    parameters = context.get("parameters") or {}
    action = str(parameters.get("action") or "").strip()
    arguments = _arguments(action, parameters)
    token = str(parameters.get("authorization_token") or "")
    operator_message = str(context.get("operator_message") or "")

    if not consume_authorization(token, action, arguments, operator_message):
        token = request_authorization(action, arguments)
        return {
            "status": "approval_required",
            "action": action,
            "arguments": arguments,
            "authorization_token": token,
            "expires_in_seconds": 600,
            "confirmation": f"AUTHORIZE {token}",
            "message": "Echo the confirmation exactly in a new operator message to execute this action.",
        }

    return _execute(action, arguments)
