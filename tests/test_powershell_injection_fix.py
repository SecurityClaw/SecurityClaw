"""Tests that verify the PowerShell injection fixes in endpoint_security and endpoint_response.

These tests validate:
1. _validate_limit rejects non-integer and non-positive values
2. PowerShell command strings use parameterized arguments, not string interpolation
3. The _execute function in endpoint_response uses param($ip) for neighbor deletion
"""

from __future__ import annotations

import subprocess

from skills.endpoint_telemetry import logic as endpoint_security
from skills.endpoint_response import logic as endpoint_response


# ── _validate_limit tests ──────────────────────────────────────────────────


def test_validate_limit_accepts_positive_integer():
    assert endpoint_security._validate_limit(10) == 10
    assert endpoint_security._validate_limit(1) == 1
    assert endpoint_security._validate_limit(5000) == 5000


def test_validate_limit_rejects_zero():
    try:
        endpoint_security._validate_limit(0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "positive integer" in str(exc)


def test_validate_limit_rejects_negative():
    try:
        endpoint_security._validate_limit(-1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "positive integer" in str(exc)


def test_validate_limit_rejects_non_int():
    try:
        endpoint_security._validate_limit("100")  # type: ignore[arg-type]
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "positive integer" in str(exc)

    try:
        endpoint_security._validate_limit(1.5)  # type: ignore[arg-type]
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "positive integer" in str(exc)

    try:
        endpoint_security._validate_limit(None)  # type: ignore[arg-type]
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "positive integer" in str(exc)


# ── PowerShell command structure tests ──────────────────────────────────────
# These verify the generated PowerShell scripts use safe_limit (validated int)
# and that the endpoint_response _execute uses parameterized PowerShell.


def test_network_connections_powershell_uses_safe_limit(monkeypatch):
    """The PowerShell script should contain the validated limit as a literal %d."""
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")
    captured_script = []

    def fake_run(command, timeout=15):
        captured_script.append(command[-1])
        return "[]"

    monkeypatch.setattr(endpoint_security, "_run", fake_run)

    endpoint_security.collect_network_connections(limit=42)
    assert len(captured_script) == 1
    script = captured_script[0]
    assert "Select-Object -First 42" in script, (
        f"Expected limit 42 in script, got: {script}"
    )


def test_network_defense_powershell_uses_safe_limit(monkeypatch):
    """All three network defense scripts should use the validated limit."""
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")
    captured_scripts = []

    def fake_run(command, timeout=15):
        captured_scripts.append(command[-1])
        return "[]"

    monkeypatch.setattr(endpoint_security, "_run", fake_run)

    endpoint_security.collect_network_defense(limit=99)
    assert len(captured_scripts) == 3
    for script in captured_scripts:
        assert "Select-Object -First 99" in script, (
            f"Expected limit 99 in script, got: {script}"
        )


def test_software_inventory_powershell_uses_safe_limit(monkeypatch):
    """The software inventory PowerShell script should use the validated limit."""
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")
    captured_script = []

    def fake_run(command, timeout=15):
        captured_script.append(command[-1])
        return "[]"

    monkeypatch.setattr(endpoint_security, "_run", fake_run)

    endpoint_security.collect_software_inventory(limit=77)
    assert len(captured_script) == 1
    script = captured_script[0]
    assert "Select-Object -First 77" in script, (
        f"Expected limit 77 in script, got: {script}"
    )


def test_security_posture_powershell_uses_safe_limit(monkeypatch):
    """The security posture services script should use the validated limit."""
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")
    captured_scripts = []

    def fake_run(command, timeout=15):
        captured_scripts.append(command[-1])
        return "[]"

    monkeypatch.setattr(endpoint_security, "_run", fake_run)

    endpoint_security.collect_security_posture(limit=33)
    # firewall and defender scripts don't use limit, only services does
    services_script = [s for s in captured_scripts if "Select-Object -First" in s]
    assert len(services_script) == 1
    assert "Select-Object -First 33" in services_script[0], (
        f"Expected limit 33 in services script, got: {services_script[0]}"
    )


# ── endpoint_response _execute PowerShell parameterization test ─────────────


def test_endpoint_response_neighbor_eviction_uses_parameterized_powershell(monkeypatch):
    """The _execute function for delete_neighbor_entry on Windows must use
    param($ip) parameter binding, not string interpolation of the IP."""
    monkeypatch.setattr(endpoint_response, "platform_name", lambda: "windows")
    captured_args = []

    def fake_subprocess_run(args, **kwargs):
        captured_args.append(args)
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(endpoint_response.subprocess, "run", fake_subprocess_run)

    # Call _execute directly with a crafted IP that would break string interpolation
    result = endpoint_response._execute("delete_neighbor_entry", {
        "ip": "192.0.2.1",
        "interface": "eth0",
    })

    assert result["status"] == "ok"
    assert len(captured_args) == 1
    cmd = captured_args[0]

    # The command must use param($ip) parameter binding, not f-string interpolation
    assert "param($ip)" in cmd[4], (
        f"Expected param($ip) in PowerShell command, got: {cmd[4]}"
    )
    # The IP must be passed as a separate argument, not embedded in the script
    assert "-ip" in cmd, f"Expected -ip argument in command, got: {cmd}"
    ip_index = cmd.index("-ip")
    assert cmd[ip_index + 1] == "192.0.2.1", (
        f"Expected IP as separate argument, got: {cmd}"
    )


def test_endpoint_response_neighbor_eviction_rejects_crafted_ip_via_arguments(monkeypatch):
    """The _arguments function validates IP via ipaddress.ip_address(),
    which rejects anything that isn't a valid IP address."""
    monkeypatch.setattr(endpoint_response, "platform_name", lambda: "linux")

    # Valid IP should pass
    args = endpoint_response._arguments("delete_neighbor_entry", {
        "ip": "192.0.2.1",
        "interface": "eth0",
    })
    assert args["ip"] == "192.0.2.1"

    # Crafted injection string should be rejected by ipaddress.ip_address()
    try:
        endpoint_response._arguments("delete_neighbor_entry", {
            "ip": "'; Remove-Item C:\\ -Recurse -Force; '",
            "interface": "eth0",
        })
        assert False, "expected ValueError for crafted IP"
    except ValueError:
        pass
