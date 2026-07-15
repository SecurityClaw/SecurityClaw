from pathlib import Path

from core import endpoint_security
from core.chat_router.logic import _build_skill_catalog
from core.skill_loader import SkillLoader
import core.skill_loader as skill_loader_module
from core import skill_manifest
from skills.endpoint_response import logic as endpoint_response
from web.api.server import ChatStreamParser
from skills.forensic_examiner import logic as canonical_forensic_logic
import sys


def test_endpoint_skills_are_discovered_with_cross_platform_contracts():
    discovered = SkillLoader().discover()
    expected = {
        "host_inventory", "process_monitor", "network_monitor",
        "persistence_scanner", "file_integrity_monitor",
        "software_inventory", "security_posture", "vulnerability_scanner",
        "endpoint_threat_hunter",
        "network_defense_monitor",
    }
    assert expected.issubset(discovered)

    catalog = _build_skill_catalog(
        [{"name": name, "description": ""} for name in expected],
        {
            name: {
                "supported_platforms": ["linux", "windows"],
                "risk_level": "read_only",
            }
            for name in expected
        },
    )
    assert all(item["supported_platforms"] == ["linux", "windows"] for item in catalog)
    assert all(item["risk_level"] == "read_only" for item in catalog)


def test_platform_contract_rejects_incompatible_skill(monkeypatch):
    monkeypatch.setattr(skill_manifest, "current_platform_name", lambda: "linux")
    assert skill_manifest.manifest_supports_current_platform({"supported_platforms": ["linux"]})
    assert not skill_manifest.manifest_supports_current_platform({"supported_platforms": ["windows"]})
    assert skill_manifest.manifest_supports_current_platform({})


def test_loader_does_not_activate_unsupported_skill(monkeypatch, tmp_path: Path):
    skill_dir = tmp_path / "windows_only"
    skill_dir.mkdir()
    (skill_dir / "logic.py").write_text("def run(context):\n    return {'status': 'ok'}\n", encoding="utf-8")
    (skill_dir / "instruction.md").write_text("# Windows only\n", encoding="utf-8")
    (skill_dir / "manifest.yaml").write_text(
        "name: windows_only\nsupported_platforms: [windows]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        skill_loader_module,
        "manifest_supports_current_platform",
        lambda manifest: "linux" in (manifest.get("supported_platforms") or []),
    )

    assert "windows_only" not in SkillLoader(skills_dir=tmp_path).discover()


def test_skill_discovery_preserves_canonical_module_identity():
    before = sys.modules["skills.forensic_examiner.logic"]

    discovered = SkillLoader().discover()

    assert before is canonical_forensic_logic
    assert sys.modules["skills.forensic_examiner.logic"] is before
    assert discovered["forensic_examiner"].run is before.run


def test_file_integrity_hashes_requested_file(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("trusted", encoding="utf-8")

    result = endpoint_security.collect_file_integrity([str(target)])

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["files"][0]["path"] == str(target)
    assert len(result["files"][0]["sha256"]) == 64


def test_windows_network_collection_uses_read_only_powershell(monkeypatch):
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")
    commands = []

    def fake_run(command, timeout=15):
        commands.append(command)
        return '[{"State":"Established","LocalAddress":"127.0.0.1"}]'

    monkeypatch.setattr(endpoint_security, "_run", fake_run)
    result = endpoint_security.collect_network_connections(limit=10)

    assert result["count"] == 1
    assert commands[0][:4] == ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
    assert "Get-NetTCPConnection" in commands[0][-1]


def test_windows_network_defense_collects_neighbors_routes_and_interfaces(monkeypatch):
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "windows")

    def fake_run(command, timeout=15):
        script = command[-1]
        if "Get-NetNeighbor" in script:
            return '[{"IPAddress":"192.0.2.1","LinkLayerAddress":"AA-BB-CC-DD-EE-FF"}]'
        if "Get-NetRoute" in script:
            return '[{"DestinationPrefix":"0.0.0.0/0","NextHop":"192.0.2.1"}]'
        return '[{"Name":"Ethernet","Status":"Up"}]'

    monkeypatch.setattr(endpoint_security, "_run", fake_run)
    result = endpoint_security.collect_network_defense(limit=10)

    assert result["neighbor_count"] == 1
    assert result["route_count"] == 1
    assert result["interface_count"] == 1
    assert result["gateways"][0]["NextHop"] == "192.0.2.1"


def test_linux_process_collection_tolerates_unreadable_executable(monkeypatch, tmp_path: Path):
    proc = tmp_path / "42"
    proc.mkdir()
    (proc / "stat").write_text("42 (worker) S 1 0 0", encoding="utf-8")
    (proc / "status").write_text("Name:\tworker", encoding="utf-8")
    (proc / "cmdline").write_bytes(b"worker\0--safe")
    monkeypatch.setattr(endpoint_security, "platform_name", lambda: "linux")
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [proc])
    monkeypatch.setattr(endpoint_security.os, "readlink", lambda path: (_ for _ in ()).throw(PermissionError()))

    result = endpoint_security.collect_processes(limit=5)

    assert result["count"] == 1
    assert result["processes"][0]["executable"] is None
    assert result["processes"][0]["command"] == "worker --safe"


def test_endpoint_action_requires_exact_operator_authorization(monkeypatch):
    executed = []
    monkeypatch.setattr(
        endpoint_response,
        "_execute",
        lambda action, arguments: executed.append((action, arguments)) or {"status": "ok"},
    )
    request = endpoint_response.run({
        "parameters": {"action": "terminate_process", "pid": 4242},
        "operator_message": "Terminate process 4242",
    })
    assert request["status"] == "approval_required"
    assert executed == []

    result = endpoint_response.run({
        "parameters": {
            "action": "terminate_process",
            "pid": 4242,
            "authorization_token": request["authorization_token"],
        },
        "operator_message": request["confirmation"],
    })
    assert result["status"] == "ok"
    assert executed == [("terminate_process", {"pid": 4242})]


def test_endpoint_action_rejects_token_not_echoed_by_current_operator(monkeypatch):
    monkeypatch.setattr(
        endpoint_response,
        "_execute",
        lambda *_: (_ for _ in ()).throw(AssertionError("action must not execute")),
    )
    request = endpoint_response.run({
        "parameters": {"action": "terminate_process", "pid": 4343},
        "operator_message": "Prepare termination",
    })
    retry = endpoint_response.run({
        "parameters": {
            "action": "terminate_process",
            "pid": 4343,
            "authorization_token": request["authorization_token"],
        },
        "operator_message": "Continue without quoting the confirmation",
    })
    assert retry["status"] == "approval_required"


def test_neighbor_eviction_requires_operator_authorization(monkeypatch):
    monkeypatch.setattr(endpoint_response, "platform_name", lambda: "linux")
    executed = []
    monkeypatch.setattr(endpoint_response, "_execute", lambda action, arguments: executed.append((action, arguments)) or {"status": "ok"})
    request = endpoint_response.run({
        "parameters": {"action": "delete_neighbor_entry", "ip": "192.0.2.1", "interface": "eth0"},
        "operator_message": "Clear the suspicious neighbor entry",
    })
    assert request["status"] == "approval_required"
    assert executed == []


def test_observed_stream_step_keeps_tool_debug_separate():
    payload = ChatStreamParser.to_step_payload(
        "observed",
        {"skills": ["host_inventory"], "results": {"host_inventory": {"status": "ok"}}},
        1,
        4,
    )
    assert payload["kind"] == "tool"
    assert payload["skills"] == ["host_inventory"]
    assert payload["debug"]["host_inventory"]["status"] == "ok"


def test_vulnerability_scan_returns_only_grounded_osv_advisories(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"vulns": [{
                "id": "OSV-TEST-1",
                "aliases": ["CVE-2026-12345"],
                "summary": "Grounded test advisory",
                "severity": [{"type": "CVSS_V3", "score": "9.1"}],
                "references": [{"type": "ADVISORY", "url": "https://example.invalid/advisory"}],
            }]}]}

    calls = []
    monkeypatch.setattr(endpoint_security.requests, "post", lambda url, json, timeout: calls.append((url, json)) or Response())
    result = endpoint_security.scan_vulnerabilities([{"name": "sample", "version": "1.0", "ecosystem": "Debian"}])

    assert result["status"] == "ok"
    assert result["vulnerabilities"][0]["cves"] == ["CVE-2026-12345"]
    assert result["vulnerabilities"][0]["package"] == "sample"
    assert result["coverage"]["queried"] == 1
    assert calls[0][0] == "https://api.osv.dev/v1/querybatch"


def test_vulnerability_scan_reports_external_correlation_failure(monkeypatch):
    monkeypatch.setattr(endpoint_security.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("offline")))
    result = endpoint_security.scan_vulnerabilities([{"name": "sample", "version": "1.0"}])

    assert result["status"] == "partial"
    assert result["count"] == 0
    assert "unavailable" in result["errors"][0]


def test_vulnerability_severity_uses_published_numeric_score():
    score, label, source = endpoint_security.classify_vulnerability_severity({
        "severity": [{"type": "CVSS_V3", "score": "9.1"}],
    })
    assert score == 9.1
    assert label == "critical"
    assert source == "published_score"


def test_vulnerability_severity_calculates_cvss_vector_without_guessing():
    score, label, source = endpoint_security.classify_vulnerability_severity({
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
    })
    assert score == 9.8
    assert label == "critical"
    assert source == "cvss_v3_vector"


def test_vulnerability_severity_is_unknown_when_advisory_has_no_rating():
    assert endpoint_security.classify_vulnerability_severity({}) == (None, "unknown", "not_provided")
