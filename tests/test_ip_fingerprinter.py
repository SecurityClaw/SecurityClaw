from __future__ import annotations

from pathlib import Path

from core.chat_router.logic import _supervisor_evaluate_satisfaction, format_response
from skills.ip_fingerprinter.logic import run
from skills.ip_fingerprinter.port_registry import load_port_registry


class _Cfg:
    def __init__(self, root: Path):
        self.values = {
            ("port_registry", "cache_path"): str(root / "service-names-port-numbers.csv"),
            ("port_registry", "download_url"): "https://example.test/iana.csv",
            ("port_registry", "timeout_seconds"): 5,
            ("port_registry", "update_interval_days"): 30,
            ("db", "logs_index"): "securityclaw-logs",
            ("ip_fingerprinter", "lookback_hours"): 24,
            ("ip_fingerprinter", "query_size"): 500,
        }

    def get(self, section: str, key: str, default=None):
        return self.values.get((section, key), default)


class _Response:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _Registry:
    action = "ready"
    source = "test"
    cache_path = None
    warning = None

    def classify(self, port: int, protocol: str | None = None):
        service_map = {
            80: ("http", True, "system", "unlikely"),
            443: ("https", True, "system", "unlikely"),
            445: ("microsoft-ds", True, "system", "unlikely"),
            3389: ("ms-wbt-server", True, "user", "unlikely"),
            548: ("afp", True, "system", "unlikely"),
        }
        service_name, registered, range_class, ephemeral = service_map.get(
            port,
            (None, False, "dynamic" if port >= 49152 else "user", "high" if port >= 49152 else "possible" if port >= 32768 else "unlikely"),
        )
        return {
            "port": port,
            "protocol": protocol,
            "range_class": range_class,
            "registered": registered,
            "service_name": service_name,
            "description": service_name,
            "registration_source": "test",
            "status": "registered" if registered else "unregistered",
            "ephemeral_likelihood": ephemeral,
            "ephemeral_reason": ephemeral,
            "ephemeral_by_family": {},
        }


def test_port_registry_downloads_iana_csv_on_first_load(monkeypatch, tmp_path):
    csv_text = "Service Name,Port Number,Transport Protocol,Description\nhttps,443,tcp,HTTP over TLS\ndomain,53,udp,Domain Name System\n"
    cfg = _Cfg(tmp_path)

    monkeypatch.setattr(
        "skills.ip_fingerprinter.port_registry.requests.get",
        lambda *args, **kwargs: _Response(csv_text),
    )

    registry = load_port_registry(cfg, force_update=True)

    assert registry.action == "downloaded"
    assert Path(cfg.get("port_registry", "cache_path")).exists()
    https = registry.classify(443, "tcp")
    assert https["registered"] is True
    assert https["service_name"] == "https"
    dynamic = registry.classify(55000, "tcp")
    assert dynamic["range_class"] == "dynamic"
    assert dynamic["ephemeral_likelihood"] == "high"


def test_ip_fingerprinter_identifies_likely_server_and_windows(monkeypatch, tmp_path):
    """Test that the skill analyzes aggregated ports and infers Windows server role."""
    monkeypatch.setattr("skills.ip_fingerprinter.logic.load_port_registry", lambda *args, **kwargs: _Registry())

    # Pre-aggregated ports (simulating opensearch_querier output)
    # These ports were observed on 10.0.0.15 as destination (listening)
    aggregated_ports = {
        445: {"observations": 2, "protocols": ["TCP"], "peers": {"10.0.0.20", "10.0.0.22"}},
        3389: {"observations": 1, "protocols": ["TCP"], "peers": {"10.0.0.21"}},
    }

    result = run(
        {
            "config": _Cfg(tmp_path),
            "parameters": {"ip": "10.0.0.15", "aggregated_ports": aggregated_ports},
            "memory": None,
        }
    )

    assert result["status"] == "ok"
    assert result["likely_role"]["classification"] == "likely_server"
    assert set(result["port_summary"]["registered_ports"]) == {445, 3389}
    assert result["os_family_likelihoods"][0]["family"] == "Windows"



def test_ip_fingerprinter_identifies_inconclusive_with_no_destination_ports(monkeypatch, tmp_path):
    """When a host has no destination ports (not listening on anything), classification is inconclusive."""
    monkeypatch.setattr("skills.ip_fingerprinter.logic.load_port_registry", lambda *args, **kwargs: _Registry())

    # Empty aggregated ports - 10.0.0.40 was source IP only, not destination (server)
    aggregated_ports = {}

    result = run(
        {
            "config": _Cfg(tmp_path),
            "parameters": {"ip": "10.0.0.40", "aggregated_ports": aggregated_ports},
            "memory": None,
        }
    )

    # No destination ports for 10.0.0.40 means no listening services = inconclusive
    assert result["status"] == "no_data"
    assert result.get("reason") == "No listening ports found in aggregated data for this IP."


def test_ip_fingerprinter_uses_aggregated_ports_from_previous_opensearch_results(monkeypatch, tmp_path):
    monkeypatch.setattr("skills.ip_fingerprinter.logic.load_port_registry", lambda *args, **kwargs: _Registry())

    result = run(
        {
            "config": _Cfg(tmp_path),
            "parameters": {"ip": "192.168.0.16"},
            "previous_results": {
                "opensearch_querier": {
                    "status": "ok",
                    "aggregated_ports": {
                        137: {"observations": 12, "protocols": ["udp"], "is_known": True},
                        138: {"observations": 8, "protocols": ["udp"], "is_known": True},
                        1900: {"observations": 5, "protocols": ["udp"], "is_known": True},
                    },
                }
            },
            "memory": None,
        }
    )

    assert result["status"] == "ok"
    assert set(result["port_summary"]["listening_ports"]) == {137, 138, 1900}


def test_format_response_renders_ip_fingerprinter_results_without_llm():
    """Core component test: format_response renders ip_fingerprinter results with proper port prioritization."""
    class _LLM:
        def chat(self, messages):
            raise AssertionError("LLM should not be used for passive IP fingerprint formatting")

    # Test case 1: Standard Windows server fingerprint
    rendered = format_response(
        "fingerprint 10.0.0.15",
        {"skills": ["ip_fingerprinter"], "parameters": {}},
        {
            "ip_fingerprinter": {
                "status": "ok",
                "ip": "10.0.0.15",
                "ports": [
                    {"port": 445, "service_name": "microsoft-ds", "registered": True, "observations": 100},
                    {"port": 3389, "service_name": "ms-wbt-server", "registered": True, "observations": 100},
                ],
                "port_summary": {
                    "listening_ports": [445, 3389],
                    "registered_ports": [445, 3389],
                    "unregistered_ports": [],
                },
                "likely_role": {"classification": "likely_server", "confidence": 88},
                "os_family_likelihoods": [{"family": "Windows", "confidence": 81}],
            }
        },
        _LLM(),
    )

    assert "Passive fingerprint for 10.0.0.15" in rendered
    assert "likely_server" in rendered
    assert "445 (microsoft-ds)" in rendered and "3389 (ms-wbt-server)" in rendered
    assert "Windows" in rendered

    # Test case 2: Linux server with notable stable ports
    rendered_linux = format_response(
        "fingerprint 192.168.0.17",
        {"skills": ["ip_fingerprinter"], "parameters": {}},
        {
            "ip_fingerprinter": {
                "status": "ok",
                "ip": "192.168.0.17",
                "ports": [
                    {"port": 1562, "service_name": "pconnectmgr", "registered": True, "ephemeral_likelihood": "unlikely", "observations": 100},
                    {"port": 22, "service_name": "ssh", "registered": True, "ephemeral_likelihood": "unlikely", "observations": 400},
                    {"port": 9200, "service_name": "opensearch", "registered": True, "ephemeral_likelihood": "unlikely", "observations": 657},
                    {"port": 5601, "service_name": "esmagent", "registered": True, "ephemeral_likelihood": "unlikely", "observations": 500},
                ],
                "likely_role": {"classification": "likely_server", "confidence": 91},
                "os_family_likelihoods": [{"family": "Linux", "confidence": 72}],
            }
        },
        _LLM(),
    )

    # Should prioritize notable/stable ports in output
    assert "9200 (opensearch)" in rendered_linux
    assert "22 (ssh)" in rendered_linux


def test_format_response_prefers_ip_fingerprinter_no_data_for_fingerprint_question():
    class _LLM:
        def chat(self, messages):
            raise AssertionError("LLM should not be used for fingerprint formatting")

    rendered = format_response(
        "fingerprint 192.168.0.16",
        {"skills": ["fields_querier", "opensearch_querier", "ip_fingerprinter"], "parameters": {}},
        {
            "opensearch_querier": {
                "status": "ok",
                "results_count": 57772,
                "results": [{"destination.port": 137, "destination.ip": "192.168.0.16"}],
                "summary_results": [{"destination.port": 137, "destination.ip": "192.168.0.16"}],
                "observed_ports": [137],
            },
            "ip_fingerprinter": {
                "status": "no_data",
                "ip": "192.168.0.16",
                "reason": "No listening ports found in aggregated data for this IP.",
            },
        },
        _LLM(),
    )

    assert rendered == "No matching port observations were found for 192.168.0.16 in the available records."


def test_supervisor_evaluation_requires_fingerprint_result_when_only_raw_hits_exist():
    eval_result = _supervisor_evaluate_satisfaction(
        user_question="fingerprint 192.168.0.16",
        llm=None,
        instruction="test",
        conversation_history=[],
        skill_results={
            "opensearch_querier": {
                "status": "ok",
                "results_count": 57772,
                "results": [{"destination.ip": "192.168.0.16", "destination.port": 137}],
            },
            "ip_fingerprinter": {
                "status": "no_data",
                "ip": "192.168.0.16",
                "reason": "No listening ports found in aggregated data for this IP.",
            },
        },
        step=1,
        max_steps=4,
    )

    assert eval_result["satisfied"] is False
    assert "passive fingerprint analysis" in " ".join(eval_result["missing"]).lower()