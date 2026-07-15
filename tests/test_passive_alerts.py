from core.alert_store import AlertStore
from core.scheduler import AgentScheduler
from skills.network_defense_monitor import logic as network_defense


def test_alert_store_persists_manifest_contracted_findings(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    alert = store.emit(
        "passive_sensor",
        {"finding_count": 2, "findings": [{"entity": "host-a"}, {"entity": "host-b"}]},
        {"count_field": "finding_count", "findings_field": "findings", "default_severity": "high"},
    )

    assert alert["severity"] == "high"
    assert store.list()[0]["count"] == 2
    assert store.update_status(alert["id"], "investigating")["status"] == "investigating"


def test_alert_store_ignores_clean_passive_run(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    assert store.emit("sensor", {"finding_count": 0, "findings": []}, {}) is None
    assert store.list() == []


def test_alert_store_remembers_acknowledged_finding_and_suppresses_repeats(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    result = {"finding_count": 1, "findings": [{"entity": "host-a", "timestamp": "first"}]}
    contract = {"count_field": "finding_count", "findings_field": "findings", "default_severity": "high"}
    first = store.emit("passive_sensor", result, contract)
    store.update_status(first["id"], "resolved")

    repeated = {"finding_count": 1, "findings": [{"timestamp": "second", "entity": "host-a"}]}
    assert store.emit("passive_sensor", repeated, contract) is None
    alerts = store.list()
    assert len(alerts) == 1
    assert alerts[0]["status"] == "resolved"
    assert alerts[0]["suppressed_repeats"] == 1


def test_alert_store_emits_again_when_grounded_evidence_changes(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    contract = {"default_severity": "medium"}
    first = store.emit("sensor", {"finding_count": 1, "findings": [{"entity": "host-a"}]}, contract)
    store.update_status(first["id"], "resolved")

    second = store.emit("sensor", {"finding_count": 1, "findings": [{"entity": "host-b"}]}, contract)
    assert second is not None
    assert len(store.list()) == 2


def test_scheduler_reports_manual_and_scheduled_results():
    observed = []
    scheduler = AgentScheduler()
    scheduler.set_result_callback(lambda name, result: observed.append((name, result)))
    scheduler.register("hunt", lambda context: {"finding_count": 1}, 300)

    result = scheduler.dispatch("hunt")

    assert result == {"finding_count": 1}
    assert observed == [("hunt", result)]
    assert scheduler.job_status["hunt"]["last_result"] == result
    assert scheduler.job_status["hunt"]["last_run"]


def test_network_defense_detects_gateway_mac_change(monkeypatch, tmp_path):
    state = tmp_path / "network.json"
    state.write_text('{"bindings":{"192.0.2.1":"aa:bb:cc:dd:ee:01"}}', encoding="utf-8")
    monkeypatch.setattr(network_defense, "STATE_PATH", state)
    monkeypatch.setattr(network_defense, "collect_network_defense", lambda limit: {
        "status": "ok",
        "neighbors": [{"dst": "192.0.2.1", "lladdr": "aa:bb:cc:dd:ee:99"}],
        "gateways": [{"dst": "default", "gateway": "192.0.2.1"}],
        "routes": [], "interfaces": [], "errors": [],
    })

    result = network_defense.run({})

    assert result["finding_count"] == 1
    assert result["findings"][0]["type"] == "gateway_mac_changed"
    assert result["findings"][0]["severity"] == "high"
