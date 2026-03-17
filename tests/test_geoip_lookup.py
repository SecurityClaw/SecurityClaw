from __future__ import annotations

import io
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.chat_router.logic import format_response
from skills.geoip_lookup.logic import run


class _Cfg:
    def __init__(self, db_path: Path, license_key: str | None = "test-license", update_days: int = 7):
        self.values = {
            ("geoip", "db_path"): str(db_path),
            ("geoip", "edition_id"): "GeoLite2-City",
            ("geoip", "license_key"): license_key,
            ("geoip", "download_url"): "https://download.maxmind.com/app/geoip_download",
            ("geoip", "update_interval_days"): update_days,
            ("geoip", "timeout_seconds"): 10,
        }

    def get(self, section: str, key: str, default=None):
        return self.values.get((section, key), default)


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _Reader:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def city(self, ip: str):
        return self.response


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_archive(mmdb_bytes: bytes, name: str = "GeoLite2-City.mmdb") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = io.BytesIO(mmdb_bytes)
        info = tarfile.TarInfo(name=f"GeoLite2-City_20260306/{name}")
        info.size = len(mmdb_bytes)
        archive.addfile(info, payload)
    return buffer.getvalue()


def _fake_geo_response():
    subdivision = _Obj(name="California", iso_code="CA")
    return _Obj(
        continent=_Obj(name="North America"),
        country=_Obj(name="United States", iso_code="US"),
        registered_country=_Obj(name="United States"),
        city=_Obj(name="Mountain View"),
        postal=_Obj(code="94043"),
        location=_Obj(time_zone="America/Los_Angeles", latitude=37.386, longitude=-122.0838, accuracy_radius=20),
        subdivisions=_Obj(most_specific=subdivision),
    )


def test_first_run_downloads_database_and_returns_geo(monkeypatch, tmp_path):
    db_path = tmp_path / "GeoLite2-City.mmdb"
    cfg = _Cfg(db_path)
    archive = _make_archive(b"fake-mmdb")

    monkeypatch.setattr(
        "skills.geoip_lookup.logic.requests.get",
        lambda *args, **kwargs: _Response(archive),
    )
    monkeypatch.setattr(
        "skills.geoip_lookup.logic._open_reader",
        lambda path: _Reader(_fake_geo_response()),
    )

    result = run({"config": cfg, "parameters": {"ip": "8.8.8.8"}, "memory": None})

    assert result["status"] == "ok"
    assert result["action"] == "downloaded"
    assert db_path.exists()
    assert result["geo"]["country"] == "United States"
    assert result["geo"]["subdivision"] == "California"
    assert result["geo"]["city"] == "Mountain View"


def test_fresh_database_skips_download(monkeypatch, tmp_path):
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"existing-mmdb")
    cfg = _Cfg(db_path)

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("download should not be called for a fresh DB")

    monkeypatch.setattr("skills.geoip_lookup.logic.requests.get", _unexpected_download)
    monkeypatch.setattr(
        "skills.geoip_lookup.logic._open_reader",
        lambda path: _Reader(_fake_geo_response()),
    )

    result = run({"config": cfg, "parameters": {"question": "What country is 1.1.1.1 from?"}, "memory": None})

    assert result["status"] == "ok"
    assert result["action"] == "ready"
    assert result["ip"] == "1.1.1.1"


def test_weekly_run_updates_stale_database(monkeypatch, tmp_path):
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"old-mmdb")
    old = datetime.now(timezone.utc) - timedelta(days=10)
    ts = old.timestamp()
    db_path.chmod(0o644)
    import os
    os.utime(db_path, (ts, ts))

    archive = _make_archive(b"new-mmdb")
    monkeypatch.setattr(
        "skills.geoip_lookup.logic.requests.get",
        lambda *args, **kwargs: _Response(archive),
    )

    result = run({"config": _Cfg(db_path), "parameters": {}, "memory": None})

    assert result["status"] == "ok"
    assert result["action"] == "updated"
    assert db_path.read_bytes() == b"new-mmdb"


def test_missing_license_and_missing_database_returns_error(tmp_path):
    db_path = tmp_path / "GeoLite2-City.mmdb"

    result = run({"config": _Cfg(db_path, license_key=None), "parameters": {"ip": "8.8.8.8"}, "memory": None})

    assert result["status"] == "error"
    assert "MAXMIND_LICENSE_KEY" in result["error"]


def test_geoip_lookup_uses_previous_results_for_followup_country_question(monkeypatch, tmp_path):
    db_path = tmp_path / "GeoLite2-City.mmdb"
    db_path.write_bytes(b"existing-mmdb")
    cfg = _Cfg(db_path)

    monkeypatch.setattr(
        "skills.geoip_lookup.logic._open_reader",
        lambda path: _Reader(_fake_geo_response()),
    )

    result = run(
        {
            "config": cfg,
            "parameters": {"question": "what country are these ips from?"},
            "previous_results": {
                "opensearch_querier": {
                    "results": [
                        {"src_ip": "8.8.8.8", "dest_ip": "1.1.1.1"},
                        {"source.ip": "8.8.4.4", "destination.ip": "1.0.0.1"},
                    ]
                }
            },
            "memory": None,
        }
    )

    assert result["status"] == "ok"
    assert len(result.get("lookups", [])) == 4
    assert result["lookups"][0]["geo"]["country"] == "United States"


def test_format_response_renders_geoip_result_with_single_and_multiple_ips():
    """Core component test: format_response renders both single and multi-IP geoip results without LLM."""
    class _LLM:
        def chat(self, messages):
            raise AssertionError("LLM should not be used for geoip formatter")

    # Test single IP result
    single_result = format_response(
        "What country is 8.8.8.8 from?",
        {"skills": ["geoip_lookup"], "parameters": {}},
        {
            "geoip_lookup": {
                "status": "ok",
                "action": "ready",
                "ip": "8.8.8.8",
                "geo": {
                    "country": "United States",
                    "country_iso_code": "US",
                    "subdivision": "California",
                    "city": "Mountain View",
                    "timezone": "America/Los_Angeles",
                    "latitude": 37.386,
                    "longitude": -122.0838,
                },
            }
        },
        _LLM(),
    )

    assert "8.8.8.8" in single_result
    assert "United States" in single_result
    assert "California" in single_result

    # Test multi-IP result
    multi_result = format_response(
        "What country are these IPs from?",
        {"skills": ["geoip_lookup"], "parameters": {}},
        {
            "geoip_lookup": {
                "status": "ok",
                "action": "ready",
                "db_path": "/tmp/GeoLite2-City.mmdb",
                "lookups": [
                    {"status": "ok", "ip": "8.8.8.8", "geo": {"country": "United States", "subdivision": "California", "city": "Mountain View"}},
                    {"status": "not_found", "ip": "192.168.0.16"},
                ],
            }
        },
        _LLM(),
    )

    assert "8.8.8.8" in multi_result
    assert "United States" in multi_result
    assert "192.168.0.16: not found" in multi_result
