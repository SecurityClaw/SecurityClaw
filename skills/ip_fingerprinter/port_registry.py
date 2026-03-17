from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[2]
DEFAULT_CACHE_DIR = ROOT_DIR / "data" / "port_registry"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "service-names-port-numbers.csv"
DEFAULT_REGISTRY_URL = (
    "https://www.iana.org/assignments/service-names-port-numbers/"
    "service-names-port-numbers.csv"
)

IANA_DYNAMIC_START = 49152
IANA_DYNAMIC_END = 65535
LINUX_EPHEMERAL_START = 32768
LINUX_EPHEMERAL_END = 60999

COMMON_SERVICE_OVERRIDES = [
    (20, 20, "tcp", "ftp-data", "File Transfer Protocol data channel"),
    (21, 21, "tcp", "ftp", "File Transfer Protocol control channel"),
    (22, 22, "tcp", "ssh", "Secure Shell"),
    (25, 25, "tcp", "smtp", "Simple Mail Transfer Protocol"),
    (53, 53, "udp", "domain", "Domain Name System"),
    (53, 53, "tcp", "domain", "Domain Name System"),
    (67, 67, "udp", "dhcp-server", "Dynamic Host Configuration Protocol server"),
    (68, 68, "udp", "dhcp-client", "Dynamic Host Configuration Protocol client"),
    (80, 80, "tcp", "http", "Hypertext Transfer Protocol"),
    (88, 88, "tcp", "kerberos", "Kerberos authentication"),
    (110, 110, "tcp", "pop3", "Post Office Protocol v3"),
    (111, 111, "tcp", "rpcbind", "RPC portmapper"),
    (123, 123, "udp", "ntp", "Network Time Protocol"),
    (135, 135, "tcp", "msrpc", "Microsoft RPC endpoint mapper"),
    (137, 139, "udp", "netbios", "NetBIOS service ports"),
    (137, 139, "tcp", "netbios", "NetBIOS service ports"),
    (143, 143, "tcp", "imap", "Internet Message Access Protocol"),
    (389, 389, "tcp", "ldap", "Lightweight Directory Access Protocol"),
    (443, 443, "tcp", "https", "HTTP over TLS"),
    (445, 445, "tcp", "microsoft-ds", "SMB over TCP"),
    (465, 465, "tcp", "submissions", "SMTP over implicit TLS"),
    (514, 514, "udp", "syslog", "Syslog"),
    (548, 548, "tcp", "afp", "Apple Filing Protocol"),
    (587, 587, "tcp", "submission", "Mail submission"),
    (636, 636, "tcp", "ldaps", "LDAP over TLS"),
    (993, 993, "tcp", "imaps", "IMAP over TLS"),
    (995, 995, "tcp", "pop3s", "POP3 over TLS"),
    (1433, 1433, "tcp", "ms-sql-s", "Microsoft SQL Server"),
    (1521, 1521, "tcp", "oracle", "Oracle database listener"),
    (2049, 2049, "tcp", "nfs", "Network File System"),
    (3306, 3306, "tcp", "mysql", "MySQL database"),
    (3389, 3389, "tcp", "ms-wbt-server", "Remote Desktop Protocol"),
    (5432, 5432, "tcp", "postgresql", "PostgreSQL database"),
    (5900, 5900, "tcp", "vnc", "Virtual Network Computing"),
    (5985, 5985, "tcp", "wsman", "Windows Remote Management HTTP"),
    (5986, 5986, "tcp", "wsmans", "Windows Remote Management HTTPS"),
    (62078, 62078, "tcp", "iphone-sync", "Apple lockdown / iPhone sync"),
    (6379, 6379, "tcp", "redis", "Redis"),
    (6443, 6443, "tcp", "kubernetes-api", "Kubernetes API server"),
    (8080, 8080, "tcp", "http-alt", "Alternative HTTP"),
    (8443, 8443, "tcp", "https-alt", "Alternative HTTPS"),
    (9200, 9200, "tcp", "opensearch", "OpenSearch / Elasticsearch API"),
    (9300, 9300, "tcp", "opensearch-transport", "OpenSearch / Elasticsearch transport"),
]


@dataclass(frozen=True)
class RegistryRecord:
    start_port: int
    end_port: int
    protocol: str
    service_name: str
    description: str
    source: str


@dataclass
class PortRegistry:
    records: list[RegistryRecord]
    source: str
    action: str
    cache_path: str | None = None
    warning: str | None = None

    def lookup(self, port: int, protocol: str | None = None) -> RegistryRecord | None:
        normalized = str(protocol or "").strip().lower()
        exact: list[RegistryRecord] = []
        generic: list[RegistryRecord] = []
        for record in self.records:
            if record.start_port <= port <= record.end_port:
                if normalized and record.protocol == normalized:
                    exact.append(record)
                elif not normalized or not record.protocol:
                    generic.append(record)
        if exact:
            return sorted(exact, key=lambda entry: (entry.end_port - entry.start_port, entry.source != "override"))[0]
        if generic:
            return sorted(generic, key=lambda entry: (entry.end_port - entry.start_port, entry.source != "override"))[0]
        return None

    def classify(self, port: int, protocol: str | None = None) -> dict[str, Any]:
        record = self.lookup(port, protocol)
        range_class = classify_port_range(port)
        ephemeral = classify_ephemeral_likelihood(port)
        return {
            "port": port,
            "protocol": str(protocol or "").strip().lower() or None,
            "range_class": range_class,
            "registered": record is not None,
            "service_name": record.service_name if record else None,
            "description": record.description if record else None,
            "registration_source": record.source if record else None,
            "status": "registered" if record else "unregistered",
            "ephemeral_likelihood": ephemeral["likelihood"],
            "ephemeral_reason": ephemeral["reason"],
            "ephemeral_by_family": ephemeral["by_family"],
        }


def _cfg_get(cfg: Any, section: str, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(section, key, default=default)
        except TypeError:
            return getter(section, key, default)
    return default


def _settings_from_config(cfg: Any) -> dict[str, Any]:
    cache_path = Path(_cfg_get(cfg, "port_registry", "cache_path", default=str(DEFAULT_CACHE_PATH)))
    if not cache_path.is_absolute():
        cache_path = ROOT_DIR / cache_path
    return {
        "cache_path": cache_path,
        "download_url": _cfg_get(cfg, "port_registry", "download_url", default=DEFAULT_REGISTRY_URL),
        "timeout_seconds": int(_cfg_get(cfg, "port_registry", "timeout_seconds", default=30) or 30),
        "update_interval_days": int(_cfg_get(cfg, "port_registry", "update_interval_days", default=30) or 30),
    }


def classify_port_range(port: int) -> str:
    if 0 <= port <= 1023:
        return "system"
    if 1024 <= port <= 49151:
        return "user"
    return "dynamic"


def classify_ephemeral_likelihood(port: int) -> dict[str, Any]:
    if IANA_DYNAMIC_START <= port <= IANA_DYNAMIC_END:
        likelihood = "high"
        reason = "Falls in the IANA dynamic/private range and common modern client ephemeral range."
    elif LINUX_EPHEMERAL_START <= port < IANA_DYNAMIC_START:
        likelihood = "possible"
        reason = "Falls below the IANA dynamic range but inside the common Linux default ephemeral range."
    else:
        likelihood = "unlikely"
        reason = "Outside the common default ephemeral ranges and more consistent with a stable service port."
    return {
        "likelihood": likelihood,
        "reason": reason,
        "by_family": {
            "linux_default": LINUX_EPHEMERAL_START <= port <= LINUX_EPHEMERAL_END,
            "windows_modern": IANA_DYNAMIC_START <= port <= IANA_DYNAMIC_END,
            "macos_modern": IANA_DYNAMIC_START <= port <= IANA_DYNAMIC_END,
            "iana_dynamic_private": IANA_DYNAMIC_START <= port <= IANA_DYNAMIC_END,
        },
    }


def _builtin_records() -> list[RegistryRecord]:
    return [
        RegistryRecord(
            start_port=start,
            end_port=end,
            protocol=protocol,
            service_name=service_name,
            description=description,
            source="override",
        )
        for start, end, protocol, service_name, description in COMMON_SERVICE_OVERRIDES
    ]


def _parse_port_range(value: str) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        if start_text.strip().isdigit() and end_text.strip().isdigit():
            return int(start_text.strip()), int(end_text.strip())
        return None
    if text.isdigit():
        port = int(text)
        return port, port
    return None


def _parse_registry_csv(csv_text: str) -> list[RegistryRecord]:
    records = _builtin_records()
    reader = csv.DictReader(csv_text.splitlines())
    for row in reader:
        port_range = _parse_port_range(row.get("Port Number", ""))
        protocol = str(row.get("Transport Protocol", "") or "").strip().lower()
        service_name = str(row.get("Service Name", "") or "").strip().lower()
        description = str(row.get("Description", "") or "").strip()
        if port_range is None or not protocol:
            continue
        if service_name in {"", "unknown", "unassigned"}:
            service_name = ""
        records.append(
            RegistryRecord(
                start_port=port_range[0],
                end_port=port_range[1],
                protocol=protocol,
                service_name=service_name,
                description=description,
                source="iana",
            )
        )
    return records


def _download_registry(settings: dict[str, Any]) -> str:
    response = requests.get(
        settings["download_url"],
        timeout=settings["timeout_seconds"],
    )
    response.raise_for_status()
    text = response.text
    if "Service Name" not in text or "Port Number" not in text:
        raise RuntimeError("Downloaded port registry did not look like the IANA CSV format")
    cache_path: Path = settings["cache_path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def _is_stale(path: Path, update_interval_days: int) -> bool:
    if not path.exists():
        return True
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - modified >= timedelta(days=update_interval_days)


def load_port_registry(cfg: Any = None, *, force_update: bool = False) -> PortRegistry:
    settings = _settings_from_config(cfg)
    cache_path: Path = settings["cache_path"]
    cache_exists = cache_path.exists()
    stale = _is_stale(cache_path, settings["update_interval_days"])
    warning: str | None = None
    action = "ready"

    csv_text: str | None = None
    if cache_exists and not stale and not force_update:
        csv_text = cache_path.read_text(encoding="utf-8")
    else:
        try:
            csv_text = _download_registry(settings)
            action = "downloaded" if not cache_exists else "updated"
        except Exception as exc:
            logger.warning("[port_registry] Could not refresh IANA registry: %s", exc)
            warning = str(exc)
            if cache_exists:
                csv_text = cache_path.read_text(encoding="utf-8")
                action = "stale"

    if csv_text is None:
        return PortRegistry(
            records=_builtin_records(),
            source="builtin",
            action="builtin_fallback",
            warning=warning,
        )

    return PortRegistry(
        records=_parse_registry_csv(csv_text),
        source="iana",
        action=action,
        cache_path=str(cache_path),
        warning=warning,
    )