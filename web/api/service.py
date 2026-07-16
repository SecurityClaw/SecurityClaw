from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import Config
from core.db_connector import OpenSearchConnector
from core.graph_enrichment import set_enrichers
from core.llm_provider import build_llm_provider
from core.runner import Runner
from core.skill_loader import SkillLoader
from core.skill_manifest import SkillManifestLoader


@dataclass
class ServiceContext:
    cfg: Config
    db: Any
    llm: Any
    runner: Runner
    skill_loader: SkillLoader
    manifest_loader: SkillManifestLoader


class SecurityClawService:
    """Long-lived runtime holder for the web service."""

    def __init__(self, *, enable_scheduler: bool = True) -> None:
        self.enable_scheduler = enable_scheduler
        self._lock = threading.RLock()
        self._context: ServiceContext | None = None

    def start(self) -> None:
        with self._lock:
            if self._context and self._context.runner.is_running:
                return

            Config.reset()
            cfg = Config()
            db = OpenSearchConnector()
            llm = build_llm_provider()
            runner = Runner(db_connector=db, llm_provider=llm)
            runner.setup()
            if self.enable_scheduler:
                runner.start(register_signals=False)

            self._context = ServiceContext(
                cfg=cfg,
                db=db,
                llm=llm,
                runner=runner,
                skill_loader=SkillLoader(),
                manifest_loader=SkillManifestLoader(),
            )

            # Register enrichment callables to avoid hardcoded skill imports in core.
            from skills.geoip_lookup.logic import run as geoip_run
            from skills.threat_analyst.reputation_intel import get_domain_reputation, get_ip_reputation
            set_enrichers(
                geoip=geoip_run,
                ip_reputation=get_ip_reputation,
                domain_reputation=get_domain_reputation,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._context:
                return
            self._context.runner.stop()
            self._context = None

    def restart(self) -> None:
        with self._lock:
            self.stop()
            self.start()

    @property
    def context(self) -> ServiceContext:
        with self._lock:
            if self._context is None:
                self.start()
            assert self._context is not None
            return self._context
