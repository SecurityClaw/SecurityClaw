"""
core/scheduler.py — Cron-like task scheduler built on APScheduler.

Supports:
  - interval-based jobs (every N seconds/minutes/hours)
  - manual one-shot dispatch
  - graceful shutdown
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.config import Config

logger = logging.getLogger(__name__)


class AgentScheduler:
    """
    Wraps APScheduler to provide a clean interface for the Runner.

    Jobs are registered with a name, a callable, and an interval in
    seconds.  The callable receives a single dict `context` argument
    (populated by the Runner before each invocation).
    """

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(
            job_defaults={"coalesce": True, "max_instances": 1}
        )
        self._jobs: dict[str, dict] = {}
        self._context_factory: Callable[[], dict] = dict
        self._result_callback: Callable[[str, Any], None] | None = None

    def set_context_factory(self, factory: Callable[[], dict]) -> None:
        """Supply a callable that returns a fresh context dict per run."""
        self._context_factory = factory

    def set_result_callback(self, callback: Callable[[str, Any], None] | None) -> None:
        self._result_callback = callback

    def _record_result(self, name: str, result: Any) -> None:
        self._jobs[name]["last_result"] = result
        self._jobs[name]["last_error"] = None
        self._jobs[name]["last_run"] = datetime.now(timezone.utc).isoformat()
        if self._result_callback:
            self._result_callback(name, result)

    def register(
        self,
        name: str,
        fn: Callable[[dict], Any],
        interval_seconds: int,
        *,
        run_immediately: bool = False,
    ) -> None:
        """Register a recurring job."""
        if name in self._jobs:
            logger.warning("Job %r already registered — replacing.", name)
            self._scheduler.remove_job(name)

        def _wrapper() -> None:
            ctx = self._context_factory()
            try:
                logger.debug("Running job: %s", name)
                result = fn(ctx)
                self._record_result(name, result)
                logger.debug("Job %s completed: %s", name, result)
            except Exception as exc:
                self._jobs[name]["last_error"] = str(exc)
                self._jobs[name]["last_run"] = datetime.now(timezone.utc).isoformat()
                logger.error("Job %s raised: %s", name, exc, exc_info=True)

        trigger = IntervalTrigger(seconds=interval_seconds)
        self._scheduler.add_job(
            _wrapper,
            trigger=trigger,
            id=name,
            name=name,
        )
        self._jobs[name] = {"fn": fn, "interval": interval_seconds}
        logger.info("Registered job %r every %ds", name, interval_seconds)

        if run_immediately:
            self._run_once(name, fn)

    def register_cron(
        self,
        name: str,
        fn: Callable[[dict], Any],
        **cron_kwargs: Any,
    ) -> None:
        """Register a cron-expression job (hour, minute, day_of_week…)."""
        from apscheduler.triggers.cron import CronTrigger

        def _wrapper() -> None:
            ctx = self._context_factory()
            try:
                result = fn(ctx)
                self._record_result(name, result)
            except Exception as exc:
                self._jobs[name]["last_error"] = str(exc)
                self._jobs[name]["last_run"] = datetime.now(timezone.utc).isoformat()
                logger.error("Cron job %s raised: %s", name, exc, exc_info=True)

        self._scheduler.add_job(
            _wrapper,
            trigger=CronTrigger(**cron_kwargs),
            id=name,
            name=name,
        )
        self._jobs[name] = {"fn": fn, "cron": cron_kwargs}
        logger.info("Registered cron job %r: %s", name, cron_kwargs)

    def dispatch(self, name: str, context: Optional[dict] = None) -> Any:
        """Immediately invoke a registered job (outside its schedule)."""
        if name not in self._jobs:
            raise KeyError(f"No job named {name!r}")
        fn = self._jobs[name]["fn"]
        ctx = context or self._context_factory()
        try:
            result = fn(ctx)
            self._record_result(name, result)
            return result
        except Exception as exc:
            self._jobs[name]["last_error"] = str(exc)
            self._jobs[name]["last_run"] = datetime.now(timezone.utc).isoformat()
            raise

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Scheduler started with %d job(s).", len(self._jobs))

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    @property
    def job_names(self) -> list[str]:
        return list(self._jobs.keys())

    @property
    def job_status(self) -> dict[str, dict]:
        """Return serializable runtime status without exposing job callables."""
        return {
            name: {key: value for key, value in details.items() if key != "fn"}
            for name, details in self._jobs.items()
        }

    # ------------------------------------------------------------------

    def _run_once(self, name: str, fn: Callable[[dict], Any]) -> None:
        from threading import Thread

        def _go() -> None:
            ctx = self._context_factory()
            try:
                fn(ctx)
            except Exception as exc:
                logger.error("Immediate run %s failed: %s", name, exc)

        Thread(target=_go, daemon=True, name=f"immediate-{name}").start()
