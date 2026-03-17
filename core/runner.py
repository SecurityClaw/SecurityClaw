"""
core/runner.py — The Conductor.

Orchestrates skill discovery, scheduler setup, memory management,
and the main event loop.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from core.config import Config
from core.memory import CheckpointBackedMemory
from core.scheduler import AgentScheduler
from core.skill_loader import Skill, SkillLoader

logger = logging.getLogger(__name__)
console = Console()


class Runner:
    """
    The SecurityClaw Conductor.

    Lifecycle:
        1. Discover skills from /skills directory.
        2. Register each skill's `run` function with the scheduler.
        3. Build a context factory that injects shared services.
        4. Start the scheduler and enter a blocking loop.
        5. On SIGINT/SIGTERM, flush memory and shut down gracefully.

    Skills declare their own scheduling via schedule_interval_seconds or
    schedule_cron_expr in their instruction.md front-matter. The core
    discovers this information at load time and has no hardcoded knowledge
    of specific skill names.
    """

    def __init__(
        self,
        db_connector: Any = None,
        llm_provider: Any = None,
        skills_dir=None,
        memory_path=None,
    ) -> None:
        self.cfg = Config()
        self.memory = CheckpointBackedMemory(path=memory_path)
        self.scheduler = AgentScheduler()
        self.loader = SkillLoader(skills_dir=skills_dir)
        self.db = db_connector
        self.llm = llm_provider
        self._running = False
        self._skills: dict[str, Skill] = {}
        # Path to per-skill startup completion tracking
        self._startup_marker_path = Path("data") / ".startup_complete"

    def _is_first_startup_for_skill(self, skill_name: str) -> bool:
        """Check if a specific skill's first startup has completed."""
        if not self._startup_marker_path.exists():
            return True
        
        try:
            data = json.loads(self._startup_marker_path.read_text(encoding="utf-8"))
            return skill_name not in data
        except Exception:
            # If marker file is corrupted, treat as first startup
            return True

    def _mark_skill_startup_complete(self, skill_name: str) -> None:
        """Mark that a specific skill's first startup has completed successfully."""
        self._startup_marker_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or start fresh
        existing_data = {}
        if self._startup_marker_path.exists():
            try:
                existing_data = json.loads(
                    self._startup_marker_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing_data = {}
        
        # Add this skill's completion timestamp
        existing_data[skill_name] = time.time()
        
        # Write back the updated data
        self._startup_marker_path.write_text(
            json.dumps(existing_data, indent=2)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Discover skills and wire up the scheduler."""
        discovered_skills = self.loader.discover()
        disabled_skills = self.cfg.get("agent", "disabled_skills", default=[])
        disabled_set = {
            str(skill_name).strip()
            for skill_name in (disabled_skills if isinstance(disabled_skills, list) else [])
            if str(skill_name).strip()
        }

        if disabled_set:
            logger.info("Disabled skills from config: %s", ", ".join(sorted(disabled_set)))

        self._skills = {
            name: skill
            for name, skill in discovered_skills.items()
            if name not in disabled_set
        }

        if not self._skills:
            logger.warning("No skills found — check your skills/ directory.")

        # Run first-startup skills only on their first startup
        # They will be scheduled via cron/intervals after that
        startup_errors = []
        skills_run_on_startup = []
        
        for name, skill in self._skills.items():
            if skill.run_on_first_startup:
                # Check if this specific skill has already completed first startup
                if not self._is_first_startup_for_skill(name):
                    # Already completed first startup for this skill in a prior session
                    logger.debug(
                        "[%s] Skipping first-startup run (already completed in prior session)",
                        name,
                    )
                    continue
                
                logger.info("[%s] Running on startup…", name)
                skills_run_on_startup.append(name)
                try:
                    context = self._build_context()
                    # Pass force_refresh=true for startup runs to ensure initialization
                    context.setdefault("parameters", {})["force_refresh"] = True
                    result = skill.run(context)
                    
                    # Check if result indicates success
                    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
                    if status in ("ok", "initialized"):
                        logger.info("[%s] Startup run completed successfully (status=%s).", name, status)
                        # Mark this specific skill as startup-complete
                        self._mark_skill_startup_complete(name)
                    else:
                        error_msg = f"[{name}] Startup run returned status='{status}' (result: {result})"
                        logger.error(error_msg)
                        startup_errors.append(error_msg)
                except Exception as exc:
                    error_msg = f"[{name}] Startup run failed: {exc}"
                    logger.error(error_msg)
                    startup_errors.append(error_msg)
        
        # Log summary of startup runs
        if skills_run_on_startup:
            logger.info(
                "First-startup skills completed: %s. Future runs will use scheduler only.",
                ", ".join(skills_run_on_startup),
            )
        
        # If any startup skills failed, raise an error to prevent normal operation
        if startup_errors:
            error_summary = "\n".join(startup_errors)
            raise RuntimeError(f"Startup initialization failed:\n{error_summary}")

        self.scheduler.set_context_factory(self._build_context)

        for name, skill in self._skills.items():
            if skill.schedule_cron_expr:
                # Parse cron expression and use cron-based scheduling
                # Format: "minute hour day month day_of_week"
                # Example: "0 2 * * tue,fri"
                parts = skill.schedule_cron_expr.split()
                if len(parts) == 5:
                    cron_kwargs = {
                        "minute": parts[0],
                        "hour": parts[1],
                        "day": parts[2],
                        "month": parts[3],
                        "day_of_week": parts[4],
                    }
                    self.scheduler.register_cron(
                        name=name,
                        fn=skill.run,
                        **cron_kwargs
                    )
                else:
                    logger.error(
                        "Invalid cron expression for skill %s: %s (expected: minute hour day month day_of_week)",
                        name,
                        skill.schedule_cron_expr,
                    )
            else:
                # Use interval-based scheduling only when the skill declares one.
                # Skills without interval/cron metadata are treated as manual.
                if skill.schedule_interval_seconds is None:
                    logger.info("Skill %s is manual (on-demand only).", name)
                    continue

                interval = skill.schedule_interval_seconds
                self.scheduler.register(
                    name=name,
                    fn=skill.run,
                    interval_seconds=int(interval),
                    run_immediately=False,
                )

        self._print_skill_table()

    def start(self, *, register_signals: bool = True) -> None:
        """Start the scheduler without blocking the current thread."""
        if self._running:
            logger.info("Runner already active.")
            return

        self._running = True
        self.scheduler.start()
        self.memory.set_status("ACTIVE")
        self.memory.add_decision("Agent started.")

        if register_signals:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

    def run(self) -> None:
        """Start the scheduler and block until signal received."""
        self.start(register_signals=True)

        console.print("[bold green]SecurityClaw is running.[/] Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(1)
        finally:
            self.stop()

    def dispatch(self, skill_name: str, context: Optional[dict] = None) -> Any:
        """Manually fire a skill for testing or CLI invocation."""
        if skill_name not in self._skills:
            raise KeyError(f"Skill {skill_name!r} not loaded.")
        ctx = context or self._build_context()
        return self._skills[skill_name].run(ctx)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_context(self) -> dict:
        """Construct the shared context injected into every skill run."""
        return {
            "db": self.db,
            "llm": self.llm,
            "memory": self.memory,
            "config": self.cfg,
            "skills": self._skills,
        }

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Signal %d received — shutting down.", signum)
        self._running = False

    def stop(self) -> None:
        """Stop the scheduler and update agent memory."""
        if not self._running:
            if hasattr(self.memory, "close"):
                self.memory.close()
            return

        self._running = False
        console.print("[yellow]Shutting down SecurityClaw…[/]")
        self.scheduler.stop()
        self.memory.set_status("IDLE")
        self.memory.add_decision("Agent shut down cleanly.")
        if hasattr(self.memory, "close"):
            self.memory.close()
        console.print("[green]Done.[/]")

    @property
    def is_running(self) -> bool:
        return self._running

    def _print_skill_table(self) -> None:
        table = Table(title="Loaded Skills", show_lines=True)
        table.add_column("Skill", style="cyan")
        table.add_column("Schedule", style="magenta")
        table.add_column("Instruction Preview", style="white", max_width=60)
        for name, skill in self._skills.items():
            if skill.schedule_cron_expr:
                schedule = f"cron: {skill.schedule_cron_expr}"
            elif skill.schedule_interval_seconds is not None:
                schedule = f"every {skill.schedule_interval_seconds}s"
            else:
                schedule = "manual (on-demand)"
            preview = (skill.instruction[:80] + "…") if len(skill.instruction) > 80 else skill.instruction
            table.add_row(name, schedule, preview.replace("\n", " "))
        console.print(table)
