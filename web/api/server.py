from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import Config
from core.skill_onboarding import discover_skill_requirements, get_missing_skill_variables
from core.chat_router.logic import (
    add_to_history,
    get_context_summary,
    list_conversations,
    load_conversation_history,
    run_graph,
)
from web.api.service import SecurityClawService

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
DIST_DIR = WEB_ROOT / "dist"
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"
SKILLS_DIR = ROOT / "skills"

load_dotenv(ENV_PATH)

SECRET_NAME_RE = re.compile(r"(password|secret|token|api[_-]?key|client[_-]?secret|license[_-]?key)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Input Validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_conversation_id(conversation_id: str) -> None:
    """Validate conversation_id contains only safe characters; raise HTTPException if invalid."""
    if not conversation_id or not all(c.isalnum() or c in '-_' for c in conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")


def _validate_skill_name(skill_name: str) -> None:
    """Validate skill_name contains only safe characters; raise HTTPException if invalid."""
    if not skill_name or not all(c.isalnum() or c == '_' for c in skill_name):
        raise HTTPException(status_code=400, detail="Invalid skill name format")


def _mask_value(name: str, value: str | None) -> str:
    if not value:
        return ""
    if SECRET_NAME_RE.search(name):
        return "••••••••"
    return value


def _read_text(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def _short_json(payload: dict, limit: int = 1200) -> str:
    rendered = json.dumps(payload, indent=2, default=str)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit].rstrip() + " ..."


def _disabled_skill_names() -> set[str]:
    disabled = Config().get("agent", "disabled_skills", default=[])
    if not isinstance(disabled, list):
        return set()
    return {str(skill_name).strip() for skill_name in disabled if str(skill_name).strip()}


def _is_skill_enabled(skill_name: str) -> bool:
    return skill_name not in _disabled_skill_names()


def _update_skill_enabled_state(skill_name: str, enabled: bool) -> None:
    current = yaml.safe_load(_read_text(CONFIG_PATH, "")) or {}
    if not isinstance(current, dict):
        current = {}

    agent_cfg = current.setdefault("agent", {})
    disabled = agent_cfg.get("disabled_skills", [])
    if not isinstance(disabled, list):
        disabled = []

    normalized = [str(name).strip() for name in disabled if str(name).strip()]
    if enabled:
        normalized = [name for name in normalized if name != skill_name]
    elif skill_name not in normalized:
        normalized.append(skill_name)

    agent_cfg["disabled_skills"] = sorted(dict.fromkeys(normalized))
    CONFIG_PATH.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    Config.reset()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class SaveTextRequest(BaseModel):
    content: str


class SaveConfigRequest(BaseModel):
    content: str


class SaveEnvRequest(BaseModel):
    values: dict[str, str]


class RestartRequest(BaseModel):
    reason: str | None = None


class SkillToggleRequest(BaseModel):
    enabled: bool


class ChatStreamParser:
    @staticmethod
    def to_step_payload(event: str, data: dict[str, Any] | None, step: int, max_steps: int) -> dict[str, Any]:
        data = data or {}
        if event == "deciding":
            return {
                "kind": "thinking",
                "label": f"Thinking · step {step}/{max_steps}",
                "detail": data.get("reasoning", "Planning next move"),
                "skills": data.get("skills", []),
                "step": step,
                "max_steps": max_steps,
            }
        if event == "running":
            skills = data.get("skills", [])
            label = "Fetching" if skills else "Processing"
            return {
                "kind": "fetching",
                "label": label,
                "detail": ", ".join(skills) if skills else "Running selected skills",
                "skills": skills,
                "step": step,
                "max_steps": max_steps,
            }
        if event == "evaluated":
            satisfied = bool(data.get("satisfied", False))
            return {
                "kind": "evaluating",
                "label": "Evaluating",
                "detail": data.get("reasoning", "Checking if the answer is sufficient"),
                "satisfied": satisfied,
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "step": step,
                "max_steps": max_steps,
            }
        return {
            "kind": "processing",
            "label": event.title(),
            "detail": "Working",
            "step": step,
            "max_steps": max_steps,
        }


def _skill_dirs() -> list[Path]:
    if not SKILLS_DIR.exists():
        return []
    return sorted([
        p for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "logic.py").exists()
    ])


def _parse_instruction_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---\n"):
        return {}, content
    try:
        _, rest = content.split("---\n", 1)
        frontmatter, body = rest.split("\n---\n", 1)
        return yaml.safe_load(frontmatter) or {}, body
    except ValueError:
        return {}, content


def _get_skill_description(skill_name: str) -> str:
    manifest_path = SKILLS_DIR / skill_name / "manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        return manifest.get("description", "Security analysis skill")
    return "Security analysis skill"


def _build_available_skills() -> list[dict[str, Any]]:
    skills = []
    for skill_dir in _skill_dirs():
        if skill_dir.name == "chat_router":
            continue
        if not _is_skill_enabled(skill_dir.name):
            continue
        skills.append({
            "name": skill_dir.name,
            "description": _get_skill_description(skill_dir.name),
        })
    return skills


def _skill_payload(skill_dir: Path) -> dict[str, Any]:
    manifest_path = skill_dir / "manifest.yaml"
    instruction_path = skill_dir / "instruction.md"
    manifest_raw = _read_text(manifest_path)
    instruction_raw = _read_text(instruction_path)
    manifest = yaml.safe_load(manifest_raw) if manifest_raw else {}
    manifest = manifest or {}
    frontmatter, _ = _parse_instruction_frontmatter(instruction_raw)
    return {
        "name": skill_dir.name,
        "enabled": _is_skill_enabled(skill_dir.name),
        "manifest": manifest,
        "manifest_raw": manifest_raw,
        "instruction_raw": instruction_raw,
        "description": manifest.get("description", "Security analysis skill"),
        "schedule_interval_seconds": frontmatter.get("schedule_interval_seconds"),
        "schedule_cron_expr": frontmatter.get("schedule_cron_expr"),
        "required_env_vars": manifest.get("required_env_vars", []),
    }


def _all_skills_payload() -> list[dict[str, Any]]:
    return [_skill_payload(skill_dir) for skill_dir in _skill_dirs()]


def _env_payload() -> dict[str, Any]:
    raw = dotenv_values(ENV_PATH)
    for key in [
        "DB_USERNAME",
        "DB_PASSWORD",
        "OLLAMA_BASE_URL",
        "ABUSEIPDB_API_KEY",
        "ALIENVAULT_API_KEY",
        "VIRUSTOTAL_API_KEY",
        "TALOS_CLIENT_ID",
        "TALOS_CLIENT_SECRET",
        "MAXMIND_LICENSE_KEY",
    ]:
        raw.setdefault(key, "")

    skill_requirements = discover_skill_requirements()
    for vars_for_skill in skill_requirements.values():
        for spec in vars_for_skill.values():
            env_key = spec.get("env_key") or spec.get("name")
            if env_key:
                raw.setdefault(env_key, "")

    payload = {}
    for key, value in raw.items():
        payload[key] = {
            "value": _mask_value(key, value),
            "is_secret": bool(SECRET_NAME_RE.search(key)),
            "set": bool(value),
        }
    return payload


def _write_env(values: dict[str, str]) -> None:
    current = dict(dotenv_values(ENV_PATH))
    for key, value in values.items():
        if value == "••••••••":
            continue
        current[key] = value
    lines = [f"{key}={value}" for key, value in current.items() if value is not None]
    ENV_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    load_dotenv(ENV_PATH, override=True)
    Config.reset()


def _chat_history_for_router(conversation_id: str) -> list[dict[str, Any]]:
    history = load_conversation_history(conversation_id)
    return history[-8:] if history else []


def create_app(*, enable_scheduler: bool = True) -> FastAPI:
    service = SecurityClawService(enable_scheduler=enable_scheduler)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        app.state.service = service

        # Persistent SQLite checkpointer shared across all chat requests
        import sqlite3 as _sqlite3
        _conversations_db = ROOT / "data" / "conversations.db"
        _conversations_db.parent.mkdir(parents=True, exist_ok=True)
        _conn = _sqlite3.connect(str(_conversations_db), check_same_thread=False)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver
            app.state.checkpointer = _SqliteSaver(_conn)
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver as _MemorySaver
            app.state.checkpointer = _MemorySaver()
            _conn.close()
            _conn = None

        yield

        service.stop()
        if _conn is not None:
            _conn.close()

    app = FastAPI(title="SecurityClaw Service", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["content-type"],
        allow_credentials=False,
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        ctx = app.state.service.context
        return {
            "agent_name": ctx.cfg.get("agent", "name", default="SecurityClaw"),
            "version": ctx.cfg.get("agent", "version", default="1.0.0"),
            "scheduler_running": ctx.runner.is_running,
            "skill_count": len(ctx.runner._skills),
            "skills_loaded": sorted(ctx.runner._skills.keys()),
            "missing_skill_vars": get_missing_skill_variables(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/conversations")
    async def conversations() -> dict[str, Any]:
        return {"items": list_conversations()}

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> dict[str, Any]:
        # Security: Validate conversation_id contains only safe characters
        _validate_conversation_id(conversation_id)
        
        return {
            "id": conversation_id,
            "messages": load_conversation_history(conversation_id),
            "summary": get_context_summary(conversation_id),
        }

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, str]:
        # Security: Validate conversation_id contains only safe characters
        _validate_conversation_id(conversation_id)
        
        conv_path = (ROOT / "conversations" / f"{conversation_id}.json").resolve()
        safe_dir = (ROOT / "conversations").resolve()
        
        # Security: Ensure resolved path is within conversations directory (prevent path traversal)
        if not str(conv_path).startswith(str(safe_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if conv_path.exists():
            conv_path.unlink()
        return {"status": "ok"}

    @app.get("/api/skills")
    async def skills() -> dict[str, Any]:
        return {"items": _all_skills_payload()}

    @app.get("/api/skills/{skill_name}")
    async def skill_detail(skill_name: str) -> dict[str, Any]:
        _validate_skill_name(skill_name)
        skill_dir = SKILLS_DIR / skill_name
        if not skill_dir.exists():
            raise HTTPException(status_code=404, detail="Skill not found")
        return _skill_payload(skill_dir)

    @app.put("/api/skills/{skill_name}/enabled")
    async def set_skill_enabled(skill_name: str, body: SkillToggleRequest) -> dict[str, Any]:
        _validate_skill_name(skill_name)
        skill_dir = SKILLS_DIR / skill_name
        if not skill_dir.exists():
            raise HTTPException(status_code=404, detail="Skill not found")

        _update_skill_enabled_state(skill_name, body.enabled)
        app.state.service.restart()
        return {"status": "ok", "skill": skill_name, "enabled": body.enabled}

    @app.put("/api/skills/{skill_name}/manifest")
    async def save_skill_manifest(skill_name: str, body: SaveTextRequest) -> dict[str, str]:
        # Security: Validate skill_name contains only safe characters
        _validate_skill_name(skill_name)
        
        skill_dir = SKILLS_DIR / skill_name
        safe_dir = SKILLS_DIR.resolve()
        
        # Security: Ensure resolved path is within skills directory (prevent path traversal)
        if not skill_dir.resolve().parent == safe_dir:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not skill_dir.exists():
            raise HTTPException(status_code=404, detail="Skill not found")
        
        yaml.safe_load(body.content or "{}")
        (skill_dir / "manifest.yaml").write_text(body.content, encoding="utf-8")
        return {"status": "ok"}

    @app.put("/api/skills/{skill_name}/instruction")
    async def save_skill_instruction(skill_name: str, body: SaveTextRequest) -> dict[str, str]:
        # Security: Validate skill_name contains only safe characters
        _validate_skill_name(skill_name)
        
        skill_dir = SKILLS_DIR / skill_name
        safe_dir = SKILLS_DIR.resolve()
        
        # Security: Ensure resolved path is within skills directory (prevent path traversal)
        if not skill_dir.resolve().parent == safe_dir:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not skill_dir.exists():
            raise HTTPException(status_code=404, detail="Skill not found")
        
        (skill_dir / "instruction.md").write_text(body.content, encoding="utf-8")
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "config_raw": _read_text(CONFIG_PATH, ""),
            "env": _env_payload(),
            "required_env_vars": discover_skill_requirements(),
            "missing_skill_vars": get_missing_skill_variables(),
            "disabled_skills": sorted(_disabled_skill_names()),
        }

    @app.put("/api/config")
    async def save_config(body: SaveConfigRequest) -> dict[str, str]:
        yaml.safe_load(body.content or "{}")
        CONFIG_PATH.write_text(body.content, encoding="utf-8")
        Config.reset()
        app.state.service.restart()
        return {"status": "ok"}

    @app.put("/api/env")
    async def save_env(body: SaveEnvRequest) -> dict[str, str]:
        _write_env(body.values)
        app.state.service.restart()
        return {"status": "ok"}

    @app.get("/api/crons")
    async def crons() -> dict[str, Any]:
        items = []
        for skill in _all_skills_payload():
            schedule_type = "manual"
            if skill.get("schedule_cron_expr"):
                schedule_type = "cron"
            elif skill.get("schedule_interval_seconds") is not None:
                schedule_type = "interval"
            items.append({
                "name": skill["name"],
                "description": skill["description"],
                "enabled": skill.get("enabled", True),
                "type": schedule_type,
                "interval_seconds": skill.get("schedule_interval_seconds"),
                "cron_expr": skill.get("schedule_cron_expr"),
            })
        return {"items": items}

    @app.post("/api/restart")
    async def restart(_: RestartRequest | None = None) -> dict[str, str]:
        app.state.service.restart()
        return {"status": "ok", "message": "SecurityClaw service restarted"}

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest):
        conversation_id = body.conversation_id or str(uuid.uuid4())[:8]
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        done = asyncio.Event()
        result_box: dict[str, Any] = {}
        error_box: dict[str, str] = {}

        def callback(event: str, data: dict[str, Any], step: int, max_steps: int) -> None:
            payload = ChatStreamParser.to_step_payload(event, data, step, max_steps)
            loop.call_soon_threadsafe(queue.put_nowait, ("step", payload))

        def worker() -> None:
            try:
                ctx = app.state.service.context
                instruction = _read_text(ROOT / "core" / "chat_router" / "instruction.md")
                orchestration = run_graph(
                    user_question=body.message,
                    available_skills=_build_available_skills(),
                    runner=ctx.runner,
                    llm=ctx.llm,
                    instruction=instruction,
                    cfg=ctx.cfg,
                    conversation_history=_chat_history_for_router(conversation_id),
                    step_callback=callback,
                    checkpointer=app.state.checkpointer,
                    thread_id=f"{conversation_id}-{uuid.uuid4().hex[:8]}",
                )
                result_box.update(orchestration)
                add_to_history(
                    conversation_id,
                    body.message,
                    orchestration.get("response", ""),
                    orchestration.get("routing", {}),
                    orchestration.get("skill_results", {}),
                )
            except Exception as exc:
                error_box["message"] = str(exc)
            finally:
                loop.call_soon_threadsafe(done.set)

        asyncio.get_running_loop().run_in_executor(None, worker)

        async def event_stream():
            yield _sse("meta", {"conversation_id": conversation_id})
            while not done.is_set() or not queue.empty():
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=0.25)
                    yield _sse(event, payload)
                except asyncio.TimeoutError:
                    continue
            if error_box:
                yield _sse("error", {"message": error_box["message"]})
            else:
                yield _sse("response", {
                    "conversation_id": conversation_id,
                    "response": result_box.get("response", ""),
                    "routing": result_box.get("routing", {}),
                    "trace": result_box.get("trace", []),
                    "skill_results": result_box.get("skill_results", {}),
                })
            yield _sse("done", {"conversation_id": conversation_id})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="web")
    else:
        @app.get("/")
        async def root() -> dict[str, str]:
            return {
                "message": "SecurityClaw web frontend is not built yet.",
                "hint": "Run `python main.py web-build` and then `python main.py service`."
            }

    return app


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


app = create_app(enable_scheduler=os.getenv("SECURITYCLAW_API_ONLY") != "1")


def run_service(host: str = "0.0.0.0", port: int = 7799, enable_scheduler: bool = True) -> None:
    os.environ["SECURITYCLAW_API_ONLY"] = "0" if enable_scheduler else "1"
    uvicorn.run(
        create_app(enable_scheduler=enable_scheduler),
        host=host,
        port=port,
        reload=False,
    )
