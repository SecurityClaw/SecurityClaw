"""
main.py — SecurityClaw CLI entrypoint.

Usage:
    python main.py onboard              # Interactive setup wizard
    python main.py run                  # Start full agent loop
    python main.py service              # Start web service + API + scheduler
    python main.py web-build            # Build the React frontend in /web
    python main.py web-dev              # Start the frontend dev server
    python main.py dispatch <skill>     # Fire a skill once
    python main.py chat                 # Interactive chat with skill routing
    python main.py status               # Print the compact agent memory snapshot
    python main.py list-skills          # List discovered skills
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional

# Configure logging BEFORE any other imports that use logging
from rich.console import Console
from rich.text import Text

_console = Console()

class _GreyLogHandler(logging.Handler):
    """Custom log handler that renders all logs in grey using ANSI codes."""
    
    def emit(self, record):
        """Emit a log record, styling it as grey with ANSI codes."""
        try:
            message = self.format(record)
            # Use ANSI escape codes for bright black (grey) - most reliable across terminals
            # \033[90m = bright black (grey), \033[0m = reset to default
            import sys
            sys.stderr.write(f"\033[90m{message}\033[0m\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)

# Set up grey logging IMMEDIATELY, before any imports
_root_logger = logging.getLogger()

# Clear ALL existing handlers from root logger
for _h in list(_root_logger.handlers):
    _root_logger.removeHandler(_h)

# Add ONLY our grey handler
_grey_handler = _GreyLogHandler()
_grey_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root_logger.addHandler(_grey_handler)
_root_logger.setLevel(logging.DEBUG)

# Override basicConfig to not add extra handlers
_original_basicConfig = logging.basicConfig
def _no_op_basicConfig(*args, **kwargs):
    """Ignore basicConfig calls from other modules."""
    pass
logging.basicConfig = _no_op_basicConfig

# Now safe to import other modules
import click
import yaml
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm

from core.config import Config
from core.db_connector import OpenSearchConnector
from core.llm_provider import build_llm_provider
from core.memory import CheckpointBackedMemory
from core.runner import Runner

console = _console
logger = logging.getLogger(__name__)

THOUGHT_TOKEN_PHASES = {"skills_check", "think", "reflect"}
FINAL_TOKEN_PHASES = {"direct_answer", "answer", "response_final"}


def _setup_logging(level: str) -> None:
    """Update logging level during runtime."""
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def _highlight_response(response: str) -> str:
    """Highlight IPs, ports, and timestamps in response text with different colors.
    
    Returns Rich markup string with colors:
    - IPs: cyan
    - Ports: yellow  
    - Timestamps: green
    """
    import re
    
    # Pattern for IPv4 addresses
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # Pattern for ISO timestamps FIRST (so we mark them before looking for ports)
    timestamp_pattern = r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b'
    
    # Pattern for ports - match "port 443", "ports 22,80", etc. 
    # Use word-specific port keyword pattern to avoid false matches
    port_pattern = r'\bports?\s+([0-9]{1,5})'
    
    # Apply highlighting in order: timestamps first, then IPs, then ports
    # This prevents port patterns from matching timestamps
    response = re.sub(timestamp_pattern, r'[green]\g<0>[/green]', response)
    response = re.sub(ip_pattern, r'[cyan]\g<0>[/cyan]', response)
    response = re.sub(port_pattern, r'port \g<1>[yellow]\1[/yellow]', response)
    
    return response


def _build_runner() -> Runner:
    cfg = Config()
    db = OpenSearchConnector()
    llm = build_llm_provider()
    return Runner(db_connector=db, llm_provider=llm)


@click.group()
@click.option("--log-level", default=None, help="Logging level (DEBUG/INFO/WARNING/ERROR)")
@click.pass_context
def cli(ctx, log_level):
    cfg = Config()
    _setup_logging(log_level or cfg.get("agent", "log_level", default="INFO"))


@cli.command()
def onboard():
    """Interactive setup wizard for OpenSearch and LLM configuration."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/]")
    console.print("[bold yellow]SecurityClaw Configuration Wizard[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Database Configuration
    # ──────────────────────────────────────────────────────────────────────────
    console.print("[bold green]Step 1: Database Configuration[/]\n")
    db_provider = Prompt.ask(
        "Database provider",
        choices=["opensearch", "elasticsearch"],
        default="opensearch"
    )

    db_host = Prompt.ask("Database host", default="localhost")
    db_port = Prompt.ask("Database port", default="9200")

    use_ssl = Confirm.ask("Use SSL/TLS?", default=False)
    verify_certs = Confirm.ask("Verify SSL certificates?", default=False) if use_ssl else False

    has_auth = Confirm.ask("Require authentication?", default=False)
    db_user = ""
    db_pass = ""
    if has_auth:
        db_user = Prompt.ask("Database username")
        db_pass = Prompt.ask("Database password", password=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Index Configuration
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Index Configuration[/]\n")
    logs_index = Prompt.ask(
        "Network logs index pattern",
        default="securityclaw-logs"
    )
    anomaly_index = Prompt.ask(
        "Anomaly detection findings index",
        default="securityclaw-anomalies"
    )
    vector_index = Prompt.ask(
        "RAG vector embeddings index",
        default="securityclaw-vectors"
    )

    # Test DB connection
    console.print("\n[cyan]Testing database connection…[/]")
    if _test_opensearch_connection(db_host, int(db_port), db_user, db_pass, use_ssl, verify_certs):
        console.print("[green]✓ Database connection successful![/]\n")
    else:
        console.print("[yellow]⚠ Database connection test failed. Proceeding anyway…[/]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: LLM Configuration
    # ──────────────────────────────────────────────────────────────────────────
    console.print("[bold green]Step 2: Ollama Configuration[/]\n")
    llm_provider = "ollama"
    ollama_url = Prompt.ask("Ollama base URL", default="http://localhost:11434")
    ollama_model = Prompt.ask("Ollama chat model name", default="llama3")
    console.print(
        "[dim]The embed model is used exclusively for RAG vector embeddings (not for chat).\n"
        "Use a small, fast model such as [italic]nomic-embed-text:latest[/italic].[/]"
    )
    ollama_embed_model = Prompt.ask("Ollama embed model name", default="nomic-embed-text:latest")
    llm_config = {
        "ollama_base_url": ollama_url,
        "ollama_model": ollama_model,
        "ollama_embed_model": ollama_embed_model,
    }
    # Test Ollama
    console.print("[cyan]Testing Ollama connection…[/]")
    if _test_ollama_connection(ollama_url):
        console.print("[green]✓ Ollama connection successful![/]\n")
    else:
        console.print("[yellow]⚠ Ollama connection test failed. Proceeding anyway…[/]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: External Reputation Intelligence (Optional)
    # ──────────────────────────────────────────────────────────────────────────
    console.print("[bold green]Step 3: External Threat Intelligence APIs (Optional)[/]\n")
    console.print(
        "[dim]SecurityClaw can enrich threat analysis with external reputation data:\n"
        "  • AbuseIPDB (IP abuse history)\n"
        "  • AlienVault OTX (threat intelligence pulses)\n"
        "  • VirusTotal (multi-engine malware detection)\n"
        "  • Cisco Talos (IP/domain intelligence)\n\n"
        "These are [bold]optional[/] — threat_analyst works without them.\n"
        "Setup takes 2 minutes. [/]\n"
    )
    
    setup_apis = Confirm.ask("Configure external threat intelligence APIs?", default=False)
    api_keys: dict = {}
    
    if setup_apis:
        console.print("\n[cyan]Registering APIs (links provided):[/]\n")
        
        # AbuseIPDB
        if Confirm.ask("Setup AbuseIPDB (IP reputation)?", default=True):
            console.print(
                "[dim]Sign up free at: https://www.abuseipdb.com/api\n"
                "Free tier: 50 queries/day[/]"
            )
            api_key = Prompt.ask("AbuseIPDB API Key", default="", show_default=False)
            if api_key:
                api_keys["ABUSEIPDB_API_KEY"] = api_key
        
        # AlienVault OTX
        if Confirm.ask("\nSetup AlienVault OTX (threat pulses)?", default=True):
            console.print(
                "[dim]Sign up free at: https://otx.alienvault.com/api\n"
                "Free tier: Unlimited[/]"
            )
            api_key = Prompt.ask("AlienVault API Key", default="", show_default=False)
            if api_key:
                api_keys["ALIENVAULT_API_KEY"] = api_key
        
        # VirusTotal
        if Confirm.ask("\nSetup VirusTotal (malware detection)?", default=True):
            console.print(
                "[dim]Sign up free at: https://www.virustotal.com\n"
                "Free tier: 500 queries/day[/]"
            )
            api_key = Prompt.ask("VirusTotal API Key", default="", show_default=False)
            if api_key:
                api_keys["VIRUSTOTAL_API_KEY"] = api_key
        
        # Cisco Talos
        if Confirm.ask("\nSetup Cisco Talos (enterprise intelligence)?", default=False):
            console.print(
                "[dim]Register at: https://dashboard.cisco.com/webex\n"
                "Free tier: Available with registration[/]"
            )
            client_id = Prompt.ask("Talos Client ID", default="", show_default=False)
            client_secret = Prompt.ask("Talos Client Secret", default="", show_default=False)
            if client_id:
                api_keys["TALOS_CLIENT_ID"] = client_id
            if client_secret:
                api_keys["TALOS_CLIENT_SECRET"] = client_secret
        
        if api_keys:
            console.print(f"\n[green]✓ {len(api_keys)} API key(s) configured[/]")
        else:
            console.print("\n[yellow]No API keys configured; threat_analyst will work with local baselines only[/]")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Write Configuration & Create Initial Files
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold green]Step 4: Saving Configuration[/]\n")
    _write_config(
        db_provider, db_host, db_port, db_user, db_pass, use_ssl, verify_certs,
        logs_index, anomaly_index, vector_index,
        llm_provider, llm_config, api_keys
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5: Skill Variable Configuration (Optional)
    # ──────────────────────────────────────────────────────────────────────────
    console.print("\n[bold green]Step 5: Skill Configuration (Optional)[/]\n")
    from core.skill_onboarding import discover_skill_requirements, prompt_for_skill_variables, _write_env_vars
    
    skill_requirements = discover_skill_requirements()
    if skill_requirements:
        console.print("[dim]Some skills require additional configuration variables:[/]\n")
        for skill_name, var_specs in skill_requirements.items():
            console.print(f"  • {skill_name}")
            for var_name, var_spec in var_specs.items():
                optional_label = "[optional]" if var_spec.get("optional") else "[required]"
                console.print(f"    - {var_name} {optional_label}")
        
        configure_skills = Confirm.ask("\nConfigure skill variables now?", default=False)
        if configure_skills:
            # Prompt for all variables
            collected = prompt_for_skill_variables(skill_requirements)
            if collected:
                _write_env_vars(collected)
                console.print("[green]✓ Skill variables saved to .env[/]")
        else:
            console.print("[dim]You can configure these later by running:[/]")
            console.print("  [yellow]python main.py onboard[/]\n")
    else:
        console.print("[dim]No skill-specific variables required.[/]\n")

    console.print("[green bold]✓ Configuration complete![/]")
    console.print("\n[cyan]You can now run:[/]")
    console.print("  [yellow]python main.py chat[/]              # Start in chat mode")
    console.print("  [yellow]python main.py service[/]              # Start the agent")
    console.print("  [yellow]python main.py list-skills[/]      # See available skills")
    console.print("  [yellow]python main.py dispatch <skill>[/] # Fire a skill\n")


@cli.command()
def run():
    """Start the full SecurityClaw agent loop."""
    runner = _build_runner()
    runner.setup()
    runner.run()


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host interface for the web service")
@click.option("--port", default=7799, show_default=True, type=int, help="Port for the web service")
@click.option("--api-only", is_flag=True, help="Serve the API/UI without running scheduled skills")
def service(host: str, port: int, api_only: bool):
    """Start the web interface service (API + UI + optional scheduler)."""
    from web.api.server import run_service

    # Security warning: 0.0.0.0 exposes the service to all network interfaces
    if host == "0.0.0.0":
        console.print("\n[bold yellow]⚠  SECURITY WARNING[/]")
        console.print("[yellow]The API is binding to 0.0.0.0 (all network interfaces).[/]")
        console.print("[yellow]Ensure your firewall restricts access to trusted IPs only.[/]")
        console.print("[yellow]For local-only access, use: [cyan]--host 127.0.0.1[/][/]\n")

    console.print(f"[green]Starting SecurityClaw service at {host}:{port}...[/]")
    if not api_only:
        console.print("[green]Background scheduler is enabled[/]")
    else:
        console.print("[dim]Background scheduler is disabled (--api-only)[/]")
    
    run_service(host=host, port=port, enable_scheduler=not api_only)


@cli.command("web-build")
def web_build():
    """Install web dependencies and build the React frontend."""
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        console.print("[red]Error:[/] web/ directory not found.")
        raise SystemExit(1)

    console.print("[cyan]Installing web dependencies…[/]")
    subprocess.run(["npm", "install"], cwd=web_dir, check=True)
    console.print("[cyan]Building frontend…[/]")
    subprocess.run(["npm", "run", "build"], cwd=web_dir, check=True)
    console.print("[green]✓ Web frontend built successfully.[/]")


@cli.command("web-dev")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=5173, show_default=True, type=int)
def web_dev(host: str, port: int):
    """Run the Vite development server for the web frontend."""
    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        console.print("[red]Error:[/] web/ directory not found.")
        raise SystemExit(1)

    console.print("[cyan]Starting Vite dev server…[/]")
    subprocess.run(["npm", "run", "dev", "--", "--host", host, "--port", str(port)], cwd=web_dir, check=True)


@cli.command()
@click.argument("skill_name")
def dispatch(skill_name):
    """Fire a single skill immediately and print the result."""
    runner = _build_runner()
    runner.setup()
    try:
        result = runner.dispatch(skill_name)
        console.print_json(data=result)
    except KeyError as e:
        console.print(f"[red]Error:[/] Skill {e} not found.")
        sys.exit(1)


@cli.command()
def status():
    """Print the compact structured agent memory snapshot."""
    memory = CheckpointBackedMemory()
    try:
        console.print(memory.read())
    finally:
        memory.close()


@cli.command("list-skills")
def list_skills():
    """Discover and list all available skills."""
    from core.skill_loader import SkillLoader
    loader = SkillLoader()
    skills = loader.discover()
    if not skills:
        console.print("[yellow]No skills found.[/]")
        return
    for name, skill in skills.items():
        if skill.schedule_cron_expr:
            schedule = f"cron: [magenta]{skill.schedule_cron_expr}[/]"
        elif skill.schedule_interval_seconds is None:
            schedule = "manual [magenta](on-demand)[/]"
        else:
            interval = skill.schedule_interval_seconds
            schedule = f"every [magenta]{interval}s[/]"
        console.print(f"  [cyan]{name}[/] — {schedule}")


@cli.command()
def chat():
    """Interactive chat with the SOC agent—ask questions and route to skills."""
    from pathlib import Path
    from datetime import datetime
    from core.chat_router.logic import (
        route_question,
        execute_skill_workflow,
        format_response,
        run_graph,
        load_conversation_history,
        add_to_history,
        get_context_summary,
        list_conversations,
    )
    from core.skill_onboarding import ensure_skill_variables_onboarded
    
    # Ensure all skill variables are configured on first chat
    ensure_skill_variables_onboarded()
    from core.skill_loader import SkillLoader

    cfg = Config()
    db = OpenSearchConnector()
    llm = build_llm_provider()
    from core.runner import Runner
    runner = Runner(db_connector=db, llm_provider=llm)
    runner.setup()

    # Load chat_router skill instruction
    instruction_path = Path(__file__).parent / "core" / "chat_router" / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")

    # Define available skills for routing
    skill_loader = SkillLoader()
    discovered_skills = skill_loader.discover()
    available_skills = [
        {
            "name": name,
            "description": skill.description if hasattr(skill, "description") else "Security analysis skill",
        }
        for name, skill in discovered_skills.items()
        if name != "chat_router"  # Don't route to ourselves
    ]

    # Welcome message
    console.print("\n[bold cyan]═════════════════════════════════════════════════════════[/]")
    console.print("[bold yellow]SecurityClaw — SOC Chatbot[/]")
    console.print("[bold cyan]═════════════════════════════════════════════════════════[/]")
    console.print("[dim]Type /help for commands, /new for new conversation, /exit to quit[/]\n")

    # Open persistent SQLite checkpointer for the whole chat session
    import uuid
    import sqlite3
    _conversations_db = Path(__file__).parent / "data" / "conversations.db"
    _conversations_db.parent.mkdir(parents=True, exist_ok=True)
    _sqlite_conn = sqlite3.connect(str(_conversations_db), check_same_thread=False)
    try:
        _SqliteSaver = getattr(import_module("langgraph.checkpoint.sqlite"), "SqliteSaver")
        _checkpointer = _SqliteSaver(_sqlite_conn)
    except ImportError:
        _MemorySaver = getattr(import_module("langgraph.checkpoint.memory"), "MemorySaver")
        _checkpointer = _MemorySaver()
        logger.warning("langgraph-checkpoint-sqlite not installed; using in-memory checkpointer")

    # Conversation management
    conversation_id = str(uuid.uuid4())[:8]
    console.print(f"[dim]Conv ID: {conversation_id}[/]")

    # Main chat loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/]").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "/exit":
                console.print("[yellow]Goodbye![/]")
                break

            if user_input.lower() == "/new":
                conversation_id = str(uuid.uuid4())[:8]
                console.print(f"[green]✓ New conversation started. ID: {conversation_id}[/]")
                continue

            if user_input.lower() == "/help":
                console.print("\n[bold cyan]Commands:[/]")
                console.print("  /new    - Start a new conversation")
                console.print("  /history - Show past conversations")
                console.print("  /context - Show recent conversation context")
                console.print("  /skills - List available skills")
                console.print("  /exit   - Exit chat mode\n")
                continue

            if user_input.lower() == "/skills":
                console.print("\n[bold cyan]Available Skills:[/]")
                for skill in available_skills:
                    console.print(f"  • {skill['name']}: {skill['description']}")
                console.print()
                continue

            if user_input.lower() == "/history":
                convs = list_conversations()
                if not convs:
                    console.print("[yellow]No past conversations.[/]\n")
                else:
                    console.print("\n[bold cyan]Past Conversations:[/]")
                    for conv in convs[-10:]:  # Show last 10
                        console.print(
                            f"  {conv['id']}: {conv['messages']} messages — "
                            f"{conv['first_question'][:40]}..."
                        )
                    console.print()
                continue

            if user_input.lower() == "/context":
                context = get_context_summary(conversation_id, last_n=3)
                if context:
                    console.print("\n[bold cyan]Recent Context:[/]")
                    console.print(context)
                else:
                    console.print("[dim]No recent context.[/]")
                console.print()
                continue

            # Load conversation history for context (last 2 turns)
            conversation_history = load_conversation_history(conversation_id)
            recent_history = conversation_history[-4:] if conversation_history else []  # Last 2 Q&A pairs
            
            # Route/orchestrate with conversation context
            console.print()

            llm_stream_open = False
            llm_stream_phase: str | None = None

            def _close_llm_stream_line() -> None:
                nonlocal llm_stream_open, llm_stream_phase
                if llm_stream_open:
                    console.print()
                    llm_stream_open = False
                    llm_stream_phase = None

            def _stream_format_for_phase(phase: str) -> tuple[str, str, str]:
                if phase in FINAL_TOKEN_PHASES:
                    return "Final answer", "white", "white"
                if phase in THOUGHT_TOKEN_PHASES:
                    return "LLM thought", "dim", "grey70"
                return "LLM thought", "dim", "grey70"

            def _supervisor_callback(event: str, data: dict, step: int, max_steps: int) -> None:
                """Print a structured supervisor planning trace in real-time."""
                nonlocal llm_stream_open, llm_stream_phase
                if event != "token":
                    _close_llm_stream_line()

                if event == "skills_check":
                    console.print(f"[dim]┌ Skills evaluation[/]")
                    console.print(f"[dim]│ Analyzing: Do available skills match this question?[/]")
                elif event == "skills_needed":
                    reasoning = data.get("reasoning", "")
                    matched = data.get("matched_skills", [])
                    console.print(f"[dim]│ Result: Question NEEDS tools[/]")
                    if matched:
                        console.print(f"[dim]│ Matched skills: {', '.join(matched)}[/]")
                    console.print(f"[dim]│ Reasoning: {reasoning}[/]")
                    console.print(f"[dim]└ Proceeding to skill routing...[/]")
                    console.print()
                elif event == "skills_not_needed":
                    reasoning = data.get("reasoning", "")
                    console.print(f"[dim]│ Result: Question does NOT need tools[/]")
                    console.print(f"[dim]│ Reasoning: {reasoning}[/]")
                    console.print(f"[dim]└ Answering directly with knowledge...[/]")
                    console.print()
                elif event == "skills_check_failed":
                    error = data.get("error", "Unknown error")
                    console.print(f"[yellow]│ Skills check error: {error}[/]")
                    console.print(f"[yellow]└ Defaulting to skill routing...[/]")
                    console.print()
                elif event == "deciding":
                    reasoning = data.get("reasoning", "")
                    skills = data.get("skills", [])
                    planner_trace = data.get("planner_trace") or {}
                    question_grounding = planner_trace.get("question_grounding") or {}
                    initial_candidate = planner_trace.get("initial_candidate") or {}
                    reviews = planner_trace.get("reviews") or []
                    console.print(f"[dim]┌ Supervisor step {step}/{max_steps}[/]")
                    if question_grounding:
                        summary = str(question_grounding.get("summary") or "").strip()
                        immediate_need = str(question_grounding.get("immediate_need") or "").strip()
                        preserve = list(question_grounding.get("must_preserve") or [])
                        blocked = list(question_grounding.get("must_not_reframe_as") or [])
                        if summary:
                            console.print(f"[dim]│ Ask: {summary}[/]")
                        if immediate_need:
                            console.print(f"[dim]│ Immediate need: {immediate_need}[/]")
                        if preserve:
                            console.print(f"[dim]│ Must preserve: {', '.join(str(item) for item in preserve[:4])}[/]")
                        if blocked:
                            console.print(f"[dim]│ Must not reframe as: {', '.join(str(item) for item in blocked[:3])}[/]")
                    if initial_candidate:
                        initial_skills = initial_candidate.get("skills") or []
                        initial_reasoning = str(initial_candidate.get("reasoning") or "").strip()
                        if initial_skills:
                            console.print(f"[dim]│ Initial candidate: {', '.join(initial_skills)}[/]")
                        if initial_reasoning:
                            console.print(f"[dim]│ Initial reasoning: {initial_reasoning}[/]")
                    for review in reviews:
                        stage = str(review.get("stage") or "review")
                        proposed_skills = review.get("proposed_skills") or []
                        verdict = "accept" if review.get("valid", False) and review.get("should_execute", False) else "reject"
                        if stage == "grounding_group_rejection":
                            console.print(f"[dim]│ Grounding rejection: {review.get('issue', '')}[/]")
                            continue
                        review_reasoning = str(review.get("reasoning") or "").strip()
                        review_issue = str(review.get("issue") or "").strip()
                        confidence = float(review.get("confidence") or 0.0)
                        console.print(
                            f"[dim]│ Review round {review.get('round', '?')}: {verdict} "
                            f"({confidence:.0%}) for [{', '.join(proposed_skills) if proposed_skills else 'no skills'}][/]")
                        if review_issue:
                            console.print(f"[dim]│ Review issue: {review_issue}[/]")
                        elif review_reasoning:
                            console.print(f"[dim]│ Review reasoning: {review_reasoning}[/]")
                    if reasoning:
                        console.print(f"[dim]│ Final reasoning: {reasoning}[/]")
                    if skills:
                        console.print(f"[dim]│ → Invoking: {', '.join(skills)}[/]")
                    else:
                        console.print(f"[dim]│ → No skills selected — finalizing[/]")
                elif event == "evaluated":
                    satisfied = data.get("satisfied", False)
                    confidence = float(data.get("confidence") or 0)
                    reasoning = data.get("reasoning", "")
                    icon = "✓" if satisfied else "✗"
                    console.print(f"[dim]└ {icon} {'Satisfied' if satisfied else 'Not satisfied'} ({confidence:.0%}) — {reasoning}[/]")
                    console.print()
                elif event == "token":
                    phase = str(data.get("phase") or "")
                    token = str(data.get("token") or "")
                    if not token:
                        return
                    label, label_style, token_style = _stream_format_for_phase(phase)
                    if not llm_stream_open or llm_stream_phase != phase:
                        _close_llm_stream_line()
                        console.print(f"[{label_style}]│ {label}: [/]", end="")
                        llm_stream_open = True
                        llm_stream_phase = phase
                    console.print(token, end="", style=token_style, markup=False, highlight=False)

            orchestration = run_graph(
                user_question=user_input,
                available_skills=available_skills,
                runner=runner,
                llm=llm,
                instruction=instruction,
                cfg=cfg,
                conversation_history=recent_history,
                step_callback=_supervisor_callback,
                checkpointer=_checkpointer,
                thread_id=f"{conversation_id}-{uuid.uuid4().hex[:8]}",
            )

            routing = orchestration.get("routing", {"skills": []})
            skill_results = orchestration.get("skill_results", {})
            response = orchestration.get("response", "Unable to produce a response.")

            _close_llm_stream_line()

            # Highlight IPs, ports, and timestamps in the response
            highlighted_response = _highlight_response(response)
            console.print(f"[bold cyan]Agent[/]: {highlighted_response}\n")

            # Save to history
            add_to_history(conversation_id, user_input, response, routing, skill_results)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            break
        except Exception as e:
            console.print(f"Error: {e}", style="red")
            logging.getLogger(__name__).exception("Chat error")

    # Clean up SQLite connection when chat session ends
    try:
        _sqlite_conn.close()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Onboarding Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _test_opensearch_connection(
    host: str, port: int, user: str, password: str, use_ssl: bool, verify_certs: bool
) -> bool:
    """Test connectivity to OpenSearch/Elasticsearch."""
    try:
        from opensearchpy import OpenSearch
        
        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password) if user and password else None,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
        )
        # Simple ping test
        info = client.info()
        return info is not None
    except Exception as e:
        console.print(f"  [dim]{type(e).__name__}: {e}[/]")
        return False


def _test_ollama_connection(base_url: str) -> bool:
    """Test connectivity to Ollama."""
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        console.print(f"  [dim]{type(e).__name__}: {e}[/]")
        return False


def _write_config(
    db_provider: str,
    db_host: str,
    db_port: str,
    db_user: str,
    db_pass: str,
    use_ssl: bool,
    verify_certs: bool,
    logs_index: str,
    anomaly_index: str,
    vector_index: str,
    llm_provider: str,
    llm_config: dict,
    api_keys: dict = None,
) -> None:
    """Update config.yaml and .env with user settings (Credentials only in .env)."""
    if api_keys is None:
        api_keys = {}
    
    config_path = Path(__file__).parent / "config.yaml"
    example_path = Path(__file__).parent / "config.yaml.example"
    env_path = Path(__file__).parent / ".env"

    # Read existing config, or use example as template if config doesn't exist
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif example_path.exists():
        with open(example_path) as f:
            config = yaml.safe_load(f)
    else:
        # Fallback: create minimal config structure
        config = {
            "agent": {"name": "SecurityClaw", "version": "1.0.0", "skills_dir": "skills", "log_level": "INFO"},
            "scheduler": {"heartbeat_interval_seconds": 60, "memory_build_interval_hours": 6},
            "db": {"provider": "opensearch", "index_prefix": "securityclaw"},
            "llm": {"provider": "ollama"},
            "rag": {"embedding_model": "all-MiniLM-L6-v2", "top_k": 5, "similarity_threshold": 0.65},
        }

    # Update DB section (NO credentials in config — only in .env)
    config["db"]["provider"] = db_provider
    config["db"]["host"] = db_host
    config["db"]["port"] = int(db_port)
    config["db"]["use_ssl"] = use_ssl
    config["db"]["verify_certs"] = verify_certs
    config["db"]["logs_index"] = logs_index
    config["db"]["anomaly_index"] = anomaly_index
    config["db"]["vector_index"] = vector_index

    # Update LLM section
    config["llm"]["provider"] = "ollama"
    for key, val in llm_config.items():
        config["llm"][key] = val

    # Write config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    console.print(f"  [dim]Written to {config_path.name}[/]")

    # Write .env with credentials AND API keys
    env_lines = []
    if db_user:
        env_lines.append(f"DB_USERNAME={db_user}")
    if db_pass:
        env_lines.append(f"DB_PASSWORD={db_pass}")
    
    # Add external API keys
    for api_key_name, api_key_value in api_keys.items():
        if api_key_value:
            env_lines.append(f"{api_key_name}={api_key_value}")

    if env_lines:
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        console.print(f"  [dim]Written to {env_path.name} (credentials)[/]")

    # Clear the Config singleton so it reloads on next use
    Config.reset()


if __name__ == "__main__":
    cli()
