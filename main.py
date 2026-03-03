"""
main.py — SecurityClaw CLI entrypoint.

Usage:
    python main.py onboard              # Interactive setup wizard
    python main.py run                  # Start full agent loop
    python main.py dispatch <skill>     # Fire a skill once
    python main.py chat                 # Interactive chat with skill routing
    python main.py status               # Print SITUATION.md
    python main.py list-skills          # List discovered skills
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

from core.config import Config
from core.db_connector import OpenSearchConnector
from core.llm_provider import build_llm_provider
from core.memory import AgentMemory
from core.runner import Runner

console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    console.print("[bold green]Step 2: LLM Provider Configuration[/]\n")
    llm_provider = Prompt.ask(
        "LLM provider",
        choices=["ollama", "openai"],
        default="ollama"
    )

    llm_config: dict = {}
    if llm_provider == "ollama":
        ollama_url = Prompt.ask("Ollama base URL", default="http://localhost:11434")
        ollama_model = Prompt.ask("Ollama chat model name", default="llama3")
        console.print(
            "[dim]The embed model is used exclusively for RAG vector embeddings (not for chat).\n"
            "Use a small, fast model such as [italic]tinyllama[/italic] or [italic]nomic-embed-text[/italic].[/]"
        )
        ollama_embed_model = Prompt.ask("Ollama embed model name", default="tinyllama")
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
    else:
        openai_key = Prompt.ask("OpenAI API Key", password=True)
        openai_model = Prompt.ask("OpenAI model", default="gpt-4o")
        llm_config = {
            "openai_api_key_env": "OPENAI_API_KEY",
            "openai_model": openai_model,
        }
        # Store API key in env var (we'll write to .env)
        os.environ["OPENAI_API_KEY"] = openai_key

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: Write Configuration & Create Initial Files
    # ──────────────────────────────────────────────────────────────────────────
    console.print("[bold green]Step 3: Saving Configuration[/]\n")
    _write_config(
        db_provider, db_host, db_port, db_user, db_pass, use_ssl, verify_certs,
        logs_index, anomaly_index, vector_index,
        llm_provider, llm_config
    )

    # Create SITUATION.md if it doesn't exist
    _create_situation_file()

    console.print("[green bold]✓ Configuration complete![/]")
    console.print("\n[cyan]You can now run:[/]")
    console.print("  [yellow]python main.py run[/]              # Start the agent")
    console.print("  [yellow]python main.py list-skills[/]      # See available skills")
    console.print("  [yellow]python main.py dispatch <skill>[/] # Fire a skill\n")


@cli.command()
def run():
    """Start the full SecurityClaw agent loop."""
    runner = _build_runner()
    runner.setup()
    runner.run()


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
    """Print the current SITUATION.md to terminal."""
    memory = AgentMemory()
    console.print(Markdown(memory.read()))


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
        interval = skill.schedule_interval_seconds or "?"
        console.print(f"  [cyan]{name}[/] — every [magenta]{interval}s[/]")


@cli.command()
def chat():
    """Interactive chat with the SOC agent—ask questions and route to skills."""
    from pathlib import Path
    from datetime import datetime
    from skills.chat_router.logic import (
        route_question,
        execute_skill_workflow,
        format_response,
        load_conversation_history,
        add_to_history,
        get_context_summary,
        list_conversations,
    )
    from core.skill_loader import SkillLoader

    cfg = Config()
    db = OpenSearchConnector()
    llm = build_llm_provider(cfg)
    memory = AgentMemory()
    from core.runner import Runner
    runner = Runner(db_connector=db, llm_provider=llm)
    runner.setup()

    # Load chat_router skill instruction
    instruction_path = Path(__file__).parent / "skills" / "chat_router" / "instruction.md"
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

    # Conversation management
    import uuid
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

            # Route the question
            console.print("\n[dim]Analyzing question...[/]")
            routing = route_question(user_input, available_skills, llm, instruction)

            console.print(f"[dim]→ Reasoning: {routing.get('reasoning', 'N/A')}[/]")

            # Execute skill workflow
            if routing.get("skills"):
                skills_str = ", ".join(routing["skills"])
                console.print(f"[dim]→ Skills: {skills_str}[/]\n")

                skill_results = execute_skill_workflow(routing["skills"], runner, {}, routing)

                # Format response
                response = format_response(user_input, routing, skill_results, llm)
            else:
                # Direct response without skills
                console.print("[dim]→ No skills needed\n[/]")
                response = format_response(user_input, routing, {}, llm)

            console.print(f"[bold green]Agent[/]: {response}\n")

            # Save to history
            add_to_history(conversation_id, user_input, response, routing, 
                          skill_results if routing.get("skills") else {})

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            logger.exception("Chat error")


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
) -> None:
    """Update config.yaml and .env with user settings(Credentials only in .env)."""
    config_path = Path(__file__).parent / "config.yaml"
    env_path = Path(__file__).parent / ".env"

    # Read existing config
    with open(config_path) as f:
        config = yaml.safe_load(f)

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
    config["llm"]["provider"] = llm_provider
    for key, val in llm_config.items():
        if key != "openai_api_key_env":  # Don't write the env var name to config
            config["llm"][key] = val

    # Write config.yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    console.print(f"  [dim]Written to {config_path.name}[/]")

    # Write .env with credentials
    env_lines = []
    if db_user:
        env_lines.append(f"DB_USERNAME={db_user}")
    if db_pass:
        env_lines.append(f"DB_PASSWORD={db_pass}")
    if llm_provider == "openai" and "OPENAI_API_KEY" in os.environ:
        env_lines.append(f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}")

    if env_lines:
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        console.print(f"  [dim]Written to {env_path.name} (credentials)[/]")

    # Clear the Config singleton so it reloads on next use
    Config.reset()


def _create_situation_file() -> None:
    """Create SITUATION.md with initialization status if it doesn't exist."""
    situation_path = Path(__file__).parent / "SITUATION.md"
    
    if situation_path.exists():
        return  # Don't overwrite existing
    
    initial_situation = """# SecurityClaw Situation Report

## Status
Agent initialized and ready.

## Current Focus
- None (awaiting input)

## Recent Findings
- System online
- RAG index ready
- Baseline generation available

## Notes
- This file is automatically updated by the agent as it runs
- Key findings and escalations tracked here
- Regenerated on each major discovery or escalation
"""
    
    situation_path.write_text(initial_situation, encoding="utf-8")
    console.print(f"  [dim]Created {situation_path.name}[/]")


if __name__ == "__main__":
    cli()
