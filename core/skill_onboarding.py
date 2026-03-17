"""
core/skill_onboarding.py — Dynamic skill variable discovery and onboarding.

Scans skills for required environment variables, tracks onboarding state,
and prompts for missing variables on first chat.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONBOARDING_STATE_PATH = _ROOT / ".onboarding_state.json"

console = Console()


@staticmethod
def _load_onboarding_state() -> dict[str, Any]:
    """Load the onboarding state tracking file."""
    if not _ONBOARDING_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_ONBOARDING_STATE_PATH.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _save_onboarding_state(state: dict[str, Any]) -> None:
    """Save the onboarding state tracking file."""
    _ONBOARDING_STATE_PATH.write_text(json.dumps(state, indent=2))


def discover_skill_requirements() -> dict[str, dict[str, Any]]:
    """
    Scan all skills for required environment variables.
    
    Each skill can declare required variables in manifest.yaml:
    
      required_env_vars:
        - name: MY_VAR
          description: "What this variable is for"
          env_key: MY_VAR  (optional, defaults to name)
    
    Returns:
        {skill_name: {var_name: {description, env_key, ...}}}
    """
    skills_dir = _ROOT / "skills"
    requirements: dict[str, dict[str, Any]] = {}

    if not skills_dir.exists():
        return requirements

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue

        try:
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
            required_vars = manifest.get("required_env_vars", [])

            if required_vars:
                skill_name = skill_dir.name
                requirements[skill_name] = {}
                for var_spec in required_vars:
                    if isinstance(var_spec, dict):
                        var_name = var_spec.get("name")
                        if var_name:
                            requirements[skill_name][var_name] = var_spec
                    elif isinstance(var_spec, str):
                        # Simple string format: just the variable name
                        requirements[skill_name][var_spec] = {"name": var_spec, "env_key": var_spec}
        except (yaml.YAMLError, IOError) as e:
            logger.warning(f"Error reading manifest for {skill_dir.name}: {e}")

    return requirements


def get_missing_skill_variables() -> dict[str, list[str]]:
    """
    Find which skill variables are required but not yet set.
    
    Returns:
        {skill_name: [missing_var_names]}
    """
    load_dotenv()  # Reload .env to get latest vars
    requirements = discover_skill_requirements()
    missing: dict[str, list[str]] = {}

    for skill_name, var_specs in requirements.items():
        missing_vars = []
        for var_name, var_spec in var_specs.items():
            env_key = var_spec.get("env_key", var_name)
            if not os.getenv(env_key):
                missing_vars.append(var_name)

        if missing_vars:
            missing[skill_name] = missing_vars

    return missing


def prompt_for_skill_variables(skill_requirements: dict[str, dict[str, Any]]) -> dict[str, str]:
    """
    Interactively prompt user for missing skill variables.
    
    Returns:
        {env_key: value}
    """
    collected: dict[str, str] = {}

    for skill_name, var_specs in skill_requirements.items():
        console.print(f"\n[bold cyan]{skill_name}[/]")
        
        for var_name, var_spec in var_specs.items():
            env_key = var_spec.get("env_key", var_name)
            description = var_spec.get("description", var_name)
            is_secret = var_spec.get("is_secret", False)

            prompt_text = f"{description}"
            if var_spec.get("optional"):
                prompt_text += " [optional]"

            value = Prompt.ask(prompt_text, default="", show_default=False, password=is_secret)
            if value:
                collected[env_key] = value

    return collected


def ensure_skill_variables_onboarded() -> None:
    """
    Check if all skill variables are configured.
    If not, prompt user to onboard them (on first chat only).
    """
    load_dotenv()

    # Never trigger interactive onboarding in CI or other non-interactive runs.
    # Chat integration tests pipe stdin, which is not a TTY, so prompting here
    # would consume the scripted chat input and abort the subprocess.
    if os.getenv("CI") or os.getenv("SECURITYCLAW_SKIP_SKILL_ONBOARDING"):
        return
    if not sys.stdin or not sys.stdin.isatty():
        return
    
    # Check onboarding state
    state = _load_onboarding_state()
    
    # Discover skill requirements
    requirements = discover_skill_requirements()
    
    # Find missing variables
    missing_by_skill: dict[str, list[str]] = {}
    for skill_name, var_specs in requirements.items():
        missing_vars = []
        for var_name, var_spec in var_specs.items():
            is_optional = var_spec.get("optional", False)
            if not is_optional:
                env_key = var_spec.get("env_key", var_name)
                if not os.getenv(env_key):
                    missing_vars.append(var_name)

        if missing_vars:
            # Check if this skill has been onboarded before
            if skill_name not in state.get("skills_onboarded", []):
                missing_by_skill[skill_name] = missing_vars

    if missing_by_skill:
        console.print(
            "\n[bold yellow]⚠ Missing Required Skill Variables[/]\n"
            "[dim]Some skills need additional configuration. "
            "You can configure them now or skip for later.[/]\n"
        )
        
        # Build requirements dict for only missing items
        prompt_requirements = {}
        for skill_name, missing_vars in missing_by_skill.items():
            requirements_entry = requirements.get(skill_name, {})
            prompt_requirements[skill_name] = {
                var: requirements_entry[var]
                for var in missing_vars
                if var in requirements_entry
            }

        # Prompt for variables
        if prompt_requirements:
            from rich.prompt import Confirm
            
            setup_now = Confirm.ask("Configure these variables now?", default=True)
            if setup_now:
                collected = prompt_for_skill_variables(prompt_requirements)
                
                # Write to .env
                if collected:
                    _write_env_vars(collected)
                    console.print("[green]✓ Variables saved to .env[/]")
                    
                    # Update onboarding state
                    state.setdefault("skills_onboarded", [])
                    for skill_name in missing_by_skill.keys():
                        if skill_name not in state["skills_onboarded"]:
                            state["skills_onboarded"].append(skill_name)
                    _save_onboarding_state(state)
            else:
                console.print(
                    "[yellow]You can configure these variables later by running:[/]\n"
                    "  [dim]python main.py onboard[/]\n"
                )


def _write_env_vars(vars_dict: dict[str, str]) -> None:
    """Append variables to .env file."""
    env_path = _ROOT / ".env"
    
    # Read existing vars
    existing = {}
    if env_path.exists():
        load_dotenv(env_path)
        existing = {k: v for k, v in os.environ.items() if k.isupper()}
    
    # Update with new values
    existing.update(vars_dict)
    
    # Write back
    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n")
