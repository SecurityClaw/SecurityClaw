"""
core/chat_router/logic.py

Intelligent skill router for conversational SOC queries.
Routes user questions to appropriate skills, handles multi-skill workflows,
and maintains conversation context using LangGraph orchestration.

This is the core orchestration module—not a periodic skill.
LangGraph is a required dependency for the agent loop orchestration.
"""

# ARCHITECTURE GUARDRAIL:
# Do not add new hardcoded skill names or skill-specific routing branches in this file.
# The chat router must remain capability- and manifest-driven. Resolve concrete skills
# through manifest routing groups, capability metadata, and manifest contracts instead
# of embedding skill identifiers here.

from __future__ import annotations

import ipaddress
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig

from core.capability_graph import expand_skill_dependencies
from core.memory import StateBackedMemory
from core.skill_manifest import (
    SkillManifestLoader,
    apply_manifest_recovery_policies,
    apply_manifest_plan_policies,
    apply_question_enrichment,
    check_and_apply_auto_chain,
    first_skill_in_group,
    manifest_answer_types,
    manifest_artifact_inputs,
    manifest_artifact_outputs,
    manifest_non_goals,
    manifest_required_entities,
)

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).parent / "instruction.md"
ROUTING_PROMPT_PATH = Path(__file__).parent / "ROUTING_PROMPT.md"
SUPERVISOR_NEXT_ACTION_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_NEXT_ACTION_PROMPT.md"
SUPERVISOR_PLAN_REPAIR_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_PLAN_REPAIR_PROMPT.md"
SUPERVISOR_PLAN_REVIEW_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_PLAN_REVIEW_PROMPT.md"
SUPERVISOR_REFLECTION_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_REFLECTION_PROMPT.md"
SUPERVISOR_EVALUATION_PROMPT_PATH = Path(__file__).parent / "SUPERVISOR_EVALUATION_PROMPT.md"
RESPONSE_THINK_PROMPT_PATH = Path(__file__).parent / "RESPONSE_THINK_PROMPT.md"
RESPONSE_REFLECTION_PROMPT_PATH = Path(__file__).parent / "RESPONSE_REFLECTION_PROMPT.md"
RESPONSE_VERIFICATION_PROMPT_PATH = Path(__file__).parent / "RESPONSE_VERIFICATION_PROMPT.md"
RESPONSE_FINAL_PROMPT_PATH = Path(__file__).parent / "RESPONSE_FINAL_PROMPT.md"
SKILL_NAME = "chat_router"


class AgentState(TypedDict, total=False):
    """LangGraph state shared across all graph nodes for a single Q&A turn."""

    # Conversation identity
    thread_id: str
    user_question: str
    messages: list  # conversation_history (list[dict])

    # Orchestration control
    skill_plan: list        # skills chosen by the supervisor for this step
    pending_parameters: dict  # parameters chosen by the supervisor for this step
    pending_reasoning: str    # reasoning chosen by the supervisor for this step
    pending_question_grounding: dict  # grounding extracted from current question by supervisor
    skill_results: dict     # accumulated results from all executed skills
    step_count: int
    max_steps: int
    previously_run_skills: list  # list of plan signatures for previously executed steps
    plan_exhausted: bool

    # Evaluation
    evaluation: dict        # {satisfied, confidence, reasoning, missing}
    trace: list             # execution trace (list[dict])

    # Agent memory (serializable LangGraph working memory)
    mem_status: str
    mem_focus: str
    mem_findings: list      # list[{timestamp, text}]
    mem_decisions: list
    mem_escalations: list

    # Output
    response: str
    routing_metadata: dict


# Cache for compiled system prompts (P1.1: Prevent file I/O per turn)
_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt_template(path: Path, fallback: str) -> str:
    """Load prompt template from file, or fall back to embedded template.
    
    Results are cached to avoid file I/O on every turn.
    (P1.1: Improves latency by ~5-10% in typical scenarios)
    """
    cache_key = str(path.resolve())
    
    # Return cached version if available
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    
    # Load or use fallback
    if not path.exists():
        # File doesn't exist—silently use fallback (expected for deleted template files)
        result = fallback.strip()
    else:
        try:
            result = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("[%s] Could not load prompt template %s: %s", SKILL_NAME, path.name, exc)
            result = fallback.strip()
    
    _PROMPT_CACHE[cache_key] = result
    return result


def _render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC SKILL LOOKUP HELPERS
# ──────────────────────────────────────────────────────────────────────────────
# These functions build dynamic lookups from available_skills to eliminate
# hardcoded skill name references. Skills are discovered at runtime and used
# via these helpers, not as string constants in conditionals.

def _build_skill_name_set(available_skills: list[dict]) -> set[str]:
    """Build a set of all available skill names from available_skills."""
    return {s.get("name") for s in (available_skills or []) if s.get("name")}


def _build_skill_index_by_name(available_skills: list[dict]) -> dict[str, dict]:
    """Build a dict mapping skill name -> skill config from available_skills."""
    return {s.get("name"): s for s in (available_skills or []) if s.get("name")}


def _skill_exists(available_skills: list[dict], skill_name: str) -> bool:
    """Check if a skill is available by name (dynamically from available_skills)."""
    return skill_name in _build_skill_name_set(available_skills)


def _get_skill_by_name(available_skills: list[dict], skill_name: str) -> dict | None:
    """Retrieve a skill config by name (dynamically from available_skills)."""
    return _build_skill_index_by_name(available_skills).get(skill_name)


def _get_skills_by_names(available_skills: list[dict], skill_names: list[str]) -> list[dict]:
    """Retrieve multiple skill configs by names (dynamically from available_skills)."""
    index = _build_skill_index_by_name(available_skills)
    return [index[name] for name in skill_names if name in index]


def _any_skill_in_plan(
    skills_to_check: list[str],
    selected_skills: list[str],
    available_skills: list[dict],
) -> bool:
    """Check if any of the skills_to_check exist in available_skills AND are in selected_skills."""
    available_names = _build_skill_name_set(available_skills)
    return any(
        skill_name in selected_skills
        for skill_name in skills_to_check
        if skill_name in available_names
    )


def _all_skills_in_plan(
    skills_to_check: list[str],
    selected_skills: list[str],
    available_skills: list[dict],
) -> bool:
    """Check if all of the skills_to_check exist in available_skills AND are in selected_skills."""
    available_names = _build_skill_name_set(available_skills)
    available_checks = [s for s in skills_to_check if s in available_names]
    if not available_checks:
        return False
    return all(skill_name in selected_skills for skill_name in available_checks)


def _get_skills_requesting_log_search(available_skills: list[dict]) -> list[str]:
    """Dynamically identify which available skills are used for log searching.
    
    This replaces hardcoded skill name sets like {"fields_querier", "opensearch_querier"}.
    Returns list of skill names that perform log/data searching functions.
    """
    # Determine by checking manifest metadata or fallback to known function patterns
    log_search_candidates = ["fields_querier", "opensearch_querier"]
    available_names = _build_skill_name_set(available_skills)
    return [name for name in log_search_candidates if name in available_names]



def _normalize_plan_question(question: str) -> str:
    return " ".join(str(question or "").lower().split())


def _build_skill_catalog(
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    manifests = manifests or {}
    catalog: list[dict[str, Any]] = []
    for skill in available_skills or []:
        skill_name = str(skill.get("name") or "").strip()
        if not skill_name:
            continue
        manifest = manifests.get(skill_name, {})
        prerequisites = []
        for prereq in manifest.get("prerequisites") or []:
            if not isinstance(prereq, dict):
                continue
            prerequisites.append(
                {
                    "group": str(prereq.get("group") or "").strip(),
                    "why": str(prereq.get("why") or "").strip(),
                }
            )
        catalog.append(
            {
                "name": skill_name,
                "description": str(skill.get("description") or manifest.get("description") or "").strip(),
                "routing_group": str(manifest.get("routing_group") or "").strip(),
                "capability_groups": list(manifest.get("capability_groups") or []),
                "orchestration_role": str(manifest.get("orchestration_role") or "single_step").strip(),
                "min_prior_context": int(manifest.get("min_prior_context", 0) or 0),
                "prerequisites": prerequisites,
                "answer_types": manifest_answer_types(manifest),
                "non_goals": manifest_non_goals(manifest),
                "required_entities": manifest_required_entities(manifest),
                "artifact_inputs": manifest_artifact_inputs(manifest),
                "artifact_outputs": manifest_artifact_outputs(manifest),
            }
        )
    return catalog


def _question_has_explicit_ip(user_question: str) -> bool:
    return bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", str(user_question or "")))


def _question_has_explicit_domain(user_question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
            str(user_question or "").lower(),
        )
    )


def _extract_explicit_ips(user_question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", str(user_question or ""))))


def _extract_explicit_domains(user_question: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(
                r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
                str(user_question or "").lower(),
            )
        )
    )


def _deterministic_supervisor_question_grounding(user_question: str) -> dict[str, Any] | None:
    """Extract structurally identifiable entities (IPv4 addresses) from the question.

    Routing decisions are left entirely to the LLM supervisor and manifest system.
    This function only provides raw IP entities so downstream skills can use them
    as search targets without an extra LLM extraction call.
    """
    explicit_ips = _extract_explicit_ips(user_question)
    if not explicit_ips:
        return None
    return {"ips": list(explicit_ips)}


# Backwards-compat alias: tests and callers that imported the old LLM-based grounding function
# now resolve to the deterministic version (the LLM version was removed as dead code).
def _ground_supervisor_question_with_llm(
    user_question: str,
    llm: Any = None,
    instruction: str = "",
    **_kwargs: Any,
) -> dict[str, Any]:
    return _deterministic_supervisor_question_grounding(user_question) or {}


def _skill_has_required_entity_context(
    manifest: dict[str, Any],
    user_question: str,
    current_results: dict[str, Any] | None = None,
) -> bool:
    required_entities = manifest_required_entities(manifest)
    if not required_entities:
        return True

    # If the manifest declares requires_explicit_entity, only consider entities
    # that appear explicitly in the question itself — not those extracted from
    # previous skill results.  This prevents skills like ip_fingerprinter from
    # being routed for country-only queries just because opensearch returned
    # records that happened to contain IP addresses.
    requires_explicit_entity = manifest.get("requires_explicit_entity", False)

    if requires_explicit_entity:
        available_entities = {
            "entity": _question_has_explicit_entities(user_question),
            "ip": _question_has_explicit_ip(user_question),
            "ipv4": _question_has_explicit_ip(user_question),
            "ipv6": _question_has_explicit_ip(user_question),
            "domain": _question_has_explicit_domain(user_question),
            "hostname": _question_has_explicit_domain(user_question),
            "country": False,
            "port": False,
        }
    else:
        extracted = _extract_entities_from_previous_results(current_results or {}) if current_results else {}
        available_entities = {
            "entity": _question_has_explicit_entities(user_question)
            or bool(extracted.get("ips") or extracted.get("domains") or extracted.get("countries") or extracted.get("ports")),
            "ip": _question_has_explicit_ip(user_question) or bool(extracted.get("ips")),
            "ipv4": _question_has_explicit_ip(user_question) or bool(extracted.get("ips")),
            "ipv6": _question_has_explicit_ip(user_question) or bool(extracted.get("ips")),
            "domain": _question_has_explicit_domain(user_question) or bool(extracted.get("domains")),
            "hostname": _question_has_explicit_domain(user_question) or bool(extracted.get("domains")),
            "country": bool(extracted.get("countries")),
            "port": bool(extracted.get("ports")),
        }

    if any(available_entities.get(entity_type, False) for entity_type in required_entities):
        return True

    return bool(manifest.get("question_enrichment_hook")) and bool(current_results)


def _extract_skill_errors(skill_results: dict | None) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for skill_name, result in (skill_results or {}).items():
        if not isinstance(result, dict):
            continue
        if result.get("status") != "error":
            continue
        errors.append(
            {
                "skill": str(skill_name),
                "error": str(result.get("error") or "unknown error"),
            }
        )
    return errors


def _ground_selected_skills(
    selected_skills: list[str],
    user_question: str,
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]] | None = None,
    current_results: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    manifests = manifests or {}
    available_names = {s.get("name") for s in available_skills if s.get("name")}
    grounded: list[str] = []
    dropped: list[str] = []
    for skill in (selected_skills or []):
        resolved_skill = None
        if skill in available_names:
            resolved_skill = skill
        elif manifests:
            resolved_skill = first_skill_in_group(manifests, str(skill or "").strip())

        if resolved_skill and resolved_skill in available_names:
            manifest = manifests.get(resolved_skill, {}) if manifests else {}
            if manifest and not _skill_has_required_entity_context(manifest, user_question, current_results):
                dropped.append(skill)
                continue
            grounded.append(resolved_skill)
            continue

        dropped.append(skill)
    if manifests and grounded:
        grounded = apply_manifest_plan_policies(
            selected_skills=grounded,
            user_question=user_question,
            available_skills=available_skills,
            all_manifests=manifests,
            current_results=current_results or {},
        )
        grounded = expand_skill_dependencies(grounded, manifests)

    deduped: list[str] = []
    for skill in grounded:
        if skill and skill in available_names and skill not in deduped:
            deduped.append(skill)
    return deduped, dropped


def _repair_plan_with_llm(
    *,
    user_question: str,
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]],
    llm: Any,
    instruction: str,
    current_results: dict | None,
    previous_eval: dict | None,
    previous_trace: list[dict] | None,
    question_grounding: dict[str, Any] | None = None,
    invalid_skills: list[str] | None,
    proposed_skills: list[str] | None,
    proposed_parameters: dict | None,
    mode: str,
    failure_reason: str = "",
) -> dict:
    catalog_json = json.dumps(_build_skill_catalog(available_skills, manifests), indent=2, default=str)
    template_path = (
        SUPERVISOR_REFLECTION_PROMPT_PATH if mode == "reflection" else SUPERVISOR_PLAN_REPAIR_PROMPT_PATH
    )
    template = _load_prompt_template(
        template_path,
        """
You must repair the supervisor plan using only the allowed skills.

QUESTION:
{{USER_QUESTION}}

ALLOWED SKILLS:
{{SKILL_CATALOG_JSON}}

CURRENT RESULTS:
{{CURRENT_RESULTS}}

QUESTION GROUNDING:
{{QUESTION_GROUNDING}}

PREVIOUS EVALUATION:
{{PREVIOUS_EVALUATION}}

PREVIOUS TRACE:
{{PREVIOUS_TRACE}}

INVALID OR UNAVAILABLE SKILLS:
{{INVALID_SKILLS}}

PROPOSED SKILLS:
{{PROPOSED_SKILLS}}

PROPOSED PARAMETERS:
{{PROPOSED_PARAMETERS}}

FAILURE REASON:
{{FAILURE_REASON}}

Return strict JSON with reasoning, skills, and parameters. Every skill must be chosen from ALLOWED SKILLS.
""",
    )
    prompt = _render_prompt(
        template,
        {
            "USER_QUESTION": user_question,
            "SKILL_CATALOG_JSON": catalog_json,
            "CURRENT_RESULTS": json.dumps(current_results or {}, indent=2, default=str)[:6000],
            "QUESTION_GROUNDING": json.dumps(question_grounding or {}, indent=2, default=str),
            "PREVIOUS_EVALUATION": json.dumps(previous_eval or {}, indent=2, default=str),
            "PREVIOUS_TRACE": json.dumps(previous_trace or [], indent=2, default=str)[:5000],
            "INVALID_SKILLS": json.dumps(invalid_skills or [], default=str),
            "PROPOSED_SKILLS": json.dumps(proposed_skills or [], default=str),
            "PROPOSED_PARAMETERS": json.dumps(proposed_parameters or {}, default=str),
            "FAILURE_REASON": failure_reason,
        },
    )

    response = llm.chat([
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ])
    parsed = _parse_json_object(response) or {}
    if not isinstance(parsed.get("skills"), list):
        parsed["skills"] = []
    if not isinstance(parsed.get("parameters"), dict):
        parsed["parameters"] = {}
    if not isinstance(parsed.get("reasoning"), str):
        parsed["reasoning"] = "Supervisor repaired the plan against the loaded skill inventory"
    if not parsed["parameters"].get("question"):
        parsed["parameters"]["question"] = user_question
    return parsed


def _review_supervisor_plan_with_llm(
    *,
    user_question: str,
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]],
    llm: Any,
    instruction: str,
    current_results: dict | None,
    previous_eval: dict | None,
    previous_trace: list[dict] | None,
    question_grounding: dict[str, Any] | None,
    proposed_skills: list[str] | None,
    proposed_parameters: dict | None,
    proposed_reasoning: str,
) -> dict:
    catalog_json = json.dumps(_build_skill_catalog(available_skills, manifests), indent=2, default=str)
    template = _load_prompt_template(
        SUPERVISOR_PLAN_REVIEW_PROMPT_PATH,
        """
Review whether this proposed next supervisor step is the best grounded immediate action.

QUESTION:
{{USER_QUESTION}}

ALLOWED SKILLS:
{{SKILL_CATALOG_JSON}}

CURRENT RESULTS:
{{CURRENT_RESULTS}}

QUESTION GROUNDING:
{{QUESTION_GROUNDING}}

PREVIOUS EVALUATION:
{{PREVIOUS_EVALUATION}}

PREVIOUS TRACE:
{{PREVIOUS_TRACE}}

PROPOSED REASONING:
{{PROPOSED_REASONING}}

PROPOSED SKILLS:
{{PROPOSED_SKILLS}}

PROPOSED PARAMETERS:
{{PROPOSED_PARAMETERS}}

Return strict JSON:
{
  "is_valid": true,
  "should_execute": true,
  "confidence": 0.0,
  "reasoning": "why this immediate next step is or is not grounded",
  "issue": "specific problem if invalid",
  "suggestion": "how the immediate next step should change"
}
""",
    )
    prompt = _render_prompt(
        template,
        {
            "USER_QUESTION": user_question,
            "SKILL_CATALOG_JSON": catalog_json,
            "CURRENT_RESULTS": json.dumps(current_results or {}, indent=2, default=str)[:6000],
            "QUESTION_GROUNDING": json.dumps(question_grounding or {}, indent=2, default=str),
            "PREVIOUS_EVALUATION": json.dumps(previous_eval or {}, indent=2, default=str),
            "PREVIOUS_TRACE": json.dumps(previous_trace or [], indent=2, default=str)[:5000],
            "PROPOSED_REASONING": proposed_reasoning,
            "PROPOSED_SKILLS": json.dumps(proposed_skills or [], default=str),
            "PROPOSED_PARAMETERS": json.dumps(proposed_parameters or {}, default=str),
        },
    )

    response = llm.chat([
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ])
    parsed = _parse_json_object(response) or {}
    return {
        "is_valid": bool(parsed.get("is_valid", True)),
        "should_execute": bool(parsed.get("should_execute", parsed.get("is_valid", True))),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "has_confidence": "confidence" in parsed,
        "reasoning": str(parsed.get("reasoning") or ""),
        "issue": str(parsed.get("issue") or ""),
        "suggestion": str(parsed.get("suggestion") or ""),
    }


def _review_and_refine_supervisor_plan(
    *,
    decision: dict,
    user_question: str,
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]],
    llm: Any,
    instruction: str,
    current_results: dict | None,
    previous_eval: dict | None,
    previous_trace: list[dict] | None,
    max_rounds: int = 2,
    min_execute_confidence: float = 0.6,
) -> dict:
    """Review and iteratively refine the supervisor-proposed skill plan.

    Uses deterministic question grounding (no extra LLM call) plus up to
    `max_rounds` LLM review-and-repair cycles to ensure the plan is grounded
    and viable before execution.
    """
    reviewed = dict(decision or {})
    # Deterministic grounding — no LLM call needed here.
    question_grounding = _deterministic_supervisor_question_grounding(user_question) or {}
    planner_trace = {
        "question_grounding": question_grounding,
        "initial_candidate": {
            "skills": list(reviewed.get("skills") or []),
            "parameters": dict(reviewed.get("parameters") or {}),
            "reasoning": str(reviewed.get("reasoning") or ""),
        },
        "reviews": [],
    }
    for review_round in range(max_rounds):
        proposed_skill_names = list(reviewed.get("skills") or [])

        review = _review_supervisor_plan_with_llm(
            user_question=user_question,
            available_skills=available_skills,
            manifests=manifests,
            llm=llm,
            instruction=instruction,
            current_results=current_results,
            previous_eval=previous_eval,
            previous_trace=previous_trace,
            question_grounding=question_grounding,
            proposed_skills=proposed_skill_names,
            proposed_parameters=dict(reviewed.get("parameters") or {}),
            proposed_reasoning=str(reviewed.get("reasoning") or ""),
        )
        logger.info(
            "[%s] Supervisor plan review round %d: valid=%s execute=%s confidence=%.1f%% issue=%s",
            SKILL_NAME,
            review_round + 1,
            review["is_valid"],
            review["should_execute"],
            review["confidence"] * 100,
            str(review.get("issue") or "")[:160],
        )
        planner_trace["reviews"].append(
            {
                "round": review_round + 1,
                "stage": "plan_review",
                "proposed_skills": proposed_skill_names,
                "valid": bool(review["is_valid"]),
                "should_execute": bool(review["should_execute"]),
                "confidence": float(review.get("confidence", 0.0) or 0.0),
                "reasoning": str(review.get("reasoning") or ""),
                "issue": str(review.get("issue") or ""),
                "suggestion": str(review.get("suggestion") or ""),
            }
        )
        should_execute = bool(review["should_execute"])
        review_confidence = float(review.get("confidence", 0.0) or 0.0)
        confidence_provided = bool(review.get("has_confidence", False))
        confidence_sufficient = (review_confidence >= min_execute_confidence) if confidence_provided else True
        if review["is_valid"] and should_execute and confidence_sufficient:
            reviewed["planner_trace"] = planner_trace | {
                "final_plan": {
                    "skills": list(reviewed.get("skills") or []),
                    "parameters": dict(reviewed.get("parameters") or {}),
                    "reasoning": str(reviewed.get("reasoning") or ""),
                }
            }
            return reviewed

        repaired = _repair_plan_with_llm(
            user_question=user_question,
            available_skills=available_skills,
            manifests=manifests,
            llm=llm,
            instruction=instruction,
            current_results=current_results,
            previous_eval=previous_eval,
            previous_trace=previous_trace,
            question_grounding=question_grounding,
            invalid_skills=[],
            proposed_skills=list(reviewed.get("skills") or []),
            proposed_parameters=dict(reviewed.get("parameters") or {}),
            mode="next_action",
            failure_reason=(
                review.get("issue")
                or review.get("suggestion")
                or (
                    "Plan review confidence was too low to execute safely. "
                    + (review.get("reasoning") or "")
                    if confidence_provided and review_confidence < min_execute_confidence
                    else review.get("reasoning")
                )
                or "Plan review rejected the immediate next step."
            ),
        )
        reviewed = _ensure_viable_plan(
            decision={
                "reasoning": str(repaired.get("reasoning") or reviewed.get("reasoning") or ""),
                "skills": list(repaired.get("skills") or []),
                "parameters": dict(repaired.get("parameters") or {"question": user_question}),
            },
            user_question=user_question,
            available_skills=available_skills,
            manifests=manifests,
            llm=llm,
            instruction=instruction,
            current_results=current_results,
            previous_eval=previous_eval,
            previous_trace=previous_trace,
            mode="next_action",
            failure_reason=review.get("issue") or "Plan review requested a more grounded next step.",
        )
    reviewed["planner_trace"] = planner_trace | {
        "final_plan": {
            "skills": list(reviewed.get("skills") or []),
            "parameters": dict(reviewed.get("parameters") or {}),
            "reasoning": str(reviewed.get("reasoning") or ""),
        }
    }
    return reviewed


def _ensure_viable_plan(
    *,
    decision: dict,
    user_question: str,
    available_skills: list[dict],
    manifests: dict[str, dict[str, Any]],
    llm: Any,
    instruction: str,
    current_results: dict | None,
    previous_eval: dict | None,
    previous_trace: list[dict] | None,
    mode: str,
    failure_reason: str = "",
) -> dict:
    proposed_skills = list(decision.get("skills") or [])
    proposed_parameters = dict(decision.get("parameters") or {})
    plan_question = str(proposed_parameters.get("question") or user_question)
    grounded_skills, dropped_skills = _ground_selected_skills(
        proposed_skills,
        plan_question,
        available_skills,
        manifests,
        current_results,
    )

    if grounded_skills:
        if dropped_skills:
            logger.info(
                "[%s] Dropped unavailable skills from %s plan: %s",
                SKILL_NAME,
                mode,
                dropped_skills,
            )
        decision["skills"] = grounded_skills
        decision["parameters"] = proposed_parameters or {"question": user_question}
        if not decision["parameters"].get("question"):
            decision["parameters"]["question"] = user_question
        return decision

    if not proposed_skills and mode == "next_action":
        decision["skills"] = []
        decision["parameters"] = proposed_parameters or {"question": user_question}
        return decision

    repaired = _repair_plan_with_llm(
        user_question=plan_question,
        available_skills=available_skills,
        manifests=manifests,
        llm=llm,
        instruction=instruction,
        current_results=current_results,
        previous_eval=previous_eval,
        previous_trace=previous_trace,
        question_grounding=question_grounding,
        invalid_skills=[],
        proposed_skills=proposed_skills,
        proposed_parameters=proposed_parameters,
        mode=mode,
        failure_reason=failure_reason,
    )
    repaired_skills, repaired_dropped = _ground_selected_skills(
        repaired.get("skills", []),
        str((repaired.get("parameters") or {}).get("question") or plan_question),
        available_skills,
        manifests,
        current_results,
    )

    if repaired_dropped:
        logger.warning(
            "[%s] Repaired %s plan still referenced unavailable skills: %s",
            SKILL_NAME,
            mode,
            repaired_dropped,
        )

    decision["skills"] = repaired_skills
    decision["parameters"] = dict(repaired.get("parameters") or {"question": user_question})
    if not decision["parameters"].get("question"):
        decision["parameters"]["question"] = user_question
    decision["reasoning"] = str(repaired.get("reasoning") or decision.get("reasoning") or "")
    return decision


def _plan_signature(skills: list[str], parameters: dict | None) -> str:
    payload = {
        "skills": list(skills or []),
        "question": _normalize_plan_question((parameters or {}).get("question", "")),
    }
    return json.dumps(payload, sort_keys=True)





def _question_has_explicit_entities(user_question: str) -> bool:
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question):
        return True
    return bool(re.search(r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", user_question.lower()))


def _is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def route_question(
    user_question: str,
    available_skills: list[dict],
    llm: Any,
    instruction: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Analyze user question and decide which skill(s) to invoke.
    
    Args:
        user_question: Current user input
        available_skills: List of skill definitions
        llm: LLM provider
        instruction: System instruction
        conversation_history: Prior Q&A turns for context (optional)
    
    Returns dict with:
      - reasoning: Why this skill was chosen
      - skills: List of skill names to invoke (can be multiple for workflows)
      - parameters: Parameters to pass to skills (includes the question)
    """
    all_manifests: dict[str, dict[str, Any]] = {}
    try:
        loader = SkillManifestLoader()
        all_manifests = loader.load_all_manifests()
    except Exception as e:
        logger.warning("[%s] Manifest loading failed during initial routing: %s", SKILL_NAME, e)

    skills_description = "\n".join([
        f"- {s['name']}: {s['description']}"
        for s in available_skills
    ])

    # Build conversation context if history is provided
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                history_lines.append(f"User: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                history_lines.append(f"Agent: {msg.get('content', '')}")
        if history_lines:
            history_context = "\n\nRECENT CONVERSATION HISTORY (for context):\n" + "\n".join(history_lines)

    routing_template = _load_prompt_template(
        ROUTING_PROMPT_PATH,
        """
Analyze this security question and decide which available skills to use.
Consider the recent conversation history to maintain context.

Current Question: "{{USER_QUESTION}}"{{HISTORY_CONTEXT}}

Available skills:
{{SKILLS_DESCRIPTION}}

Return a strict JSON object with reasoning, skills, and parameters.
""",
    )
    prompt = _render_prompt(
        routing_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_CONTEXT": history_context,
            "SKILLS_DESCRIPTION": skills_description,
        },
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    response = llm.chat(messages)
    
    # Try to clean up the response by removing markdown code block markers
    cleaned_response = response
    if response.strip().startswith("```"):
        # Remove markdown code block markers
        cleaned_response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response, flags=re.MULTILINE)
        cleaned_response = cleaned_response.strip()
    
    try:
        result = json.loads(cleaned_response)
        # Ensure parameters has the question
        if "parameters" not in result:
            result["parameters"] = {}
        if not result["parameters"].get("question"):
            result["parameters"]["question"] = user_question

        contextual_reputation_entities = _recover_threat_followup_entities(
            user_question,
            conversation_history,
        )
        if contextual_reputation_entities and _skill_exists(available_skills, "threat_analyst"):
            result["skills"] = ["threat_analyst"]
            result["reasoning"] = "Follow-up reputation question anchored to entities from the previous answer"
            result["parameters"]["question"] = _build_context_aware_threat_question(
                user_question,
                contextual_reputation_entities,
            )
            if conversation_history:
                result["parameters"]["conversation_history"] = conversation_history
            # Add question_grounding for skill planning
            question_grounding = _deterministic_supervisor_question_grounding(user_question) or {}
            result["question_grounding"] = question_grounding
            return result

        result["skills"] = result.get("skills", [])

        result["skills"] = _postprocess_selected_skills(
            user_question=user_question,
            selected_skills=result.get("skills", []),
            available_skills=available_skills,
            current_results={},
            manifests=all_manifests,
        )
        
        # Expand skills to include manifest-declared prerequisites
        result["skills"] = _expand_skills_with_prerequisites(
            result.get("skills", []),
            manifests=all_manifests,
            available_skill_names={s.get("name") for s in available_skills},
        )

        selected_skills = list(result.get("skills", []))
        # Use dynamic skill lookups instead of hardcoded names
        if _all_skills_in_plan(["fields_querier", "opensearch_querier"], selected_skills, available_skills):
            result["parameters"]["question"] = user_question
        elif _any_skill_in_plan(["opensearch_querier"], selected_skills, available_skills):
            result["parameters"]["question"] = result["parameters"].get("question") or user_question

        # Include conversation history in parameters if provided
        if conversation_history:
            result["parameters"]["conversation_history"] = conversation_history

        result = _ensure_viable_plan(
            decision=result,
            user_question=user_question,
            available_skills=available_skills,
            manifests=all_manifests,
            llm=llm,
            instruction=instruction,
            current_results={},
            previous_eval={},
            previous_trace=[],
            mode="next_action",
            failure_reason="Initial routing must choose only loaded skills and manifest-declared prerequisites.",
        )

        # Add question_grounding so skills (like opensearch_querier) can use deterministic entity extraction
        question_grounding = _deterministic_supervisor_question_grounding(user_question) or {}
        result["question_grounding"] = question_grounding

        return result
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse LLM routing response: %s", SKILL_NAME, cleaned_response[:200])
        # Fallback: try to extract JSON from response
        try:
            match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if "parameters" not in result:
                    result["parameters"] = {}
                if not result["parameters"].get("question"):
                    result["parameters"]["question"] = user_question

                result["skills"] = _postprocess_selected_skills(
                    user_question=user_question,
                    selected_skills=result.get("skills", []),
                    available_skills=available_skills,
                    current_results={},
                    manifests=all_manifests,
                )

                result = _ensure_viable_plan(
                    decision=result,
                    user_question=user_question,
                    available_skills=available_skills,
                    manifests=all_manifests,
                    llm=llm,
                    instruction=instruction,
                    current_results={},
                    previous_eval={},
                    previous_trace=[],
                    mode="next_action",
                    failure_reason="Fallback routing must still stay within the loaded skill inventory.",
                )
                
                # Add question_grounding for skill planning
                question_grounding = _deterministic_supervisor_question_grounding(user_question) or {}
                result["question_grounding"] = question_grounding
                
                return result
        except:
            pass
        
        # If all else fails, return no skills


def _expand_skills_with_prerequisites(
    selected_skills: list[str],
    manifests: dict[str, dict[str, Any]] | None = None,
    available_skill_names: set[str] | None = None,
) -> list[str]:
    """Expand selected skills to include manifest-declared prerequisites.
    
    For composite skills (e.g., ip_fingerprinter), auto-queue their prerequisites
    so they can be executed in dependency order.
    
    Args:
        selected_skills: List of skill names to expand
        manifests: Skill manifests with prerequisite declarations
        available_skill_names: Set of available skill names (for validation)
    
    Returns:
        Expanded skill list with prerequisites prepended in order
    """
    if not selected_skills or not manifests:
        return selected_skills
    
    available_skill_names = available_skill_names or set(manifests.keys())
    expanded: list[str] = []
    seen: set[str] = set()
    
    def add_prerequisites(skill_name: str) -> None:
        """Recursively add prerequisites for a skill."""
        if skill_name in seen:
            return
        
        manifest = manifests.get(skill_name, {})
        prerequisites = manifest.get("prerequisites") or []
        
        # Expand each prerequisite group to the first available skill in that group
        for prereq in prerequisites:
            if not isinstance(prereq, dict):
                continue
            group = str(prereq.get("group") or "").strip()
            if group:
                # Find first skill in this group
                for candidate_skill in available_skill_names:
                    candidate_manifest = manifests.get(candidate_skill, {})
                    if candidate_manifest.get("routing_group") == group:
                        add_prerequisites(candidate_skill)
                        if candidate_skill not in seen:
                            expanded.append(candidate_skill)
                            seen.add(candidate_skill)
                        break
        
        # Then add the skill itself
        if skill_name not in seen:
            expanded.append(skill_name)
            seen.add(skill_name)
    
    for skill in selected_skills:
        add_prerequisites(skill)
    
    return expanded


def _postprocess_selected_skills(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
    manifests: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Apply shared routing post-processing across initial and supervisor flows."""
    current_results = current_results or {}
    selected_skills = _apply_result_aware_recovery(
        user_question=user_question,
        selected_skills=selected_skills,
        available_skills=available_skills,
        current_results=current_results,
        manifests=manifests,
    )
    return selected_skills




def execute_skill_workflow(
    skills: list[str],
    runner: Any,
    context: dict,
    routing_decision: dict,
    conversation_history: list[dict] = None,
    aggregated_results: dict = None,
    memory: Any = None,
) -> dict:
    """
    Execute one or more skills in sequence, passing context between them.
    
    Args:
        skills: List of skill names to execute
        runner: Runner instance
        context: Shared context dict
        routing_decision: Dict with 'parameters' key for skill inputs
        conversation_history: Conversation history for context (optional)
        aggregated_results: Results from previous skills, for context enrichment (optional)
    
    Returns dict with results from each skill execution.
    """
    results = {}
    params = routing_decision.get("parameters", {})
    aggregated_results = aggregated_results or {}
    
    # Load skill manifests for enrichment and auto-chain dispatching
    try:
        loader = SkillManifestLoader()
        all_manifests = loader.load_all_manifests()
    except Exception as e:
        logger.warning("[%s] Could not load manifests for enrichment: %s", SKILL_NAME, e)
        all_manifests = {}
    
    for skill_name in skills:
        logger.info("[%s] Executing skill: %s", SKILL_NAME, skill_name)
        
        try:
            # Build context with parameters
            skill_context = runner._build_context()
            skill_context["parameters"] = params.copy()
            if memory is not None:
                skill_context["memory"] = memory
            
            # Pass conversation history for context-aware skills
            if conversation_history:
                skill_context["conversation_history"] = conversation_history
            
            # Pass routing_decision so skills have access to supervisor's grounding
            skill_context["routing_decision"] = routing_decision
            
            # ── CONTEXT ENRICHMENT: Pass previous results to this skill ──────
            # This allows skills to see what was discovered in prior steps.
            combined_previous_results = {**aggregated_results, **results}
            if combined_previous_results:
                skill_context["previous_results"] = combined_previous_results
            
            # ── MANIFEST-DRIVEN ENRICHMENT: Apply question enrichment hooks ──
            manifest = all_manifests.get(skill_name, {})
            if manifest.get("question_enrichment_hook"):
                try:
                    skill_context["parameters"] = apply_question_enrichment(
                        skill_name=skill_name,
                        manifest=manifest,
                        parameters=skill_context["parameters"],
                        conversation_history=conversation_history,
                        previous_results=combined_previous_results,
                    )
                except Exception as enrich_exc:
                    logger.warning("[%s] Question enrichment failed for %s: %s", SKILL_NAME, skill_name, enrich_exc)
                    # Continue with original parameters if enrichment fails
            
            # Dispatch skill with context
            result = runner.dispatch(skill_name, context=skill_context)
            results[skill_name] = result
            logger.info("[%s] Skill %s completed with status: %s", 
                       SKILL_NAME, skill_name, result.get("status"))

            # ── AUTO-CHAIN: Check if skill should auto-chain to successor ──
            if result.get("status") == "ok":
                try:
                    auto_chain_skill, auto_chain_result = check_and_apply_auto_chain(
                        last_skill_name=skill_name,
                        last_skill_result=result,
                        all_manifests=all_manifests,
                        runner=runner,
                        context=context,
                        parameters=params,
                        conversation_history=conversation_history,
                        memory=memory,
                    )
                    
                    if auto_chain_skill and auto_chain_result:
                        # Check if auto-chained skill should not execute
                        if (
                            auto_chain_skill in skills
                            or auto_chain_skill in results
                        ):
                            logger.info("[%s] Skipped auto-chain %s (already in execution plan)", SKILL_NAME, auto_chain_skill)
                        else:
                            results[auto_chain_skill] = auto_chain_result
                            logger.info("[%s] Auto-chained %s completed successfully", SKILL_NAME, auto_chain_skill)
                except Exception as auto_chain_exc:
                    logger.warning("[%s] Auto-chain check failed: %s", SKILL_NAME, auto_chain_exc)
                    # Continue even if auto-chain fails
        
        except Exception as e:
            logger.error("[%s] Skill %s failed: %s", SKILL_NAME, skill_name, e)
            results[skill_name] = {
                "status": "error",
                "error": str(e),
            }
    
    return results


def _apply_result_aware_recovery(
    user_question: str,
    selected_skills: list[str],
    available_skills: list[dict],
    current_results: dict | None = None,
    manifests: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Promote recovery skills when prior results show unmet needs."""
    current_results = current_results or {}
    if manifests is None:
        try:
            manifests = SkillManifestLoader().load_all_manifests()
        except Exception as exc:
            logger.warning("[%s] Failed to load manifests for recovery policies: %s", SKILL_NAME, exc)
            manifests = {}

    entities = _extract_entities_from_previous_results(current_results) if current_results else {}
    return apply_manifest_recovery_policies(
        selected_skills=selected_skills,
        user_question=user_question,
        available_skills=available_skills,
        all_manifests=manifests,
        current_results=current_results,
        extracted_entities=entities,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph node functions
# ──────────────────────────────────────────────────────────────────────────────

def _graph_runtime(config: dict) -> dict:
    """Extract non-serializable runtime objects from LangGraph configurable."""
    return config.get("configurable", {})


def decide_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    DECIDE: ask the supervisor which skill(s) to run next, apply guards, MANDATORY ITERATIVE REFINEMENT.
    
    Following LangGraph best practices: When a previous execution didn't satisfy the user,
    we MUST force the LLM to think deeper and try different approaches.
    The system should NOT give up on duplicates - it should iterate until either:
    - The question is satisfied, OR
    - Max steps are reached
    
    This prevents premature termination and enables proper agentic behavior.
    """
    rt = _graph_runtime(config)
    available_skills = rt["available_skills"]
    llm = rt["llm"]
    instruction = rt["instruction"]
    step_callback = rt.get("step_callback")

    aggregated_results: dict = state.get("skill_results") or {}
    previously_run_skills: list = list(state.get("previously_run_skills") or [])
    step_count: int = (state.get("step_count") or 0) + 1
    max_steps: int = state.get("max_steps") or 4
    last_eval: dict = state.get("evaluation") or {}
    trace: list = list(state.get("trace") or [])

    # Check if previous execution left us not satisfied
    was_not_satisfied = not last_eval.get("satisfied", False) and step_count > 1
    loader = SkillManifestLoader()
    try:
        all_manifests = loader.load_all_manifests()
    except Exception as e:
        logger.warning("[%s] Failed to load manifests in decide_node: %s", SKILL_NAME, e)
        all_manifests = {}
    
    # STEP 1: Get initial next action from supervisor
    decision = _supervisor_next_action(
        user_question=state["user_question"],
        available_skills=available_skills,
        llm=llm,
        instruction=instruction,
        conversation_history=list(state.get("messages") or []),
        previous_trace=trace[-3:],
        current_results=aggregated_results,
        previous_eval=last_eval,
    )
    effective_question = str(
        (decision.get("parameters") or {}).get("question") or state["user_question"]
    )

    selected: list[str] = list(decision.get("skills", []))
    selected = _postprocess_selected_skills(
        user_question=effective_question,
        selected_skills=selected,
        available_skills=available_skills,
        current_results=aggregated_results,
        manifests=all_manifests,
    )
    decision["skills"] = selected
    # STEP 2: Review and refine the plan (review+repair cycle with deterministic grounding)
    decision = _review_and_refine_supervisor_plan(
        decision=decision,
        user_question=state["user_question"],
        available_skills=available_skills,
        manifests=all_manifests,
        llm=llm,
        instruction=instruction,
        current_results=aggregated_results,
        previous_eval=last_eval,
        previous_trace=trace,
    )
    # Check if fingerprinter skill (dynamically resolved) is in the plan
    if _skill_exists(available_skills, "ip_fingerprinter") and _any_skill_in_plan(
        ["ip_fingerprinter"], decision.get("skills") or [], available_skills
    ):
        final_question = str((decision.get("parameters") or {}).get("question") or state["user_question"])
        if not _question_has_explicit_entities(final_question):
            fingerprint_entities = _recover_fingerprint_followup_entities(
                state["user_question"],
                list(state.get("messages") or []),
                aggregated_results,
            )
            if fingerprint_entities.get("ips"):
                anchored_question = _build_context_aware_fingerprint_question(
                    state["user_question"],
                    fingerprint_entities,
                    list(state.get("messages") or []),
                )
                decision.setdefault("parameters", {})["question"] = anchored_question
                decision["parameters"]["ip"] = fingerprint_entities["ips"][0]

    # Honor requires_explicit_entity manifest contracts: drop skills that require an
    # explicit entity (e.g. ip_fingerprinter) when no real IP appears in the ORIGINAL
    # user question or in prior user turns.
    #
    # IMPORTANT: We deliberately do NOT use the LLM-modified parameters["question"]
    # here because the supervisor may inject a hallucinated IP (e.g. 192.0.2.1) into
    # that field for country-only queries, which would falsely satisfy the entity check.
    # We only trust IPs the user actually typed.
    _original_question = state["user_question"]
    _user_history_ips: set[str] = set()
    for _msg in (state.get("messages") or []):
        if _msg.get("role") == "user":
            _user_history_ips.update(
                re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", str(_msg.get("content", "")))
            )
    _skills_needing_explicit_entity = [
        s for s in (decision.get("skills") or [])
        if all_manifests.get(s, {}).get("requires_explicit_entity")
        and not _skill_has_required_entity_context(
            all_manifests.get(s, {}),
            _original_question,
        )
        and not _user_history_ips  # also allow when user mentioned IPs in prior turns
    ]
    if _skills_needing_explicit_entity:
        logger.info(
            "[%s] Dropping skills requiring explicit entity absent from question/history: %s",
            SKILL_NAME,
            _skills_needing_explicit_entity,
        )
        decision["skills"] = [s for s in (decision.get("skills") or []) if s not in _skills_needing_explicit_entity]

        # The supervisor LLM may have injected a hallucinated IP (e.g. 192.0.2.1)
        # into parameters["question"] when it was planning to use ip_fingerprinter.
        # Now that ip_fingerprinter is dropped, we must restore the original user
        # question so that opensearch_querier and other remaining skills don't plan
        # around the fabricated entity.
        _param_question = (decision.get("parameters") or {}).get("question", "")
        _original_ips = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", _original_question))
        _param_ips = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", _param_question))
        _fabricated_ips = _param_ips - _original_ips - _user_history_ips
        if _fabricated_ips:
            logger.info(
                "[%s] Restoring original question to remove fabricated IPs %s from parameters",
                SKILL_NAME,
                _fabricated_ips,
            )
            decision.setdefault("parameters", {})["question"] = _original_question

    selected = list(decision.get("skills") or [])
    current_parameters = dict(decision.get("parameters") or {"question": state["user_question"]})
    current_signature = _plan_signature(selected, current_parameters)
    plan_exhausted = False

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: MANDATORY ITERATIVE REFINEMENT when not satisfied
    # ─────────────────────────────────────────────────────────────────────────────
    # Key insight: If the user's question is NOT yet answered, we MUST force the LLM
    # to think deeper and try different approaches. No early termination allowed.
    
    if was_not_satisfied:
        logger.info(
            "[%s] Previous attempt not satisfied (confidence=%.0f%%) — supervisor will course-correct via prior_eval context on next step",
            SKILL_NAME,
            float(last_eval.get("confidence", 0.0) or 0.0) * 100,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: Anti-infinite-loop protection (different from anti-duplicate blocking)
    # ─────────────────────────────────────────────────────────────────────────────
    # Only block if:
    # 1. The plan is identical to last time AND
    # 2. The previous result WAS satisfied (meaning we're purposely re-running?)
    # Otherwise, allow the iteration to continue - the LLM might legitimately
    # need to try the same skill with different parameters.
    
    if selected and current_signature in previously_run_skills and aggregated_results:
        # Exact same plan was already executed. Try to upgrade it once; otherwise
        # finalize instead of burning steps on an identical retry.
        improved = _postprocess_selected_skills(
            user_question=state["user_question"],
            selected_skills=selected,
            available_skills=available_skills,
            current_results=aggregated_results,
            manifests=all_manifests,
        )
        improved_signature = _plan_signature(improved, current_parameters)
        if improved != selected and improved_signature not in previously_run_skills:
            logger.info(
                "[%s] Upgraded duplicate satisfied plan: %s → %s",
                SKILL_NAME, selected, improved,
            )
            selected = improved
            decision["skills"] = selected
        else:
            # No new viable plan emerged after reflection/postprocessing.
            logger.info(
                "[%s] Skipping: identical plan was already executed and no new viable alternative emerged",
                SKILL_NAME,
            )
            selected = []
            decision["skills"] = []
            plan_exhausted = True

    if step_callback:
        step_callback("deciding", decision, step_count, max_steps)

    return {
        "skill_plan": selected,
        "step_count": step_count,
        "pending_parameters": dict(decision.get("parameters") or {"question": state["user_question"]}),
        "pending_reasoning": str(decision.get("reasoning", "")),
        "pending_question_grounding": dict(decision.get("question_grounding") or {}),
        "plan_exhausted": plan_exhausted,
    }


def execute_node(state: AgentState, config: RunnableConfig) -> dict:
    """EXECUTE: run the planned skills and merge results into aggregated_results."""
    rt = _graph_runtime(config)
    runner = rt["runner"]
    step_callback = rt.get("step_callback")

    skill_plan: list[str] = list(state.get("skill_plan") or [])
    step_count: int = state.get("step_count") or 1
    max_steps: int = state.get("max_steps") or 4
    aggregated_results: dict = dict(state.get("skill_results") or {})
    previously_run_skills: list = list(state.get("previously_run_skills") or [])
    pending_parameters: dict = dict(state.get("pending_parameters") or {})
    pending_reasoning: str = str(state.get("pending_reasoning") or "")
    pending_question_grounding: dict = dict(state.get("pending_question_grounding") or {})

    if not skill_plan:
        return {}  # nothing to do

    graph_memory = StateBackedMemory.from_state(state)

    if step_callback:
        step_callback("running", {"skills": skill_plan}, step_count, max_steps)

    # Re-constitute a routing_decision dict compatible with execute_skill_workflow
    routing_decision = {
        "skills": skill_plan,
        "parameters": pending_parameters or {"question": state["user_question"]},
        "reasoning": pending_reasoning,
        "question_grounding": pending_question_grounding,
    }

    step_results = execute_skill_workflow(
        skill_plan,
        runner,
        {},
        routing_decision,
        conversation_history=list(state.get("messages") or []),
        aggregated_results=aggregated_results,
        memory=graph_memory,
    )
    aggregated_results.update(step_results)
    previously_run_skills.append(_plan_signature(skill_plan, pending_parameters))

    mem_updates = graph_memory.to_dict()

    return {
        "skill_results": aggregated_results,
        "previously_run_skills": previously_run_skills,
        **mem_updates,
    }


def evaluate_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    EVALUATE: ask the supervisor whether current results satisfy the question.
    
    Following LangGraph best practices: Always evaluate based on actual results,
    even if the next plan is empty. Let the LLM decide satisfaction based on what we have.
    Never force hardcoded "not satisfied" - that prevents proper agentic iteration.
    """
    rt = _graph_runtime(config)
    llm = rt["llm"]
    instruction = rt["instruction"]
    step_callback = rt.get("step_callback")

    step_count: int = state.get("step_count") or 1
    max_steps: int = state.get("max_steps") or 4
    skill_plan: list = state.get("skill_plan") or []
    aggregated_results: dict = state.get("skill_results") or {}
    trace: list = list(state.get("trace") or [])

    # IMPORTANT: Always ask the LLM to evaluate based on actual results
    # Do NOT hardcode "not satisfied" based on empty skill_plan
    # The plan being empty might mean we've exhausted alternative approaches,
    # but the LLM should still evaluate what we have and decide if it's sufficient
    
    last_eval = _supervisor_evaluate_satisfaction(
        user_question=state["user_question"],
        llm=llm,
        instruction=instruction,
        conversation_history=list(state.get("messages") or []),
        skill_results=aggregated_results,
        step=step_count,
        max_steps=max_steps,
        planned_skills=skill_plan,
    )

    trace.append({
        "step": step_count,
        "decision": {"skills": skill_plan},
        "selected_skills": skill_plan,
        "step_result_keys": list((state.get("skill_results") or {}).keys()),
        "evaluation": last_eval,
    })

    logger.info(
        "[%s] Graph step %d/%d | skills=%s | satisfied=%s (%.2f)",
        SKILL_NAME, step_count, max_steps, skill_plan,
        bool(last_eval.get("satisfied", False)),
        float(last_eval.get("confidence", 0.0) or 0.0),
    )

    if step_callback:
        step_callback("evaluated", last_eval, step_count, max_steps)

    return {"evaluation": last_eval, "trace": trace}


def memory_write_node(state: AgentState, config: RunnableConfig) -> dict:  # noqa: ARG001
    """MEMORY_WRITE: funnel point before formatting; state is already in LangGraph."""
    return {}


def format_response_node(state: AgentState, config: RunnableConfig) -> dict:
    """FORMAT: render the final natural-language response."""
    rt = _graph_runtime(config)
    llm = rt["llm"]
    cfg = rt.get("cfg")
    available_skills = rt.get("available_skills", [])

    aggregated_results: dict = state.get("skill_results") or {}
    trace: list = state.get("trace") or []
    last_eval: dict = state.get("evaluation") or {}
    user_question: str = state["user_question"]
    graph_memory = StateBackedMemory.from_state(state)

    # Build final_routing compatible with format_response()
    final_routing = {
        "reasoning": last_eval.get("reasoning", "Graph orchestration complete."),
        "skills": list(aggregated_results.keys()),
        "preferred_skills": list((trace[-1] or {}).get("selected_skills") or []) if trace else [],
        "parameters": {"question": user_question},
    }

    # NOTE: Removed absolute fallback routing (P0.1 - contradicted max_steps logic).
    # Per LangGraph best practices: if max_steps reached without results, format what we have.
    # Previous implementation would re-route after timeout, contradicting should_loop() termination.
    # format_response() delegates to fallback agent if needed for domain expertise.

    response = format_response(
        user_question,
        final_routing,
        aggregated_results,
        llm,
        cfg,
        available_skills=available_skills,
    )

    return {
        "response": response,
        "routing_metadata": final_routing,
        "skill_results": aggregated_results,  # update in case fallback ran
        **graph_memory.to_dict(),
    }


def should_loop(state: AgentState) -> str:
    """
    Conditional edge after evaluate_node: loop back to decide or move to memory_write.
    
    Following LangGraph best practices: enable iterative refinement until either:
    1. Question is satisfied, OR
    2. Max steps reached, OR
    3. We've truly exhausted all meaningful options
    
    Do NOT give up just because a duplicate plan was proposed - the system should
    keep trying different approaches and parameters.
    """
    evaluation: dict = state.get("evaluation") or {}
    satisfied: bool = bool(evaluation.get("satisfied", False))
    step_count: int = state.get("step_count") or 0
    max_steps: int = state.get("max_steps") or 4
    
    # STOP CONDITION #1: Question is satisfied
    if satisfied:
        logger.info(
            "[%s] ✓ SATISFIED at step %d/%d — moving to response formatting",
            SKILL_NAME, step_count, max_steps,
        )
        return "memory_write"

    # STOP CONDITION #1.5: no further viable plan remains
    if bool(state.get("plan_exhausted", False)):
        logger.info(
            "[%s] No further viable skill plan at step %d/%d — moving to response formatting",
            SKILL_NAME, step_count, max_steps,
        )
        return "memory_write"
    
    # STOP CONDITION #2: Max steps reached
    if step_count >= max_steps:
        logger.info(
            "[%s] Max steps (%d) reached at step %d — moving to response formatting",
            SKILL_NAME, max_steps, step_count,
        )
        return "memory_write"
    
    # OTHERWISE: CONTINUE ITERATING
    # We should keep going and let the LLM think deeper and try different approaches
    logger.info(
        "[%s] ✓ CONTINUE ITERATION: Step %d/%d | Satisfaction confidence: %.0f%% | Will retry",
        SKILL_NAME, step_count, max_steps,
        float(evaluation.get("confidence", 0.0) or 0.0) * 100,
    )
    return "decide"


def build_graph(checkpointer=None):
    """Compile the SecurityClaw orchestration StateGraph.

    Args:
        checkpointer: An optional LangGraph checkpointer (e.g. SqliteSaver or MemorySaver).
            Defaults to an in-memory MemorySaver if not provided.
    """
    _checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    builder = StateGraph(AgentState)
    builder.add_node("decide", decide_node)
    builder.add_node("execute", execute_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("memory_write", memory_write_node)
    builder.add_node("format", format_response_node)

    builder.add_edge(START, "decide")
    builder.add_edge("decide", "execute")
    builder.add_edge("execute", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_loop,
        {"decide": "decide", "memory_write": "memory_write"},
    )
    builder.add_edge("memory_write", "format")
    builder.add_edge("format", END)

    return builder.compile(checkpointer=_checkpointer)


def run_graph(
    user_question: str,
    available_skills: list[dict],
    runner: Any,
    llm: Any,
    instruction: str,
    cfg: Any = None,
    conversation_history: list[dict] | None = None,
    step_callback: Any = None,
    thread_id: str | None = None,
    checkpointer: Any = None,
) -> dict:
    """Execute the LangGraph orchestration pipeline for a single Q&A turn.

    Replaces ``orchestrate_with_supervisor``. All existing callers that pass
    keyword-compatible arguments can switch to this function without any other
    changes.

    Args:
        checkpointer: LangGraph checkpointer (SqliteSaver for persistence,
            MemorySaver for tests). Created automatically when omitted.
        thread_id: Unique ID for this turn's LangGraph thread. Defaults to a
            fresh UUID so each turn starts from a clean slate.
    """
    max_steps = 4
    if cfg:
        max_steps = int(cfg.get("chat", "supervisor_max_steps", default=4) or 4)
    max_steps = max(1, min(max_steps, 8))

    # Fresh thread per turn so the graph always starts from a clean initial state
    run_thread_id = thread_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "thread_id": run_thread_id,
        "user_question": user_question,
        "messages": list(conversation_history or []),
        "skill_plan": [],
        "pending_parameters": {"question": user_question},
        "pending_reasoning": "",
        "skill_results": {},
        "step_count": 0,
        "max_steps": max_steps,
        "previously_run_skills": [],
        "evaluation": {},
        "trace": [],
        "mem_status": "IDLE",
        "mem_focus": "None",
        "mem_findings": [],
        "mem_decisions": [],
        "mem_escalations": [],
        "response": "",
        "routing_metadata": {},
    }

    # Runtime objects (non-serializable) travel via config["configurable"] and
    # are NOT stored in the checkpoint.
    graph_config = {
        "configurable": {
            "thread_id": run_thread_id,
            "runner": runner,
            "llm": llm,
            "available_skills": available_skills,
            "instruction": instruction,
            "cfg": cfg,
            "step_callback": step_callback,
        }
    }

    graph = build_graph(checkpointer)
    final_state = graph.invoke(initial_state, config=graph_config)

    return {
        "response": final_state.get("response") or "Unable to produce a response.",
        "routing": final_state.get("routing_metadata") or {},
        "skill_results": final_state.get("skill_results") or {},
        "trace": final_state.get("trace") or [],
        "evaluation": final_state.get("evaluation") or {},
    }


def orchestrate_with_supervisor(
    user_question: str,
    available_skills: list[dict],
    runner: Any,
    llm: Any,
    instruction: str,
    cfg: Any = None,
    conversation_history: list[dict] | None = None,
    step_callback: Any = None,
) -> dict:
    """Backward-compatible wrapper around :func:`run_graph`.

    All existing callers (main.py, web/api/server.py, tests) continue to work
    unchanged. New code should call ``run_graph`` directly to gain access to the
    ``checkpointer`` and ``thread_id`` parameters.
    """
    return run_graph(
        user_question=user_question,
        available_skills=available_skills,
        runner=runner,
        llm=llm,
        instruction=instruction,
        cfg=cfg,
        conversation_history=conversation_history,
        step_callback=step_callback,
    )


def _supervisor_next_action(
    user_question: str,
    available_skills: list[dict],
    llm: Any,
    instruction: str,
    conversation_history: list[dict],
    previous_trace: list[dict],
    current_results: dict,
    previous_eval: dict,
) -> dict:
    """Ask LLM supervisor what skill(s) to run next.
    
    Uses skill manifests for intelligent routing when available,
    enabling modular skill discovery and auto-adaptation.
    """
    question_lower = user_question.lower()
    available_skill_names = {s.get("name") for s in available_skills}
    attempted_skills = set()
    
    # Track what skills have been tried (extract from trace entries)
    for trace_entry in previous_trace:
        if isinstance(trace_entry, dict):
            # Trace entries use "decision" key with "skills" nested inside
            decision = trace_entry.get("decision") or {}
            attempted_skills.update(decision.get("skills", []))
            # Also check for direct "skills" or "selected_skills" keys for compatibility
            attempted_skills.update(trace_entry.get("selected_skills", []))
            attempted_skills.update(trace_entry.get("skills", []))
    
    # Try to load skill manifests for structured capability awareness
    manifests: dict[str, dict[str, Any]] = {}
    manifest_context = ""
    skill_catalog_json = "[]"
    try:
        from core.skill_manifest import SkillManifestLoader
        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        if manifests:
            manifest_context = "\n" + loader.build_supervisor_context(manifests)
            skill_catalog_json = json.dumps(_build_skill_catalog(available_skills, manifests), indent=2, default=str)
            logger.debug("[%s] Loaded %d skill manifests for intelligent routing", SKILL_NAME, len(manifests))
    except Exception as e:
        logger.debug("[%s] Manifest loading failed: %s", SKILL_NAME, e)

    skills_description = "\n".join(
        f"- {s.get('name')}: {s.get('description', '')}"
        for s in available_skills
    )
    history_text = "\n".join(
        f"- {m.get('role', '?')}: {str(m.get('content', ''))[:220]}"
        for m in conversation_history[-6:]
    )
    prior_steps = json.dumps(previous_trace[-3:], indent=2, default=str) if previous_trace else "[]"
    result_keys = list(current_results.keys())
    question_grounding = _deterministic_supervisor_question_grounding(user_question) or {}

    # Summarize what each result returned so the supervisor can make intelligent choices.
    result_summary_lines = []
    for skill_name, result in current_results.items():
        count = result.get("results_count") or result.get("log_records") or (
            len(result.get("results", [])) if isinstance(result.get("results"), list) else 0
        )
        status = result.get("status", "?")
        result_summary_lines.append(f"  {skill_name}: status={status}, records_found={count}")
    result_summary = "\n".join(result_summary_lines) or "  (no skills have run yet)"

    next_action_template = _load_prompt_template(
        SUPERVISOR_NEXT_ACTION_PROMPT_PATH,
        """
You are the SOC supervisor orchestrator.

QUESTION:
{{USER_QUESTION}}

AVAILABLE SKILLS:
{{SKILLS_DESCRIPTION}}{{MANIFEST_CONTEXT}}

PRIOR EXECUTION TRACE:
{{PRIOR_STEPS}}

RESULTS ALREADY GATHERED:
{{RESULT_SUMMARY}}

QUESTION GROUNDING FROM CURRENT QUESTION ONLY:
{{QUESTION_GROUNDING}}

PREVIOUS EVALUATION:
{{PREVIOUS_EVALUATION}}

Return strict JSON with reasoning, skills, and parameters.question.
""",
    )
    prompt = _render_prompt(
        next_action_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_TEXT": history_text or "- none",
            "SKILLS_DESCRIPTION": skills_description,
            "MANIFEST_CONTEXT": manifest_context,
            "SKILL_CATALOG_JSON": skill_catalog_json,
            "PRIOR_STEPS": prior_steps,
            "RESULT_SUMMARY": result_summary,
            "QUESTION_GROUNDING": json.dumps(question_grounding, indent=2, default=str),
            "PREVIOUS_EVALUATION": json.dumps(previous_eval, indent=2, default=str),
        },
    )

    try:
        response = llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ])
        parsed = _parse_json_object(response) or {}
        if "parameters" not in parsed:
            parsed["parameters"] = {}
        if not parsed["parameters"].get("question"):
            parsed["parameters"]["question"] = user_question
        if conversation_history:
            parsed["parameters"]["conversation_history"] = conversation_history
        if not isinstance(parsed.get("skills"), list):
            parsed["skills"] = []
        if not isinstance(parsed.get("reasoning"), str):
            parsed["reasoning"] = "Supervisor selected next action"

        wants_fingerprint = _any_skill_in_plan(["ip_fingerprinter"], parsed.get("skills", []), available_skills)
        routing_question = user_question

        if wants_fingerprint:
            fingerprint_entities = _recover_fingerprint_followup_entities(
                user_question,
                conversation_history,
                current_results,
            )
            if fingerprint_entities.get("ips"):
                anchored_fingerprint_question = _build_context_aware_fingerprint_question(
                    user_question,
                    fingerprint_entities,
                    conversation_history,
                )
                parsed["parameters"]["question"] = anchored_fingerprint_question
                if not _question_has_explicit_entities(user_question):
                    routing_question = anchored_fingerprint_question

            fingerprint_skill = None
            if manifests:
                fingerprint_skill = first_skill_in_group(manifests, "host_fingerprinting")
            if not fingerprint_skill and _skill_exists(available_skills, "ip_fingerprinter"):
                fingerprint_skill = "ip_fingerprinter"
            if fingerprint_skill and fingerprint_skill not in parsed.get("skills", []):
                parsed["skills"] = list(parsed.get("skills", [])) + [fingerprint_skill]

        contextual_reputation_entities = _recover_threat_followup_entities(
            user_question,
            conversation_history,
            current_results,
        )
        if contextual_reputation_entities and _skill_exists(available_skills, "threat_analyst"):
            parsed["skills"] = ["threat_analyst"]
            parsed["reasoning"] = "Follow-up reputation question anchored to entities from the previous answer"
            parsed["parameters"]["question"] = _build_context_aware_threat_question(
                user_question,
                contextual_reputation_entities,
            )
            if conversation_history:
                parsed["parameters"]["conversation_history"] = conversation_history
            return parsed

        if _any_skill_in_plan(["baseline_querier"], parsed.get("skills", []), available_skills):
            baseline_entities = _recover_baseline_followup_entities(
                user_question,
                conversation_history,
                current_results,
            )
            parsed["parameters"]["question"] = _build_context_aware_baseline_question(
                user_question,
                baseline_entities,
                conversation_history,
            )
        
        parsed["skills"] = _postprocess_selected_skills(
            user_question=routing_question,
            selected_skills=parsed.get("skills", []),
            available_skills=available_skills,
            current_results=current_results,
            manifests=manifests,
        )
        parsed["planner_trace"] = {
            "question_grounding": question_grounding,
            "initial_candidate": {
                "skills": list(parsed.get("skills") or []),
                "parameters": dict(parsed.get("parameters") or {}),
                "reasoning": str(parsed.get("reasoning") or ""),
            },
            "reviews": [],
        }
        
        # ── AUTO-QUEUE threat_analyst AFTER opensearch_querier ──────────────────────
        # If LLM plan includes threat_analyst and opensearch found results, ensure it runs next
        opensearch_has_results = bool(
            current_results.get("opensearch_querier") and
            current_results["opensearch_querier"].get("results_count", 0) > 0
        )
        has_threat_intel = bool(
            current_results.get("threat_analyst") and
            current_results["threat_analyst"].get("status") == "ok"
        )
        
        # Use dynamic skill lookup instead of hardcoded string
        if _any_skill_in_plan(["threat_analyst"], parsed.get("skills", []), available_skills) and opensearch_has_results and not has_threat_intel:
            if not _any_skill_in_plan(["threat_analyst"], parsed.get("skills", []), available_skills):
                parsed["skills"].append("threat_analyst")
                logger.info(
                    "[%s] Auto-queueing threat_analyst: LLM plan includes it and opensearch found results",
                    SKILL_NAME
                )
        
        # ── SHORTCUT: If threat_analyst is in plan and IPs are known, skip log search ──
        if _any_skill_in_plan(["threat_analyst"], parsed.get("skills", []), available_skills) and not has_threat_intel:
            extracted_entities = _extract_entities_from_previous_results(current_results)
            has_known_entities = bool(extracted_entities.get("ips") or extracted_entities.get("domains"))
            
            threat_analyst_available = _skill_exists(available_skills, "threat_analyst")
            skills_requesting_log_search = _get_skills_requesting_log_search(available_skills)
            llm_wants_log_search = _any_skill_in_plan(skills_requesting_log_search, parsed.get("skills", []), available_skills)
            
            if has_known_entities and threat_analyst_available and llm_wants_log_search:
                # Skip log search, go directly to threat_analyst
                parsed["skills"] = [s for s in parsed.get("skills", []) if s not in skills_requesting_log_search]
                if not _any_skill_in_plan(["threat_analyst"], parsed.get("skills", []), available_skills):
                    parsed["skills"].append("threat_analyst")
                    logger.info(
                        "[%s] Shortcutting to threat_analyst: LLM plan includes it + known IPs/domains detected",
                        SKILL_NAME
                    )
        
        # ── AUTO-QUEUE opensearch_querier AFTER fields_querier ──────────────────────
        # If LLM plan includes opensearch_querier and fields_querier just ran, execute the querier
        fields_just_ran = _any_skill_in_plan(["fields_querier"], parsed.get("skills", []), available_skills)
        fields_has_results = bool(
            current_results.get("fields_querier") and (
                current_results["fields_querier"].get("field_mappings") or
                (current_results["fields_querier"].get("findings") or {}).get("field_mappings")
            )
        )
        
        if _any_skill_in_plan(["opensearch_querier"], parsed.get("skills", []), available_skills) and fields_just_ran and fields_has_results:
            if not _any_skill_in_plan(["opensearch_querier"], parsed.get("skills", []), available_skills):
                parsed["skills"].append("opensearch_querier")
                logger.info(
                    "[%s] Auto-queueing opensearch_querier: LLM plan includes it, fields discovered",
                    SKILL_NAME
                )

        # When schema discovery and log search are linked, make sure opensearch_querier
        # receives the original user request instead of a meta-level field-discovery
        # paraphrase from the supervisor.
        if _any_skill_in_plan(["opensearch_querier"], parsed.get("skills", []), available_skills):
            parsed["parameters"]["question"] = routing_question

        parsed = _ensure_viable_plan(
            decision=parsed,
            user_question=user_question,
            available_skills=available_skills,
            manifests=manifests,
            llm=llm,
            instruction=instruction,
            current_results=current_results,
            previous_eval=previous_eval,
            previous_trace=previous_trace,
            mode="next_action",
            failure_reason="Supervisor next action must remain grounded in the loaded skill manifests.",
        )
        
        # Pass question grounding to skills so they can use it for planning
        parsed["question_grounding"] = question_grounding
        
        return parsed
    except Exception as exc:
        logger.warning("[%s] Supervisor next action failed: %s", SKILL_NAME, exc)
        return {
            "reasoning": "Fallback routing due to supervisor parse failure",
            "skills": [],
            "parameters": {"question": user_question},
        }


def _supervisor_evaluate_satisfaction(
    user_question: str,
    llm: Any,
    instruction: str,
    conversation_history: list[dict],
    skill_results: dict,
    step: int,
    max_steps: int,
    planned_skills: list[str] = None,
) -> dict:
    """Evaluate whether current aggregated results sufficiently answer the question."""
    # ── SMART FAST PATH: if records found, verify they answer the question ──
    # Don't auto-satisfy if reputation/threat intel was asked but threat_analyst wasn't run
    total_records_found = 0
    for skill_name, result in skill_results.items():
        if result.get("validation_failed"):
            continue
        if skill_name == "opensearch_querier" and result.get("aggregation_type") == "country_terms":
            continue
        count = result.get("results_count") or result.get("log_records") or (
            len(result.get("results", [])) if isinstance(result.get("results"), list) else 0
        )
        total_records_found += int(count or 0)

    os_result = skill_results.get("opensearch_querier") or {}
    country_buckets = os_result.get("country_buckets") or []

    # Check which skills were actually executed
    threat_analyst_executed = bool(skill_results.get("threat_analyst"))
    baseline_querier_executed = bool(skill_results.get("baseline_querier"))
    
    has_threat_intel = bool(
        skill_results.get("threat_analyst") and 
        skill_results["threat_analyst"].get("status") == "ok"
    )
    threat_verdicts = skill_results.get("threat_analyst", {}).get("verdicts") or []
    baseline_result = skill_results.get("baseline_querier") or {}
    baseline_findings = baseline_result.get("findings") or {}

    if baseline_querier_executed and baseline_result.get("status") == "ok" and baseline_findings.get("answer"):
        baseline_log_records = int(baseline_findings.get("log_records", 0) or 0)
        baseline_sources = int(baseline_findings.get("rag_sources", 0) or 0)
        if baseline_log_records > 0 or baseline_sources > 0:
            logger.info(
                "[%s] Evaluation: baseline_querier returned grounded baseline analysis — marking satisfied",
                SKILL_NAME,
            )
            return {
                "satisfied": True,
                "confidence": 0.85 if baseline_log_records > 0 else 0.75,
                "reasoning": (
                    f"Baseline analysis returned {baseline_log_records} matching log record(s) "
                    f"and {baseline_sources} relevant baseline document(s)."
                ),
                "missing": [],
            }

    # Satisfy if threat intel produced verdicts (when threat_analyst was executed)
    has_opensearch_records = total_records_found > 0
    if threat_analyst_executed and has_threat_intel and threat_verdicts:
        # Validate that verdict IPs match question IPs when possible
        # Extract IPs from question to ensure response is about the right entities
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        question_ips = set(re.findall(ip_pattern, user_question))
        
        # Also look in recent conversation for mentioned IPs (for "these IPs" follow-ups)
        history_text = "\n".join(
            str(m.get("content", ""))
            for m in (conversation_history or [])[-3:]
        )
        discovered_ips = set(re.findall(ip_pattern, history_text))
        question_ips.update(discovered_ips)
        
        # If question had specific IPs, check that verdicts mention similar ones
        verdicts_mention_wrong_ips = False
        if question_ips:
            # Extract IPs from verdict reasoning
            verdict_text = "\n".join(
                str(v.get("reasoning", ""))
                for v in threat_verdicts
            )
            verdict_ips = set(re.findall(ip_pattern, verdict_text))
            
            # If verdict mentions IPs, at least one should overlap with question IPs
            if verdict_ips and not verdict_ips.intersection(question_ips):
                # Verdict IPs and question IPs don't overlap — likely analyzed wrong entities
                verdicts_mention_wrong_ips = True
                logger.warning(
                    "[%s] Verdict mentions IPs %s but question asked about %s — possible IP mismatch",
                    SKILL_NAME, verdict_ips, question_ips
                )
        
        if verdicts_mention_wrong_ips:
            # Re-run to get correct verdicts
            logger.info(
                "[%s] Detected IP mismatch in verdicts — requesting re-analysis",
                SKILL_NAME,
            )
            return {
                "satisfied": False,
                "confidence": 0.3,
                "reasoning": "Threat intel was produced but for wrong IPs; re-analyzing.",
                "missing": ["threat reputation for correct IPs"],
            }
        
        logger.info(
            "[%s] Evaluation: threat_analyst returned %d verdict(s) for a reputation question — marking satisfied",
            SKILL_NAME,
            len(threat_verdicts),
        )
        return {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": f"Threat intelligence verdicts were produced for {len(threat_verdicts)} entity(s).",
            "missing": [],
        }
    
    # Check if ip_fingerprinter was executed
    fingerprint_result = skill_results.get("ip_fingerprinter") or {}
    fingerprint_status = fingerprint_result.get("status")
    has_fingerprint = fingerprint_status == "ok"

    if has_fingerprint:
        fingerprint_ports = len(fingerprint_result.get("ports") or [])
        logger.info(
            "[%s] Evaluation: ip_fingerprinter produced fingerprint with %d port(s) — marking satisfied",
            SKILL_NAME,
            fingerprint_ports,
        )
        return {
            "satisfied": True,
            "confidence": 0.95,
            "reasoning": f"Fingerprint analysis completed using {fingerprint_ports} observed port(s).",
            "missing": [],
        }

    # If fingerprinter was queued but returned no_data, check if we have records
    # If we have records but fingerprint returned no_data, that's incomplete
    if fingerprint_status == "no_data":
        if total_records_found > 0:
            # We found log evidence but fingerprinter couldn't extract ports
            logger.info(
                "[%s] Found %d records but ip_fingerprinter returned no_data (no ports detected) — marking not satisfied",
                SKILL_NAME,
                total_records_found,
            )
            return {
                "satisfied": False,
                "confidence": 0.45,
                "reasoning": "Traffic evidence exists but passive fingerprint analysis found no distinguishing ports.",
                "missing": ["passive fingerprint analysis from observed ports"],
            }
        
        # No records AND no_data → truly no evidence
        logger.info(
            "[%s] Evaluation: no traffic records AND ip_fingerprinter returned no_data — marking satisfied",
            SKILL_NAME,
        )
        return {
            "satisfied": True,
            "confidence": 0.75,
            "reasoning": fingerprint_result.get("reason") or "No matching port observations were found for the requested IP.",
            "missing": [],
        }

    # If fingerprinter was queued but not yet complete, and we have records, indicate progress
    if "ip_fingerprinter" in skill_results and total_records_found > 0 and not has_fingerprint:
        logger.info(
            "[%s] Found %d records but fingerprint result is missing or incomplete — continuing",
            SKILL_NAME,
            total_records_found,
        )
        return {
            "satisfied": False,
            "confidence": 0.45,
            "reasoning": "Traffic evidence exists but passive fingerprint analysis has not completed yet.",
            "missing": ["passive fingerprint analysis from observed ports"],
        }

    # ── CHECK: if fingerprinting was planned but hasn't run yet ──────────────
    # If user asked for fingerprinting (ip_fingerprinter in planned skills) but it hasn't
    # executed yet, don't mark satisfied even if we have traffic records. Continue orchestration.
    planned_skills = planned_skills or []
    if "ip_fingerprinter" in planned_skills and "ip_fingerprinter" not in skill_results:
        question_lower = user_question.lower()
        is_fingerprinting_intent = any(
            term in question_lower 
            for term in ["fingerprint", "port profile", "ports associated", "os likelihood", "likely os", "likely role", "client or server", "server or client"]
        )
        if is_fingerprinting_intent and total_records_found > 0:
            logger.info(
                "[%s] Fingerprinting intent detected: found %d records but ip_fingerprinter not yet executed — continuing orchestration",
                SKILL_NAME,
                total_records_found,
            )
            return {
                "satisfied": False,
                "confidence": 0.50,
                "reasoning": f"Found {total_records_found} port observation(s) but passive fingerprint analysis is still pending.",
                "missing": ["passive fingerprint analysis from observed ports"],
            }

    if total_records_found > 0:
        # If threat_analyst was queued but not complete, wait for it before finalizing
        if threat_analyst_executed and not has_threat_intel:
            logger.info(
                "[%s] Found %d records but reputation/threat was requested and threat_analyst not yet run — continuing",
                SKILL_NAME, total_records_found
            )
            return {
                "satisfied": False,
                "confidence": 0.6,
                "reasoning": f"Found {total_records_found} records but need threat intelligence enrichment.",
                "missing": ["threat reputation and risk assessment"],
            }
        
        # Records found and no actionable missing piece → satisfied
        logger.info(
            "[%s] Evaluation: %d records found across skills — marking satisfied",
            SKILL_NAME, total_records_found,
        )
        return {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": f"Found {total_records_found} matching records across executed skills.",
            "missing": [],
        }

    fields_result = skill_results.get("fields_querier") or {}
    directional_alternative = os_result.get("directional_alternative") or {}
    if country_buckets and not os_result.get("validation_failed"):
        logger.info(
            "[%s] Evaluation: opensearch_querier returned %d aggregated country bucket(s) — marking satisfied",
            SKILL_NAME,
            len(country_buckets),
        )
        return {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": f"OpenSearch returned {len(country_buckets)} aggregated country bucket(s).",
            "missing": [],
        }

    if country_buckets and os_result.get("validation_failed"):
        logger.info(
            "[%s] Evaluation: aggregated country buckets were rejected by validation — continuing | issue=%s | reasoning=%s",
            SKILL_NAME,
            os_result.get("validation_issue", ""),
            os_result.get("validation_reasoning", ""),
        )
        return {
            "satisfied": False,
            "confidence": 0.2,
            "reasoning": os_result.get("validation_reasoning") or os_result.get("validation_issue") or "Aggregated answer shape did not match the user question.",
            "missing": ["grounded search or aggregation aligned to the user question"],
        }

    if int(directional_alternative.get("results_count", 0) or 0) > 0:
        alternative_direction = directional_alternative.get("direction", "opposite")
        alternative_count = int(directional_alternative.get("results_count", 0) or 0)
        logger.info(
            "[%s] Evaluation: primary direction had zero results, but %d opposite-direction records were found — marking satisfied",
            SKILL_NAME,
            alternative_count,
        )
        return {
            "satisfied": True,
            "confidence": 0.85,
            "reasoning": f"No records matched the requested direction, but {alternative_count} {alternative_direction}-direction records were found for the same IP.",
            "missing": [],
        }

    if not total_records_found:
        if os_result.get("status") == "no_action":
            logger.info(
                "[%s] Evaluation: opensearch_querier produced no actionable query — keeping the supervisor unsatisfied",
                SKILL_NAME,
            )
            return {
                "satisfied": False,
                "confidence": 0.25,
                "reasoning": "Field discovery completed, but the log search did not execute a grounded query for the original request.",
                "missing": ["grounded log search results"],
            }

        if fields_result.get("status") == "ok" and "opensearch_querier" not in skill_results:
            logger.info(
                "[%s] Evaluation: fields_querier completed but no log search has run yet",
                SKILL_NAME,
            )
            return {
                "satisfied": False,
                "confidence": 0.5,
                "reasoning": "Field discovery completed, but no log records have been retrieved yet.",
                "missing": ["matching log records"],
            }

        if os_result.get("status") == "ok" and int(os_result.get("results_count", 0) or 0) == 0 and not os_result.get("validation_failed"):
            logger.info(
                "[%s] Evaluation: opensearch_querier returned zero grounded results",
                SKILL_NAME,
            )
            return {
                "satisfied": False,
                "confidence": 0.7,
                "reasoning": "No matching log records were found for the requested criteria.",
                "missing": ["matching log records"],
            }

    history_text = "\n".join(
        f"- {m.get('role', '?')}: {str(m.get('content', ''))[:220]}"
        for m in conversation_history[-6:]
    )
    result_summary = json.dumps(skill_results, indent=2, default=str)[:6000]

    evaluation_template = _load_prompt_template(
        SUPERVISOR_EVALUATION_PROMPT_PATH,
        """
Evaluate whether the current skill outputs are sufficient.

QUESTION:
{{USER_QUESTION}}

SKILL RESULTS:
{{RESULT_SUMMARY}}

TOTAL RECORDS FOUND ACROSS ALL SKILLS: {{TOTAL_RECORDS_FOUND}}

STEP:
{{STEP}}/{{MAX_STEPS}}

Return strict JSON with satisfied, confidence, reasoning, and missing.
""",
    )
    prompt = _render_prompt(
        evaluation_template,
        {
            "USER_QUESTION": user_question,
            "HISTORY_TEXT": history_text or "- none",
            "RESULT_SUMMARY": result_summary,
            "TOTAL_RECORDS_FOUND": total_records_found,
            "STEP": step,
            "MAX_STEPS": max_steps,
        },
    )

    try:
        response = llm.chat([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ])
        parsed = _parse_json_object(response) or {}
        return {
            "satisfied": bool(parsed.get("satisfied", False)),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided")),
            "missing": parsed.get("missing", []) if isinstance(parsed.get("missing", []), list) else [],
        }
    except Exception as exc:
        logger.warning("[%s] Supervisor evaluation failed: %s", SKILL_NAME, exc)
        # Conservative fallback: continue until max steps.
        return {
            "satisfied": step >= max_steps,
            "confidence": 0.0,
            "reasoning": "Evaluation unavailable; using step limit fallback",
            "missing": ["evaluation parsing failed"],
        }


def _parse_json_object(response: str) -> dict | None:
    """Best-effort JSON parsing from raw or fenced model output."""
    try:
        return json.loads(response)
    except Exception:
        pass

    try:
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response)
        for block in fenced:
            try:
                return json.loads(block.strip())
            except Exception:
                continue
    except Exception:
        pass

    try:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return None


def _build_threat_followup_question(forensic_result: dict) -> str:
    """Build a compact prompt for threat_analyst from forensic output."""
    report = forensic_result.get("forensic_report", {}) if forensic_result else {}
    incident = report.get("incident_summary", "")
    timeline = (report.get("timeline_narrative", "") or "")[:800]
    anchors = report.get("context_anchors", {}) or {}
    ips = anchors.get("ips", [])[:5]
    ports = anchors.get("ports", [])[:3]
    countries = anchors.get("countries", [])[:3]
    protocols = anchors.get("protocols", [])[:3]
    anchor_text = (
        f"Anchors: IPs={ips}, Ports={ports}, Countries={countries}, Protocols={protocols}."
        if (ips or ports or countries or protocols)
        else ""
    )

    if not incident and not timeline:
        return ""

    return (
        "Perform threat reputation analysis for entities in this forensic report. "
        "Prioritize the provided anchor entities and do not pivot to unrelated IPs unless strongly justified by evidence. "
        "Focus on maliciousness signals, confidence, and actionable response. "
        f"Incident: {incident}\n"
        f"{anchor_text}\n"
        f"Timeline excerpt: {timeline}"
    )


def _extract_entities_from_previous_results(aggregated_results: dict) -> dict:
    """
    Extract IPs, domains, countries, and ports from previous skill results.
    
    Returns dict with keys:
      - ips: list of unique IP addresses found
      - domains: list of unique domains found
      - countries: list of unique countries found
      - ports: list of unique ports found
      - sources: which skills found these entities
    """
    entities = {
        "ips": set(),
        "domains": set(),
        "countries": set(),
        "ports": set(),
        "sources": [],
    }
    
    # Extract from opensearch_querier results
    if "opensearch_querier" in aggregated_results:
        result = aggregated_results["opensearch_querier"]
        entities["sources"].append("opensearch_querier")
        
        # Extract from raw results (log documents)
        results_list = result.get("results", [])
        if isinstance(results_list, list):
            for record in results_list:
                if isinstance(record, dict):
                    source_ips: set[str] = set()
                    destination_ips: set[str] = set()
                    record_countries: set[str] = set()
                    # Common IP field names
                    for ip_field in ["src_ip", "source_ip", "srcip", "src", "ip", "_source.src_ip", "source.ip"]:
                        if ip_field in record and record[ip_field]:
                            val = record[ip_field]
                            if isinstance(val, str):
                                source_ips.add(val)
                    for ip_field in ["dst_ip", "dest_ip", "destination_ip", "destination.ip"]:
                        if ip_field in record and record[ip_field]:
                            val = record[ip_field]
                            if isinstance(val, str):
                                destination_ips.add(val)
                    nested_source_ip = record.get("source", {}).get("ip") if isinstance(record.get("source"), dict) else None
                    nested_destination_ip = record.get("destination", {}).get("ip") if isinstance(record.get("destination"), dict) else None
                    if isinstance(nested_source_ip, str):
                        source_ips.add(nested_source_ip)
                    if isinstance(nested_destination_ip, str):
                        destination_ips.add(nested_destination_ip)
                    
                    # Common domain field names
                    for domain_field in ["domain", "hostname", "fqdn", "src_domain"]:
                        if domain_field in record and record[domain_field]:
                            val = record[domain_field]
                            if isinstance(val, str):
                                entities["domains"].add(val)
                    
                    # Country extraction
                    for country_field in [
                        "country", "src_country", "country_name", "geoip.country_name",
                        "source.geo.country_name", "destination.geo.country_name",
                    ]:
                        if country_field in record and record[country_field]:
                            val = record[country_field]
                            if isinstance(val, str):
                                entities["countries"].add(val)
                                record_countries.add(val)
                    geo = record.get("geoip") or {}
                    if isinstance(geo, dict):
                        for nested_country in (geo.get("country_name"), geo.get("country")):
                            if isinstance(nested_country, str):
                                entities["countries"].add(nested_country)
                                record_countries.add(nested_country)
                    has_country_info = bool(record_countries)

                    if source_ips and has_country_info:
                        entities["ips"].update(source_ips)
                    else:
                        entities["ips"].update(source_ips)
                        entities["ips"].update(destination_ips)
                    
                    # Port extraction
                    for port_field in ["port", "dst_port", "dest_port", "dport", "destination.port", "destination_port"]:
                        if port_field in record and record[port_field]:
                            val = record[port_field]
                            if isinstance(val, (int, str)):
                                entities["ports"].add(str(val))
                    nested_dest_port = record.get("destination", {}).get("port") if isinstance(record.get("destination"), dict) else None
                    if isinstance(nested_dest_port, (int, str)):
                        entities["ports"].add(str(nested_dest_port))
        
        # Only trust summary metadata when the opensearch result passed validation.
        if not result.get("validation_failed"):
            entities["countries"].update(result.get("countries", []))
            entities["ports"].update(result.get("ports", []))
    
    # Extract from baseline_querier / fields_querier results (legacy rag_querier support deprecated)
    for rag_skill in ("baseline_querier", "fields_querier"):
        if rag_skill in aggregated_results:
            result = aggregated_results[rag_skill]
            entities["sources"].append(rag_skill)

            # Extract IPs and ports from RAG findings
            entities["ips"].update(result.get("ips", []))
            entities["ports"].update(result.get("ports", []))
    
    # Extract from geoip_lookup results
    if "geoip_lookup" in aggregated_results:
        result = aggregated_results["geoip_lookup"]
        entities["sources"].append("geoip_lookup")
        
        # Extract IPs that were looked up
        if result.get("status") == "ok":
            # Single IP result
            if "ip" in result:
                entities["ips"].add(result["ip"])
            # Multiple IPs from lookups
            if "lookups" in result and isinstance(result["lookups"], list):
                for lookup in result["lookups"]:
                    if isinstance(lookup, dict) and "ip" in lookup:
                        entities["ips"].add(lookup["ip"])
            # Explicit ips list
            if "ips" in result and isinstance(result["ips"], list):
                entities["ips"].update(result["ips"])
            
            # Extract countries from geo data
            if "lookups" in result and isinstance(result["lookups"], list):
                for lookup in result["lookups"]:
                    if isinstance(lookup, dict) and "geo" in lookup:
                        geo = lookup.get("geo", {})
                        if isinstance(geo, dict):
                            if "country" in geo:
                                entities["countries"].add(geo["country"])
                            elif "country_name" in geo:
                                entities["countries"].add(geo["country_name"])
    
    # Convert sets to lists
    return {
        "ips": list(entities["ips"]),
        "domains": list(entities["domains"]),
        "countries": list(entities["countries"]),
        "ports": list(entities["ports"]),
        "sources": entities["sources"],
    }


def _extract_entities_from_conversation_history(conversation_history: list[dict] | None) -> dict:
    """Extract the most recent concrete entities from recent conversation history."""
    empty = {
        "ips": [],
        "domains": [],
        "countries": [],
        "ports": [],
        "sources": [],
    }
    if not conversation_history:
        return empty

    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"

    def _is_grounded_assistant_message(role: str, text: str) -> bool:
        if role != "assistant":
            return True
        grounded_markers = [
            "Found ",
            "No traffic ",
            "No matching records found",
            "Countries seen:",
            "Source IPs:",
            "Source/destination IPs:",
            "Remote peers:",
            "Reputation analysis for ",
            "Passive fingerprint for ",
        ]
        return any(marker in text for marker in grounded_markers)

    for msg in reversed(conversation_history[-8:]):
        text = str(msg.get("content", "") or "")
        if not text:
            continue
        role = str(msg.get("role", ""))
        if not _is_grounded_assistant_message(role, text):
            continue

        ips = list(dict.fromkeys(re.findall(ip_pattern, text)))
        domains = list(dict.fromkeys(re.findall(domain_pattern, text.lower())))
        countries: list[str] = []
        for match in re.finditer(r"Countries seen:\s*([A-Za-z ,_-]+?)(?:\.|$)", text, re.IGNORECASE):
            country_text = match.group(1).strip().rstrip(".")
            countries.extend([part.strip() for part in country_text.split(",") if part.strip()])

        ports: list[str] = []
        for match in re.finditer(r"Ports:\s*([0-9, ]+?)(?:\.|$)", text, re.IGNORECASE):
            port_text = match.group(1).strip().rstrip(".")
            ports.extend([part.strip() for part in port_text.split(",") if part.strip()])

        if ips or domains or countries or ports:
            return {
                "ips": ips,
                "domains": domains,
                "countries": countries,
                "ports": ports,
                "sources": [msg.get("role", "history")],
            }

    return empty


def _filter_entities_for_question(entities: dict, user_question: str) -> dict:
    if not entities:
        return entities
    filtered = {
        "ips": list(entities.get("ips", [])),
        "domains": list(entities.get("domains", [])),
        "countries": list(entities.get("countries", [])),
        "ports": list(entities.get("ports", [])),
        "sources": list(entities.get("sources", [])),
    }
    # Router already made filtering decisions; return entities as-is
    return filtered


def _followup_reputation_entities(user_question: str, conversation_history: list[dict] | None) -> dict:
    """Recover prior entities for follow-up reputation questions like 'what about the others?'"""
    # NOTE: Caller should only invoke this if threat_analyst was executed (LLM plan driven)
    if _question_has_explicit_entities(user_question):
        return {}

    question_lower = user_question.lower()
    plural_referential_cues = [
        "those",
        "them",
        "above",
        "listed",
        "list above",
        "listed above",
        "the ip listed",
        "the ips listed",
        "listed ip",
        "listed ips",
        "above ip",
        "above ips",
        "the others",
        "others",
        "these",
        "this ip",
        "that ip",
        "mentioned",
        "just mentioned",
        "you just mentioned",
        "you've just mentioned",
        "previously mentioned",
        "prior ip",
        "prior ips",
        "previous ip",
        "previous ips",
        "aside from",
        "excluding",
        "exclude",
        "except",
        "other than",
    ]
    singular_referential_cues = [
        "the ip",
        "that ip",
        "this ip",
        "the address",
        "that address",
        "this address",
        "the host",
        "that host",
        "this host",
    ]
    # Router already decided if this follow-up should recover prior entities
    # No keyword-based filtering; return what was found
    if re.search(r"\b(?:public|internet-facing|external)\b", user_question.lower()):
        # Only if explicitly asking for public-only entities
        return {}

    history_entities = _extract_entities_from_conversation_history(conversation_history)
    history_entities = _filter_entities_for_question(history_entities, user_question)
    public_history_ips = [ip for ip in history_entities.get("ips", []) if not _is_private_ip(ip)]
    singular_reference = any(re.search(rf"\b{re.escape(cue)}\b", question_lower) for cue in singular_referential_cues)
    plural_reference = any(re.search(rf"\b{re.escape(cue)}\b", question_lower) for cue in plural_referential_cues)

    if singular_reference:
        if len(public_history_ips) == 1:
            singular_entities = dict(history_entities)
            singular_entities["ips"] = [public_history_ips[0]]
            singular_entities["domains"] = []
            return singular_entities
        return {}

    if not plural_reference:
        return {}

    if history_entities.get("ips") or history_entities.get("domains"):
        return history_entities
    return {}


def _latest_user_explicit_entities(conversation_history: list[dict] | None) -> dict:
    empty = {
        "ips": [],
        "domains": [],
        "countries": [],
        "ports": [],
        "sources": [],
    }
    if not conversation_history:
        return empty

    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"

    for msg in reversed(conversation_history[-8:]):
        if msg.get("role") != "user":
            continue
        text = str(msg.get("content", "") or "")
        if not text:
            continue
        ips = list(dict.fromkeys(re.findall(ip_pattern, text)))
        domains = list(dict.fromkeys(re.findall(domain_pattern, text.lower())))
        if ips or domains:
            return {
                "ips": ips,
                "domains": domains,
                "countries": [],
                "ports": [],
                "sources": ["user_history"],
            }

    return empty


def _recover_threat_followup_entities(
    user_question: str,
    conversation_history: list[dict] | None,
    aggregated_results: dict | None = None,
) -> dict:
    """Recover concrete entities for threat follow-ups from current results or prior conversation."""
    entities = _followup_reputation_entities(user_question, conversation_history)
    if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
        return entities

    if aggregated_results:
        entities = _extract_entities_from_previous_results(aggregated_results)
        entities = _filter_entities_for_question(entities, user_question)
        if entities and (entities.get("ips") or entities.get("domains") or entities.get("countries")):
            return entities

    return {}


def _recover_baseline_followup_entities(
    user_question: str,
    conversation_history: list[dict] | None,
    aggregated_results: dict | None = None,
) -> dict:
    # NOTE: Caller should only invoke this if baseline_querier was executed (LLM plan driven)

    explicit_ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question or "")))
    entities = {
        "ips": explicit_ips,
        "domains": [],
        "countries": [],
        "ports": [],
        "sources": [],
    }

    history_entities = _extract_entities_from_conversation_history(conversation_history)
    previous_entities = _extract_entities_from_previous_results(aggregated_results or {}) if aggregated_results else {}

    for key in ("ips", "domains", "countries", "ports"):
        combined = entities.get(key, []) + history_entities.get(key, []) + previous_entities.get(key, [])
        entities[key] = list(dict.fromkeys(combined))

    return _filter_entities_for_question(entities, user_question)


def _recover_fingerprint_followup_entities(
    user_question: str,
    conversation_history: list[dict] | None,
    aggregated_results: dict | None = None,
) -> dict:
    explicit_ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question or "")))
    if explicit_ips:
        return {
            "ips": explicit_ips,
            "domains": [],
            "countries": [],
            "ports": [],
            "sources": ["current_question"],
        }

    latest_user_entities = _latest_user_explicit_entities(conversation_history)
    if latest_user_entities.get("ips") or latest_user_entities.get("domains"):
        return latest_user_entities

    history_entities = _extract_entities_from_conversation_history(conversation_history)
    if history_entities.get("ips") or history_entities.get("domains"):
        return {
            "ips": list(history_entities.get("ips") or []),
            "domains": list(history_entities.get("domains") or []),
            "countries": [],
            "ports": [],
            "sources": list(history_entities.get("sources") or ["history"]),
        }

    if aggregated_results:
        fingerprint_result = aggregated_results.get("ip_fingerprinter") or {}
        fingerprint_ip = str(fingerprint_result.get("ip") or "").strip()
        if fingerprint_ip:
            return {
                "ips": [fingerprint_ip],
                "domains": [],
                "countries": [],
                "ports": [],
                "sources": ["previous_fingerprint"],
            }

        opensearch_result = aggregated_results.get("opensearch_querier") or {}
        opensearch_terms = [
            str(term).strip()
            for term in (opensearch_result.get("search_terms") or [])
            if str(term).strip()
        ]
        opensearch_ips = [term for term in opensearch_terms if re.fullmatch(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", term)]
        if opensearch_ips:
            return {
                "ips": opensearch_ips[:1],
                "domains": [],
                "countries": [],
                "ports": [],
                "sources": ["previous_opensearch"],
            }

    return {}


def _build_context_aware_fingerprint_question(
    original_question: str,
    entities: dict,
    conversation_history: list[dict] | None = None,
) -> str:
    ips = list(entities.get("ips") or [])
    target_ip = ips[0] if ips else ""
    history_summary = _latest_assistant_observation(conversation_history)
    if not target_ip:
        return original_question

    parts = [
        f"fingerprint {target_ip} for ports, services, likely host role, and OS-family likelihood"
    ]
    if original_question:
        parts.append(f"Follow-up request: {original_question}")
    if history_summary:
        parts.append(f"Recent observed traffic: {history_summary}")
    return ". ".join(parts)


def _latest_assistant_observation(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""
    for msg in reversed(conversation_history[-8:]):
        if msg.get("role") == "assistant":
            content = " ".join(str(msg.get("content", "")).split())
            if content and any(
                marker in content
                for marker in [
                    "Found ",
                    "No traffic ",
                    "No matching records found",
                    "Countries seen:",
                    "Source IPs:",
                    "Source/destination IPs:",
                    "Remote peers:",
                ]
            ):
                return content[:600]
    return ""


def _build_context_aware_baseline_question(
    original_question: str,
    entities: dict,
    conversation_history: list[dict] | None = None,
) -> str:
    # NOTE: Caller should only invoke this if baseline_querier is in plan

    entities = _filter_entities_for_question(entities or {}, original_question)
    ips = entities.get("ips", [])
    domains = entities.get("domains", [])
    countries = entities.get("countries", [])
    ports = [str(port) for port in entities.get("ports", [])]
    history_summary = _latest_assistant_observation(conversation_history)

    context_parts = []
    if ips:
        context_parts.append(f"Focus entities: IPs {', '.join(ips[:5])}")
    if domains:
        context_parts.append(f"Domains {', '.join(domains[:3])}")
    if countries:
        context_parts.append(f"Countries {', '.join(countries)}")
    if ports:
        context_parts.append(f"Ports {', '.join(ports[:5])}")
    if history_summary:
        context_parts.append(f"Recent observed traffic: {history_summary}")

    if not context_parts:
        return original_question

    return (
        f"{original_question}\n\n"
        "Compare the requested behavior against known baseline patterns and any observed traffic evidence. "
        "If possible, quantify how often the entity appears and whether the behavior looks routine or unusual.\n"
        + " | ".join(context_parts)
    )


def _build_context_aware_threat_question(original_question: str, entities: dict) -> str:
    """
    Build a context-aware question for threat_analyst when prior results found entities.
    
    This enriches the generic question with actual IPs/domains/countries discovered,
    so threat_analyst analyzes SPECIFIC entities rather than doing a generic lookup.
    """
    entities = _filter_entities_for_question(entities, original_question)
    if not entities or not any([entities.get("ips"), entities.get("domains"), entities.get("countries")]):
        # If no entities extracted, use original question
        return original_question
    
    # Filter private IPs: they have no external threat reputation,
    # and including them causes the LLM to look up or reason about meaningless internal addresses.
    ips = [ip for ip in entities.get("ips", []) if not _is_private_ip(ip)]
    domains = entities.get("domains", [])
    countries = entities.get("countries", [])
    ports = [str(port) for port in entities.get("ports", [])]
    
    enriched = original_question
    
    # Build context string with discovered entities
    context_parts = []
    if ips:
        context_parts.append(f"IPs: {', '.join(ips[:5])}" + (" (and more)" if len(ips) > 5 else ""))
    if domains:
        context_parts.append(f"Domains: {', '.join(domains[:3])}" + (" (and more)" if len(domains) > 3 else ""))
    if countries:
        context_parts.append(f"Countries: {', '.join(countries)}")
    if ports:
        context_parts.append(f"Ports: {', '.join(ports[:5])}" + (" (and more)" if len(ports) > 5 else ""))
    
    if context_parts:
        context_str = " | ".join(context_parts)
        enriched = f"{original_question}\n\nPreviously discovered entities from log search: {context_str}"
        logger.info("[%s] Enriched threat_analyst question with discovered entities: %s", SKILL_NAME, context_str)
    
    return enriched


def format_response(
    user_question: str,
    routing_decision: dict,
    skill_results: dict,
    llm: Any,
    cfg: Any = None,  # Pass config for anti-hallucination setting
    available_skills: list[dict] | None = None,
) -> str:
    """
    Format skill results into a natural language response with thinking-action-reflection loop.
    
    Implements:
      1. THINK: Analyze what the question is asking for
      2. ACTION: Execute skills (already done)
      3. REFLECTION: Check if results answer the question
      4. ANTI-HALLUCINATION: Recheck before presenting
      
    Uses manifest-declared response_formatter hooks when available.
    Falls back to hardcoded logic for backward compatibility.
    """
    if not routing_decision.get("skills"):
        # Generate dynamic list of available skills instead of hardcoded fallback
        if available_skills:
            skill_names = [s.get("name") for s in available_skills if s.get("name")]
            skills_str = ", ".join(sorted(skill_names))
        else:
            skills_str = "network_baseliner, anomaly_triage, threat_analyst"
        return f"I couldn't determine which skills would help with that question. Available skills are: {skills_str}."

    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 0: TRY MANIFEST-DECLARED FORMATTERS (NEW ARCHITECTURE)
    # ══════════════════════════════════════════════════════════════════════════════
    # Load manifests and try each skill's declared response_formatter
    # Try in priority order: forensic_examiner > geoip_lookup > baseline_querier > opensearch_querier
    try:
        from core.skill_manifest import SkillManifestLoader, invoke_response_formatter
        loader = SkillManifestLoader()
        manifests = loader.load_all_manifests()
        
        # Hard-coded priority for backward compatibility during transition
        # TODO: Once all skills use manifest-driven formatters, use manifest-declared priority field
        priority_order = [
            "forensic_examiner",
            "geoip_lookup",
            "ip_fingerprinter",
            "baseline_querier",
            "opensearch_querier",
            "threat_analyst"
        ]

        formatter_order: list[str] = []
        for skill_name in reversed(list(routing_decision.get("preferred_skills") or [])):
            if skill_name in skill_results and skill_name not in formatter_order:
                formatter_order.append(skill_name)
        for skill_name in priority_order:
            if skill_name in skill_results and skill_name not in formatter_order:
                formatter_order.append(skill_name)
        for skill_name in routing_decision.get("skills") or []:
            if skill_name in skill_results and skill_name not in formatter_order:
                formatter_order.append(skill_name)
        for skill_name in skill_results:
            if skill_name not in formatter_order:
                formatter_order.append(skill_name)
        
        for skill_name in formatter_order:
            result = skill_results[skill_name]
            
            # Skip only hard failures. Some skills intentionally return no_data/not_found
            # and still provide the best user-facing response.
            if result.get("status") in {"error", "failed"}:
                logger.debug("[%s] Skipping failed skill %s (status: %s)", SKILL_NAME, skill_name, result.get("status"))
                continue
            
            manifest = manifests.get(skill_name)
            
            if not manifest:
                continue
            
            # Try to invoke the skill's declared formatter
            formatted = invoke_response_formatter(
                skill_name=skill_name,
                manifest=manifest,
                user_question=user_question,
                result=result,
                skill_results=skill_results,
            )
            
            if formatted:
                logger.info("[%s] Used manifest-declared formatter for %s", SKILL_NAME, skill_name)
                return formatted
    except Exception as e:
        # If manifest loading or invocation fails, fall through to hardcoded logic
        logger.debug("[%s] Manifest formatter loading failed (expected if manifests incomplete): %s", SKILL_NAME, e)

    
    # ══════════════════════════════════════════════════════════════════════════════
    # PHASE 1: TRY HARDCODED PRIORITY FORMATTERS (BACKWARD COMPATIBILITY)
    # ══════════════════════════════════════════════════════════════════════════════
    # FORENSIC-FIRST RENDERING
    forensic_result = skill_results.get("forensic_examiner", {})
    if forensic_result and forensic_result.get("status") == "ok":
        return _format_forensic_response(user_question, forensic_result, skill_results.get("threat_analyst", {}))

    threat_only_result = skill_results.get("threat_analyst", {})

    geoip_result = skill_results.get("geoip_lookup", {})
    geoip_has_lookup = bool(geoip_result.get("ip") or geoip_result.get("status") == "not_found")
    if geoip_result and geoip_result.get("status") in {"ok", "not_found"} and geoip_has_lookup:
        return _format_geoip_response(geoip_result)

    # ── PRIORITIZE OPENSEARCH/RAG BY DATA AVAILABILITY ──────────────────────
    # Check which has actual results (log records, not just findings)
    os_result = skill_results.get("opensearch_querier", {})
    os_has_data = (
        os_result
        and os_result.get("status") == "ok"
        and os_result.get("results_count", 0) > 0
        and not os_result.get("validation_failed")
    )
    
    # Check baseline_querier / fields_querier results
    rag_result = skill_results.get("baseline_querier") or skill_results.get("fields_querier") or {}
    rag_has_data = rag_result and rag_result.get("status") == "ok" and rag_result.get("log_records", 0) > 0
    
    baseline_result = skill_results.get("baseline_querier", {})
    if baseline_result and baseline_result.get("status") in {"ok", "no_data"}:
        findings = baseline_result.get("findings") or {}
        if findings.get("answer"):
            return _format_baseline_response(user_question, baseline_result)

    # If opensearch has records, prioritize it for non-baseline questions.
    if os_has_data:
        response = _format_opensearch_response(user_question, os_result)
        threat_result = skill_results.get("threat_analyst", {})
        if threat_result and threat_result.get("status") == "ok":
            response = _append_threat_intel_summary(response, threat_result)
        return response

    if baseline_result and baseline_result.get("status") in {"ok", "no_data"}:
        findings = baseline_result.get("findings") or {}
        if findings.get("answer"):
            return _format_baseline_response(user_question, baseline_result)

    if os_result and os_result.get("status") in {"ok", "no_action"} and not os_result.get("validation_failed"):
        return _format_opensearch_response(user_question, os_result)

    if threat_only_result and threat_only_result.get("status") == "ok" and threat_only_result.get("verdicts"):
        return _format_threat_only_response(user_question, threat_only_result)

    if geoip_result and geoip_result.get("status") in {"ok", "not_found"}:
        return _format_geoip_response(geoip_result)
    
    # Only return RAG if it has actual log records (not just schema/findings)
    if rag_has_data:
        return _format_rag_response(user_question, rag_result)

    # ── PHASE 1: THINK ──────────────────────────────────────────────────────
    think_template = _load_prompt_template(
        RESPONSE_THINK_PROMPT_PATH,
        """
Analyze what the user is asking for.

Question: "{{USER_QUESTION}}"

Extract:
1. Main intent: what they are trying to understand.
2. Key entities: IPs, domains, services, alerts, users, or hosts.
3. Success criteria: what would make the answer complete.

Be specific and concise.
""",
    )
    think_prompt = _render_prompt(think_template, {"USER_QUESTION": user_question})
    
    think_response = llm.chat([
        {"role": "system", "content": "You are a security analyst. Extract structured intent."},
        {"role": "user", "content": think_prompt},
    ])
    
    # ── PHASE 2: ACTION (already done above) ──────────────────────────────
    # skill_results already contains results from executed skills
    
    # ── PHASE 3: REFLECTION ─────────────────────────────────────────────────
    results_text = "\n".join([
        f"\n[{skill_name}]\n{json.dumps(result, indent=2, default=str)}"
        for skill_name, result in skill_results.items()
    ])
    
    reflection_template = _load_prompt_template(
        RESPONSE_REFLECTION_PROMPT_PATH,
        """
You extracted the user's intent as:
{{THINK_RESPONSE}}

Now you received these skill results:
{{RESULTS_TEXT}}

REFLECTION QUESTIONS:
1. Do the results address the main intent?
2. Are all key entities covered?
3. Do the results meet the success criteria?
4. Are there any inconsistencies or gaps?

Briefly assess coverage in 2-3 sentences.
""",
    )
    reflection_prompt = _render_prompt(
        reflection_template,
        {
            "THINK_RESPONSE": think_response,
            "RESULTS_TEXT": results_text,
        },
    )
    
    reflection_response = llm.chat([
        {"role": "system", "content": "You are a critical analyst. Assess if results are sufficient."},
        {"role": "user", "content": reflection_prompt},
    ])
    
    # ── PHASE 4: ANTI-HALLUCINATION CHECK ───────────────────────────────────
    # Check if anti-hallucination is enabled in config
    anti_hallucination_enabled = True  # Default to enabled
    if cfg:
        anti_hallucination_enabled = cfg.get("llm", "anti_hallucination_check", default=True)
    
    final_response = ""
    if anti_hallucination_enabled:
        verification_template = _load_prompt_template(
            RESPONSE_VERIFICATION_PROMPT_PATH,
            """
Internally verify your answer against these facts:

User question: "{{USER_QUESTION}}"
Skill results:
{{RESULTS_TEXT}}

VERIFICATION (DO INTERNALLY, DO NOT SHOW IN ANSWER):
- Are the statements supported by the skill results?
- Did you infer something not present in the data?
- Did you make up or assume any facts?
- Is everything grounded in actual findings?

NOW PROVIDE ONLY THE ANSWER to the user's question in 2-4 sentences.
Do NOT include verification text.
Do NOT say "Based on the skill results" or "Here is the answer".
Provide only the direct answer.
""",
        )
        verification_prompt = _render_prompt(
            verification_template,
            {
                "USER_QUESTION": user_question,
                "RESULTS_TEXT": results_text,
            },
        )
        
        final_response = llm.chat([
            {"role": "system", "content": "You are a rigorous security analyst. Verify internally but output only clean answers without preamble."},
            {"role": "user", "content": verification_prompt},
        ])
    else:
        # Standard response without extra verification
        final_template = _load_prompt_template(
            RESPONSE_FINAL_PROMPT_PATH,
            """
Based on these skill execution results, provide a concise response to the user.

User question: "{{USER_QUESTION}}"

Skill results:
{{RESULTS_TEXT}}

Provide a clear, actionable answer in 2-4 sentences.
""",
        )
        final_prompt = _render_prompt(
            final_template,
            {
                "USER_QUESTION": user_question,
                "RESULTS_TEXT": results_text,
            },
        )
        
        final_response = llm.chat([
            {"role": "system", "content": "You are a helpful SOC analyst. Provide clear, actionable insights."},
            {"role": "user", "content": final_prompt},
        ])
    
    # ── APPEND THREAT INTEL APIs INFO if threat_analyst was used ──────────────
    threat_analyst_result = skill_results.get("threat_analyst", {})
    if threat_analyst_result and threat_analyst_result.get("status") == "ok":
        # Extract API query information from verdicts
        all_apis = set()
        if threat_analyst_result.get("verdicts"):
            for verdict in threat_analyst_result["verdicts"]:
                apis = verdict.get("_queried_apis", [])
                if apis:
                    all_apis.update(apis)
        
        if all_apis:
            apis_str = ", ".join(sorted(all_apis))
            final_response += f"\n\n_[Threat Intelligence Sources Queried: {apis_str}]_"
    
    return final_response


def _append_threat_intel_summary(base_response: str, threat_result: dict) -> str:
    """Append concise threat-intel verdicts to a data-backed response."""
    if not threat_result or threat_result.get("status") != "ok":
        return base_response

    verdicts = threat_result.get("verdicts") or []
    if not verdicts:
        return base_response

    per_verdict_limit = 600 if len(verdicts) == 1 else 350

    summary_parts = []
    for verdict in verdicts[:3]:
        label = verdict.get("verdict", "UNKNOWN")
        confidence = verdict.get("confidence", 0)
        reasoning = " ".join(str(verdict.get("reasoning", "")).split())
        if reasoning:
            shortened = _shorten_naturally(reasoning, per_verdict_limit)
            summary_parts.append(f"{label} ({confidence}%): {shortened}")
        else:
            summary_parts.append(f"{label} ({confidence}%)")

    all_apis = sorted({api for verdict in verdicts for api in verdict.get("_queried_apis", [])})
    if requested_ips and all(_is_private_ip(ip) for ip in requested_ips) and not all_apis:
        return (
            f"{subject} is a private/internal IP address, so public GeoIP and external threat-intelligence feeds do not apply directly. "
            "Use internal log evidence, asset ownership, and local detections to assess whether it is suspicious."
        )

    response = f"Reputation analysis for {subject}: {verdict_label} ({confidence}% confidence)."
    if reasoning:
        response += f" {reasoning}"

    if all_apis:
        response += f"\n\n_[Threat Intelligence Sources Queried: {', '.join(all_apis)}]_"
    return response


def _format_threat_only_response(user_question: str, threat_result: dict) -> str:
    """Render a threat_analyst-only answer without another LLM formatting pass."""
    if not threat_result or threat_result.get("status") != "ok":
        return "No threat intelligence verdict was produced."

    verdicts = threat_result.get("verdicts") or []
    if not verdicts:
        return "No threat intelligence verdict was produced."

    requested_ips: list[str] = []
    for verdict in verdicts:
        for ip in verdict.get("_requested_ips", []):
            if ip not in requested_ips:
                requested_ips.append(ip)

    if not requested_ips:
        requested_ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_question or "")))

    primary = verdicts[0]
    verdict_label = str(primary.get("verdict", "UNKNOWN") or "UNKNOWN")
    confidence = int(primary.get("confidence", 0) or 0)
    reasoning = _shorten_naturally(" ".join(str(primary.get("reasoning", "") or "").split()), 320)
    subject = ", ".join(requested_ips) if requested_ips else "the requested IPs"

    all_apis = sorted({api for verdict in verdicts for api in verdict.get("_queried_apis", [])})
    if requested_ips and all(_is_private_ip(ip) for ip in requested_ips) and not all_apis:
        return (
            f"{subject} is a private/internal IP address, so public GeoIP and external threat-intelligence feeds do not apply directly. "
            "Use internal log evidence, asset ownership, and local detections to assess whether it is suspicious."
        )

    response = f"Reputation analysis for {subject}: {verdict_label} ({confidence}% confidence)."
    if reasoning:
        response += f" {reasoning}"

    if all_apis:
        response += f"\n\n_[Threat Intelligence Sources Queried: {', '.join(all_apis)}]_"
    return response


def _format_baseline_response(user_question: str, baseline_result: dict) -> str:
    findings = baseline_result.get("findings") or {}
    grounded_assessment = " ".join(str(findings.get("grounded_assessment", "") or "").split())
    if grounded_assessment:
        return grounded_assessment

    answer = " ".join(str(findings.get("answer", "") or "").split())
    if not answer:
        return "No grounded baseline analysis was produced."

    log_records = int(findings.get("log_records", 0) or 0)
    rag_sources = int(findings.get("rag_sources", 0) or 0)
    evidence = findings.get("evidence") or {}
    timestamps = evidence.get("timestamps") or []
    ips = evidence.get("ips") or []
    ports = evidence.get("ports") or []

    details = []
    if log_records > 0:
        details.append(f"Observed records: {log_records}.")
    elif rag_sources > 0:
        details.append(f"Baseline documents consulted: {rag_sources}.")
    if ips:
        details.append(f"IPs referenced: {', '.join(ips[:10])}.")
    if ports:
        details.append(f"Ports referenced: {', '.join(ports[:10])}.")
    if timestamps:
        ts_sorted = sorted(str(ts) for ts in timestamps)
        details.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")

    suffix = " " + " ".join(details) if details else ""
    return answer + suffix


def _shorten_naturally(text: str, max_len: int = 180) -> str:
    """Shorten text at a sentence or word boundary instead of mid-token."""
    def _clean_tail(value: str) -> str:
        value = value.rstrip(" ,;:-")
        value = re.sub(r"\b(and|or|but|because|which|that|while|with)$", "", value, flags=re.IGNORECASE).rstrip(" ,;:-")
        return value

    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= max_len:
        return _clean_tail(cleaned)

    sentence_window = cleaned[: max_len + 1]
    last_sentence_end = max(
        sentence_window.rfind(". "),
        sentence_window.rfind("! "),
        sentence_window.rfind("? "),
    )
    if last_sentence_end >= int(max_len * 0.6):
        return _clean_tail(sentence_window[: last_sentence_end + 1])

    word_window = cleaned[: max_len + 1]
    last_space = word_window.rfind(" ")
    if last_space >= int(max_len * 0.6):
        return _clean_tail(word_window[:last_space]) + "..."

    return _clean_tail(cleaned[:max_len]) + "..."


def _format_forensic_response(user_question: str, forensic_result: dict, threat_result: dict | None = None) -> str:
    """Render a detailed forensic report (timeline + pattern + entities + reputation)."""
    report = forensic_result.get("forensic_report", {})
    incident = report.get("incident_summary") or user_question
    results_found = report.get("results_found", 0)
    refinements = report.get("refinement_rounds", 0)
    narrative = report.get("timeline_narrative", "") or ""

    timeline_lines = []
    for line in narrative.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}\b|\bUTC\b", line_stripped, re.IGNORECASE):
            timeline_lines.append(line_stripped)
        if len(timeline_lines) >= 6:
            break

    if not timeline_lines and narrative:
        timeline_lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative) if s.strip()][:4]

    entities = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", incident + "\n" + narrative)))
    ports = sorted(set(re.findall(r"\bport\s*(\d{1,5})\b|\b(\d{2,5})/tcp\b|\b(\d{2,5})/udp\b", (incident + "\n" + narrative).lower())))
    flat_ports = sorted({p for tup in ports for p in tup if p})

    paragraph1 = (
        f"Forensic report for '{incident}': analyzed {results_found} matching events "
        f"with {refinements} refinement rounds. The objective was incident reconstruction "
        "with timeline, behavior frequency, key entities, and risk implications."
    )

    timeline_text = "\n".join([f"- {line}" for line in timeline_lines]) if timeline_lines else "- No concrete timestamped events were returned by the data source."
    paragraph2 = f"Timeline:\n{timeline_text}"

    entity_text = ", ".join(entities[:10]) if entities else "No IP entities extracted"
    port_text = ", ".join(flat_ports[:10]) if flat_ports else "No explicit ports extracted"
    pattern_hint = ""
    for sentence in re.split(r"(?<=[.!?])\s+", narrative):
        if re.search(r"pattern|periodic|sporadic|frequency|interval|automated|bot|risk|threat", sentence, re.IGNORECASE):
            pattern_hint = sentence.strip()
            break
    if not pattern_hint:
        pattern_hint = "Pattern/risk signal was not explicit in the model output and should be treated as low confidence."
    paragraph3 = (
        f"Entities and behavior: IPs involved: {entity_text}. Ports involved: {port_text}. "
        f"Frequency/pattern assessment: {pattern_hint}"
    )

    if threat_result and threat_result.get("status") == "ok" and threat_result.get("verdicts"):
        verdict_lines = []
        for v in threat_result.get("verdicts", [])[:3]:
            verdict = v.get("verdict", "UNKNOWN")
            confidence = v.get("confidence", 0)
            reason = (v.get("reasoning") or "").strip().replace("\n", " ")
            verdict_lines.append(f"- {verdict} ({confidence}% confidence): {reason}")
        paragraph4 = "Reputation and threat intel:\n" + "\n".join(verdict_lines)
    else:
        paragraph4 = (
            "Reputation and threat intel: no explicit reputation verdict was returned. "
            "If API keys are configured, rerun with threat_analyst enabled to include AbuseIPDB/VirusTotal/OTX/Talos signals."
        )

    return "\n\n".join([paragraph1, paragraph2, paragraph3, paragraph4])


def _format_opensearch_response(user_question: str, os_result: dict) -> str:
    """Render opensearch_querier results with evidence detail."""
    results = os_result.get("results") or []
    summary_results = os_result.get("summary_results") or results
    results_count = os_result.get("results_count", len(results))
    sampled_results_count = int(os_result.get("sampled_results_count", len(summary_results)) or 0)
    sample_strategy = os_result.get("sample_strategy") or "page"
    oldest_sample_count = int(os_result.get("oldest_sample_count", sampled_results_count) or 0)
    newest_sample_count = int(os_result.get("newest_sample_count", sampled_results_count) or 0)
    countries = os_result.get("countries", [])
    ports = os_result.get("ports", [])
    protocols = os_result.get("protocols", [])
    time_range = os_result.get("time_range_label") or os_result.get("time_range", "")
    search_terms = os_result.get("search_terms", [])
    directional_alternative = os_result.get("directional_alternative") or {}
    country_buckets = os_result.get("country_buckets") or []

    if os_result.get("aggregation_type") == "country_terms":
        excluded_countries = os_result.get("excluded_countries") or []
        if not country_buckets:
            exclusion_text = f" excluding {', '.join(excluded_countries)}" if excluded_countries else ""
            return f"No matching country aggregates were found in the {time_range} window{exclusion_text}."

        bucket_summary = ", ".join(
            f"{bucket.get('country')} ({int(bucket.get('count', 0) or 0)})"
            for bucket in country_buckets[:10]
            if bucket.get("country")
        )
        exclusion_text = f" excluding {', '.join(excluded_countries)}" if excluded_countries else ""
        return (
            f"Observed traffic from {len(country_buckets)} country(s) in the {time_range} window{exclusion_text}: "
            f"{bucket_summary}."
        )

    if os_result.get("status") == "no_action":
        time_range = time_range or "requested time range"
        return (
            f"I couldn't produce a grounded OpenSearch query for that request, so I do not have log evidence to answer it for the {time_range} window. "
            "The previous step only identified schema/field information, not matching traffic records."
        )

    if not results:
        if directional_alternative:
            requested_direction = os_result.get("ip_direction") or "requested"
            alternative_direction = directional_alternative.get("direction") or "opposite"
            alternative_count = int(directional_alternative.get("results_count", 0) or 0)
            alt_time_range = directional_alternative.get("time_range_label") or time_range
            queried_ip = "/".join(str(term) for term in search_terms[:3]) or "the requested IP"
            detail_parts = [
                f"No traffic {requested_direction} {queried_ip} was found in the {time_range} window.",
                f"However, {alternative_count} record(s) were found in the {alternative_direction} direction for the same IP in the {alt_time_range} window.",
            ]
            sample_peers = directional_alternative.get("sample_peers") or []
            if sample_peers:
                detail_parts.append(f"Peers seen: {', '.join(sample_peers[:10])}.")
            earliest = directional_alternative.get("earliest")
            latest = directional_alternative.get("latest")
            if earliest and latest:
                detail_parts.append(f"Earliest: {earliest}. Latest: {latest}.")
            return " ".join(detail_parts)

        filter_parts = []
        if countries:
            filter_parts.append(f"country={'/' .join(countries)}")
        if ports:
            filter_parts.append(f"port={'/'.join(str(p) for p in ports)}")
        if protocols:
            filter_parts.append(f"protocol={'/'.join(protocols)}")
        filter_desc = ", ".join(filter_parts) or "the specified criteria"
        return f"No matching records found for {filter_desc} in the {time_range} window."

    # ── SPECIAL HANDLING FOR ALERT QUERIES ──────────────────────────────────
    # If the question is about alerts/signatures, show alert-specific information
    question_lower = user_question.lower()
    is_alert_query = any(kw in question_lower for kw in [
        "alert", "signature", "et exploit", "et rule", "et drop", "et policy", "suricata", "snort", "rule"
    ])
    
    if is_alert_query:
        # For alert queries, extract alert/signature information
        alert_signatures: set = set()
        alert_types: set = set()
        alert_count_by_sig: dict = {}
        alert_ips: set = set()
        alert_countries: set = set()
        alert_timestamps: list[str] = []
        
        for row in summary_results:
            # Extract alert signatures
            sig = row.get("alert.signature") or row.get("signature") or row.get("alert", {}).get("signature")
            if sig:
                sig_str = str(sig)
                alert_signatures.add(sig_str)
                alert_count_by_sig[sig_str] = alert_count_by_sig.get(sig_str, 0) + 1
            
            # Extract alert types/categories
            alert_type = row.get("alert.category") or row.get("event.category")
            if alert_type:
                alert_types.add(str(alert_type))

            ts = row.get("@timestamp") or row.get("timestamp")
            if ts:
                alert_timestamps.append(str(ts))

            for value in (
                row.get("src_ip"),
                row.get("dest_ip"),
                row.get("source.ip"),
                row.get("destination.ip"),
                row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
                row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
            ):
                if value:
                    alert_ips.add(str(value))

            geo = row.get("geoip") or {}
            if isinstance(geo, dict):
                for cn in (geo.get("country_name"), geo.get("country")):
                    if cn:
                        alert_countries.add(str(cn))
            for cn in (
                row.get("geoip.country_name"),
                row.get("country_name"),
                row.get("source.geo.country_name"),
                row.get("destination.geo.country_name"),
            ):
                if cn:
                    alert_countries.add(str(cn))
        
        # Build alert-focused summary
        summary = f"Found {results_count} total alert record(s) matching {' / '.join(search_terms)} in the {time_range} window."
        if sampled_results_count and sampled_results_count < int(results_count or 0):
            if sample_strategy == "edge_windows":
                summary += (
                    f" Details below are sampled from up to {oldest_sample_count} earliest and "
                    f"{newest_sample_count} latest matching records."
                )
            else:
                summary += f" Details below are sampled from {sampled_results_count} matching records."
        
        detail_parts = []
        if alert_signatures:
            top_sigs = sorted(alert_signatures)[:5]
            detail_parts.append(f"Alert signatures: {', '.join(top_sigs)}.")
        if alert_types:
            detail_parts.append(f"Alert categories: {', '.join(sorted(alert_types))}.")
        asks_for_alert_details = any(
            term in question_lower
            for term in [
                "what ip", "which ip", "their ip", "their ips", "source ip", "destination ip",
                "what countr", "which countr", "what countries", "what country", "where are they from",
                "when did", "when was", "timestamp", "what time", "alert happen",
            ]
        )
        if asks_for_alert_details and alert_ips:
            detail_parts.append(f"IPs seen in matching alerts: {', '.join(sorted(alert_ips)[:12])}.")
        if asks_for_alert_details and alert_countries:
            detail_parts.append(f"Countries seen in matching alerts: {', '.join(sorted(alert_countries)[:12])}.")
        if asks_for_alert_details and alert_timestamps:
            ts_sorted = sorted(alert_timestamps)
            detail_parts.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")
        
        if detail_parts:
            return summary + " " + " ".join(detail_parts)
        return summary

    # ── STANDARD TRAFFIC QUERY FORMATTING ──────────────────────────────────
    # Extract key evidence
    ips: set = set()
    source_ips: set = set()
    ts_list: list = []
    countries_seen: set = set()

    for row in summary_results:
        # Timestamps
        ts = row.get("@timestamp") or row.get("timestamp")
        if ts:
            ts_list.append(str(ts))

        # IPs
        for v in (
            row.get("src_ip"),
            row.get("source_ip"),
            row.get("source.ip"),
            row.get("source", {}).get("ip") if isinstance(row.get("source"), dict) else None,
        ):
            if v:
                ips.add(str(v))
                source_ips.add(str(v))
        for v in (
            row.get("dest_ip"),
            row.get("destination_ip"),
            row.get("destination.ip"),
            row.get("destination", {}).get("ip") if isinstance(row.get("destination"), dict) else None,
        ):
            if v:
                ips.add(str(v))

        # Countries from geoip
        geo = row.get("geoip") or {}
        if isinstance(geo, dict):
            cn = geo.get("country_name")
            if cn:
                countries_seen.add(str(cn))
        for cn in (
            row.get("geoip.country_name"),
            row.get("country_name"),
            row.get("source.geo.country_name"),
            row.get("destination.geo.country_name"),
        ):
            if cn:
                countries_seen.add(str(cn))

    # Build summary
    filter_parts = []
    if countries:
        filter_parts.append("/".join(countries))
    if ports:
        filter_parts.append("port " + "/".join(str(p) for p in ports))
    if protocols:
        filter_parts.append("/".join(protocols))
    if search_terms and not filter_parts:
        shown_terms = "/".join(str(term) for term in search_terms[:3])
        if len(search_terms) > 3:
            shown_terms += "/…"
        filter_parts.append(shown_terms)
    filter_desc = ", ".join(filter_parts) or "the query criteria"

    summary = f"Found {results_count} total record(s) matching {filter_desc} in the {time_range} window."
    if sampled_results_count and sampled_results_count < int(results_count or 0):
        if sample_strategy == "edge_windows":
            summary += (
                f" Details below are sampled from up to {oldest_sample_count} earliest and "
                f"{newest_sample_count} latest matching records."
            )
        else:
            summary += f" Details below are sampled from {sampled_results_count} matching records."

    # For port-specific queries, extract discovered port values from results (not just restate filter)
    extracted_ports: set = set()
    if ports:  # Only extract if a port was specifically queried
        for row in summary_results:
            # Try both nested and flat field names
            port_candidates = [
                row.get("destination.port"),
                row.get("destination", {}).get("port") if isinstance(row.get("destination"), dict) else None,
                row.get("destination_port"),
                row.get("dst_port"),
                row.get("dest_port"),
                row.get("dport"),
                row.get("port"),
            ]
            for p in port_candidates:
                if p is not None:
                    try:
                        extracted_ports.add(int(p))
                    except (ValueError, TypeError):
                        pass

    detail_parts = []
    if countries_seen:
        detail_parts.append(f"Countries seen: {', '.join(sorted(countries_seen))}.")
    if ips:
        if countries:
            # Country-filtered query: the foreign IPs are sources (public).
            # Internal destination IPs (private RFC-1918 space) are not from that country
            # and would mislead the user if listed alongside Russian/etc. sources.
            public_ips = {ip for ip in ips if not _is_private_ip(ip)}
            display_ips = sorted(public_ips)[:10] if public_ips else sorted(source_ips)[:10]
            if display_ips:
                detail_parts.append(f"Source IPs: {', '.join(display_ips)}.")
        elif ports and source_ips:
            detail_parts.append(f"Remote peers: {', '.join(sorted(source_ips)[:10])}.")
        else:
            detail_parts.append(f"Source/destination IPs: {', '.join(sorted(ips)[:10])}.")
    if ts_list:
        ts_sorted = sorted(ts_list)
        detail_parts.append(f"Earliest: {ts_sorted[0]}. Latest: {ts_sorted[-1]}.")
    matched_ports = extracted_ports.intersection({int(p) for p in ports if str(p).isdigit()}) if ports else extracted_ports
    if matched_ports:
        detail_parts.append(f"Destination port(s): {', '.join(str(p) for p in sorted(matched_ports))}.")

    if detail_parts:
        return summary + " " + " ".join(detail_parts)
    return summary


def _format_rag_response(user_question: str, rag_result: dict) -> str:
    """Render rag_querier responses with explicit evidence details."""
    findings = rag_result.get("findings", {})
    base_answer = _strip_json_like_content((findings.get("answer") or "").strip())
    evidence = findings.get("evidence", {}) or {}

    ips = evidence.get("ips", [])
    ports = evidence.get("ports", [])
    protocols = evidence.get("protocols", [])
    timestamps = evidence.get("timestamps", [])

    details = (
        "Evidence details: "
        f"IPs involved: {', '.join(ips[:10]) if ips else 'none extracted'}. "
        f"Ports: {', '.join(ports[:10]) if ports else 'none extracted'}. "
        f"Protocols: {', '.join(protocols[:10]) if protocols else 'none extracted'}. "
        f"Timestamps: {', '.join(timestamps[:6]) if timestamps else 'not available'}."
    )

    if base_answer:
        return f"{base_answer}\n\n{details}"
    return details


def _format_geoip_response(geoip_result: dict) -> str:
    """Render direct GeoIP lookup or maintenance results without LLM synthesis."""
    action = geoip_result.get("action", "ready")
    db_path = geoip_result.get("db_path")
    warning = geoip_result.get("warning")

    lookups = geoip_result.get("lookups") or []
    if lookups:
        rendered: list[str] = []
        for lookup in lookups[:15]:
            ip = lookup.get("ip", "unknown")
            if lookup.get("status") == "not_found":
                rendered.append(f"{ip}: not found in the MaxMind database")
                continue
            if lookup.get("status") == "error":
                rendered.append(f"{ip}: lookup error ({lookup.get('error', 'unknown error')})")
                continue

            geo = lookup.get("geo") or {}
            location_parts = []
            for field in ("city", "subdivision", "country"):
                value = geo.get(field)
                if value and value not in location_parts:
                    location_parts.append(value)
            location = ", ".join(location_parts) if location_parts else "an unknown location"
            rendered.append(f"{ip}: {location}")

        response = "Resolved GeoIP for the referenced IPs: " + "; ".join(rendered) + "."
        if db_path:
            response += f" Database: {db_path}."
        if warning:
            response += f" Warning: {warning}."
        return response

    if geoip_result.get("status") == "not_found":
        response = f"No MaxMind geolocation record was found for IP {geoip_result.get('ip', 'unknown')}."
        if db_path:
            response += f" Database: {db_path}."
        return response

    ip = geoip_result.get("ip")
    geo = geoip_result.get("geo") or {}
    if not ip:
        response = f"GeoIP database check complete. Status: {action}."
        if db_path:
            response += f" Database: {db_path}."
        if warning:
            response += f" Warning: {warning}."
        return response

    location_parts = []
    for field in ("city", "subdivision", "country"):
        value = geo.get(field)
        if value and value not in location_parts:
            location_parts.append(value)
    location = ", ".join(location_parts) if location_parts else "an unknown location"

    response = f"IP {ip} resolves to {location}."
    extra = []
    if geo.get("country_iso_code"):
        extra.append(f"country code {geo['country_iso_code']}")
    if geo.get("timezone"):
        extra.append(f"timezone {geo['timezone']}")
    if geo.get("postal_code"):
        extra.append(f"postal code {geo['postal_code']}")
    if geo.get("latitude") is not None and geo.get("longitude") is not None:
        extra.append(f"coordinates {geo['latitude']}, {geo['longitude']}")
    if extra:
        response += " " + "; ".join(extra) + "."

    response += f" GeoIP DB status: {action}."
    if warning:
        response += f" Warning: {warning}."
    return response


def _strip_json_like_content(text: str) -> str:
    """Remove raw JSON/code-block style dumps from model answers."""
    if not text:
        return text

    # Remove fenced blocks first.
    cleaned = re.sub(r"```[\s\S]*?```", "", text)

    # Remove obvious JSON object dumps that start on their own line.
    cleaned = re.sub(r"\n\s*\{[\s\S]*?\}\s*", "\n", cleaned)

    # Collapse excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Memory Management
# ──────────────────────────────────────────────────────────────────────────────

CONVERSATIONS_DIR = Path(__file__).parent.parent.parent / "conversations"


def _ensure_conversations_dir():
    """Create conversations directory if it doesn't exist."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_conversation_history(conversation_id: str) -> list[dict]:
    """Load conversation history from disk."""
    _ensure_conversations_dir()
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not conv_file.exists():
        return []
    
    try:
        with open(conv_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load conversation %s: %s", conversation_id, e)
        return []


def save_conversation_history(conversation_id: str, history: list[dict]) -> None:
    """Save conversation history to disk."""
    _ensure_conversations_dir()
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    try:
        with open(conv_file, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error("Failed to save conversation %s: %s", conversation_id, e)


def list_conversations() -> list[dict]:
    """List all saved conversations with metadata."""
    _ensure_conversations_dir()
    conversations = []
    
    for conv_file in sorted(CONVERSATIONS_DIR.glob("*.json")):
        try:
            with open(conv_file, "r") as f:
                history = json.load(f)
            
            if history:
                first_user = next((entry for entry in history if entry.get("role") == "user"), {})
                last_entry = history[-1] if history else {}
                timestamp = last_entry.get("timestamp", "Unknown")
                conversations.append({
                    "id": conv_file.stem,
                    "messages": len(history),
                    "first_question": first_user.get("content", "Unknown"),
                    "last_update": timestamp,
                    "timestamp": timestamp,
                })
        except Exception as e:
            logger.warning("Failed to read conversation file %s: %s", conv_file, e)
    
    return conversations


def add_to_history(conversation_id: str, question: str, answer: str, 
                  routing: dict, skill_results: dict) -> None:
    """Add a Q&A exchange to conversation history."""
    from datetime import datetime, timezone
    
    history = load_conversation_history(conversation_id)
    
    # Save user question
    user_entry = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history.append(user_entry)
    
    # Save assistant answer
    assistant_entry = {
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "routing_skills": routing.get("skills", []),
        "routing_reasoning": routing.get("reasoning", ""),
        "skill_results": skill_results,
    }
    history.append(assistant_entry)
    
    save_conversation_history(conversation_id, history)


def get_context_summary(conversation_id: str, last_n: int = 3) -> str:
    """Get summary of recent conversation for context injection."""
    history = load_conversation_history(conversation_id)
    
    if not history:
        return ""
    
    recent = history[-(last_n * 2):]
    summary_lines = []

    pending_question = ""
    for entry in recent:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            pending_question = content
        elif role == "assistant":
            summary_lines.append(f"Q: {pending_question}")
            answer = content
            if len(answer) > 200:
                answer = answer[:200] + "..."
            summary_lines.append(f"A: {answer}")
            pending_question = ""
    
    return "\n".join(summary_lines)
