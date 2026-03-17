"""Load and manage skill manifests for modular supervisor routing."""

import importlib
import logging
import re
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)



class SkillManifestLoader:
    """Load skill manifests to enable modular supervisor decision-making."""

    def __init__(self, skills_dir: Path | str = None):
        """Initialize with path to skills directory."""
        if skills_dir is None:
            # Default to skills/ directory relative to this file
            skills_dir = Path(__file__).parent.parent / "skills"
        self.skills_dir = Path(skills_dir)

    def load_all_manifests(self) -> dict[str, dict[str, Any]]:
        """Load manifest.yaml from all skill directories.
        
        Returns:
            Dict mapping skill_name -> manifest content
        """
        manifests = {}
        
        if not self.skills_dir.exists():
            logger.warning("Skills directory not found: %s", self.skills_dir)
            return manifests
        
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            
            manifest_path = skill_dir / "manifest.yaml"
            if not manifest_path.exists():
                continue
            
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = yaml.safe_load(f)
                
                if not manifest:
                    logger.warning("Empty manifest in %s", skill_dir.name)
                    continue
                
                skill_name = manifest.get("name", skill_dir.name)
                manifests[skill_name] = manifest
                logger.debug("Loaded manifest for skill: %s", skill_name)
            
            except Exception as e:
                logger.warning("Failed to load manifest in %s: %s", skill_dir.name, e)
        
        return manifests

    def build_supervisor_context(self, manifests: dict[str, dict]) -> str:
        """Generate supervisor prompt context from manifests.
        
        This creates a structured guide the LLM can use to understand
        which skill to choose based on declared capabilities, artifacts,
        and prerequisites.
        
        Args:
            manifests: Dict from load_all_manifests()
        
        Returns:
            Formatted string for inclusion in supervisor prompt
        """
        if not manifests:
            return ""
        
        lines = [
            "## Skill Contracts (Dynamically Loaded from Manifests)\n",
            "### Capability, Prerequisite, and Artifact Summary\n",
        ]
        
        for skill_name, manifest in sorted(manifests.items()):
            lines.append(f"\n**{skill_name}**")

            description = str(manifest.get("description") or "").strip()
            if description:
                lines.append(f"  - Purpose: {description}")

            answer_types = manifest_answer_types(manifest)
            if answer_types:
                lines.append(f"  - Answers: {', '.join(answer_types)}")

            non_goals = manifest_non_goals(manifest)
            if non_goals:
                lines.append(f"  - Not for: {', '.join(non_goals)}")

            routing_group = str(manifest.get("routing_group") or "").strip()
            capability_groups = [str(group).strip() for group in (manifest.get("capability_groups") or []) if str(group).strip()]
            groups = [group for group in [routing_group, *capability_groups] if group]
            if groups:
                lines.append(f"  - Groups: {', '.join(dict.fromkeys(groups))}")

            required_entities = manifest_required_entities(manifest)
            if required_entities:
                lines.append(f"  - Required entities: any of {', '.join(required_entities)}")

            artifact_inputs = manifest_artifact_inputs(manifest)
            if artifact_inputs:
                lines.append(f"  - Needs prior artifacts from: {', '.join(artifact_inputs)}")

            artifact_outputs = manifest_artifact_outputs(manifest)
            if artifact_outputs:
                lines.append(f"  - Produces: {', '.join(artifact_outputs)}")
            
            min_context = manifest.get("min_prior_context", 0)
            if min_context > 0:
                lines.append(f"  - Works best with {min_context}+ prior results")
        
        lines.append("\n")
        return "\n".join(lines)


def manifest_for_skill(manifests: dict[str, dict[str, Any]], skill_name: str) -> dict[str, Any]:
    """Return a manifest dict for a skill name, or an empty mapping when absent."""
    if not skill_name:
        return {}
    manifest = manifests.get(skill_name)
    return manifest if isinstance(manifest, dict) else {}


def manifest_answer_types(manifest: dict[str, Any]) -> list[str]:
    declared = manifest.get("answer_types") or manifest.get("can_answer") or []
    return [str(item).strip() for item in declared if str(item).strip()]


def manifest_non_goals(manifest: dict[str, Any]) -> list[str]:
    declared = manifest.get("non_goals") or manifest.get("cannot_answer") or []
    return [str(item).strip() for item in declared if str(item).strip()]


def manifest_required_entities(manifest: dict[str, Any]) -> list[str]:
    declared = manifest.get("required_entities") or []
    normalized = [str(item).strip().lower() for item in declared if str(item).strip()]
    if normalized:
        return normalized
    if manifest.get("requires_explicit_entity"):
        return ["entity"]
    return []


def manifest_artifact_inputs(manifest: dict[str, Any]) -> list[str]:
    declared = manifest.get("artifact_inputs") or []
    artifact_inputs = [str(item).strip() for item in declared if str(item).strip()]
    if artifact_inputs:
        return artifact_inputs

    prerequisite_groups: list[str] = []
    for prereq in manifest.get("prerequisites") or []:
        if not isinstance(prereq, dict):
            continue
        group = str(prereq.get("group") or "").strip()
        if group and group not in prerequisite_groups:
            prerequisite_groups.append(group)
    return prerequisite_groups


def manifest_artifact_outputs(manifest: dict[str, Any]) -> list[str]:
    declared = manifest.get("artifact_outputs") or manifest.get("returns") or []
    return [str(item).strip() for item in declared if str(item).strip()]


def first_skill_in_group(manifests: dict[str, dict[str, Any]], group: str) -> str | None:
    """Resolve the first declared skill for a manifest routing or capability group."""
    requested_group = str(group or "").strip()
    if not requested_group:
        return None

    for skill_name, manifest in manifests.items():
        if str(manifest.get("routing_group") or "").strip() == requested_group:
            return skill_name

    for skill_name, manifest in manifests.items():
        capability_groups = manifest.get("capability_groups") or []
        if requested_group in capability_groups:
            return skill_name

    return None


def _matches_manifest_pattern(pattern_def: Any, text: str) -> bool:
    if not text:
        return False
    if isinstance(pattern_def, dict):
        regex = str(pattern_def.get("regex") or "").strip()
        if not regex:
            return False
        try:
            return bool(re.search(regex, text, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Invalid manifest regex '%s': %s", regex, exc)
            return False
    return str(pattern_def or "").lower() in text.lower()


def question_has_explicit_entity(user_question: str) -> bool:
    question = str(user_question or "")
    if not question:
        return False

    ipv4_pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ipv6_pattern = r"(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}"
    domain_pattern = r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b"

    if re.search(ipv4_pattern, question, re.IGNORECASE):
        return True
    if re.search(ipv6_pattern, question, re.IGNORECASE):
        return True
    if re.search(domain_pattern, question.lower()):
        return True
    if re.search(r"\b(?:hostname|domain|fqdn|host)\s+\S+", question.lower()):
        return True
    if re.search(r"\b(?:geolocate|where\s+is)\s+\S+", question.lower()):
        return True
    return False


def _question_has_explicit_field_syntax(user_question: str) -> bool:
    return bool(re.search(r"\b[\w.]+\s*[=:<>!]+", str(user_question or "")))


def _group_has_satisfied_results(
    group: str,
    manifests: dict[str, dict[str, Any]],
    current_results: dict[str, Any],
) -> bool:
    if not group or not current_results:
        return False

    for skill_name, result in current_results.items():
        if not isinstance(result, dict) or result.get("status") == "error":
            continue

        manifest = manifests.get(skill_name, {})
        routing_group = str(manifest.get("routing_group") or "").strip()
        capability_groups = set(manifest.get("capability_groups") or [])
        if group != routing_group and group not in capability_groups:
            continue

        if group == "schema_discovery":
            if result.get("field_mappings") or (result.get("findings") or {}).get("field_mappings"):
                return True
            continue

        return True

    return False


def _group_has_any_results(
    group: str,
    manifests: dict[str, dict[str, Any]],
    current_results: dict[str, Any],
) -> bool:
    if not group or not current_results:
        return False

    for skill_name, result in current_results.items():
        if not isinstance(result, dict):
            continue

        manifest = manifests.get(skill_name, {})
        routing_group = str(manifest.get("routing_group") or "").strip()
        capability_groups = set(manifest.get("capability_groups") or [])
        if group == routing_group or group in capability_groups:
            return True

    return False


def _get_nested_value(data: Any, path: str) -> Any:
    current = data
    for part in str(path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _result_predicate_matches(predicate: dict[str, Any], current_results: dict[str, Any]) -> bool:
    skill_name = str(predicate.get("skill") or "").strip()
    if not skill_name:
        return False

    result = current_results.get(skill_name)
    if not isinstance(result, dict):
        return False

    path = str(predicate.get("path") or "").strip()
    value = _get_nested_value(result, path) if path else None

    if predicate.get("truthy") is True and not value:
        return False

    if "equals" in predicate and value != predicate.get("equals"):
        return False

    if "min" in predicate:
        try:
            if float(value or 0) < float(predicate.get("min") or 0):
                return False
        except (TypeError, ValueError):
            return False

    any_paths_truthy = predicate.get("any_paths_truthy") or []
    if any_paths_truthy and not any(_get_nested_value(result, candidate) for candidate in any_paths_truthy):
        return False

    any_text_paths = predicate.get("any_text_paths") or []
    if any_text_paths:
        haystack = " ".join(
            str(_get_nested_value(result, candidate) or "") for candidate in any_text_paths
        ).lower()
        contains_any = [str(token).lower() for token in (predicate.get("contains_any") or []) if str(token).strip()]
        if contains_any and not any(token in haystack for token in contains_any):
            return False
        contains_all = [str(token).lower() for token in (predicate.get("contains_all") or []) if str(token).strip()]
        if contains_all and not all(token in haystack for token in contains_all):
            return False

    return True


def apply_manifest_plan_policies(
    selected_skills: list[str],
    user_question: str,
    available_skills: list[dict],
    all_manifests: dict[str, dict[str, Any]],
    current_results: dict[str, Any] | None = None,
) -> list[str]:
    """Apply generic manifest-declared planning policies to a selected skill list."""
    if not selected_skills or not all_manifests:
        return selected_skills

    current_results = current_results or {}
    available_names = {s.get("name") for s in available_skills if s.get("name")}
    ordered: list[str] = []

    for skill_name in selected_skills:
        manifest = all_manifests.get(skill_name, {})
        for policy in manifest.get("conditional_prerequisites") or []:
            if not isinstance(policy, dict):
                continue

            when_patterns = policy.get("when_any_question_patterns") or []
            if when_patterns and not any(_matches_manifest_pattern(pattern, user_question) for pattern in when_patterns):
                continue

            unless_patterns = policy.get("unless_any_question_patterns") or []
            if unless_patterns and any(_matches_manifest_pattern(pattern, user_question) for pattern in unless_patterns):
                continue

            if policy.get("skip_if_explicit_field_syntax") and _question_has_explicit_field_syntax(user_question):
                continue

            skip_groups = policy.get("skip_if_result_exists_for_groups") or []
            if any(_group_has_satisfied_results(group, all_manifests, current_results) for group in skip_groups):
                continue

            for group in policy.get("groups") or []:
                prerequisite_skill = first_skill_in_group(all_manifests, str(group or "").strip())
                if prerequisite_skill and prerequisite_skill in available_names and prerequisite_skill != skill_name and prerequisite_skill not in ordered:
                    ordered.append(prerequisite_skill)

        if skill_name in available_names and skill_name not in ordered:
            ordered.append(skill_name)

        for policy in manifest.get("conditional_successors") or []:
            if not isinstance(policy, dict):
                continue

            when_patterns = policy.get("when_any_question_patterns") or []
            if when_patterns and not any(_matches_manifest_pattern(pattern, user_question) for pattern in when_patterns):
                continue

            unless_patterns = policy.get("unless_any_question_patterns") or []
            if unless_patterns and any(_matches_manifest_pattern(pattern, user_question) for pattern in unless_patterns):
                continue

            if policy.get("skip_if_explicit_field_syntax") and _question_has_explicit_field_syntax(user_question):
                continue

            skip_groups = policy.get("skip_if_result_exists_for_groups") or []
            if any(_group_has_satisfied_results(group, all_manifests, current_results) for group in skip_groups):
                continue

            for group in policy.get("groups") or []:
                successor_skill = first_skill_in_group(all_manifests, str(group or "").strip())
                if successor_skill and successor_skill in available_names and successor_skill != skill_name and successor_skill not in ordered:
                    ordered.append(successor_skill)

    return ordered


def apply_manifest_recovery_policies(
    selected_skills: list[str],
    user_question: str,
    available_skills: list[dict],
    all_manifests: dict[str, dict[str, Any]],
    current_results: dict[str, Any] | None = None,
    extracted_entities: dict[str, list[Any]] | None = None,
) -> list[str]:
    """Apply generic manifest-declared recovery policies after partial results."""
    if not all_manifests:
        return selected_skills

    current_results = current_results or {}
    extracted_entities = extracted_entities or {}
    available_names = {s.get("name") for s in available_skills if s.get("name")}
    ordered: list[str] = [skill for skill in selected_skills if skill]

    for skill_name, manifest in all_manifests.items():
        if skill_name not in available_names:
            continue

        for policy in manifest.get("conditional_recovery") or []:
            if not isinstance(policy, dict):
                continue

            when_patterns = policy.get("when_any_question_patterns") or []
            if when_patterns and not any(_matches_manifest_pattern(pattern, user_question) for pattern in when_patterns):
                continue

            unless_patterns = policy.get("unless_any_question_patterns") or []
            if unless_patterns and any(_matches_manifest_pattern(pattern, user_question) for pattern in unless_patterns):
                continue

            skip_skills = {
                str(name).strip() for name in (policy.get("skip_if_result_exists_for_skills") or []) if str(name).strip()
            }
            if skip_skills and any(name in current_results for name in skip_skills):
                continue

            skip_groups = policy.get("skip_if_result_exists_for_groups") or []
            if any(_group_has_any_results(str(group or "").strip(), all_manifests, current_results) for group in skip_groups):
                continue

            required_entities = [
                str(name).strip() for name in (policy.get("requires_extracted_entities") or []) if str(name).strip()
            ]
            if required_entities and not all(extracted_entities.get(name) for name in required_entities):
                continue

            required_groups = policy.get("requires_result_for_groups") or []
            if required_groups and not all(
                _group_has_satisfied_results(str(group or "").strip(), all_manifests, current_results)
                for group in required_groups
            ):
                continue

            required_skills = {
                str(name).strip() for name in (policy.get("requires_result_for_skills") or []) if str(name).strip()
            }
            if required_skills and not all(name in current_results for name in required_skills):
                continue

            predicates = policy.get("requires_result_predicates") or []
            if predicates and not all(
                isinstance(predicate, dict) and _result_predicate_matches(predicate, current_results)
                for predicate in predicates
            ):
                continue

            ordered = [name for name in ordered if name != skill_name]
            position = str(policy.get("position") or "back").strip().lower()
            if position == "front":
                ordered.insert(0, skill_name)
            else:
                ordered.append(skill_name)

            for group in policy.get("add_groups_after") or []:
                chained_skill = first_skill_in_group(all_manifests, str(group or "").strip())
                if not chained_skill or chained_skill == skill_name or chained_skill in current_results:
                    continue
                if chained_skill not in available_names:
                    continue
                if chained_skill not in ordered:
                    ordered.append(chained_skill)

            for chained_skill in policy.get("add_skills_after") or []:
                chained_name = str(chained_skill or "").strip()
                if not chained_name or chained_name == skill_name or chained_name in current_results:
                    continue
                if chained_name not in available_names:
                    continue
                if chained_name not in ordered:
                    ordered.append(chained_name)

    deduped: list[str] = []
    for skill in ordered:
        if skill and skill not in deduped:
            deduped.append(skill)
    return deduped


def invoke_response_formatter(skill_name: str, manifest: dict[str, Any], user_question: str, result: dict, skill_results: dict | None = None) -> str | None:
    """Invoke a skill's response_formatter hook if declared in manifest.
    
    Args:
        skill_name: Name of the skill
        manifest: Skill manifest dict (should include response_formatter field)
        user_question: Original user query
        result: Skill result dict
        skill_results: All aggregated skill results
    
    Returns:
        Formatted response string, or None if no formatter found/callable
    """
    formatter_path = manifest.get("response_formatter")
    if not formatter_path:
        return None
    
    try:
        # Parse formatter_path as "module.path:function_name"
        if ":" not in formatter_path:
            logger.warning("[%s] Invalid formatter path (no colon): %s", skill_name, formatter_path)
            return None
        
        module_path, func_name = formatter_path.rsplit(":", 1)
        
        # Dynamically import the module
        try:
            # Import using importlib
            import importlib
            module = importlib.import_module(module_path)
            formatter_func = getattr(module, func_name, None)
            
            if not callable(formatter_func):
                logger.warning(
                    "[%s] Formatter function not callable: %s:%s", 
                    skill_name, module_path, func_name
                )
                return None
            
            # Invoke formatter with available arguments
            return formatter_func(
                user_question=user_question,
                result=result,
                skill_results=skill_results,
            )
        except ImportError as e:
            logger.warning("[%s] Could not import formatter module %s: %s", skill_name, module_path, e)
            return None
        except AttributeError as e:
            logger.warning("[%s] Formatter function %s not found in %s: %s", skill_name, func_name, module_path, e)
            return None
        except TypeError as e:
            # Function signature mismatch—try with fewer args
            try:
                return formatter_func(user_question, result)
            except Exception as e2:
                logger.warning("[%s] Formatter signature error: %s", skill_name, e2)
                return None
    except Exception as e:
        logger.error("[%s] Unexpected error invoking formatter: %s", skill_name, e)
        return None


def apply_routing_guards(
    selected_skills: list[str],
    user_question: str,
    available_skills: list[dict],
    all_manifests: dict[str, dict[str, Any]],
) -> list[str]:
    """Apply manifest-declared routing guards to filter/reorder selected skills.
    
    Routing guards include:
    1. cannot_answer: Filter out skills that declare they cannot answer this type of question
    2. requires_explicit_keywords: Skill only auto-routes if keywords present
    3. explicit_only: Skill must be explicitly requested; replaced with fallback_on_block
    4. reserved_for_followup_only: Skill only used for follow-up research; replaced with fallback
    5. priority_override_keywords: Skill promoted/selected if keywords present; can co-route with other skills
    
    Args:
        selected_skills: List of skill names selected by router
        user_question: Original user question
        available_skills: List of available skill definitions
        all_manifests: Dict mapping skill names to manifest dicts
    
    Returns:
        Filtered and reordered skill list
    """
    SKILL_NAME = "routing_guards"
    
    if not all_manifests:
        return selected_skills
    
    question_lower = user_question.lower().strip()
    
    # ════════════════════════════════════════════════════════════════════════════
    # GUARD -1: FILTER OUT SKILLS THAT CANNOT ANSWER THIS QUESTION
    # ════════════════════════════════════════════════════════════════════════════
    # Check each skill's cannot_answer patterns and remove the skill if it matches
    filtered_by_cannot_answer = []
    for skill_name in selected_skills:
        manifest = all_manifests.get(skill_name, {})
        cannot_answer_patterns = manifest.get("cannot_answer", [])
        
        skill_cannot_answer = False
        for pattern in cannot_answer_patterns:
            pattern_lower = str(pattern).lower()
            if pattern_lower in question_lower:
                logger.info(
                    "[%s] Filtering out %s: cannot_answer pattern '%s' matches question '%s'",
                    SKILL_NAME, skill_name, pattern_lower, user_question[:80]
                )
                skill_cannot_answer = True
                break
        
        if not skill_cannot_answer:
            filtered_by_cannot_answer.append(skill_name)
    
    selected_skills = filtered_by_cannot_answer
    filtered = []
    override_skills = []  # Skills promoted by priority_override_keywords
    search_filters_present = False
    
    # Check for search filter patterns (used by forensic_examiner co-routing)
    search_filter_regex_patterns = []
    for skill_name in all_manifests:
        manifest = all_manifests.get(skill_name, {})
        filter_patterns = manifest.get("search_filter_patterns", [])
        for pattern_def in filter_patterns:
            if isinstance(pattern_def, dict) and "regex" in pattern_def:
                search_filter_regex_patterns.append(pattern_def["regex"])
    
    for pattern in search_filter_regex_patterns:
        if re.search(pattern, question_lower):
            search_filters_present = True
            break
    
    # ════════════════════════════════════════════════════════════════════════════
    # GUARD 0: CHECK FOR PRIORITY OVERRIDES (before processing regular skills)
    # ════════════════════════════════════════════════════════════════════════════
    # Some skills (like forensic_examiner) can override the selection entirely
    # if their priority_override_keywords are detected
    priority_override_triggered = False
    
    for skill_name, manifest in all_manifests.items():
        priority_override_keywords = manifest.get("priority_override_keywords", [])
        if not priority_override_keywords:
            continue
        
        if not {s.get("name") for s in available_skills if s.get("name") == skill_name}:
            # Skill not available
            continue
        
        keywords_match = False
        for keyword_def in priority_override_keywords:
            if isinstance(keyword_def, dict):
                if "regex" in keyword_def:
                    try:
                        if re.search(keyword_def["regex"], question_lower):
                            keywords_match = True
                            break
                    except re.error as e:
                        logger.warning("[%s] Invalid regex in manifest %s: %s", SKILL_NAME, skill_name, e)
            else:
                # String keyword
                if keyword_def.lower() in question_lower:
                    keywords_match = True
                    break
        
        if keywords_match:
            # Priority override triggered - skill replaces the selection
            priority_override_triggered = True
            override_skills = []
            
            # Check if should co-route
            co_route_skills = manifest.get("co_route_skills_if_filters_present", [])
            if co_route_skills and search_filters_present:
                override_skills.extend(co_route_skills)
                logger.info(
                    "[%s] Priority override: %s with co-routed %s (search filters detected)",
                    SKILL_NAME, skill_name, co_route_skills
                )
            
            override_skills.append(skill_name)
            logger.info(
                "[%s] Priority override triggered: %s → %s",
                SKILL_NAME, selected_skills, override_skills
            )
            break  # Only one priority override at a time
    
    # If priority override was triggered, use the override skills; otherwise process normally
    if priority_override_triggered:
        return override_skills
    
    # ════════════════════════════════════════════════════════════════════════════
    # NORMAL SKILL PROCESSING (when no priority overrides)
    # ════════════════════════════════════════════════════════════════════════════
    
    if not selected_skills:
        return selected_skills
    
    for skill in selected_skills:
        manifest = all_manifests.get(skill, {})
        
        # ════════════════════════════════════════════════════════════════════
        # GUARD 1: requires_explicit_keywords (blocks auto-routing)
        # ════════════════════════════════════════════════════════════════════
        requires_explicit = manifest.get("requires_explicit_keywords", [])
        if requires_explicit:
            keywords_found = any(kw.lower() in question_lower for kw in requires_explicit)
            if not keywords_found:
                logger.info(
                    "[%s] Blocked %s - requires explicit keywords not found",
                    SKILL_NAME, skill
                )
                continue

        if manifest.get("requires_explicit_entity", False) and not question_has_explicit_entity(user_question):
            logger.info(
                "[%s] Blocked %s - requires explicit entity but none was found",
                SKILL_NAME, skill
            )
            fallback = manifest.get("fallback_on_block")
            if fallback and fallback not in filtered:
                filtered.append(fallback)
            continue
        
        # ════════════════════════════════════════════════════════════════════
        # GUARD 2: explicit_only (requires explicit mention)
        # ════════════════════════════════════════════════════════════════════
        if manifest.get("explicit_only", False):
            explicit_keywords = manifest.get("explicit_keywords", [])
            keywords_found = any(kw.lower() in question_lower for kw in explicit_keywords)
            
            if not keywords_found:
                logger.info(
                    "[%s] Blocked %s (explicit-only skill, keywords not found)",
                    SKILL_NAME, skill
                )
                # Add fallback skill if specified
                fallback = manifest.get("fallback_on_block")
                if fallback and fallback not in filtered:
                    filtered.append(fallback)
                    logger.info(
                        "[%s] Added fallback %s for blocked %s",
                        SKILL_NAME, fallback, skill
                    )
                continue
        
        # ════════════════════════════════════════════════════════════════════
        # GUARD 3: reserved_for_followup_only (follow-up research only)
        # ════════════════════════════════════════════════════════════════════
        if manifest.get("reserved_for_followup_only", False):
            followup_keywords = manifest.get("followup_keywords", [])
            keywords_found = any(kw.lower() in question_lower for kw in followup_keywords)
            
            if not keywords_found:
                logger.info(
                    "[%s] Blocked %s (follow-up only, keywords not found)",
                    SKILL_NAME, skill
                )
                # Add fallback skill if specified
                fallback = manifest.get("fallback_on_block")
                if fallback and fallback not in filtered:
                    filtered.append(fallback)
                    logger.info(
                        "[%s] Added fallback %s for blocked %s",
                        SKILL_NAME, fallback, skill
                    )
                continue
        
        # Skill passed all guards
        filtered.append(skill)
    
    if filtered != selected_skills:
        logger.info(
            "[%s] Routing guards applied: %s → %s",
            SKILL_NAME, selected_skills, filtered
        )
    
    return filtered


def apply_question_enrichment(
    skill_name: str,
    manifest: dict[str, Any],
    parameters: dict,
    conversation_history: list[dict],
    previous_results: dict,
) -> dict:
    """Apply manifest-declared question enrichment to skill parameters.
    
    If a skill declares a question_enrichment_hook in its manifest, this function
    invokes that hook to potentially enrich the question with context from prior
    results or conversation history.
    
    Args:
        skill_name: Name of the skill being enriched
        manifest: Skill manifest dict (should include question_enrichment_hook field)
        parameters: Skill parameters dict (contains 'question' key)
        conversation_history: Conversation history for context
        previous_results: Results from prior skills
    
    Returns:
        Updated parameters dict (may have modified 'question' key)
    """
    enrichment_hook_path = manifest.get("question_enrichment_hook")
    if not enrichment_hook_path:
        return parameters
    
    try:
        # Parse enrichment_hook_path as "module.path:function_name"
        if ":" not in enrichment_hook_path:
            logger.warning("[%s] Invalid enrichment hook path (no colon): %s", skill_name, enrichment_hook_path)
            return parameters
        
        module_path, func_name = enrichment_hook_path.rsplit(":", 1)
        
        # Dynamically import the module
        try:
            module = importlib.import_module(module_path)
            enrichment_func = getattr(module, func_name, None)
            
            if not callable(enrichment_func):
                logger.warning(
                    "[%s] Enrichment hook not callable: %s:%s",
                    skill_name, module_path, func_name
                )
                return parameters
            
            # Invoke enrichment hook with available context
            original_question = parameters.get("question", "")
            enriched_question = enrichment_func(
                original_question=original_question,
                conversation_history=conversation_history,
                previous_results=previous_results,
            )
            
            if enriched_question and enriched_question != original_question:
                parameters = parameters.copy()
                parameters["question"] = enriched_question
                logger.info(
                    "[%s] Question enriched by %s hook",
                    skill_name, func_name
                )
            
            return parameters
        
        except ImportError as e:
            logger.warning("[%s] Could not import enrichment hook module %s: %s", skill_name, module_path, e)
            return parameters
        except AttributeError as e:
            logger.warning("[%s] Enrichment hook function %s not found in %s: %s", skill_name, func_name, module_path, e)
            return parameters
        except TypeError as e:
            logger.warning("[%s] Enrichment hook signature error: %s", skill_name, e)
            return parameters
    
    except Exception as e:
        logger.error("[%s] Unexpected error in question enrichment: %s", skill_name, e)
        return parameters


def check_and_apply_auto_chain(
    last_skill_name: str,
    last_skill_result: dict,
    all_manifests: dict[str, dict[str, Any]],
    runner: Any,
    context: dict,
    parameters: dict,
    conversation_history: list[dict],
    memory: Any = None,
) -> tuple[str | None, dict | None]:
    """Check if last skill should auto-chain to a successor skill.
    
    If a skill declares auto_chain_successor and successor_question_builder_hook,
    this function invokes the builder to create a question for the successor,
    then executes the successor skill.
    
    Args:
        last_skill_name: Name of the skill that just executed
        last_skill_result: Result dict from that skill
        all_manifests: Dict mapping skill names to manifest dicts
        runner: Runner instance for skill execution
        context: Shared context dict
        parameters: Original parameters dict
        conversation_history: Conversation history
        memory: Optional memory instance
    
    Returns:
        Tuple of (successor_skill_name, successor_result) or (None, None) if no auto-chain
    """
    manifest = all_manifests.get(last_skill_name, {})
    auto_chain_successor = manifest.get("auto_chain_successor")
    
    if not auto_chain_successor:
        return None, None
    
    if last_skill_result.get("status") != "ok":
        logger.info("[%s] Auto-chain skipped: predecessor skill failed", last_skill_name)
        return None, None
    
    builder_hook_path = manifest.get("successor_question_builder_hook")
    if not builder_hook_path:
        logger.warning("[%s] auto_chain_successor declared but no successor_question_builder_hook", last_skill_name)
        return None, None
    
    try:
        # Parse builder_hook_path as "module.path:function_name"
        if ":" not in builder_hook_path:
            logger.warning("[%s] Invalid builder hook path (no colon): %s", last_skill_name, builder_hook_path)
            return None, None
        
        module_path, func_name = builder_hook_path.rsplit(":", 1)
        
        # Dynamically import the module
        try:
            module = importlib.import_module(module_path)
            builder_func = getattr(module, func_name, None)
            
            if not callable(builder_func):
                logger.warning(
                    "[%s] Builder function not callable: %s:%s",
                    last_skill_name, module_path, func_name
                )
                return None, None
            
            # Invoke builder to create successor question
            successor_question = builder_func(last_skill_result)
            
            if not successor_question:
                logger.info("[%s] Builder returned empty question; skipping auto-chain", last_skill_name)
                return None, None
            
            # Execute auto-chained successor skill
            logger.info("[%s] Auto-chaining to %s", last_skill_name, auto_chain_successor)
            
            successor_context = runner._build_context()
            if memory is not None:
                successor_context["memory"] = memory
            
            successor_params = dict(parameters)
            successor_params["question"] = successor_question
            successor_context["parameters"] = successor_params
            
            if conversation_history:
                successor_context["conversation_history"] = conversation_history
            
            successor_result = runner.dispatch(auto_chain_successor, context=successor_context)
            logger.info(
                "[%s] Auto-chained %s completed with status: %s",
                last_skill_name,
                auto_chain_successor,
                successor_result.get("status"),
            )
            
            return auto_chain_successor, successor_result
        
        except ImportError as e:
            logger.warning("[%s] Could not import builder module %s: %s", last_skill_name, module_path, e)
            return None, None
        except AttributeError as e:
            logger.warning("[%s] Builder function %s not found in %s: %s", last_skill_name, func_name, module_path, e)
            return None, None
        except TypeError as e:
            logger.warning("[%s] Builder signature error: %s", last_skill_name, e)
            return None, None
    
    except Exception as e:
        logger.error("[%s] Unexpected error in auto-chain: %s", last_skill_name, e)
        return None, None

