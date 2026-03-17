"""Capability-oriented dependency helpers for dynamic skill orchestration."""

from __future__ import annotations

from typing import Any

from core.skill_manifest import first_skill_in_group, manifest_for_skill


def manifest_prerequisite_groups(manifest: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    for prereq in manifest.get("prerequisites") or []:
        if not isinstance(prereq, dict):
            continue
        group = str(prereq.get("group") or "").strip()
        if group and group not in groups:
            groups.append(group)
    return groups


def manifest_owns_composite_graph(manifest: dict[str, Any]) -> bool:
    return bool(str(manifest.get("graph_builder") or "").strip())


def expand_skill_dependencies(
    selected_skills: list[str],
    manifests: dict[str, dict[str, Any]],
) -> list[str]:
    """Expand selected skills through manifest-declared prerequisite groups.

    Skills that declare a composite graph builder own their dependency flow and
    are therefore not expanded externally.
    """
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(skill_name: str) -> None:
        if not skill_name or skill_name in visited:
            return
        if skill_name in visiting:
            return

        visiting.add(skill_name)
        manifest = manifest_for_skill(manifests, skill_name)

        if not manifest_owns_composite_graph(manifest):
            for group in manifest_prerequisite_groups(manifest):
                prerequisite_skill = first_skill_in_group(manifests, group)
                if prerequisite_skill and prerequisite_skill != skill_name:
                    visit(prerequisite_skill)

        visiting.remove(skill_name)
        visited.add(skill_name)
        if skill_name not in ordered:
            ordered.append(skill_name)

    for selected in selected_skills or []:
        visit(selected)

    return ordered