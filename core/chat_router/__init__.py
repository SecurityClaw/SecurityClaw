"""
core.chat_router

Intelligent skill router for conversational SOC queries.
Routes user questions to appropriate skills, handles multi-skill workflows,
and maintains conversation context using LangGraph orchestration.

This module was previously located in skills/chat_router/ but has been
moved to core/ because it is fundamental orchestration infrastructure,
not a periodic skill.

LangGraph is a required dependency for orchestration.
"""

from core.chat_router.logic import (
    orchestrate_with_supervisor,
    run_graph,
    build_graph,
)

__all__ = [
    "orchestrate_with_supervisor",
    "run_graph",
    "build_graph",
]
