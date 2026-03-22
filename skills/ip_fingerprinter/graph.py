"""
skills/ip_fingerprinter/graph.py

Minimal subgraph for IP fingerprinting.

The supervisor handles prerequisite orchestration based on manifest declarations.
This graph is just the execution layer - it receives all required artifacts
(schema_discovery results, evidence_search results) and executes the fingerprinting analysis.

Flow:
1. Supervisor ensures fields_querier and opensearch_querier ran  
2. Those results land in previous_results
3. This graph extracts aggregated ports and delegates to ip_fingerprinter logic
4. Returns the fingerprinting analysis
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SKILL_NAME = "ip_fingerprinter"


def build_graph(config: dict):
    """
    Build minimal execution graph for IP fingerprinting.
    
    The supervisor has already ensured that:
    - fields_querier discovered field names
    - opensearch_querier returned ports for the target IP
    
    This graph just:
    1. Extract aggregated ports from prior results
    2. Execute fingerprinter analysis
    3. Return results
    """
    from langgraph.graph import StateGraph, START, END
    
    class FingerprintState(dict):
        """Simple state passthrough."""
        pass
    
    graph = StateGraph(FingerprintState)
    
    def execute_fingerprinter(state):
        """Execute fingerprinting with data from prior skills."""
        from skills.ip_fingerprinter.logic import execute as fp_execute
        
        user_question = state.get("user_question", "")
        parameters = state.get("parameters", {})
        previous_results = state.get("previous_results", {})
        conversation_history = state.get("conversation_history", [])
        
        try:
            result = fp_execute(
                user_question=user_question,
                parameters=parameters,
                previous_results=previous_results,
                conversation_history=conversation_history
            )
            state["final_result"] = result
            return state
        except Exception as exc:
            logger.error("[%s] Fingerprinting failed: %s", SKILL_NAME, exc)
            state["final_result"] = {"status": "error", "error": str(exc)}
            return state
    
    graph.add_node("execute", execute_fingerprinter)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    
    return graph.compile()

