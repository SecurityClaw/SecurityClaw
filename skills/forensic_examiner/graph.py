"""
skills/forensic_examiner/graph.py

Composite skill graph for forensic_examiner.

Orchestrates prerequisite skills (schema_discovery via fields_querier)
before executing the main forensic timeline reconstruction.

This graph ensures that field documentation is available before
forensic_examiner attempts to parse and search log data.
"""
from __future__ import annotations

import json
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END

import logging

logger = logging.getLogger(__name__)

SKILL_NAME = "forensic_examiner"


class ForensicGraphState(TypedDict, total=False):
    """State for forensic_examiner subgraph orchestration."""
    
    # Context from supervisor
    user_question: str
    conversation_history: list
    parameters: dict
    previous_results: dict
    
    # Execution state
    schema_discovery_done: bool
    schema_discovery_result: dict
    forensic_result: dict
    execution_trace: list
    
    # Final output
    final_result: dict


def build_graph(config: dict) -> StateGraph:
    """
    Build and return the forensic_examiner composite skill graph.
    
    The graph orchestrates:
    1. Ensure schema_discovery results exist (field mappings from fields_baseliner)
    2. Execute forensic_examiner with those field mappings
    3. Return the forensic timeline report
    
    Args:
        config: Configuration dict with db, llm, runner, etc.
    
    Returns:
        StateGraph: Executable LangGraph for forensic examination
    """
    graph = StateGraph(ForensicGraphState)
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 1: Check if schema_discovery results are available
    # ────────────────────────────────────────────────────────────────────────
    def ensure_schema_discovery(state: ForensicGraphState) -> ForensicGraphState:
        """
        Check if we have schema_discovery results (field documentation).
        If already present, skip. If missing, we'll rely on forensic_examiner's
        internal fetch mechanism.
        """
        previous_results = state.get("previous_results", {})
        execution_trace = state.get("execution_trace", [])
        
        # Check if fields_baseliner or fields_querier results are already available
        schema_result = previous_results.get("schema_discovery", {})
        if schema_result.get("field_mappings") or schema_result.get("fields_documented"):
            logger.info(
                "[%s] Schema discovery results already available from previous step",
                SKILL_NAME
            )
            state["schema_discovery_done"] = True
            state["schema_discovery_result"] = schema_result
            execution_trace.append({
                "step": "ensure_schema_discovery",
                "status": "skipped_already_available",
                "source": "previous_results"
            })
        else:
            logger.info(
                "[%s] No prior schema discovery results; forensic_examiner will "
                "fetch field documentation from RAG",
                SKILL_NAME
            )
            state["schema_discovery_done"] = False
            state["schema_discovery_result"] = {}
            execution_trace.append({
                "step": "ensure_schema_discovery",
                "status": "will_fetch_from_rag",
                "note": "forensic_examiner.logic._fetch_field_documentation()"
            })
        
        state["execution_trace"] = execution_trace
        return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 2: Execute forensic_examiner skill
    # ────────────────────────────────────────────────────────────────────────
    def execute_forensic_examiner(state: ForensicGraphState) -> ForensicGraphState:
        """
        Execute the forensic_examiner skill logic with aggregated prior context.
        
        The skill's run() function will:
        1. Fetch field documentation (from RAG if not in previous_results)
        2. Extract incident context (IPs, domains, ports from question)
        3. Run iterative log investigation
        4. Build timeline narrative
        5. Return forensic_report dict
        """
        logger.info("[%s] Executing forensic_examiner skill", SKILL_NAME)
        
        # Import here to avoid circular dependencies
        try:
            from skills.forensic_examiner import logic as forensic_logic
        except ImportError as e:
            logger.error("[%s] Failed to import forensic_examiner.logic: %s", SKILL_NAME, e)
            state["forensic_result"] = {
                "status": "error",
                "error": f"Failed to import forensic_examiner logic: {e}"
            }
            return state
        
        # Prepare context for the skill's run() function
        skill_context = {
            "db": config.get("db"),
            "llm": config.get("llm"),
            "config": config.get("config"),
            "parameters": state.get("parameters", {}),
            "conversation_history": state.get("conversation_history", []),
            "previous_results": state.get("previous_results", {}),
        }
        
        try:
            result = forensic_logic.run(skill_context)
            state["forensic_result"] = result
            state["execution_trace"].append({
                "step": "execute_forensic_examiner",
                "status": result.get("status", "unknown"),
                "results_found": result.get("forensic_report", {}).get("results_found"),
            })
            return state
        except Exception as e:
            logger.error("[%s] Forensic examiner execution failed: %s", SKILL_NAME, e)
            state["forensic_result"] = {
                "status": "error",
                "error": str(e)
            }
            state["execution_trace"].append({
                "step": "execute_forensic_examiner",
                "status": "error",
                "error": str(e)
            })
            return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 3: Format final output
    # ────────────────────────────────────────────────────────────────────────
    def format_output(state: ForensicGraphState) -> ForensicGraphState:
        """
        Package the forensic_examiner result into the final_result field.
        This ensures compatibility with the supervisor graph's skill_results dict.
        """
        forensic_result = state.get("forensic_result", {})
        
        # If execution was successful, extract the forensic_report
        if forensic_result.get("status") == "ok":
            final_result = {
                "skill_name": SKILL_NAME,
                "status": "success",
                "forensic_report": forensic_result.get("forensic_report", {}),
                "execution_trace": state.get("execution_trace", []),
            }
        else:
            # Error case
            final_result = {
                "skill_name": SKILL_NAME,
                "status": "failed",
                "error": forensic_result.get("error", "Unknown error"),
                "execution_trace": state.get("execution_trace", []),
            }
        
        state["final_result"] = final_result
        return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Build the graph topology
    # ────────────────────────────────────────────────────────────────────────
    graph.add_node("ensure_schema_discovery", ensure_schema_discovery)
    graph.add_node("execute_forensic_examiner", execute_forensic_examiner)
    graph.add_node("format_output", format_output)
    
    # Wire the nodes in sequence
    graph.add_edge(START, "ensure_schema_discovery")
    graph.add_edge("ensure_schema_discovery", "execute_forensic_examiner")
    graph.add_edge("execute_forensic_examiner", "format_output")
    graph.add_edge("format_output", END)
    
    return graph
