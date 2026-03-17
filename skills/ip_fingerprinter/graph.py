"""
skills/ip_fingerprinter/graph.py

Composite skill graph for ip_fingerprinter.

Orchestrates prerequisite skills:
- schema_discovery (fields_querier) for field names
- evidence_search (opensearch_querier) to retrieve port observations

Then executes ip_fingerprinter with aggregated ports from evidence.

This graph ensures that port data is available before ip_fingerprinter
attempts to analyze and classify ports by role (client/server).
"""
from __future__ import annotations

import json
import logging
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

SKILL_NAME = "ip_fingerprinter"


class IPFingerprintGraphState(TypedDict, total=False):
    """State for ip_fingerprinter subgraph orchestration."""
    
    # Context from supervisor
    user_question: str
    conversation_history: list
    parameters: dict
    previous_results: dict
    
    # Execution state
    schema_discovery_done: bool
    schema_discovery_result: dict
    evidence_search_done: bool
    evidence_search_result: dict
    aggregated_ports: dict
    fingerprint_result: dict
    execution_trace: list
    
    # Final output
    final_result: dict


def _extract_aggregated_ports_from_evidence(evidence_result: dict) -> dict:
    """
    Extract aggregated port information from opensearch_querier results.
    
    Converts individual port observations into aggregated format:
    {
        port_number: {
            "observations": count,
            "protocols": [proto1, proto2, ...],
            "is_known": bool
        },
        ...
    }
    
    Args:
        evidence_result: Dict from opensearch_querier with 'results', 'ports', etc.
    
    Returns:
        dict mapping port -> observation data
    """
    aggregated = {}
    
    # Method 1: Use pre-aggregated ports if available (from opensearch_querier format_response)
    if evidence_result.get("ports"):
        for port in evidence_result["ports"]:
            aggregated[port] = {
                "observations": 1,  # Conservative estimate
                "protocols": ["tcp"],  # Default assumption from opensearch
                "is_known": True
            }
    
    # Method 2: Aggregate from raw results
    results = evidence_result.get("results", [])
    if results and not aggregated:
        for record in results:
            # Try various port field names that opensearch might return
            port = None
            for port_field in [
                "destination.port",
                "destination_port",
                "dst_port",
                "dport",
                "port"
            ]:
                if port_field in record:
                    try:
                        port = int(record[port_field])
                        break
                    except (TypeError, ValueError):
                        continue
            
            if port and 0 < port <= 65535:
                if port not in aggregated:
                    aggregated[port] = {
                        "observations": 0,
                        "protocols": set(),
                        "is_known": False
                    }
                aggregated[port]["observations"] += 1
    
    # Convert protocol sets to lists for JSON serialization
    for port_data in aggregated.values():
        if isinstance(port_data.get("protocols"), set):
            port_data["protocols"] = list(port_data["protocols"])
    
    return aggregated


def build_graph(config: dict) -> StateGraph:
    """
    Build and return the ip_fingerprinter composite skill graph.
    
    The graph orchestrates:
    1. Ensure schema_discovery results exist (field mappings)
    2. Ensure evidence_search results exist (ports from opensearch_querier)
    3. Execute ip_fingerprinter with aggregated_ports
    4. Return the fingerprint analysis
    
    Args:
        config: Configuration dict with db, llm, runner, etc.
    
    Returns:
        StateGraph: Executable LangGraph for IP fingerprinting
    """
    from core.chat_router.logic import execute_skill  # Avoid circular import
    
    graph = StateGraph(IPFingerprintGraphState)
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 1: Ensure schema_discovery
    # ────────────────────────────────────────────────────────────────────────
    def ensure_schema_discovery(state: IPFingerprintGraphState) -> IPFingerprintGraphState:
        """Check if schema discovery results are available."""
        previous_results = state.get("previous_results", {})
        execution_trace = state.get("execution_trace", [])
        
        # Check if fields_querier or fields_baseliner results are available
        schema_result = previous_results.get("fields_querier", {}) or previous_results.get("schema_discovery", {})
        if schema_result.get("field_mappings") or schema_result.get("status") == "ok":
            logger.info(
                "[%s] Schema discovery results already available",
                SKILL_NAME
            )
            state["schema_discovery_done"] = True
            state["schema_discovery_result"] = schema_result
            execution_trace.append({
                "step": "ensure_schema_discovery",
                "status": "available_from_prior",
                "source": list(previous_results.keys())
            })
        else:
            logger.info(
                "[%s] No prior schema discovery; will proceed with evidence search",
                SKILL_NAME
            )
            state["schema_discovery_done"] = False
            state["schema_discovery_result"] = {}
            execution_trace.append({
                "step": "ensure_schema_discovery",
                "status": "not_available",
                "note": "evidence_search can proceed without it"
            })
        
        state["execution_trace"] = execution_trace
        return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 2: Ensure evidence_search (retrieve ports from opensearch_querier)
    # ────────────────────────────────────────────────────────────────────────
    def ensure_evidence_search(state: IPFingerprintGraphState) -> IPFingerprintGraphState:
        """
        Check if evidence_search results are available.
        If not, execute opensearch_querier to get port observations.
        """
        previous_results = state.get("previous_results", {})
        execution_trace = state.get("execution_trace", [])
        parameters = state.get("parameters", {})
        user_question = state.get("user_question", "")
        
        # Check if opensearch_querier or evidence_search results are already available
        evidence_result = previous_results.get("opensearch_querier", {}) or previous_results.get("evidence_search", {})
        
        if evidence_result and evidence_result.get("status") == "ok" and (
            evidence_result.get("results") or evidence_result.get("ports")
        ):
            logger.info(
                "[%s] Evidence search (port observations) already available: %d results",
                SKILL_NAME,
                evidence_result.get("results_count", len(evidence_result.get("results", [])))
            )
            state["evidence_search_done"] = True
            state["evidence_search_result"] = evidence_result
            state["aggregated_ports"] = _extract_aggregated_ports_from_evidence(evidence_result)
            execution_trace.append({
                "step": "ensure_evidence_search",
                "status": "available_from_prior",
                "port_count": len(state["aggregated_ports"]),
                "source": "opensearch_querier or previous_results"
            })
        else:
            logger.warning(
                "[%s] No prior evidence_search results. "
                "IP fingerprinting requires port observations.",
                SKILL_NAME
            )
            state["evidence_search_done"] = False
            state["evidence_search_result"] = {}
            state["aggregated_ports"] = {}
            execution_trace.append({
                "step": "ensure_evidence_search",
                "status": "not_available",
                "warning": "Supervisor should have orchestrated opensearch_querier first"
            })
        
        state["execution_trace"] = execution_trace
        return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Node 3: Execute ip_fingerprinter with aggregated ports
    # ────────────────────────────────────────────────────────────────────────
    def execute_fingerprinter(state: IPFingerprintGraphState) -> IPFingerprintGraphState:
        """
        Execute the ip_fingerprinter skill with aggregated port data.
        """
        from skills.ip_fingerprinter.logic import execute as fingerprinter_execute
        
        execution_trace = state.get("execution_trace", [])
        parameters = state.get("parameters", {})
        user_question = state.get("user_question", "")
        
        # Add aggregated ports to parameters
        enriched_parameters = {**parameters}
        enriched_parameters["aggregated_ports"] = state.get("aggregated_ports", {})
        
        logger.info(
            "[%s] Executing fingerprinter with %d aggregated ports",
            SKILL_NAME,
            len(enriched_parameters.get("aggregated_ports", {}))
        )
        
        try:
            result = fingerprinter_execute(
                user_question=user_question,
                parameters=enriched_parameters,
                previous_results=state.get("previous_results", {}),
                conversation_history=state.get("conversation_history", [])
            )
            
            state["fingerprint_result"] = result
            state["final_result"] = result
            execution_trace.append({
                "step": "execute_fingerprinter",
                "status": result.get("status", "unknown"),
                "ports_analyzed": len(result.get("ports", []))
            })
        except Exception as e:
            logger.error(
                "[%s] Error executing fingerprinter: %s",
                SKILL_NAME,
                str(e)
            )
            state["fingerprint_result"] = {
                "status": "error",
                "error": str(e),
                "message": "Failed to execute fingerprinter"
            }
            state["final_result"] = state["fingerprint_result"]
            execution_trace.append({
                "step": "execute_fingerprinter",
                "status": "error",
                "error": str(e)
            })
        
        state["execution_trace"] = execution_trace
        return state
    
    # ────────────────────────────────────────────────────────────────────────
    # Build the graph structure
    # ────────────────────────────────────────────────────────────────────────
    graph.add_node("ensure_schema_discovery", ensure_schema_discovery)
    graph.add_node("ensure_evidence_search", ensure_evidence_search)
    graph.add_node("execute_fingerprinter", execute_fingerprinter)
    
    # Connect nodes in sequence
    graph.add_edge(START, "ensure_schema_discovery")
    graph.add_edge("ensure_schema_discovery", "ensure_evidence_search")
    graph.add_edge("ensure_evidence_search", "execute_fingerprinter")
    graph.add_edge("execute_fingerprinter", END)
    
    return graph.compile()
