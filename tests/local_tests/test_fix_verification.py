#!/usr/bin/env python3
"""
Direct test of the fix: Verify that when ip_fingerprinter fails (status=error),
opensearch_querier's formatter is used instead (status=ok).

This is the EXACT scenario from the user's chat logs:
- opensearch_querier found 5984456 results with ports
- ip_fingerprinter failed with error
- BUG: Returned "No passive fingerprint..." (ip_fingerprinter's formatter)
- FIX: Should return port information (opensearch_querier's formatter)
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(levelname)s: %(message)s'
)

def test_response_formatter_selection():
    """
    Directly test the response formatter selection logic with imported code.
    Uses realistic skill_results matching the user's chat scenario.
    
    SCENARIO: opensearch_querier succeeded with port data, but ip_fingerprinter failed.
    EXPECTED: Use opensearch_querier's formatter, not ip_fingerprinter's error message.
    """
    
    from core.chat_router import logic
    
    # Simulate the EXACT scenario from user's chat logs:
    # - opensearch_querier: Found 5984456 results with port 53
    # - ip_fingerprinter: Failed with status=error
    
    user_question = "What ports are associated with this traffic?"
    
    skill_results = {
        "fields_querier": {
            "status": "ok"
        },
        "opensearch_querier": {
            "status": "ok",
            "results_count": 5984456,
            "sampled_results_count": 400,
            "results": [
                {
                    "src_ip": "192.168.0.130",
                    "destination.port": 53,
                    "timestamp": "2026-03-14T22:20:12.382Z"
                }
            ],
            "ports": [53]
        },
        "ip_fingerprinter": {
            "status": "error",
            "error": "No target IP was provided. LLM should extract and pass parameters.ip"
        }
    }
    
    # Call the actual response formatter selection function
    response = logic.format_response(
        user_question=user_question,
        routing_decision={"skills": ["opensearch_querier", "ip_fingerprinter"]},
        skill_results=skill_results,
        llm=None,
        cfg=None,
        available_skills=[]
    )
    
    # Check what we got
    has_port_info = "port" in response.lower() or "53" in response
    has_error_msg = "no passive fingerprint" in response.lower()
    starts_with_found = response.lower().startswith("found")
    
    # ASSERTIONS
    assert not has_error_msg, (
        "Response contains ip_fingerprinter error message. "
        f"Expected port info, got: {response[:200]}"
    )
    assert has_port_info, (
        "Response should contain port information. "
        f"Got: {response[:200]}"
    )
    assert starts_with_found, (
        "Response should start with opensearch_querier format (Found ...). "
        f"Got: {response[:50]}"
    )


if __name__ == "__main__":
    # For direct execution (not pytest)
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).parent.parent.parent)
    )
    sys.exit(result.returncode)
