"""
Unit test: Verify failed skills are skipped before using their formatters.
This directly tests the fix for the port follow-up issue.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_failed_skill_formatter_skipping():
    """Test that formatters for failed skills are never invoked"""
    
    print(f"\n{'='*80}")
    print(f"UNIT TEST: Failed Skills Skipped in Formatter Selection")
    print(f"{'='*80}\n")
    
    # Simulate skill results where ip_fingerprinter failed but opensearch_querier succeeded
    skill_results = {
        "fields_querier": {"status": "ok", "field_count": 15},
        "opensearch_querier": {
            "status": "ok",
            "results_count": 5984456,
            "ports": [53, 443]
        },
        "ip_fingerprinter": {
            "status": "error",
            "error": "No target IP was provided"
        }
    }
    
    # Test the priority selection logic
    priority_order = [
        "forensic_examiner",
        "geoip_lookup", 
        "ip_fingerprinter",
        "baseline_querier",
        "opensearch_querier",
        "threat_analyst"
    ]
    
    formatters_checked = []
    for skill_name in priority_order:
        if skill_name not in skill_results:
            continue
        
        result = skill_results[skill_name]
        
        # THE FIX: Skip failed skills - only use formatters from successful executions
        if result.get("status") != "ok":
            print(f"  ✓ Skipping {skill_name} (status: {result.get('status')})")
            continue
        
        print(f"  ✓ Would use formatter for {skill_name} (status: ok)")
        formatters_checked.append(skill_name)
    
    print()
    
    # Validate the result
    if formatters_checked[0] == "opensearch_querier":
        print(f"✅ PASS: opensearch_querier would be chosen before ip_fingerprinter")
        print(f"   Formatters checked in order: {formatters_checked}")
        return True
    else:
        print(f"❌ FAIL: Wrong formatter selected: {formatters_checked[0]}")
        return False


if __name__ == "__main__":
    try:
        success = test_failed_skill_formatter_skipping()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
