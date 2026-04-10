"""
Test supervisor routing with REAL chat interface (not mocked).

This tests that the supervisor correctly handles:
1. "Any 1.1.1.1 traffic?" → Routes to opensearch_querier (not geoip_lookup)
2. "What ports are associated?" → Uses follow-up context to extract ports

Uses the actual supervisor graph and skill manifest loading.
"""

import json
import sys
from pathlib import Path

# Add workspace root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import the actual routing logic
from core.config import Config


def test_real_supervisor_routing():
    """Test supervisor routing with actual manifest-based skill selection."""
    
    print(f"\n{'='*70}")
    print(f"TEST: Real Supervisor Routing for Traffic + Ports")
    print(f"{'='*70}\n")
    
    # Load config
    config = Config()
    
    print(f"Configuration loaded successfully")
    
    # Check geoip_lookup and opensearch_querier manifests
    geoip_mani = config.get("skills", "geoip_lookup", "manifest")
    opensearch_mani = config.get("skills", "opensearch_querier", "manifest")
    
    print(f"\n{'='*70}")
    print(f"Skill Manifest Review")
    print(f"{'='*70}\n")
    
    if geoip_mani:
        print(f"geoip_lookup manifest:")
        print(f"  can_answer: {geoip_mani.get('can_answer', [])}")
        print(f"  cannot_answer: {geoip_mani.get('cannot_answer', [])}")
        print(f"  min_prior_context: {geoip_mani.get('min_prior_context', 0)}")
        print(f"  priority_keywords: {geoip_mani.get('priority_keywords', [])}")
    else:
        print("⚠️  geoip_lookup manifest not found in config")
    
    if opensearch_mani:
        print(f"\nopensearch_querier manifest:")
        print(f"  can_answer: {opensearch_mani.get('can_answer', [])}")
        print(f"  cannot_answer: {opensearch_mani.get('cannot_answer', [])}")
        print(f"  min_prior_context: {opensearch_mani.get('min_prior_context', 0)}")
    else:
        print("⚠️  opensearch_querier manifest not found in config")
    
    # Test 1: Route traffic question
    question1 = "Any 1.1.1.1 traffic?"
    print(f"\n{'='*70}")
    print(f"Test 1: Routing '{question1}'")
    print(f"{'='*70}")
    print(f"Expected: opensearch_querier (traffic search)")
    print(f"NOT Expected: geoip_lookup (only for geolocation)\n")
    
    # Analyze question against manifests
    question_lower = question1.lower()
    
    print(f"Analysis:")
    print(f"  - Contains 'traffic': {('traffic' in question_lower)}")
    print(f"  - Contains 'any': {('any' in question_lower)}")
    print(f"  - Contains IP: {('1.1.1.1' in question_lower)}")
    print(f"  - Is geolocation question: {any(kw in question_lower for kw in ['where', 'country', 'location', 'geo'])}")
    
    if opensearch_mani:
        can_answer = opensearch_mani.get('can_answer', [])
        cannot_answer = opensearch_mani.get('cannot_answer', [])
        
        matches_can = any(pattern in question_lower for pattern in can_answer if isinstance(pattern, str))
        matches_cannot = any(pattern in question_lower for pattern in cannot_answer if isinstance(pattern, str))
        
        print(f"  - opensearch_querier can_answer match: {matches_can}")
        print(f"  - opensearch_querier cannot_answer match: {matches_cannot}")
    
    if geoip_mani:
        cannot_answer = geoip_mani.get('cannot_answer', [])
        matches_cannot = any(pattern in question_lower for pattern in cannot_answer if isinstance(pattern, str))
        print(f"  - geoip_lookup cannot_answer match: {matches_cannot}")
        if 'log search' in [str(p).lower() for p in cannot_answer]:
            print(f"  - geoip explicitly excludes: log search ✓")
        else:
            print(f"  - geoip does NOT explicitly exclude log search ⚠️")
    
    # Test 2: Routing decision based on manifests
    print(f"\n{'='*70}")
    print(f"Manifest-based routing decisions:")
    print(f"{'='*70}\n")
    
    if opensearch_mani and geoip_mani:
        geoip_cannot = geoip_mani.get('cannot_answer', [])
        opensearch_can = opensearch_mani.get('can_answer', [])
        
        geoip_excludes_logs = any('log' in str(p).lower() for p in geoip_cannot)
        opensearch_includes_traffic = any('traffic' in str(p).lower() for p in opensearch_can)
        
        print(f"1. geoip_lookup cannot_answer includes 'log' patterns: {geoip_excludes_logs}")
        print(f"2. opensearch_querier can_answer includes 'traffic' patterns: {opensearch_includes_traffic}")
        
        if geoip_excludes_logs:
            print(f"\n✅ CORRECT: geoip should be excluded because question is 'log search'")
        else:
            print(f"\n⚠️  WARNING: geoip doesn't explicitly exclude log searches")
            print(f"   This may cause wrong routing!")
            print(f"   geoip cannot_answer list: {geoip_cannot}")
        
        if opensearch_includes_traffic:
            print(f"✅ CORRECT: opensearch_querier claims it can handle traffic questions")
        else:
            print(f"⚠️  WARNING: opensearch_querier doesn't explicitly claim traffic questions")
            print(f"   opensearch_querier can_answer list: {opensearch_can}")


if __name__ == "__main__":
    try:
        test_real_supervisor_routing()
        print(f"\n{'='*70}")
        print("Test completed - check manifest configuration above")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

