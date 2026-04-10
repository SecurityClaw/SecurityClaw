"""Test if cannot_answer filtering works for traffic questions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_manifest import apply_routing_guards, SkillManifestLoader


def test_cannot_answer_filtering():
    """Test that geoip_lookup is filtered out for traffic questions."""
    
    # Load actual manifests
    loader = SkillManifestLoader()
    all_manifests = loader.load_all_manifests()
    
    print(f"Loaded {len(all_manifests)} skill manifests\n")
    
    # Get geoip manifest
    geoip_manifest = all_manifests.get("geoip_lookup", {})
    print(f"geoip_lookup cannot_answer: {geoip_manifest.get('cannot_answer', [])}\n")
    
    # Define fake available_skills list (just for the test)
    available_skills = [
        {"name": "geoip_lookup", "description": "IP geolocation"},
        {"name": "opensearch_querier", "description": "Log search"},
        {"name": "fields_querier", "description": "Field discovery"},
    ]
    
    question = "Any 1.1.1.1 traffic?"
    
    print(f"Question: '{question}'")
    print(f"Question lower: '{question.lower()}'")
    print()
    
    # Test case 1: LLM selected geoip_lookup (wrong)
    selected_before = ["geoip_lookup", "opensearch_querier"]
    print(f"Before filtering: {selected_before}")
    
    selected_after = apply_routing_guards(
        selected_skills=selected_before,
        user_question=question,
        available_skills=available_skills,
        all_manifests=all_manifests,
    )
    
    print(f"After filtering:  {selected_after}\n")
    
    # Check result
    if "geoip_lookup" not in selected_after:
        print("✅ PASS: geoip_lookup was correctly filtered out")
        if "opensearch_querier" in selected_after:
            print("✅ PASS: opensearch_querier remains in selection")
        else:
            print("❌ FAIL: opensearch_querier was removed too!")
    else:
        print("❌ FAIL: geoip_lookup was NOT filtered out")
        print(f"   This means the cannot_answer pattern 'traffic' didn't match the question")
    

if __name__ == "__main__":
    try:
        test_cannot_answer_filtering()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
