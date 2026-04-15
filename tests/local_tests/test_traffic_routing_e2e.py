"""
End-to-end test: Verify "Any 1.1.1.1 traffic?" routes correctly through the full supervisor pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import route_question
from core.llm_provider import build_llm_provider
from core.config import Config
from core.skill_loader import SkillLoader


def test_traffic_question_routing_e2e():
    """Test the complete routing pipeline for "Any 1.1.1.1 traffic?" """
    
    print(f"\n{'='*70}")
    print(f"END-TO-END TEST: Traffic Question Routing")
    print(f"{'='*70}\n")
    
    # Setup
    config = Config()
    llm = build_llm_provider(config)
    
    # Load skills
    skill_loader = SkillLoader()
    discovered_skills = skill_loader.discover()
    available_skills = [
        {
            "name": name,
            "description": skill.description if hasattr(skill, "description") else "Security analysis skill",
        }
        for name, skill in discovered_skills.items()
        if name != "chat_router"
    ]
    
    print(f"Available skills: {len(available_skills)}")
    for skill in sorted(available_skills, key=lambda s: s["name"])[:5]:
        print(f"  - {skill['name']}")
    print()
    
    # Load instruction
    instruction_path = Path(__file__).parent.parent.parent / "core" / "chat_router" / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")
    
    # Test the routing
    question = "Any 1.1.1.1 traffic?"
    
    print(f"Question: '{question}'")
    print(f"\nCalling route_question with real LLM...\n")
    
    try:
        result = route_question(
            user_question=question,
            available_skills=available_skills,
            llm=llm,
            instruction=instruction,
            conversation_history=[],
        )
        
        print(f"Routing Result:")
        print(f"  Skills: {result.get('skills', [])}")
        print(f"  Reasoning: {result.get('reasoning', '')}\n")
        
        # Validate
        skills = result.get('skills', [])
        
        print("Validation:")
        if "geoip_lookup" in skills:
            print(f"  ❌ FAIL: geoip_lookup should NOT be in routing (cannot_answer: traffic)")
        else:
            print(f"  ✅ PASS: geoip_lookup correctly excluded")
        
        if "opensearch_querier" in skills:
            print(f"  ✅ PASS: opensearch_querier included (can answer traffic questions)")
        else:
            print(f"  ⚠️  WARNING: opensearch_querier not in routing")
            print(f"       Skills selected: {skills}")
        
        if "fields_querier" in skills or "opensearch_querier" in skills:
            print(f"  ✅ PASS: Log search-capable skill selected")
            return True
        else:
            print(f"  ❌ FAIL: No log search skill selected")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_traffic_question_routing_e2e()
    sys.exit(0 if success else 1)
