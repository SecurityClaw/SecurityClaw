"""Debug test to check if apply_routing_guards is being called and working"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='[%(name)s] %(levelname)s: %(message)s')

from core.chat_router.logic import route_question
from core.llm_provider import build_llm_provider
from core.config import Config
from core.skill_loader import SkillLoader


def test_debug():
    """Debug the routing guards application"""
    
    print(f"\nDEBUG TEST: Check if apply_routing_guards is called\n")
    
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
    
    # Load instruction
    instruction_path = Path(__file__).parent.parent.parent / "core" / "chat_router" / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")
    
    # Test the routing
    question = "Any 1.1.1.1 traffic?"
    
    print(f"Question: '{question}\n")
    
    result = route_question(
        user_question=question,
        available_skills=available_skills,
        llm=llm,
        instruction=instruction,
        conversation_history=[],
    )
    
    print(f"\nFinal Skills: {result.get('skills', [])}\n")
    
    if "geoip_lookup" not in result.get('skills', []):
        print("✅ PASS: geoip_lookup was filtered out")
    else:
        print("❌ FAIL: geoip_lookup still in routing despite cannot_answer")


if __name__ == "__main__":
    test_debug()
