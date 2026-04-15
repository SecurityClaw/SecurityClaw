"""Direct test of route_question with the problematic question"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import route_question
from core.llm_provider import build_llm_provider
from core.config import Config
from core.skill_loader import SkillLoader

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

question = "Any 1.1.1.1 traffic?"

print(f"Testing route_question with: '{question}'\n")

result = route_question(
    user_question=question,
    available_skills=available_skills,
    llm=llm,
    instruction=instruction,
    conversation_history=[],
)

print(f"Result skills: {result.get('skills', [])}")
print(f"Result reasoning: {result.get('reasoning', '')[:200]}...\n")

if not result.get('skills'):
    print("ERROR: No skills selected!")
elif "geoip_lookup" in result.get('skills', []):
    print("ERROR: geoip_lookup should not be selected!")
else:
    print("SUCCESS: Correct skills selected (no geoip_lookup)")
