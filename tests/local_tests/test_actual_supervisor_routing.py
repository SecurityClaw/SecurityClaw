"""
Test what the supervisor ACTUALLY routes for "Any 1.1.1.1 traffic?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import route_question
from core.config import Config


def test_actual_routing():
    """Test actual supervisor routing with real manifests."""
    
    cfg = Config()
    
    # Load the instruction file
    instruction_path = Path(__file__).parent.parent.parent / "core" / "chat_router" / "instruction.md"
    with open(instruction_path) as f:
        instruction = f.read()
    
    # Load skill manifests
    skills_dir = Path(__file__).parent.parent.parent / "skills"
    available_skills = []
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue
        
        with open(manifest_path) as f:
            import yaml
            manifest = yaml.safe_load(f)
            if manifest and manifest.get("name"):
                available_skills.append({
                    "name": manifest["name"],
                    "description": manifest.get("description", "")
                })
    
    print(f"Loaded {len(available_skills)} skills")
    
    # Create a mock LLM that just returns the raw manifest data
    class DebugLLM:
        def __init__(self, manifest_data):
            self.manifest_data = manifest_data
        
        def chat(self, messages):
            import json
            # Just return something neutral to see what router does
            return json.dumps({
                "reasoning": "Debug routing",
                "skills": [],
                "parameters": {}
            })
    
    question = "Any 1.1.1.1 traffic?"
    
    print(f"\n{'='*70}")
    print(f"Testing supervisor routing for: '{question}'")
    print(f"{'='*70}\n")
    
    print(f"Available skills:")
    for skill in sorted(available_skills, key=lambda s: s["name"]):
        # Load full manifest to show can_answer and cannot_answer
        manifest_path = skills_dir / skill["name"] / "manifest.yaml"
        if manifest_path.exists():
            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            can_answer = manifest.get("can_answer", [])
            cannot_answer = manifest.get("cannot_answer", [])
            print(f"\n  {skill['name']}:")
            print(f"    can_answer: {can_answer[:2]}...")
            print(f"    cannot_answer: {cannot_answer[:2]}...")
    
    # Now call route_question with debug LLM
    result = route_question(
        user_question=question,
        available_skills=available_skills,
        llm=DebugLLM(None),
        instruction=instruction,
        conversation_history=[],
    )
    
    print(f"\n{'='*70}")
    print(f"Supervisor Routing Result:")
    print(f"{'='*70}\n")
    print(f"Skills selected: {result.get('skills', [])}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}\n")
    
    # Check if this routes to the right place
    if "opensearch_querier" in result.get("skills", []):
        print("✅ CORRECT: Routed to opensearch_querier")
    elif "geoip_lookup" in result.get("skills", []):
        print("❌ WRONG: Routed to geoip_lookup instead of opensearch_querier")
    else:
        print(f"⚠️ Unexpected routing: {result.get('skills', [])}")


if __name__ == "__main__":
    try:
        test_actual_routing()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
