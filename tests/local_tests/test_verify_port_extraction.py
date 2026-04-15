"""
Verify that the follow-up question actually extracts and returns port numbers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import run_graph
from core.llm_provider import build_llm_provider
from core.config import Config
from core.db_connector import OpenSearchConnector
from core.skill_loader import SkillLoader


def test_ports_in_followup_response():
    """Test that follow-up response actually contains port numbers, not errors"""
    
    print(f"\n{'='*80}")
    print(f"VERIFYING PORT EXTRACTION IN FOLLOW-UP RESPONSE")
    print(f"{'='*80}\n")
    
    # Setup
    config = Config()
    llm = build_llm_provider(config)
    db = OpenSearchConnector()
    
    from core.runner import Runner
    runner = Runner(db_connector=db, llm_provider=llm)
    runner.setup()
    
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
    
    # TURN 1: Traffic search
    result1 = run_graph(
        user_question="Any 1.1.1.1 traffic?",
        available_skills=available_skills,
        runner=runner,
        llm=llm,
        instruction=instruction,
        cfg=config,
        conversation_history=[],
    )
    
    response1 = result1.get("response", "")
    print(f"TURN 1 Response:\n{response1[:500]}...\n")
    
    # TURN 2: Port follow-up
    conversation_history = [
        {"role": "user", "content": "Any 1.1.1.1 traffic?"},
        {"role": "assistant", "content": response1},
    ]
    
    result2 = run_graph(
        user_question="What ports are associated with this traffic?",
        available_skills=available_skills,
        runner=runner,
        llm=llm,
        instruction=instruction,
        cfg=config,
        conversation_history=conversation_history,
    )
    
    response2 = result2.get("response", "")
    skill_results2 = result2.get("skill_results", {})
    
    print(f"TURN 2 Response:\n{response2}\n")
    print(f"="*80)
    
    # Analyze response
    print(f"\nANALYSIS:")
    print(f"- Response starts with 'Found': {response2.startswith('Found')}")
    print(f"- Response mentions 'port': {'port' in response2.lower()}")
    print(f"- Response length: {len(response2)} chars")
    
    # Check if port numbers are in response
    import re
    port_matches = re.findall(r'Destination port\(s\): ([\d, ]+)', response2)
    if port_matches:
        print(f"✅ PORTS FOUND: {port_matches[0]}")
        return True
    else:
        # Check for 53 explicitly
        if "53" in response2 and "port" in response2.lower():
            print(f"✅ Port 53 found in response")
            return True
        else:
            print(f"❌ NO PORT INFORMATION IN RESPONSE")
            return False


if __name__ == "__main__":
    try:
        success = test_ports_in_followup_response()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
