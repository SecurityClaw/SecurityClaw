"""
Chat integration test: Full two-turn conversation flow.
Tests that follow-up questions about ports actually return port information.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import run_graph
from core.llm_provider import build_llm_provider
from core.config import Config
from core.db_connector import OpenSearchConnector
from core.skill_loader import SkillLoader


def test_full_conversation_flow():
    """Test two-turn conversation: traffic search → port follow-up"""
    
    print(f"\n{'='*80}")
    print(f"CHAT INTEGRATION TEST: IP Traffic → Ports Follow-up")
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
    print(f"TURN 1: User asks 'Any 1.1.1.1 traffic?'\n")
    
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
    skills_used1 = result1.get("routing", {}).get("skills", [])
    
    print(f"Skills used: {skills_used1}")
    print(f"Response (first 300 chars): {response1[:300]}...\n")
    
    # Check first response
    has_port_53 = "53" in response1
    has_ip_data = "1.1.1.1" in response1 or "192.168.0" in response1
    
    print(f"First response validation:")
    print(f"  - Contains port 53: {'✅' if has_port_53 else '❌'}")
    print(f"  - Contains IP data: {'✅' if has_ip_data else '❌'}")
    print()
    
    if not has_ip_data:
        print("ERROR: First question should return IP traffic data!")
        return False
    
    # TURN 2: Port follow-up
    print(f"TURN 2: User asks 'What ports are associated with this traffic?'\n")
    
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
    skills_used2 = result2.get("routing", {}).get("skills", [])
    skill_results2 = result2.get("skill_results", {})
    
    print(f"Skills used: {skills_used2}")
    print(f"Response (first 300 chars): {response2[:300]}...\n")
    
    # Check second response
    has_port_info = "53" in response2 or "port" in response2.lower()
    has_error = "error" in response2.lower() or "no passive fingerprint" in response2.lower()
    
    print(f"Second response validation:")
    print(f"  - Contains port information: {'✅' if has_port_info else '❌'}")
    print(f"  - Is error response: {'❌' if has_error else '✅'}")
    
    if skill_results2.get("opensearch_querier"):
        os_ports = skill_results2["opensearch_querier"].get("ports", [])
        print(f"  - opensearch_querier found ports: {os_ports if os_ports else 'None'}")
    
    if skill_results2.get("ip_fingerprinter"):
        fp_result = skill_results2["ip_fingerprinter"]
        print(f"  - ip_fingerprinter status: {fp_result.get('status', '?')}")
        if fp_result.get("status") == "error":
            print(f"    Error: {fp_result.get('error', 'unknown')}")
    
    print()
    
    # Final verdict
    print("="*80)
    if has_port_info and not has_error:
        print("✅ PASS: Follow-up question answered with port information")
        return True
    else:
        print("❌ FAIL: Follow-up question did not return port information")
        if has_error:
            print(f"   Response: {response2[:200]}...")
        return False


if __name__ == "__main__":
    try:
        success = test_full_conversation_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
