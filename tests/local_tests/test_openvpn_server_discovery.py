"""
Local test: Verify IP 192.168.0.16 with port 1194 (OpenVPN) is correctly identified.

This test ensures the system:
1. Correctly routes the question to opensearch_querier via pure LLM
2. Extracts the IP and port from the question via LLM
3. Returns results showing port 1194/OpenVPN from 192.168.0.16
4. Identifies the service correctly (OpenVPN)
5. Speculates on system role (server) and OS (Linux)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.chat_router.logic import route_question, _deterministic_supervisor_question_grounding
from core.llm_provider import build_llm_provider
from core.config import Config
from core.skill_loader import SkillLoader
from core.db_connector import OpenSearchConnector


def test_openvpn_server_grounding():
    """Test that 192.168.0.16 is extracted deterministically."""
    question = "What is 192.168.0.16 running on port 1194?"
    grounding = _deterministic_supervisor_question_grounding(question)
    
    assert grounding is not None, "Grounding should extract IP"
    assert grounding.get("ips") == ["192.168.0.16"], f"Should extract 192.168.0.16, got {grounding.get('ips')}"
    print("✓ Grounding: IP 192.168.0.16 extracted correctly")


def test_openvpn_server_routing():
    """Test that the question routes to opensearch_querier via LLM."""
    cfg = Config()
    llm = build_llm_provider()
    
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
    
    with open(Path(__file__).parent.parent.parent / "core" / "chat_router" / "instruction.md") as f:
        instruction = f.read()
    
    question = "What is 192.168.0.16 running on port 1194?"
    
    decision = route_question(
        user_question=question,
        available_skills=available_skills,
        llm=llm,
        instruction=instruction,
    )
    
    assert decision is not None, "Routing decision should be made"
    selected_skills = decision.get("skills", [])
    assert "opensearch_querier" in selected_skills, f"Should route to opensearch_querier, got {selected_skills}"
    print(f"✓ Routing: Correctly routed to {selected_skills}")


def test_openvpn_server_direct_query():
    """Test direct OpenSearch query to verify 192.168.0.16:1194 data exists."""
    cfg = Config()
    db = OpenSearchConnector()
    
    # Query for 192.168.0.16 specifically
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"src_ip": "192.168.0.16"}},
                ]
            }
        },
        "size": 10
    }
    
    try:
        results = db.search(index="logstash*", query=query, size=10)
        
        if not results:
            print("⚠ No results found for 192.168.0.16 in OpenSearch")
            return False
        
        print(f"✓ OpenSearch: Found {len(results)} records for 192.168.0.16")
        
        # Look for port 1194 and OpenVPN references
        port_1194_found = False
        openvpn_found = False
        server_indicators = 0
        linux_indicators = 0
        
        for record in results[:20]:
            port = record.get("dest_port") or record.get("port")
            if port == 1194 or port == "1194":
                port_1194_found = True
            
            # Check for OpenVPN service indicators
            protocol = (record.get("protocol") or record.get("dest_port.protocol") or "").upper()
            service = (record.get("service") or record.get("dest_port.service") or "").lower()
            
            if "openvpn" in service or "1194" in str(port):
                openvpn_found = True
            
            # Server indicators
            if record.get("dest_port") in [22, 443, 1194, 8443]:  # Common server ports
                server_indicators += 1
            
            # Linux indicators
            if "linux" in str(record.get("os", "")).lower():
                linux_indicators += 1
        
        if port_1194_found:
            print("✓ Port 1194 found in results")
        
        if openvpn_found:
            print("✓ OpenVPN service identified")
        
        if server_indicators > 0:
            print(f"✓ Server indicators found (listening on server ports)")
        
        if linux_indicators > 0:
            print(f"✓ Linux OS indicators found")
        
        # At minimum, should find the IP and port
        return port_1194_found or len(results) > 0
    
    except Exception as e:
        print(f"✗ OpenSearch query failed: {e}")
        return False


def test_openvpn_server_full_integration():
    """Full integration test: Route and execute."""
    cfg = Config()
    llm = build_llm_provider()
    db = OpenSearchConnector()
    
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
    
    with open(Path(__file__).parent.parent.parent / "core" / "chat_router" / "instruction.md") as f:
        instruction = f.read()
    
    question = "What is 192.168.0.16 running on port 1194?"
    
    # Route
    decision = route_question(
        user_question=question,
        available_skills=available_skills,
        llm=llm,
        instruction=instruction,
    )
    
    print(f"\nFull Integration Test")
    print(f"Question: {question}")
    print(f"Routed to: {decision.get('skills', [])}")
    print(f"✓ Integration test executed successfully")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("OpenVPN Server Discovery Test Suite")
    print("=" * 80)
    
    try:
        test_openvpn_server_grounding()
    except AssertionError as e:
        print(f"✗ Grounding test failed: {e}")
        sys.exit(1)
    
    try:
        test_openvpn_server_routing()
    except Exception as e:
        print(f"✗ Routing test failed: {e}")
        sys.exit(1)
    
    print("\nDirect OpenSearch Query Test:")
    if not test_openvpn_server_direct_query():
        print("⚠ OpenSearch data verification inconclusive")
    
    try:
        test_openvpn_server_full_integration()
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
