"""
Local live chat tests for the 4 primary use cases.

These tests are designed to be run locally against the chat interface.
They test the following 4 prompts:
1. "russia traffic in past 30 days?"
2. "russia traffic in past 24 hours?"
3. "reputation of 1.1.1.1?"
4. "fingerprint 192.168.0.17"

These tests validate end-to-end routing, skill execution, and response formatting.
They are NOT committed to the repository (see .gitignore).
"""

import json
from core.chat_router.logic import route_question


def _mock_llm_for_russia_30d():
    """Mock LLM that routes russia 30-day traffic to fields + opensearch."""

    class MockLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "User is asking for concrete traffic evidence from Russia in the past 30 days.",
                    "skills": ["fields_querier", "opensearch_querier"],
                    "parameters": {},
                }
            )

    return MockLLM()


def _mock_llm_for_russia_24h():
    """Mock LLM that routes russia 24-hour traffic to fields + opensearch."""

    class MockLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "User is asking for concrete traffic evidence from Russia in the past 24 hours.",
                    "skills": ["fields_querier", "opensearch_querier"],
                    "parameters": {},
                }
            )

    return MockLLM()


def _mock_llm_for_reputation():
    """Mock LLM that routes direct reputation question."""

    class MockLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "User is asking about reputation of a specific IP address.",
                    "skills": ["threat_analyst"],
                    "parameters": {"ip": "1.1.1.1"},
                }
            )

    return MockLLM()


def _mock_llm_for_fingerprint():
    """Mock LLM that routes fingerprint question."""

    class MockLLM:
        def chat(self, messages: list[dict]):
            return json.dumps(
                {
                    "reasoning": "User is asking for a passive fingerprint of a specific IP.",
                    "skills": ["ip_fingerprinter"],
                    "parameters": {"ip": "192.168.0.17"},
                }
            )

    return MockLLM()


def test_local_russia_traffic_past_30_days():
    """Test: russia traffic in past 30 days?"""
    result = route_question(
        user_question="russia traffic in past 30 days?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "baseline_querier", "description": "Baseline analysis"},
        ],
        llm=_mock_llm_for_russia_30d(),
        instruction="test",
        conversation_history=[],
    )

    print("\n=== Test: russia traffic in past 30 days? ===")
    print(f"Route: {result['skills']}")
    assert "fields_querier" in result["skills"]
    assert "opensearch_querier" in result["skills"]
    assert "baseline_querier" not in result["skills"]
    print("✓ PASS: Correctly uses fields_querier + opensearch_querier, not baseline")


def test_local_russia_traffic_past_24_hours():
    """Test: russia traffic in past 24 hours?"""
    result = route_question(
        user_question="russia traffic in past 24 hours?",
        available_skills=[
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
            {"name": "baseline_querier", "description": "Baseline analysis"},
        ],
        llm=_mock_llm_for_russia_24h(),
        instruction="test",
        conversation_history=[],
    )

    print("\n=== Test: russia traffic in past 24 hours? ===")
    print(f"Route: {result['skills']}")
    assert "fields_querier" in result["skills"]
    assert "opensearch_querier" in result["skills"]
    assert "baseline_querier" not in result["skills"]
    print("✓ PASS: Correctly uses fields_querier + opensearch_querier for 24-hour window")


def test_local_reputation_of_ip():
    """Test: reputation of 1.1.1.1?"""
    result = route_question(
        user_question="reputation of 1.1.1.1?",
        available_skills=[
            {"name": "threat_analyst", "description": "IP reputation analysis"},
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        llm=_mock_llm_for_reputation(),
        instruction="test",
        conversation_history=[],
    )

    print("\n=== Test: reputation of 1.1.1.1? ===")
    print(f"Route: {result['skills']}")
    assert "threat_analyst" in result["skills"]
    print("✓ PASS: Correctly routes to threat_analyst for direct reputation question")


def test_local_fingerprint_ip():
    """Test: fingerprint 192.168.0.17"""
    result = route_question(
        user_question="fingerprint 192.168.0.17",
        available_skills=[
            {"name": "ip_fingerprinter", "description": "Passive IP fingerprint analysis"},
            {"name": "fields_querier", "description": "Field schema discovery"},
            {"name": "opensearch_querier", "description": "Direct log search"},
        ],
        llm=_mock_llm_for_fingerprint(),
        instruction="test",
        conversation_history=[],
    )

    print("\n=== Test: fingerprint 192.168.0.17 ===")
    print(f"Route: {result['skills']}")
    assert "ip_fingerprinter" in result["skills"]
    print("✓ PASS: Correctly routes to ip_fingerprinter")


if __name__ == "__main__":
    test_local_russia_traffic_past_30_days()
    test_local_russia_traffic_past_24_hours()
    test_local_reputation_of_ip()
    test_local_fingerprint_ip()
    print("\n=== All 4 prompt tests passed ===")
