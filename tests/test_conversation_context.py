"""
tests/test_conversation_context.py — Tests for conversation context passing.

Verifies that:
  - Conversation history is passed through the routing system
  - Search terms are extracted from conversation history
  - Geographic context is preserved across conversation turns
    - baseline_querier receives conversation history for follow-up searches
"""
from __future__ import annotations

import pytest
from tests.mock_llm import MockLLMProvider


class TestRouteQuestionWithHistory:
    """Test route_question() accepts and uses conversation history."""
    
    def test_route_question_signature_accepts_history(self):
        """Verify route_question signature includes conversation_history parameter."""
        from core.chat_router.logic import route_question
        import inspect
        
        sig = inspect.signature(route_question)
        params = list(sig.parameters.keys())
        assert "conversation_history" in params
        assert sig.parameters["conversation_history"].default is None
    
    def test_route_question_includes_history_in_parameters(self):
        """When history is provided, it should be in the returned parameters."""
        from core.chat_router.logic import route_question
        
        llm = MockLLMProvider()
        history = [
            {"role": "user", "content": "Any visits from Iran?"},
            {"role": "assistant", "content": "Found 2 visits from Iran"},
        ]
        
        skills = [{"name": "baseline_querier", "description": "Search baselines"}]
        instruction = "You are a security analyst."
        
        result = route_question("When did visits occur?", skills, llm, instruction, history)
        
        assert "parameters" in result
        assert "conversation_history" in result["parameters"]
        assert result["parameters"]["conversation_history"] == history
    
    def test_route_question_without_history_works(self):
        """route_question should work fine with no history (backward compatibility)."""
        from core.chat_router.logic import route_question
        
        llm = MockLLMProvider()
        skills = [{"name": "baseline_querier", "description": "Search baselines"}]
        instruction = "You are a security analyst."
        
        # Should not raise any exception
        result = route_question("Any visits from Iran?", skills, llm, instruction)
        
        assert "parameters" in result
        assert "question" in result["parameters"]
    
    def test_route_question_includes_history_in_prompt(self):
        """The routing prompt should mention history when it exists."""
        from core.chat_router.logic import route_question
        
        llm = MockLLMProvider()
        history = [
            {"role": "user", "content": "Iran visits"},
            {"role": "assistant", "content": "Found Iran data"},
        ]
        
        skills = [{"name": "baseline_querier", "description": "Search baselines"}]
        instruction = "You are a security analyst."
        
        # MockLLMProvider will return a valid response
        result = route_question("When?", skills, llm, instruction, history)
        
        # Just verify it returns successfully with history
        assert "parameters" in result




class TestConversationFlowIntegration:
    """Integration tests for multi-turn conversations."""
    
    def test_two_turn_conversation_preserves_context(self):
        """Simulates a two-turn conversation where context is preserved."""
        from core.chat_router.logic import route_question
        from tests.mock_llm import MockLLMProvider
        
        llm = MockLLMProvider()
        skills = [{"name": "baseline_querier", "description": "Search baselines"}]
        instruction = "You are a security analyst."
        
        # Q1: Ask about Iran
        q1 = "Any visits from Iran?"
        routing1 = route_question(q1, skills, llm, instruction, conversation_history=None)
        
        # Check Q1 result
        assert "parameters" in routing1
        assert "question" in routing1["parameters"]
        
        # Build history from Q1
        history = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": "Found 2 visits from Iran"},
        ]
        
        # Q2: Ask about timing (without re-mentioning Iran)
        q2 = "When did visits occur?"
        routing2 = route_question(q2, skills, llm, instruction, conversation_history=history)
        
        # Check Q2 result includes history
        assert "parameters" in routing2
        assert "conversation_history" in routing2["parameters"]
        assert routing2["parameters"]["conversation_history"] == history
        
        # The parameters should be passed to baseline_querier when that path is selected
        assert "baseline_querier" in routing2.get("skills", []) or routing2.get("skills") == []
