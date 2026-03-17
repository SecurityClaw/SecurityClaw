#!/usr/bin/env python3
"""Test current fingerprinting output."""
import sys
sys.path.insert(0, '/mnt/c/Users/tsike/Desktop/SecurityClaw')

from core.skill_loader import SkillLoader
from tests.mock_llm import MockLLMProvider

# Load the fingerprinter skill
skill_loader = SkillLoader()
runner = skill_loader.discover()
fingerprinter_run = runner['ip_fingerprinter'].run

# Test with an IP
result = fingerprinter_run({
    'question': 'fingerprint 192.168.0.16',
    'ports': []  # Empty ports - this is the problem!
})

print("\n" + "="*80)
print("CURRENT IP_FINGERPRINTER OUTPUT (without port data)")
print("="*80)
print(f"\nInput: question='fingerprint 192.168.0.16', ports=[]")
print(f"\nOutput:")
print(f"  Status: {result.get('status')}")
print(f"  Result: {result.get('result', {}).get('summary', 'N/A')[:200]}")
print(f"  Confidence: {result.get('result', {}).get('confidence', 'N/A')}")
print(f"\nProblem: Returns 0 ports because no port data was gathered first!")
print("\nSolution: Route fingerprinting to: fields_querier → opensearch_querier → ip_fingerprinter")
print("="*80)
