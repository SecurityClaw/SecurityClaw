"""Debug why both skills are being filtered"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_manifest import apply_routing_guards, SkillManifestLoader

loader = SkillManifestLoader()
all_manifests = loader.load_all_manifests()

question = "Any 1.1.1.1 traffic?"
selected = ["opensearch_querier", "geoip_lookup"]  # This is what the LLM returns
available = [
    {"name": "opensearch_querier"},
    {"name": "geoip_lookup"},
]

print(f"Question: {question}\n")
print(f"Selected before filtering: {selected}\n")

# Check each skill's cannot_answer
for skill in selected:
    manifest = all_manifests.get(skill, {})
    cannot_answer = manifest.get("cannot_answer", [])
    print(f"{skill}:")
    print(f"  cannot_answer: {cannot_answer}")
    for pattern in cannot_answer:
        if pattern.lower() in question.lower():
            print(f"    MATCH: '{pattern}' found in question")

print(f"\nApplying guards...")
result = apply_routing_guards(selected, question, available, all_manifests)
print(f"Selected after filtering: {result}")
