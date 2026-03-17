"""Test if SkillManifestLoader works in route_question context"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_manifest import SkillManifestLoader

print("Testing SkillManifestLoader...")

try:
    loader = SkillManifestLoader()
    print("✓ Loader created")
    
    all_manifests = loader.load_all_manifests()
    print(f"✓ Manifests loaded: {len(all_manifests)} skills")
    
    if "geoip_lookup" in all_manifests:
        geoip = all_manifests["geoip_lookup"]
        cannot_answer = geoip.get("cannot_answer", [])
        print(f"✓ geoip_lookup found")
        print(f"  cannot_answer: {cannot_answer}")
        
        # Check if "traffic" is in cannot_answer
        if "traffic" in cannot_answer:
            print("  ✓ 'traffic' is in cannot_answer list")
        else:
            print("  ❌ 'traffic' is NOT in cannot_answer list")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
