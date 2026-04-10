"""Test if the cannot_answer filtering would work for "Any 1.1.1.1 traffic?" """

question = "Any 1.1.1.1 traffic?"
question_lower = question.lower().strip()

# geoip_lookup cannot_answer list from manifest
cannot_answer_patterns = [
    "threat reputation",
    "log searching",
    "timeline reconstruction",
    "OpenSearch field discovery",
    "behavioral baseline analysis"
]

# Check if any pattern matches
for pattern in cannot_answer_patterns:
    pattern_lower = pattern.lower()
    if pattern_lower in question_lower:
        print(f"MATCH: '{pattern_lower}' is found in '{question_lower}'")
    else:
        print(f"NO MATCH: '{pattern_lower}' is NOT in '{question_lower}'")

# Check if "traffic" contains "log searching"
print(f"\nDoes 'traffic' match 'log searching'? No")
print(f"But does a log search ask for 'traffic'? Yes, 'traffic' IS log searching")
print(f"\nProblem: 'log searching' pattern doesn't directly match 'traffic' or 'Any 1.1.1.1 traffic?'")
print(f"Solution: Need a better pattern like 'traffic' or add keywords to geoip cannot_answer")
