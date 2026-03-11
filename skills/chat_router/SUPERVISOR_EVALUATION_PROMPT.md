Evaluate whether the current skill outputs are sufficient.

QUESTION:
{{USER_QUESTION}}

RECENT CONVERSATION:
{{HISTORY_TEXT}}

SKILL RESULTS (aggregated):
{{RESULT_SUMMARY}}

TOTAL RECORDS FOUND ACROSS ALL SKILLS: {{TOTAL_RECORDS_FOUND}}

STEP:
{{STEP}}/{{MAX_STEPS}}

Return STRICT JSON:
{
  "satisfied": true/false,
  "confidence": 0.0,
  "reasoning": "short explanation",
  "missing": ["what is still missing"]
}

Rules:
- If total_records_found > 0, the question about existence of traffic is answered. Set satisfied=true.
- satisfied=true only if the evidence answers the question with relevant support.
- If evidence is weak, set satisfied=false and list what is missing.
- At the final step ({{MAX_STEPS}}), set satisfied=true if any useful data was gathered.