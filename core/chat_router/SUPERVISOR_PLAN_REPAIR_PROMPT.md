# Supervisor Plan Repair

You are repairing a supervisor plan that was invalid, unavailable, or not viable.

You must return a corrected NEXT STEP using only the loaded skills listed in the allowed catalog.

## Rules
- Use only exact skill names from the allowed skill catalog.
- Do not invent replacement skills.
- If the desired answer requires multiple stages, choose the next missing viable stage now.
- Respect manifest prerequisites and current results.
- Treat QUESTION GROUNDING as authoritative for the user's current request.
- If the failed plan reframed the request into a different answer type, repair it back to the grounded task.
- If the proposed plan included unavailable skills, explain why the repaired plan is different.
- Prefer the smallest viable step that moves the investigation forward.

## Current Question
{{USER_QUESTION}}

## Allowed Skill Catalog
```json
{{SKILL_CATALOG_JSON}}
```

## Current Results
{{CURRENT_RESULTS}}

## Question Grounding
{{QUESTION_GROUNDING}}

## Previous Evaluation
{{PREVIOUS_EVALUATION}}

## Previous Trace
{{PREVIOUS_TRACE}}

## Invalid Or Unavailable Skills
{{INVALID_SKILLS}}

## Proposed Skills
{{PROPOSED_SKILLS}}

## Proposed Parameters
{{PROPOSED_PARAMETERS}}

## Failure Reason
{{FAILURE_REASON}}

## Output
Return strict JSON only:

```json
{
  "reasoning": "Why the previous plan was invalid and why this repaired plan is the next viable step.",
  "skills": ["exact_loaded_skill_name"],
  "parameters": {
    "question": "Grounded question for the selected skill"
  }
}
```
