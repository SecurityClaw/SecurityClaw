# Supervisor Reflection Repair

You are performing a reflection step after the previous execution did not satisfy the user's question.

Your task is to produce the next viable investigation step using only the loaded skills in the allowed catalog.

## Reflection Procedure
1. Identify why the previous step failed or remained insufficient.
2. Check whether the prior plan used the wrong capability, lacked prerequisites, or produced irrelevant evidence.
3. Choose the next viable skill from the allowed catalog.
4. If the same skill should run again, change the parameters or explain why the retry is materially different.
5. Never invent a skill name.

## Current Question
{{USER_QUESTION}}

## Allowed Skill Catalog
```json
{{SKILL_CATALOG_JSON}}
```

## Current Results
{{CURRENT_RESULTS}}

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
  "reasoning": "Root cause and why this next step is now viable.",
  "skills": ["exact_loaded_skill_name"],
  "parameters": {
    "question": "Grounded question for the selected skill"
  }
}
```
