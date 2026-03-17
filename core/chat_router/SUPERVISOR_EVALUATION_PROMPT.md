# Supervisor Satisfaction Evaluation

Evaluate whether the currently accumulated skill outputs are sufficient to answer the user's question.
Evaluate whether the current skill outputs are sufficient.

## Rules
- Judge sufficiency based on the user's actual question.
- Prefer grounded evidence over optimistic interpretation.
- If a prior step failed, say what evidence or capability is still missing.
- Do not claim satisfaction merely because a skill executed successfully.
- If the current evidence supports a partial answer but not the asked answer, mark unsatisfied.

## Question
{{USER_QUESTION}}

## Recent Conversation Context
{{HISTORY_TEXT}}

## Skill Results
{{RESULT_SUMMARY}}

## Total Records Found Across Skills
{{TOTAL_RECORDS_FOUND}}

## Step Budget
{{STEP}}/{{MAX_STEPS}}

## Output
Return strict JSON only:

```json
{
  "satisfied": false,
  "confidence": 0.0,
  "reasoning": "Why the current results do or do not answer the question.",
  "missing": ["specific missing evidence or analysis"]
}
```
