Review whether this proposed next supervisor step is the best grounded immediate action.

Question:
{{USER_QUESTION}}

Allowed skills:
{{SKILL_CATALOG_JSON}}

Current results:
{{CURRENT_RESULTS}}

Question grounding:
{{QUESTION_GROUNDING}}

Previous evaluation:
{{PREVIOUS_EVALUATION}}

Previous trace:
{{PREVIOUS_TRACE}}

Proposed reasoning:
{{PROPOSED_REASONING}}

Proposed skills:
{{PROPOSED_SKILLS}}

Proposed parameters:
{{PROPOSED_PARAMETERS}}

Instructions:
- Judge the immediate next step, not the final end-to-end workflow.
- Prefer the most grounded next step for the current question and current evidence.
- Treat QUESTION GROUNDING as authoritative for what the user is asking right now.
- Reject plans that answer a different question than the user asked.
- Reject plans whose reasoning reframes the request into a different answer type such as geolocation, threat assessment, baseline comparison, or schema explanation unless that reframe is explicitly present in the grounding.
- Treat passive fingerprinting, service profiling, likely host role, and OS-likelihood questions as behavior/evidence tasks rather than geolocation.
- Reject `geoip_lookup` for fingerprinting unless the grounding explicitly asks where the IP is located.
- Reject plans that pivot to enrichment or side-analysis before the core evidence-gathering step is grounded.
- Treat the allowed skill catalog and current results as the only valid execution space.
- If the plan is invalid, explain the specific mismatch and suggest a better immediate next step.

Return strict JSON:
{
  "is_valid": true,
  "should_execute": true,
  "confidence": 0.0,
  "reasoning": "why this immediate next step is or is not grounded",
  "issue": "specific problem if invalid",
  "suggestion": "how the immediate next step should change"
}
