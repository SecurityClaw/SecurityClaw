Analyze this security question and decide which available skills to use.
Consider the recent conversation history to maintain context.

Current Question: "{{USER_QUESTION}}"{{HISTORY_CONTEXT}}

Available skills:
{{SKILLS_DESCRIPTION}}

ROUTING GUIDELINES:

PRIMARY ROUTING RULES (Use These First):
0. DIRECT IP GEOLOCATION LOOKUP: ONLY for questions asking to geolocate a SPECIFIC IP ADDRESS.
   These questions ask about a single IP/host and where it's located.
   -> geoip_lookup
   Examples: "what country is 8.8.8.8 from?", "geolocate 1.1.1.1", "what city is IP 62.60.131.168?"
   IMPORTANT: "Any traffic from Russia?" is NOT a geoip_lookup question. It asks for log records matching a country filter, not geolocation for a specific IP.

1. FIELD SCHEMA DISCOVERY FIRST: Questions about "what fields exist", "which field holds X", or "what is the field name for Y" -> fields_querier FIRST.
   Examples: "what field holds country info?", "which field stores IP addresses?"
   Then use opensearch_querier with the discovered field names.

2. NATURAL LANGUAGE LOCATION OR TEMPORAL FILTERING: Questions like "traffic from [country/region]", "logs from [time period]", or "connections on [protocol/port]" WITHOUT explicit field names -> fields_querier FIRST, THEN opensearch_querier.
   Examples: "Any traffic from Russia?", "traffic from Iran in the past 24 hours", "connections on port 443 last week"

3. DIRECT LOG SEARCH WITH KNOWN FIELDS: Questions about logs where explicit field names are already stated -> opensearch_querier directly.
   Examples: "show logs where source.ip=1.2.3.4", "filter where destination.port=443", "search geoip.country_name for Iran"

4. BASELINE ANALYSIS: After finding results, analyze normal or expected behavior -> baseline_querier.
   Use ONLY for follow-up research, not for the initial question.

5. DEPRECATED: rag_querier is legacy. Use opensearch_querier + fields_querier instead.

CRITICAL PREVENTION RULES:
- Do NOT use geoip_lookup as a first skill unless the question explicitly mentions a specific IP address or hostname and asks about its location.
- If a question asks about traffic from a location or country without mentioning specific IPs, use fields_querier -> opensearch_querier first. Only add geoip_lookup later if specific IPs are found and need geolocation.

SECONDARY ROUTING RULES:
- Use forensic_examiner to build a +/-5 minute timeline.
- Use threat_analyst for external reputation checks.
- Use opensearch_querier first to gather evidence, then threat_analyst.

KEY PRIORITY:
- Use fields_querier first for natural-language log-search questions unless the user explicitly names exact fields.
- Use opensearch_querier directly only when field names are already known.
- Use baseline_querier only for follow-up research after results already exist.

ALWAYS ANSWER WITH A JSON OBJECT ONLY:
{
  "reasoning": "Why you chose these skills and which routing rules matched",
  "skills": ["skill_name_1", "skill_name_2"],
  "parameters": {"question": "{{USER_QUESTION}}"}
}