# OpenSearch Query Planning Prompt

## Purpose
This prompt guides the LLM to convert natural language questions into structured OpenSearch query parameters.

## Task
Analyze a user's question and extract STRUCTURED fields that will be used to build an OpenSearch query.

## Input
- User question (natural language)
- Conversation context (prior Q&A)

## Output Format
Respond in STRICT JSON:
```json
{
  "reasoning": "What the user is looking for",
  "search_type": "alert|traffic|domain|ip|general",
  "detected_time_range": "Time period description (or 'none')",
  "time_range": "Elasticsearch range code (now-3M, now-1w, now-7d, now-1d, now-90d, etc.)",
  "countries": ["CountryName1", "CountryName2"],
  "exclude_countries": ["CountryName3"],
  "ports": [1194, 443],
  "protocols": ["TCP", "UDP"],
  "search_terms": ["keyword1", "keyword2"],
  "aggregation_type": "none|country_terms",
  "aggregation_field": "country|none",
  "result_limit": 10,
  "matching_strategy": "phrase|token|term|match",
  "field_analysis": "Which field categories are most relevant and why",
  "skip_search": false
}
```

## Extraction Rules

### Countries
Extract country NAMES (not codes). Examples: "Iran", "Russia", "China"
- Look for explicit country mentions: "from Iran", "in Russia", "China traffic"
- Look in conversation context for previously mentioned countries
- **Do NOT assume** — only extract if clearly stated
- If the question excludes a country ("other than the USA", "excluding Russia"), place that country in `exclude_countries` instead of `countries`

### Ports
Extract as integers. Examples: 443, 1194, 22, 53
- Look for "port [number]", "port [number]", ":[number]"
- Look for service names that map to ports (SSH=22, HTTPS=443, DNS=53, OpenVPN=1194)
- Can omit if not mentioned

### Protocols
Extract protocol names in UPPERCASE. Examples: "TCP", "UDP", "ICMP", "DNS", "TLS"
- Look for explicit mentions: "TCP connections", "UDP traffic"
- Look for protocol indicators in context
- Can omit if not mentioned

### Time Range
Extract as Elasticsearch range code (ALWAYS use lowercase time unit letters):
- "past 3 months" → `now-3M`
- "past 3 years" or "past year" → `now-3y` (lowercase 'y', NOT 'Y')
- "past 1 year" → `now-1y`
- "last week" → `now-1w`
- "today" → `now-1d`
- "last 90 days" → `now-90d`
- (no time mentioned) → `now-90d` (default)

CRITICAL: Only use lowercase: d, w, M, y. Never use uppercase D, W, Y.
Elasticsearch date math ONLY accepts lowercase time units.

### Search Terms
Extract keywords that don't fit structured fields:
- Domain names: "example.com"
- Service names that aren't standard ports: "SSH", "OpenVPN"
- Anomaly indicators: "suspicious", "alert"
- Event types: "DNS query", "connection failure"
- **Do NOT extract** country names, ports, or protocols here (they have their own fields)

### Search Type
Pick the dominant category of the user's intent:
- `ip`: direct IP lookup, IP reputation, or questions centered on specific IP addresses
- `traffic`: traffic/flow/connection/log existence questions
- `alert`: signatures, alert names, Suricata/Snort/ET rule searches
- `domain`: domain/DNS/FQDN-focused searches
- `general`: fallback when none of the above cleanly fit

### Aggregation
- Use `aggregation_type="country_terms"` and `aggregation_field="country"` when the user wants a distinct/top list of countries rather than raw matching documents
- Examples: "What countries do we get traffic from?", "What countries other than the USA do we get traffic from in the past month?", "Top 10 source countries this week"
- When using country aggregation, `countries` may be empty and `exclude_countries` should hold exclusions like USA
- Set `result_limit` from phrases like "top 5"; otherwise default to 10

### Matching Strategy
- `term`: exact values like IPs, ports, keyword fields, protocol literals
- `phrase`: exact signature or rule names where tokenization would broaden matches too much
- `token`: standard free-text matching across text fields
- `match`: fallback if none of the above fit cleanly

For IPs and ports, prefer `term`.
For alerts/signatures/rules, prefer `phrase`.
For general traffic/log searches, prefer `token`.

### Field Analysis
Briefly explain which discovered field categories matter most, for example:
- source vs destination IP fields
- alert/signature fields
- country/geo fields
- timestamp fields

## Examples

Example 1: "Show me traffic from Iran in the past 3 months"
```json
{
  "reasoning": "User wants to see network traffic originating from Iran",
  "search_type": "traffic",
  "detected_time_range": "past 3 months",
  "time_range": "now-3M",
  "countries": ["Iran"],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": [],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "token",
  "field_analysis": "Use country/geo fields plus timestamp fields for a traffic search.",
  "skip_search": false
}
```

Example 2: "Port 1194 activity in Russia last week"
```json
{
  "reasoning": "User asking for activity on port 1194 from Russia",
  "search_type": "traffic",
  "detected_time_range": "last week",
  "time_range": "now-7d",
  "countries": ["Russia"],
  "exclude_countries": [],
  "ports": [1194],
  "protocols": [],
  "search_terms": [],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use port fields, country fields, and timestamp fields.",
  "skip_search": false
}
```

Example 3: "Find TCP connections to example.com"
```json
{
  "reasoning": "User wants TCP flows to example.com domain",
  "search_type": "domain",
  "detected_time_range": "not specified",
  "time_range": "now-90d",
  "countries": [],
  "exclude_countries": [],
  "ports": [],
  "protocols": ["TCP"],
  "search_terms": ["example.com"],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use domain fields plus protocol and timestamp fields.",
  "skip_search": false
}
```

Example 4: "What fields are available for byte transfers?"
```json
{
  "reasoning": "User asking about field schema, not executing a search",
  "search_type": "general",
  "detected_time_range": "N/A",
  "time_range": "now-90d",
  "countries": [],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": [],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "token",
  "field_analysis": "Schema question only; no OpenSearch execution needed.",
  "skip_search": true
}
```

Example 5: "China TCP connections on port 443 or 22 past 90 days"
```json
{
  "reasoning": "User wants TCP connections from China on SSH or HTTPS ports",
  "search_type": "traffic",
  "detected_time_range": "past 90 days",
  "time_range": "now-90d",
  "countries": ["China"],
  "exclude_countries": [],
  "ports": [443, 22],
  "protocols": ["TCP"],
  "search_terms": [],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use country, port, protocol, and timestamp fields.",
  "skip_search": false
}
```

Example 6: "Traffic from Iran in the past 3 years"
```json
{
  "reasoning": "User wants to see network traffic from Iran going back 3 years",
  "search_type": "traffic",
  "detected_time_range": "past 3 years",
  "time_range": "now-3y",
  "countries": ["Iran"],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": [],
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "token",
  "field_analysis": "Use country/geo fields and timestamp fields for a long-range traffic search.",
  "skip_search": false
}
```

Example 7: "What countries other than the USA do we get traffic from in the past month"
```json
{
  "reasoning": "User wants a distinct list of non-US source countries seen in traffic over the past month.",
  "search_type": "traffic",
  "detected_time_range": "past month",
  "time_range": "now-30d",
  "countries": [],
  "exclude_countries": ["United States"],
  "ports": [],
  "protocols": [],
  "search_terms": [],
  "aggregation_type": "country_terms",
  "aggregation_field": "country",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use country/geo fields with a terms aggregation plus timestamp filtering.",
  "skip_search": false
}
```

## Error Cases

| Scenario | Behavior |
|----------|----------|
| Country mentioned unsure (e.g., "some country somewhere") | Leave countries=[], don't guess |
| Port range given ("ports 1000-2000") | Extract individual ports if under 10, else search_terms="port_range_1000-2000" |
| No time period mentioned | Default to now-90d |
| Question is about schema/fields | Set skip_search=true |


## Implementation Notes

Python code receives this JSON and:
1. Maps country names to ISO codes (Iran→IR, Russia→RU, China→CN)
2. Builds OpenSearch `match_phrase` queries for country names
3. Builds OpenSearch `term` queries for ports, protocols
4. Adds time range filter: `{"range": {"@timestamp": {"gte": time_range}}}`
5. Executes against the index with proper nesting and filtering
