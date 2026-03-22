# OpenSearch Query Planning Prompt

## Purpose
This prompt guides the LLM to convert natural language questions into structured OpenSearch query parameters.

## Task
Analyze a user's question and extract STRUCTURED fields that will be used to build an OpenSearch query.

## ⚠️ CRITICAL: Understand Query Intent Through Conversation Context
**IMPORTANT:** Analyze the USER'S CURRENT QUESTION with full awareness of prior context.
- Extract plan directly from what the USER IS ASKING NOW
- Conversation context is provided FOR DECISION MAKING — use it to understand query intent
- If the current question refers back to a previous result (e.g., "What were the IPs from that result?"), this is a DRILL-DOWN query:
  - Previous context showed aggregation/summary → Current question asks for details from that result
  - Example: Prior: "Any traffic from Netherlands?" → Response: "Yes, aggregation shows X buckets"
  - Current: "What were the IP addresses from the Netherlands?" → This is asking for INDIVIDUAL RECORDS, NOT another aggregation
  - **DRILL-DOWN RULE:** When you detect a drill-down query (asking for details/records from a previous aggregation), set `aggregation_type="none"` to return individual documents instead of an aggregation
 
- Do NOT invent aggregations (country_terms, etc.) that aren't explicitly requested in the current question
- Do NOT assume the user is asking for something different than what they typed

**VALIDATION: Before returning your plan, ask yourself:**
1. Is this a follow-up question asking for details from a previous aggregation result?
2. Does the current question ask for individual records/IPs/details, or for a summary?
3. If previous turn returned aggregation + current turn asks "what were the [things]", set aggregation_type="none"

## Drill-Down Query Detection
A drill-down query COMBINES current question intent with prior context:
- **Prior context** had an aggregation result (e.g., country_terms showing "X countries with traffic")
- **Current question** asks "what were the [entities]" e.g., "what were the IP addresses", "what were the countries", "what specific traffic"
- **Your decision:** Switch from aggregation_type to aggregation_type="none" to extract individual records

Examples of drill-down detection:
1. Previous: "Any traffic from Netherlands in past 3 months?" → Response: "Found aggregated results" 
   Current: "What were the IP addresses from the Netherlands?" → **DRILL-DOWN**: Use aggregation_type="none" to retrieve IP records
   
2. Previous: "What countries had traffic?" → Response: "Top 5 countries: Iran, China, Russia..."
   Current: "What was the traffic from Iran?" → **DRILL-DOWN**: Use aggregation_type="none" to get Iran traffic records

3. Previous: Not an aggregation result, just context
   Current: "Show me specific traffic events" → **NOT a drill-down necessarily**, but may still want aggregation_type="none" if asking for records

## ⚡ FINGERPRINTING DETECTION (CHECK FIRST!)
**Before analyzing anything else, check if the user is asking for IP fingerprinting:**

**CRITICAL: If the question contains ANY of these keywords + mentions an IP → MUST use aggregation_type="fingerprint_ports":**
1. "fingerprint" keyword: "fingerprint 192.168.0.17" → fingerprint_ports + search_type="ip"
2. "what ports" + IP: "What ports does 10.0.0.1 use?" → fingerprint_ports + search_type="ip"
3. "what services" + IP: "What services run on IP 172.16.0.1?" → fingerprint_ports + search_type="ip"
4. "what destination ports": "What destination ports does X connect to?" → fingerprint_ports + search_type="ip"
5. "profile" + IP: "Profile this IP 1.1.1.1" → fingerprint_ports + search_type="ip"
6. "is it a client or server" + IP: "Is this IP a client or server?" → fingerprint_ports + search_type="ip"
7. Analyzing/characterizing an IP: "Analyze 192.168.0.16" → fingerprint_ports + search_type="ip"

**RULE: When fingerprinting, ALWAYS:**
- Set `search_type="ip"` 
- Set `aggregation_type="fingerprint_ports"`
- Place target IP(s) in `search_terms`
- Set `time_range` to "now-30d" (30-day window for fingerprinting) or longer if specified
- Set `matching_strategy` to "term"
- Leave `countries`, `ports`, `protocols` empty unless explicitly mentioned

**DO NOT** confuse fingerprinting with traffic search:
- ❌ "Show me traffic from this IP" (traffic search) ≠ "Fingerprint this IP" (fingerprinting)
- ❌ "Is there activity on port 443" (traffic) ≠ "What ports does this IP use" (fingerprinting)

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
  "ip_direction": "source|destination|any",
  "aggregation_type": "none|country_terms|fingerprint_ports",
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

### IP Direction
Specify how the user's constraints (countries, IPs, ports) should be applied:
- `source`: Apply filters to SOURCE IP addresses (e.g., "traffic FROM Iran", "FROM this IP")
- `destination`: Apply filters to DESTINATION IP addresses (e.g., "traffic TO port 443")
- `any`: No preference; both source and destination (default when unclear)

Examples:
- "Traffic FROM Iran" → `ip_direction: "source"`
- "Connections TO this server" → `ip_direction: "destination"`
- "Show IP 1.1.1.1" with no FROM/TO → `ip_direction: "any"` (will try to find this IP in either direction)

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
**CRITICAL: Minimize search_terms. Most queries should have 0-2 terms MAX. At most 5 in extreme cases.**

Extract ONLY keywords that require semantic/text searching:
- Domain names: "example.com"
- Specific anomaly keywords: "ransomware", "malware", "suspicious_activity" (if explicitly mentioned in question)
- Rule/signature names: "ET MALWARE", "SURICATA.ALERT"
- Custom event descriptions mentioned by user

**DO NOT extract** in search_terms:
- EXCLUDE: Country names — use `countries` field instead
- EXCLUDE: Port numbers — use `ports` field instead
- EXCLUDE: Protocol names — use `protocols` field instead
- EXCLUDE: Generic metadata words like "logs", "traffic", "data", "record", "ip", "geolocation" — these are NOT searchable business terms
- EXCLUDE: Words that are already covered by structured fields

**RULE:** If the question only uses structured categories (countries/ports/protocols/time), leave search_terms EMPTY []

**Examples:**
- User: "Any traffic from Netherlands in past 3 months?" → search_terms=[] (only structured filters needed)
- User: "Traffic from Iran containing 'malware'" → search_terms=['malware'] (1 semantic term)
- User: "Port 1194 activity from Russia last week" → search_terms=[] (only structured)
- User: "Show me alerts matching ET_MALWARE signature" → search_terms=['ET_MALWARE'] (1 semantic term)
- User: "DNS queries to suspicious domains" → search_terms=['suspicious'] OR better search_terms=[] if you're filtering domains differently

**WHEN IN DOUBT:** Empty search_terms is safer than hallucinating irrelevant terms.

### Search Type
Pick the dominant category of the user's intent:
- `ip`: IP-focused aggregations and fingerprinting questions — when asking ABOUT an IP (its ports, services, characteristics)
  - Examples: "What ports does 192.168.0.1 use?", "Fingerprint 1.1.1.1", "What services run on this IP?"
  - Tip: If the question asks for **IP characteristics/analysis** (ports, services, protocols associated WITH the IP), use `ip`
  - Tip: If the question asks "what ports are associated with", "fingerprint", "client or server", "services on" → likely `ip` + fingerprint_ports aggregation
- `traffic`: traffic/flow/connection/log existence questions — when looking for **records/evidence OF traffic**
  - Examples: "Is there traffic from Russia?", "Show me 1.1.1.1 connections", "Any activity on port 443?"
  - Tip: If the question searches **for records where traffic occurred**, use `traffic`
- `alert`: signatures, alert names, Suricata/Snort/ET rule searches
- `domain`: domain/DNS/FQDN-focused searches
- `general`: fallback when none of the above cleanly fit

**KEY DISTINCTION:**
- "What ports are used BY 1.1.1.1?" or "What ports are associated with 1.1.1.1?" → `search_type="ip"` + `aggregation_type="fingerprint_ports"`
- "Show me traffic TO port 443" or "Is 1.1.1.1 sending traffic?" → `search_type="traffic"`

### Aggregation & Analysis Types
OpenSearch supports aggregate queries as a first-class search feature. When you request an aggregation type, the skill will:
1. Execute an aggregation query in OpenSearch (not document retrieval)
2. Return aggregated results (e.g., port counts, country distributions)
3. **You will interpret the aggregation results** - analyze what the data means and present findings to the user

Choose aggregation only when the user asks for **summarized/distinct data**:

**country_terms aggregation:**
- Use when: User wants distinct/top countries, geographic distribution, traffic sources by country
- Examples: "What countries do we get traffic from?", "Top countries this month?", "Countries other than USA?"
- Set: `aggregation_type="country_terms"`, `aggregation_field="country"`, `result_limit` (default 10)
- What you'll get back: Bucket list of countries with doc_count for each
- Your interpretation task: Analyze which countries have most activity, patterns, anomalies
- Note: `countries` may be empty; use `exclude_countries` for exclusions like USA

**fingerprint_ports aggregation:**
- Use when: User asks for **port discovery, service identification, or passive fingerprinting on a specific IP**
  - Keywords to watch for: "what ports", "what services", "fingerprint", "profiles", "is it a client or server", "what's running on"
- Examples: 
  - "Fingerprint 192.168.0.16"
  - "What ports are associated with 1.1.1.1?"
  - "What services use IP 10.0.0.1?"
  - "Is 1.1.1.1 a client or server?"  
  - "What destination ports does 1.1.1.1 connect to?"
- Set: `aggregation_type="fingerprint_ports"`, `search_type="ip"`, place target IP in `search_terms[0]`
- Time window: Default to `now-30d` (30-day window preferred for port profiling)
- What you'll get back: Port frequency distribution with observation counts for the target IP
  - Format: {port1: {observations: count}, port2: {observations: count}, ...}
- Your interpretation task: Analyze the port distribution to determine what services/protocols run on this IP, assess if it's a client/server, identify known/unknown services
- **CRITICAL**: When you see "what ports are associated with [IP]", ALWAYS set aggregation_type="fingerprint_ports" and search_type="ip", NOT search_type="traffic"

**direct document search (aggregation_type="none"):**
- Use for: Raw log/document retrieval, specific event searches, flow details
- Default behavior when aggregation not applicable

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
- source vs destination IP fields (for fingerprinting which IP is the client/server)
- alert/signature fields
- country/geo fields
- timestamp fields (for time window filtering)

## Examples

### FINGERPRINTING EXAMPLES (Highest Priority!)

Example FP1: "Fingerprint 192.168.0.16"
```json
{
  "reasoning": "User asking for IP fingerprinting - determine ports, services, and characteristics of this IP",
  "search_type": "ip",
  "detected_time_range": "not specified",
  "time_range": "now-30d",
  "countries": [],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": ["192.168.0.16"],
  "ip_direction": "any",
  "aggregation_type": "fingerprint_ports",
  "aggregation_field": "none",
  "result_limit": 100,
  "matching_strategy": "term",
  "field_analysis": "Focus on IP address fields and destination/source ports to determine services, roles (client vs server), and operating system characteristics.",
  "skip_search": false
}
```

Example FP2: "What ports does 10.0.0.1 use?"
```json
{
  "reasoning": "User asking what ports are associated with this IP - port fingerprinting intent",
  "search_type": "ip",
  "detected_time_range": "not specified",
  "time_range": "now-30d",
  "countries": [],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": ["10.0.0.1"],
  "ip_direction": "any",
  "aggregation_type": "fingerprint_ports",
  "aggregation_field": "none",
  "result_limit": 100,
  "matching_strategy": "term",
  "field_analysis": "Extract port distribution for the target IP to determine what services it uses, exposes, or connects to.",
  "skip_search": false
}
```

### TRAFFIC & LOCATION EXAMPLES

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
  "ip_direction": "source",
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
  "ip_direction": "source",
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use port fields, country fields, and timestamp fields.",
  "skip_search": false
}
```

Example 2a: "Any traffic from Netherlands in the past 3 months?"
```json
{
  "reasoning": "User asking for traffic originating from Netherlands",
  "search_type": "traffic",
  "detected_time_range": "past 3 months",
  "time_range": "now-3M",
  "countries": ["Netherlands"],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": [],
  "ip_direction": "source",
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "term",
  "field_analysis": "Use country/geo fields and timestamp fields. No text search needed.",
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
  "ip_direction": "any",
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
  "ip_direction": "any",
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
  "ip_direction": "source",
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
  "ip_direction": "source",
  "aggregation_type": "none",
  "aggregation_field": "none",
  "result_limit": 10,
  "matching_strategy": "token",
  "field_analysis": "Use country/geo fields and timestamp fields for a long-range traffic search.",
  "skip_search": false
}
```

Example 7: "What ports are associated with 1.1.1.1 traffic?"
```json
{
  "reasoning": "User is asking for passive fingerprinting: what ports are observed being used BY the IP 1.1.1.1. This is port discovery on a specific IP, not a search for traffic records.",
  "search_type": "ip",
  "detected_time_range": "not specified",
  "time_range": "now-30d",
  "countries": [],
  "exclude_countries": [],
  "ports": [],
  "protocols": [],
  "search_terms": ["1.1.1.1"],
  "ip_direction": "any",
  "aggregation_type": "fingerprint_ports",
  "aggregation_field": "destination.port",
  "result_limit": 256,
  "matching_strategy": "term",
  "field_analysis": "Use IP fields and port fields to discover which ports this IP is associated with.",
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
  "ip_direction": "source",
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
