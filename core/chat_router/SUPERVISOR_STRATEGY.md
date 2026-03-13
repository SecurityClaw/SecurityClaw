# Chat Router Supervisor Strategy

A deep-dive into the routing and orchestration strategy for the SecurityClaw chat supervisor.

## Overview

The chat router supervisor is the **central orchestration engine** responsible for:
1. **Intelligent skill selection** based on question semantics
2. **Iterative investigation loops** that refine evidence over multiple steps
3. **Satisfaction evaluation** to prevent endless loops while ensuring completeness
4. **Result synthesis** into actionable natural-language answers

This document explains the strategic decisions and heuristics behind the supervisor's logic.

---

## 1. Initial Routing Strategy

### Routing Decision Tree

```
User Question
    ├─ Explicit command (create baseline, scan fields)?
    │  └─→ Route directly to requested skill
    │
    ├─ Forensic/timeline/incident keywords?
    │  └─→ forensic_examiner (with optional opensearch_querier for supporting data)
    │
    ├─ Reputation/threat/malicious keywords?
    │  └─→ threat_analyst (or after evidence gathering)
    │
    ├─ Specific IP/domain mentioned?
    │  └─→ geoip_lookup (if just asking location)
    │  └─→ opensearch_querier (if asking for traffic/connections)
    │
    ├─ Log search / traffic / data discovery?
    │  ├─ About specific data types (alerts, signatures)?
    │  │  └─→ fields_querier → opensearch_querier
    │  └─ Natural language search?
    │     └─→ opensearch_querier
    │
    ├─ Comparison / anomaly detection?
    │  └─→ baseline_querier or anomaly_triage
    │
    └─→ Default: opensearch_querier or ask for clarification
```

### Skill Selection Criteria

| Question Type | Primary Skill | Secondary Skill(s) | Rationale |
|---|---|---|---|
| "Where is [IP]?" | geoip_lookup | - | Direct geolocation |
| "Show me traffic from [IP]" | opensearch_querier | geoip_lookup (for country context) | Log search, optionally enrich with geo |
| "What happened?" | forensic_examiner | opensearch_querier (if timeline sparse) | Incident reconstruction |
| "Is [IP] malicious?" | threat_analyst | - | Reputation verdicts |
| "Show me alerts" | fields_querier → opensearch_querier | - | Field discovery → search with discovered fields |
| "[Data type] in logs?" | opensearch_querier | fields_querier (if schema unknown) | Log search, fallback to field discovery |
| "Baseline analysis" | baseline_querier | fields_querier (if schema mismatch) | Comparative analysis |

---

## 2. Iterative Decision Strategy

### The Supervision Loop

After each skill executes, the supervisor:
1. **Evaluates** results against the original question
2. **Decides** what to do next (continue, specialize, or complete)
3. **Guards** against loops and dead ends

### State Transitions

```
[INITIAL ROUTE]
    ↓
[EXECUTE SKILLS]
    ↓
[EVALUATE] — Is the question answered?
    │
    ├─ YES (satisfied)
    │   └─→ [FORMAT] → Return answer
    │
    └─ NO (not satisfied)
        ├─ Missing evidence?
        │  └─→ [ROUTE RECOVERY]: Queue alternative or follow-up skill
        │
        ├─ Incomplete analysis?
        │  └─→ [ROUTE ENRICHMENT]: Queue threat/anomaly analysis
        │
        ├─ Step limit reached?
        │  └─→ [FORMAT] → Best-effort answer with available data
        │
        └─ Repeated failure?
            └─→ [FORMAT] → Explain limitation and partial answer
```

### Recovery Heuristics

When initial skills return no data or poor data:

| Scenario | Trigger | Recovery Action |
|---|---|---|
| Opensearch returns 0 records | No matching logs | Try baseline_querier with broader params |
| Same skill queued twice | Loop detection | Block re-execution; try alternative |
| Field schema mismatch | Validation errors | Queue fields_querier to discover correct fields |
| Missing threat verdicts | User asked for reputation | Queue threat_analyst if not already run |
| Partial timeline | Forensic found insufficient events | Retry with wider time window or different alert types |

### Enrichment Heuristics

After gathering data, enrich with:

| Primary Data | Enrichment Skill | Trigger |
|---|---|---|
| Traffic logs (IPs) | threat_analyst | User asked "are they malicious?" |
| Alert/forensic events | threat_analyst | C2 domains or suspicious IPs in results |
| Raw logs | geoip_lookup | Results have IPs; user asked for geography |
| OpenSearch results | anomaly_triage | User asked "is this anomalous?" |

---

## 3. Satisfaction Evaluation Strategy

### When to Mark SATISFIED

- **Evidence-based criteria** are met:
  - Forest/incident timeline: Forensic returned events
  - Traffic/log search: Records found with expected fields
  - Reputation: Threat verdicts returned
  - Geolocation: Country/city resolved
  
- **Comparative criteria** are met:
  - Baseline comparison: Baseline results vs. current activity
  - Anomaly detection: Anomalies flagged or normal behavior confirmed

- **Completeness criteria** are met:
  - All aspects of the question are addressed
  - All key entities are covered (or explicitly noted as missing)
  - User's intent is satisfied (even if with partial data)

### When to Continue Investigating

- **Evidence is incomplete**:
  - Found IPs but no reputation yet → Queue threat_analyst
  - Found events but missing timeline context → Broaden search
  - Found some records but query returned "truncated" → Continue with filtering

- **Question implies multiple dimensions**:
  - "What happened and who was involved?" → Forensic (timeline) + threat_analyst (reputation)
  - "Show me suspicious traffic and origins" → OpenSearch (traffic) + geoip_lookup (origins)

- **User signals incomplete satisfaction**:
  - Implicit: " What else...?" implies more investigation
  - Explicit: Further refinements in follow-up questions

### When to Force Completion

- **Max steps reached** (default 4): Return best-effort answer with available data
- **Repeated failures**: If skill failed twice, return explanation + partial data
- **Impossible to complete**: Dead-end (e.g., IP doesn't exist in any database)

---

## 4. Response Synthesis Strategy

### Prioritization by Question Type

1. **Forensic Questions**
   - Lead with timeline (chronological narrative)
   - Emphasize attack progression and key stages
   - Include context anchors (IPs, ports, affected systems)
   - Optionally append threat verdicts for discovered entities

2. **Traffic/Log Questions**
   - Lead with counts and date range
   - Detail entities (IPs, ports, countries, protocols)
   - Compare against baseline if available
   - Note any anomalous patterns

3. **Reputation Questions**
   - Lead with verdict (MALICIOUS, SUSPICIOUS, BENIGN)
   - Include confidence score and reasoning
   - Reference threat intelligence sources
   - Recommend actions (isolate, monitor, whitelist)

4. **Geolocation Questions**
   - Lead with location (city, country)
   - Include coordinates and timezone
   - Compare against expected/unexpected context
   - Note any VPN/proxy indicators if available

### Anti-Patterns

- **❌ Methodology-First**: Avoid "I searched OpenSearch and found..."
- **❌ Ungrounded Claims**: Never reference data not in the skill results
- **❌ Verbose Preambles**: Jump directly to the answer
- **❌ Repeated Findings**: Synthesize redundant results into a single statement
- **❌ False Precision**: Don't claim verdicts beyond what threat_analyst returned

---

## 5. Guard Rails & Anti-Loop Mechanisms

### Preventing Infinite Loops

1. **State Tracking**:
   - Track previously executed skill combinations
   - Block identical re-executions
   - Allow re-execution with modified parameters only

2. **Step Counter**:
   - Default max steps = 4 (configurable)
   - After max steps, force completion
   - Log warning if approaching limit

3. **Deduplication**:
   - If supervisor proposes [skill_X] and it was just run, check:
     - Did parameters change significantly? (Allow)
     - Is the context genuinely different? (Allow)
     - Otherwise, block and suggest alternative (Prevent)

### Handling Ambiguous Results

1. **Partial Matches**: If results are borderline (e.g., 30% confidence verdict):
   - Note the ambiguity in the response
   - Suggest further investigation if needed
   - Don't skip to satisfied

2. **Missing Context**: If results lack richness:
   - Check if another skill can add context
   - Otherwise, acknowledge the gap and respond with what we have

3. **Conflicting Results**: If two skills return differing conclusions:
   - Weight by reliability and recency
   - Note the discrepancy
   - Recommend manual review for high-stakes decisions

---

## 6. Multi-Skill Workflow Patterns

### Pattern A: Sequence (Discovery → Action)

```
fields_querier [Discover field structure]
    ↓
    opensearch_querier [Use discovered fields for search]
```

**Trigger**: User asks about specific data types (alerts, signatures)
**Benefit**: Avoids field mismatch errors in OpenSearch

---

### Pattern B: Depth (Evidence → Analysis)

```
opensearch_querier [Gather raw evidence]
    ↓
    threat_analyst [Analyze reputation of discovered entities]
```

**Trigger**: User asks for both evidence and threat assessment
**Benefit**: Enriches raw data with threat context

---

### Pattern C: Reconstruction (Timeline → Context)

```
forensic_examiner [Build timeline]
    ↓
    threat_analyst [Assess reputation of discovered entities]
```

**Trigger**: User asks about incident with implicit threat assessment
**Benefit**: Forensic provides anchors; threat_analyst adds verdicts

---

### Pattern D: Validation (Primary → Fallback)

```
opensearch_querier [Primary search]
    ↓
    [if 0 results]
    ↓
    baseline_querier [Alternative search]
        ↓
        [if still 0 results]
        ↓
        fields_querier [Last-resort field discovery]
```

**Trigger**: Initial query returns no results
**Benefit**: Exhausts alternatives before failing

---

## 7. Configuration & Tuning

### Key Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `supervisor_max_steps` | 4 | Max investigation iterations |
| `anti_hallucination_check` | True | Verify answers against data |
| `time_window_default` | 24h | Default if user doesn't specify |
| `record_limit` | 1000 | Max records returned per skill |

### Adjustment Guidance

- **Increase max_steps** if investigations often get cut short
- **Lower max_steps** if too many loops in slow environments
- **Enable anti_hallucination_check** for high-stakes (SOC) vs. disable for speed
- **Adjust record_limit** based on system capacity and user tolerance

---

## 8. Error Handling Strategy

### Skill Execution Failures

| Error Type | Supervisor Response |
|---|---|
| Network connectivity | Queue alternative skill if available; otherwise graceful fail |
| Invalid parameters | Notify + queue retry with corrected params |
| No data returned | Check recovery heuristics (above); try alternate skill |
| Timeout | Log + continue with partial data if available |
| Deprecated field | Queue fields_querier to auto-discover correct field |

### Graceful Degradation

If no skills can complete:
1. Return best available partial answer
2. Explain which evidence was unavailable
3. Suggest follow-up questions or manual investigation
4. Log for system diagnostics

---

## 9. Conversation Memory Integration

The supervisor leverages conversation history to:
- Infer context from prior Q&A turns
- Recover entities mentioned earlier ("What about those IPs?")
- Avoid repeating questions already answered
- Maintain investigation continuity

**Integration Points**:
- Initial routing uses conversation context to refine question
- Recovery heuristics check if entities were discovered in prior turns
- Response synthesis mentions "relating to your earlier question..."

---

## Conclusion

The chat router supervisor balances three competing goals:

1. **Completeness**: Gather enough evidence to satisfy the question
2. **Efficiency**: Avoid redundant or unnecessary skill executions
3. **Clarity**: Synthesize results into actionable insights

Success is measured by user satisfaction with the answer quality + investigation speed + system resource efficiency.
