# OpenSearch Querier Skill

## Purpose

**Simplified data access layer** for OpenSearch/Elasticsearch queries.

- Receives a query plan from the supervisor/LLM
- Executes exact search as planned
- Returns results (no diagnosis, no recovery, no heuristics)
- Lets supervisor decide next steps

## Architecture: No Heuristics

**Removed:**
- Diagnostic reasoning ("data doesn't exist in database" fallbacks)
- Strategy recovery (trying phrase→token when results are 0)
- Field name guessing (pattern-based overrides)
- IP/port detection heuristics

**Kept:**
- Simple query execution
- Field selection based on search type
- Time range filtering
- Result tabulation

## How It Works

1. **Receive Plan**: Supervisor or LLM provides search_type, search_terms, fields, strategies
2. **Validate**: Check that question exists, DB is available, fields are known
3. **Build Query**: Construct simple bool/term query based on plan
4. **Execute**: Run against OpenSearch (synchronous)
5. **Return Results**: Return what we got, even if 0 results
6. **Stop**: Do NOT attempt diagnosis or recovery

## Query Planning (LLM)

The LLM decides:
- **search_type**: "ip" | "traffic" | "alert" | "general"
- **search_terms**: List of values to search for
- **matching_strategy**: "term" | "phrase" | "wildcard"
- **time_range**: "now-90d" | "now-7d" | custom date range
- **fields**: Which schema fields to query

This is guided by PLANNING_PROMPT.md and your field_mappings discovery.

## When Results Are 0

If the query returns 0 results:
- opensearch_querier returns 0 results
- Supervisor sees "no data" evaluation
- Supervisor may decide to:
  - Retry with different supervisor/LLM planning
  - Route to a different skill
  - Tell the user "no matching records found"

**This skill does NOT guess or retry.** It returns exactly what the query found.

## For Direct User Queries

Users can query OpenSearch directly via chat:
```
User: Find all logs from IP 185.200.116.46 on port 1194
Agent: (routing to opensearch_querier)
Result: X matching documents...
```

## For Other Skills

Other skills import query_builder utilities:
```python
from core.query_builder import discover_field_mappings, build_keyword_query

# In your skill:
field_mappings = discover_field_mappings(db, llm)
query, metadata = build_keyword_query(keywords, field_mappings)
results = db.search(index, query, size=100)
```

This ensures NO hardcoded field names anywhere in the codebase.

## Query Planning Strategy

See `PLANNING_PROMPT.md` for the detailed LLM prompt that guides query planning:
- How to extract countries, ports, protocols, time_range from natural language
- Examples of question → structured fields conversion
- Error handling for ambiguous or partial information

**Architecture Decision:** The planning prompt is kept in markdown (not embedded in Python code) to:
- Enable prompt engineering without code redeploy
- Make prompt changes auditable
- Allow iterative refinement of query extraction logic

Python code (`_plan_opensearch_query_with_llm`) loads `PLANNING_PROMPT.md` at runtime and combines it with dynamic conversation context and field mappings.

### Justification for Separating Static vs Dynamic Content

| Content Type | Location | Reason |
|---|---|---|
| Static JSON examples, error handling, extraction rules | `PLANNING_PROMPT.md` | Reusable, maintainable, auditable |
| Dynamic context assembly, conversation history, runtime field mapping | `logic.py` | Changes based on actual conversation and available fields |
| Query execution, result handling | `logic.py` | Implementation detail, not guidance |
| Response formatting and data extraction | `hooks.py` | Data transformation logic |

This pattern allows the LLM prompt to evolve without code changes while keeping implementation details encapsulated.
