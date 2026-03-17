# OpenSearch Querier Skill

## Purpose

Centralized OpenSearch/Elasticsearch query execution. This is the **single point of contact** for all database searches.

Instead of each skill building its own queries (causing duplication and hardcoded field names), all skills now use opensearch_querier to execute searches consistently.

## How It Works

1. **Field Discovery**: Queries RAG for field_documentation (created by network_baseliner)
2. **Intelligent Query Building**: Uses discovered field names instead of hardcoding "source_ip", "message", etc.
3. **Execution**: Runs the constructed query against OpenSearch
4. **Results**: Returns both data and metadata about which fields were used
5. **Response Formatting**: Post-processes raw results to extract and highlight relevant details (IPs, ports, countries, etc.)

## Response Post-Processing

When opensearch_querier returns results, the formatter automatically extracts and highlights:
- **Ports**: All unique destination ports found in the matching records
- **IPs**: All source/destination IPs with distinguishing public vs. private
- **Countries**: All observed geoIP countries
- **Timestamps**: Earliest and latest records for time context

### Port Extraction Behavior (Important for Follow-ups)

When responding to **"What ports are associated with..."** follow-up questions:

1. opensearch_querier searches for traffic matching the prior IPs (with `ports: []` meaning "all ports")
2. The formatter examines the raw results for actual destination ports
3. **All unique ports** found in those results are extracted and displayed
4. This enables users to see concrete port data without needing an additional fingerprinting step

Example flow:
```
User: Find 1.1.1.1
→ Agent: Found 10 records with source/destination IPs...

User: What ports are associated with this traffic?
→ opensearch_querier runs with search_terms=[1.1.1.1], ports=[]
→ Formatter extracts actual ports from 200+ matched records
→ Returns: "Destination port(s): 443, 80, 22, 1194"
```

This design ensures opensearch_querier can handle port-related follow-ups directly without requiring external skills.

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
