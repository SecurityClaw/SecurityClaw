---
skill: ip_fingerprinter
description: >
  Passive IP fingerprinting from aggregated port observations.
  Analyzes pre-aggregated port data and infers host role, services, and OS patterns.
  Data-agnostic: works with any schema via opensearch_querier's port aggregation.
---

# IP Fingerprinter - Passive Port Analysis

## Architecture: Supervisor Orchestrates Prerequisites

The **supervisor** handles orchestration based on manifest declarations:
1. Supervisor sees user asks to "fingerprint 192.168.0.1"
2. Supervisor checks manifest → ip_fingerprinter requires "evidence_search"
3. Supervisor routes to opensearch_querier to gather port evidence for the IP
4. opensearch_querier returns aggregated ports for that IP
5. Supervisor passes those results to ip_fingerprinter via previous_results
6. ip_fingerprinter analyzes the aggregated ports
7. ip_fingerprinter returns role, services, OS likelihood, etc.

## Role
You are a **pure analysis skill** for network fingerprinting.

Your ONLY job is to enrich and interpret pre-aggregated port statistics:

1. Skip ephemeral ports (>= 32768 on Linux)
2. Classify each port using service registry
3. Infer host role (server vs client) from listening ports
4. Score OS families based on port patterns
5. Return conservative confidence scores

## Data-Agnostic Constraints
- Do NOT parse field names or access row data
- Do NOT assume any schema structure
- Do NOT perform queries or aggregations
- Do NOT discover fields or field mappings
- Accept ONLY pre-aggregated port counts via `parameters.aggregated_ports`

## Inputs
The skill receives pre-aggregated port data from the LLM-orchestrated flow:

```json
{
  "ip": "10.0.0.15",
  "aggregated_ports": {
    "443": {"observations": 127, "protocols": ["TCP"], "peers": {...}},
    "80": {"observations": 45, "protocols": ["TCP"], "peers": {...}},
    "22": {"observations": 18, "protocols": ["TCP"], "peers": {...}}
  },
  "force_update": false
}
```

**Note**: The skill does NOT:
- Extract IPs from questions
- Discover fields from schema
- Query or aggregate data
- Handle field mappings

The LLM handles all those decisions and passes only the aggregated results.

## Output Contract
Return structured JSON only:

```json
{
  "status": "ok",
  "ip": "10.0.0.15",
  "ports": [
    {
      "port": 443,
      "protocols": ["TCP"],
      "service_name": "https",
      "description": "HTTP protocol over TLS/SSL",
      "registered": true,
      "range_class": "system",
      "ephemeral_likelihood": "unlikely",
      "observations": 127,
      "peer_count": 23,
      "peers": ["192.168.1.5", "192.168.1.10", ...]
    }
  ],
  "port_summary": {
    "listening_ports": [443, 80, 22],
    "registered_ports": [443, 80, 22],
    "unregistered_ports": []
  },
  "likely_role": {
    "classification": "likely_server",
    "confidence": 89,
    "listening_score": 5.5,
    "reasons": ["Listening on service port 443 (https)", "Listening on service port 22 (ssh)"]
  },
  "os_family_likelihoods": [
    {
      "family": "Linux",
      "score": 0.78,
      "confidence": "high",
      "reasons": ["Listening on Linux-associated port 22"]
    }
  ],
  "registry_status": {
    "action": "loaded",
    "source": "cache",
    "cache_path": "/path/to/port_registry.json"
  }
}
```

## Interpretation Rules
- **Ephemeral filtering**: Linux ports >= 32768 are skipped (temporary client ports)
- **Destination-only analysis**: Only listening ports (destination IP matched) are analyzed
- **Service ports = servers**: Stable registered ports suggest server behavior
- **OS likelihood**: Conservative scoring based only on port numbers (not schema-dependent)
- **Confidence**: Low if fewer than 5 distinct service ports observed

## The Data-Agnostic Contract
This skill maintains complete separation of concerns:

| Responsibility | Handled By | This Skill? |
|---|---|---|
| Discover fields in logs | fields_querier | ❌ No |
| Select which fields to use | LLM orchestration | ❌ No |
| Build and execute queries | opensearch_querier | ❌ No |
| Aggregate port counts | opensearch_querier | ❌ No |
| Analyze aggregated data | **ip_fingerprinter** | ✅ Yes |
| Synthesize for user | LLM / chat_router | ❌ No |

By maintaining this strict separation, the skill works with **any log schema**
as long as fields_querier can identify destination IP and port fields.
