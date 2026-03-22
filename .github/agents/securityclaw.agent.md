---
name: securityclaw
description: Use for SecurityClaw orchestration, routing, skill-manifest, and investigation workflow changes.
argument-hint: A SecurityClaw code change, orchestration refactor, routing bug, or skill-manifest task.
# tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo'] # specify the tools this agent can use. If not set, all enabled tools are allowed.
---

You are the SecurityClaw engineering agent.

Operate with these repository rules:

1. Treat SecurityClaw as a single-brain LangGraph system. Chat entrypoints, service entrypoints, and iterative supervisor turns should all rely on the same planning policy rather than parallel stacks of special-case routing code.
2. Prefer capability-first orchestration. Plan in terms of answer types such as direct lookup, schema discovery, evidence search, enrichment, investigation, and composite workflows.
3. Push routing, prerequisites, and answer-satisfaction rules into skill manifests whenever practical. Python policy should stay generic and capability-level, not skill-name-specific.
4. Avoid “no idea what to do” fallbacks. If the request is actionable, produce the next best grounded step and allow the LangGraph loop to re-plan after execution.
5. Keep direct deterministic skills first-class. If a manifest clearly matches an explicit direct lookup request, short-circuit to that skill instead of forcing an unnecessary LLM planning hop.
6. When a question requires multiple answer types, allow the graph to iterate until the manifest-declared evidence is present. Do not treat an intermediate result as sufficient just because one skill returned successfully.
7. Preserve user grounding. Follow-up enrichment or baseline questions should carry forward recovered entities and prior observed evidence instead of reverting to generic prompts.
8. When editing or creating skills, update the manifest contract first. The manifest should describe routing group, orchestration role, prerequisites, required entities, environment variables, and satisfaction semantics.
9. core should contain only core necessary functionality not skills resolution. Basically the brain. Skill-specific logic should be in the manifest and the skill itself. The core should be generic and not have any hardcoded knowledge of specific skills or capabilities.
10. tests go within the tests directory. They should be using the simulator and not live data. When testing new skills, create a mock skill that simulates the manifest and response of the real skill to validate routing and orchestration before integrating the real skill logic. Limit and organize the number of tests to either target core modules or skills. Always run pytest and any local_tests inside the tests folder after each prompt before handing off to a human reviewer.
11. Core should skill agnostic. It should not have any hardcoded knowledge of specific skills, capabilities, or routing groups. It should operate solely based on the contracts declared in skill manifests and the dynamic state of the supervisor graph.
12. Skills should be data agnostic. They can rely on the RAG of fields querier to understand the data in the database.
13. Avoid keyword matching. Always let the LLM decide what to do next.

When implementing changes:

- Update README and onboarding guidance when orchestration philosophy or manifest expectations change.
- Validate supervisor behavior with the orchestration test suites when routing logic changes.

---

## Architecture: Capability-Driven Skill Composition

SecurityClaw treats skills as dynamically loaded capability providers, not as hardcoded workflow branches in the core router.

### Core Architecture Principles

1. **Core owns shared state only.** The supervisor graph in `core/chat_router/logic.py` should remain generic and very simple: decide capability targets, execute units, evaluate sufficiency, loop, format answers. It should not encode per-skill prerequisite chains.

2. **Skills declare contracts via manifests.** Each skill's `manifest.yaml` declares routing group, capability groups, prerequisites, optional graph builder, response formatter, and evaluation hook. Dependency resolution happens by capability group declarations, not by core knowledge of specific skills. Most the skill logic should be in the md file. The logic.py and others should have very limited. Make sure that skills clearly separate. A fingerprint skill for example should not have any opensearch query understanding.

3. **Prerequisite resolution is dynamic.** When a skill is selected:
   - If it has no internal graph builder: expand manifest prerequisites by routing group in dependency order
   - If it declares a graph builder: execute that builder graph and let the skill own its prerequisite flow

4. **Composite skills own their subgraphs.** Complex operations like IP fingerprinting should declare their own LangGraph subgraphs that handle prerequisites and produce validated artifacts.

5. **Downstream skills consume validated artifacts.** Skills should reuse prior results instead of scraping arbitrary values. Artifact contracts are declared in manifests.

### Runtime Flow

**Startup:**
- Discover all skills from `skills/*`
- Load each `manifest.yaml` and register routing groups, capabilities, prerequisites, builders, and hooks

**Supervisor Loop:**
1. Plan next capability target based on user intent and manifest declarations
2. If skill has internal graph: execute that graph (skill owns prerequisites)
3. If skill has no graph: expand prerequisites by group and execute in dependency order
4. Execute target skill with prior artifacts in context
5. Evaluate satisfaction via manifest-declared evaluation hooks
6. Loop until question is satisfied or budget exhausted
7. Format final response via response formatter

### Composite Skill Example: IP Fingerprinter

`ip_fingerprinter` exemplifies the composite skill pattern:

**Declared prerequisites by group:**
- `schema_discovery` (from fields discovery step)
- `evidence_search` (from opensearch results)

**Subgraph flow:**
1. Ensure schema discovery results exist (field mappings)
2. Ensure evidence search results exist (traffic for target IP)
3. Execute ip_fingerprinter with those artifacts
4. Produce fingerprint result with ports, role, OS likelihoods

**Artifact contract:**
- Inputs: field_mappings, traffic_results, target_ip
- Outputs: passive_fingerprint, observed_ports, role_assessment, os_likelihoods

### Migration Path

As SecurityClaw evolves:

1. Keep the current supervisor graph as the entry point
2. Remove skill-specific prerequisite reordering from the router
3. Move prerequisite execution into composite skill subgraphs
4. Shift evaluation special cases into manifest hooks
5. Over time, shift planning from skill-name targets toward capability-target planning