# SecurityClaw Agent Architecture

SecurityClaw uses a defensive ReAct state graph:

```text
START → think → execute_tools → observe ─┬→ think
                                        ├→ await_authorization → generate_response
                                        └→ generate_response → END
```

The model produces a structured decision on every reasoning round with an
`action` (`use_tools`, `answer`, or `ask_user`), a thought, reasoning, selected
runtime skills, and parameters. Tool results become observations and return to
the reasoning node until the objective is satisfied, the step budget is
reached, the operator must clarify something, or a privileged action requires
authorization.

The Agent console renders reasoning, tool start, tool output, evaluation,
operator guidance, approval requests, and the final answer as separate events.
Insights are derived locally from the same trace and require no additional
service or container.

The evidence graph is the central investigation surface. The Agent timeline and
Insights share a collapsible operator drawer, conversation history can be
opened independently, and each completed tool observation is merged into the
graph while the investigation is still running.

Each completed endpoint scan also generates an interactive evidence graph from
its persisted tool results. Host details, tools, processes, network peers,
persistence entries, services, packages, defensive controls, integrity records,
and grounded vulnerability advisories become related nodes in the Agent console.
The graph supports stable force-directed 2D and 3D layouts, pan, zoom, drag and
pin, fit-to-view, layout reset, fullscreen expansion, search, type filters, node
details, JSON export, and separate analyst annotations. Renderer state is not
remounted for every incoming node, and automatic fit runs only on initial
evidence or an explicit view change so an analyst's current view is preserved.
Collected evidence remains immutable when a user edits a display name, severity,
or analyst note. This intentionally avoids a graph database until
cross-investigation correlation or dataset size provides a concrete reason to
operate one.

Vulnerability nodes are grounded in OSV advisory correlation against observed
installed package versions. Severity uses a published numeric score, a
calculated CVSS 3.x vector, or a published advisory label in that order; missing
ratings remain `unknown` rather than being inferred from advisory text.

The supplied Compose stack remains supported. Its application and web
containers are the minimal development/runtime services; external telemetry
stores and local model runtimes should be added only when the selected
deployment requires them.

## Passive monitoring and investigations

Scheduled skills may declare an `alert_contract` in their runtime manifest.
The scheduler uses this structured contract to persist findings without
hardcoding skill names. SOC anomaly triage and cross-platform endpoint threat
hunting use the same notification pipeline. Operators can enable or disable
every skill, run scheduled read-only skills on demand, inspect their last
result, resolve an alert, or open a new Agent investigation with the original
evidence already attached. Containment remains approval-gated.

Alerts retain a stable evidence fingerprint across scheduler cycles. Repeated
observations update the stored alert's last-seen metadata without creating a new
notification, including after an operator marks it read, investigating, or
resolved. A materially changed finding or severity receives a new fingerprint
and can notify the operator again.

The endpoint hunter compares processes, connections, persistence, file hashes,
and defensive posture with the previous observation every five minutes. The
posture sensor runs every fifteen minutes. These defaults are editable in each
skill instruction and are inactive whenever the corresponding skill is
disabled.

The network defense monitor samples ARP/NDP neighbors, routes, interfaces, and
default gateways every two minutes. It alerts on changed IP-to-MAC bindings,
gives gateway changes higher priority, and highlights unusual MAC address
concentration while explicitly accounting for DHCP, proxy ARP, clustering,
virtualization, and legitimate failover. Operators can investigate the alert in
the Agent and, after explicit one-time authorization, block a remote address or
remove a specific neighbor-cache entry. No network containment runs unattended.

## Launchers

`securityclaw.sh` supports Linux, macOS, WSL, and POSIX-compatible shells.
`securityclaw.ps1` provides equivalent native Windows PowerShell commands. Both
support local and Compose application modes, create only the required
OpenSearch dependency, verify Ollama availability, and explicitly configure
containers with no restart-at-boot policy.

## Safety model

- Read-only endpoint skills can run autonomously while SecurityClaw is active.
- Privileged endpoint actions require a short-lived, single-use token bound to
  the exact action and arguments.
- The token must be echoed in the current operator message. The LLM cannot
  authorize an action by setting a parameter itself.
- Unsupported platform skills are excluded before scheduler registration and
  before the runtime skill catalog is given to the model.

## Attribution

The explicit ReAct node layout, structured action vocabulary, ordered timeline
events, operator-guidance flow, approval checkpoints, Agent console design, and
interactive force-directed 2D/3D evidence graph patterns are adapted from the
RedAmon agentic system maintained alongside this project.
SecurityClaw's implementation is purpose-built for defensive SOC and endpoint
operations and does not copy RedAmon's offensive tools, fireteam execution, or
container topology.
