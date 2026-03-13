# Security Policy

## Supported Versions

SecurityClaw is currently in active development. Security fixes are backported to the latest stable release only.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest | :x:               |

## Security Architecture

SecurityClaw is a locally-deployed agent designed to run inside your own network. It does not transmit data externally unless explicitly configured with optional third-party API keys (AbuseIPDB, AlienVault OTX, VirusTotal, Cisco Talos).

### Key security controls in the codebase

- **Path traversal prevention** — All file operations on conversation IDs, skill names, and config paths resolve against a safe root directory and reject requests that escape it (`HTTP 403`).
- **Input validation** — Conversation and skill name identifiers are validated to contain only safe characters before any file system operation.
- **Secrets isolation** — Credentials (DB password, API keys) are stored exclusively in `.env` (git-ignored). `config.yaml` contains no secrets.
- **Prompt construction** — Prompts are assembled from fixed templates, but they still interpolate raw user questions and raw skill outputs. Treat model responses as advisory output, not as a trust boundary.
- **SQLite checkpointing** — Conversation/chat state is persisted locally in `data/conversations.db`, and scheduler/CLI runtime memory is persisted locally in `data/runtime_memory.db`, both using LangGraph checkpointing. These files are not exposed over the network.
- **CORS restriction** — The web API allows only `localhost:3000` and `localhost:5173` as origins.

### Threat intelligence API keys

If you configure external threat intelligence integrations (`ABUSEIPDB_API_KEY`, `ALIENVAULT_API_KEY`, `VIRUSTOTAL_API_KEY`, `TALOS_CLIENT_ID`/`TALOS_CLIENT_SECRET`), IP addresses and domains discovered in your logs will be sent to those external services. Review each vendor's privacy policy before enabling.

## Reporting a Vulnerability

If you discover a security vulnerability in SecurityClaw, please **do not open a public GitHub issue**.

1. Open a [GitHub Security Advisory](https://github.com/SecurityClaw/SecurityClaw/security/advisories/new) (private disclosure).
2. Include a description of the issue, steps to reproduce, and any proof-of-concept code.
3. You can expect an initial acknowledgement within 5 business days and a status update within 14 days.

Vulnerabilities confirmed as valid will be patched in a timely manner. Credit will be given in the release notes unless you prefer to remain anonymous.
