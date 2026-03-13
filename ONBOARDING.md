# SecurityClaw Onboarding Guide

## Docker Quick Start

If you have Ollama and OpenSearch running locally, use the Docker onboarding script:

```bash
./onboard-docker.sh
```

This script will:
1. Check for an existing `config.yaml` (or offer to run the Python onboarding)
2. Extract your Ollama settings from `config.yaml`
3. Automatically convert `localhost` → `host.docker.internal` for Docker networking
4. Test that Ollama is reachable
5. Launch both the Python backend and Vite web frontend in Docker

The web UI will be available at `http://localhost:5173` and the API at `http://localhost:7799`.

## Classic Setup

## Quick Start

The interactive configuration wizard guides you through setting up SecurityClaw in minutes.

### Step 1: Run the Onboarding Wizard

```bash
.venv/bin/python main.py onboard
```

### Step 0 (Optional): Quick Ollama Setup

The current example config uses these local Ollama models:

- `qwen2.5:7b-instruct-q4_K_M`
- `nomic-embed-text:latest`

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text:latest
```

This launches an interactive CLI that will ask you about:

**Phase 1: Database Configuration**
- Which DB backend? (OpenSearch or Elasticsearch)
- Database host and port
- SSL/TLS settings
- Authentication credentials (optional)
- **Network logs index** — where to scan for historical logs (e.g., `securityclaw-logs`, `logs-*`, `filebeat-*`)
- **Anomaly detection findings index** — where AD detector results are stored (e.g., `securityclaw-anomalies` or OpenSearch's built-in `.opendistro-anomaly-results*`)
- **RAG vector index** — where to store embeddings for behavioral context (e.g., `securityclaw-vectors`)
- Tests the connection

**Phase 2: LLM Provider Configuration**
- Ollama configuration: Base URL and model name
- Tests the connection

**Phase 3: Configuration Save**
- Writes to `config.yaml` (DB, LLM, and index settings)
- Writes to `.env` (credentials)
- Resets the configuration singleton so changes take effect immediately

**Phase 4: External Threat Intelligence (Optional)**
- Setup external reputation APIs (AbuseIPDB, AlienVault OTX, VirusTotal, Talos)
- Saves API keys to `.env`

**Phase 5: Skill-Specific Variables (Optional)**
- Scans all skills in `/skills` for required environment variables
- Prompts for any missing variables declared in skill `manifest.yaml`
- Example: `threat_analyst` needs optional API keys; `geoip_lookup` needs MaxMind license key
- Can skip now and configure later by re-running `python main.py onboard`

### Step 2: Verify Configuration

After onboarding, view what was saved:

```bash
cat config.yaml     # DB and LLM provider settings
cat .env            # Secrets (core and skill-specific variables)
```

The `.env` file contains:
- Database credentials (optional)
- LLM credentials (Ollama base URL)
- External threat intelligence API keys (AbuseIPDB, AlienVault, VirusTotal, Talos)
- Skill-specific variables (MaxMind license, custom API tokens, etc.)

### Step 3: List Available Skills

```bash
.venv/bin/python main.py list-skills
```

Output example:
```
  chat_router — manual
  network_baseliner — every 21600s
  fields_baseliner — every 3600s
  anomaly_triage — manual
  threat_analyst — manual
  geoip_lookup — cron: 0 2 * * tue,fri
  opensearch_querier — manual
  forensic_examiner — manual
```

Note: `anomaly_triage` and `threat_analyst` can be converted to scheduled by adding `schedule_interval_seconds` to their `instruction.md` files.

Each skill declares:
- Its **schedule** (interval or manual)
- Its **required variables** in `manifest.yaml` (checked on first chat)

### Step 4: Start the Chat

```bash
.venv/bin/python main.py chat
```

**Conversation persistence**: Conversations and agent memory are stored in `data/conversations.db` (SQLite, created automatically on first run). Each chat session writes to this file via the LangGraph `SqliteSaver` checkpointer — no manual maintenance is needed.

**First chat startup**: SecurityClaw will check for any missing skill-specific variables and prompt you to configure them. This includes:
- MaxMind license key (for `geoip_lookup`)
- Threat intelligence API keys (for `threat_analyst`)
- Any custom variables declared by new or custom skills

You can:
1. **Configure now** — Enter values interactively
2. **Skip** — Configure later by re-running `python main.py onboard`

Once in chat, ask questions like:
- "What's the threat level for IP 192.168.1.100?"
- "Create a baseline for normal traffic patterns"
- "Compare recent activity to baseline"

The agent will automatically route your question to the appropriate skill(s).

### Step 5: (Optional) Run the Background Agent

For automated anomaly detection and analysis:

```bash
.venv/bin/python main.py run
```

The agent will:
- Discover all skills in `/skills`
- Schedule each skill according to its interval
- Poll OpenSearch for logs and anomaly findings
- Build RAG context from normal behavior
- Issue threat verdicts using the LLM

### Feature Maturity Notes

- **anomaly_triage** — in progress; currently manual-only but can be converted to scheduled
- **threat_analyst** — in progress; currently manual-only but can be converted to scheduled
- **forensic_examiner** — in progress for broader real-environment validation
- **baseline_querier** — in progress and not yet publication-hardened
- **fields_baseliner** — production; catalogs OpenSearch field schemas hourly

---

## Manual Configuration (Advanced)

If you prefer manual setup, edit these files directly:

**`config.yaml`** — Centralized configuration
```yaml
db:
  provider: opensearch              # or: elasticsearch
  host: localhost
  port: 9200
  username: ""                      # load from .env if used
  password: ""                      # load from .env if used
  use_ssl: false
  verify_certs: false

llm:
  provider: ollama
  ollama_base_url: http://localhost:11434
  ollama_model: qwen2.5:7b-instruct-q4_K_M
  ollama_embed_model: nomic-embed-text:latest
```

**`.env`** — Secret credentials (git-ignored)
```
OPENSEARCH_USERNAME=<your-opensearch-username>
OPENSEARCH_PASSWORD=<your-opensearch-password>
```

---

## CLI Commands Reference

| Command | Purpose |
|---------|---------|
| `.venv/bin/python main.py onboard` | Interactive configuration wizard |
| `.venv/bin/python main.py run` | Start the full agent (foreground) |
| `.venv/bin/python main.py list-skills` | List discovered skills and intervals |
| `.venv/bin/python main.py dispatch <skill>` | Fire a skill once (e.g., `anomaly_triage`) |
| `.venv/bin/python main.py status` | Print the compact agent memory snapshot |
| `.venv/bin/python main.py --log-level DEBUG run` | Start with debug logging |

---

## Skill-Specific Variables

SecurityClaw automatically discovers and prompts for variables that individual skills require. This enables:

1. **Dynamic Discovery** — Each skill can declare required variables in its `manifest.yaml`
2. **First-Chat Onboarding** — Missing variables are detected when you run `python main.py chat`
3. **Selective Configuration** — Only configure variables for skills you plan to use
4. **Easy Updates** — Re-run `python main.py onboard` to add more skill variables

### Example Skill Requirements

**threat_analyst** (optional API keys for enrichment):
```
ABUSEIPDB_API_KEY         — IP abuse reputation scoring
ALIENVAULT_API_KEY        — Threat intelligence pulses
VIRUSTOTAL_API_KEY        — Malware detection
TALOS_CLIENT_ID           — Cisco enterprise intelligence
TALOS_CLIENT_SECRET       — Cisco enterprise intelligence
```

**geoip_lookup** (required):
```
MAXMIND_LICENSE_KEY       — Download GeoIP database (required)
```

### Custom Skills

If you create a **new skill**, declare its required variables in the skill's `manifest.yaml`:

```yaml
name: my_custom_skill
description: "Does something special"
# ... other metadata ...

required_env_vars:
  - name: MY_API_KEY
    description: "API key for my external service"
    env_key: MY_API_KEY
    optional: false       # Set to true for optional variables
    is_secret: true       # Hides input in CLI prompts
```

On next `python main.py onboard` or `python main.py chat`, SecurityClaw will detect and prompt for `MY_API_KEY`.

---

## Troubleshooting

**"Cannot connect to OpenSearch"**
- Verify OpenSearch is running on the configured host:port
- Check firewall rules
- Ensure credentials are correct

**"Cannot connect to Ollama"**
- Ensure Ollama is running (`ollama serve`)
- Ensure the example models are present (`ollama list`)
- Check the base URL (default: `http://localhost:11434`)

**"No skills found"**
- Verify `/skills` directory exists with `skill_name/logic.py` files
- Each skill must have a `run(context) -> dict` function

**Re-running onboarding**
```bash
.venv/bin/python main.py onboard
```
Simply repeat the wizard to update any settings (existing values are shown as defaults).

**Conversation history not appearing**
- Conversation state is stored in `data/conversations.db`. If the file is missing or corrupted, delete it and restart; a fresh database will be created automatically.
- Use `/history` inside chat to browse past conversations.

---

## Architecture

- **Modular Skills**: Each skill in `/skills/<name>/` has `logic.py` (Python) + `instruction.md` (LLM system prompt)
- **Scheduler**: APScheduler fires skills on intervals (1-minute watcher, 6-hour baseliner)
- **RAG**: Embeddings stored in vector index; retrieved for contextual LLM analysis
- **Provider Agnostic**: Swap DB backends and LLMs via config without code changes

For full architecture details, see the main README or source code.
