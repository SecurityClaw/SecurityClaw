# Local Tests Directory

This directory contains local test cases that validate the chat interface behavior against the 4 primary use cases:

1. `russia traffic in past 30 days?`
2. `russia traffic in past 24 hours?`
3. `reputation of 1.1.1.1?`
4. `fingerprint 192.168.0.17`

## Purpose

These tests are designed to run locally to verify:
- Correct routing of prompts to skills
- Expected skill flow execution order
- Response formatting and data inclusion

## Design

Tests in this directory use mock LLMs to ensure deterministic routing validation without requiring live model inference.

## Usage

Run tests locally:
```bash
python -m pytest tests/local_tests/test_four_prompts_live.py -v
```

Or run directly:
```bash
python tests/local_tests/test_four_prompts_live.py
```

## Not Committed

This directory is excluded from git (see `.gitignore`). Local test results and temporary files should remain local.
