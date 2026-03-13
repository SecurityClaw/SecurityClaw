"""tests/test_memory.py - Unit tests for the structured agent memory store."""
from __future__ import annotations

from core.memory import CheckpointBackedMemory, StateBackedMemory


class TestMemoryInitialization:
    def test_initial_status_idle(self, tmp_memory):
        snap = tmp_memory.snapshot()
        assert snap["status"] == "IDLE"

    def test_initial_sections_exist(self, tmp_memory):
        content = tmp_memory.read()
        for section in ["Current Focus", "Open Findings", "Recent Decisions", "Escalation Queue"]:
            assert section in content


class TestSectionOperations:
    def test_get_section_returns_body(self, tmp_memory):
        body = tmp_memory.get_section("Current Focus")
        assert isinstance(body, str)

    def test_set_section_persists(self, tmp_memory):
        tmp_memory.set_section("Current Focus", "Investigating port scan on 10.0.1.5")
        assert "Investigating port scan" in tmp_memory.get_section("Current Focus")

    def test_set_section_overwrites(self, tmp_memory):
        tmp_memory.set_section("Current Focus", "First value")
        tmp_memory.set_section("Current Focus", "Second value")
        body = tmp_memory.get_section("Current Focus")
        assert "Second value" in body
        assert "First value" not in body

    def test_append_to_section_creates_bullets(self, tmp_memory):
        tmp_memory.append_to_section("Open Findings", "New finding A")
        tmp_memory.append_to_section("Open Findings", "New finding B")
        body = tmp_memory.get_section("Open Findings")
        assert "New finding A" in body
        assert "New finding B" in body

    def test_append_adds_timestamp(self, tmp_memory):
        tmp_memory.append_to_section("Open Findings", "Test item")
        body = tmp_memory.get_section("Open Findings")
        # timestamp is in format [YYYY-MM-DD HH:MM:SS UTC]
        assert "[20" in body  # year starts with 20xx


class TestStatusTransitions:
    def test_set_status_updates_header(self, tmp_memory):
        tmp_memory.set_status("INVESTIGATING")
        content = tmp_memory.read()
        assert "**Agent Status:** INVESTIGATING" in content

    def test_set_focus_changes_status(self, tmp_memory):
        tmp_memory.set_focus("Anomaly on 10.0.1.50")
        snap = tmp_memory.snapshot()
        assert snap["status"] == "INVESTIGATING"
        assert "Anomaly on 10.0.1.50" in snap["focus"]

    def test_clear_focus_returns_to_idle(self, tmp_memory):
        tmp_memory.set_focus("Something")
        tmp_memory.clear_focus()
        snap = tmp_memory.snapshot()
        assert snap["status"] == "IDLE"
        assert "None" in snap["focus"]

    def test_escalate_sets_escalating_status(self, tmp_memory):
        tmp_memory.escalate("Critical threat on web server")
        snap = tmp_memory.snapshot()
        assert snap["status"] == "ESCALATING"
        assert "Critical threat" in snap["escalation"]

    def test_set_status_updates_timestamp(self, tmp_memory):
        tmp_memory.set_status("ACTIVE")
        content = tmp_memory.read()
        assert "**Last Updated:**" in content
        assert "ACTIVE" in content


class TestConvenienceMethods:
    def test_add_finding(self, tmp_memory):
        tmp_memory.add_finding("Port scan detected from 192.168.1.1")
        body = tmp_memory.get_section("Open Findings")
        assert "Port scan detected" in body

    def test_add_decision(self, tmp_memory):
        tmp_memory.add_decision("Marked finding #42 as FALSE_POSITIVE")
        body = tmp_memory.get_section("Recent Decisions")
        assert "FALSE_POSITIVE" in body

    def test_multiple_findings_accumulate(self, tmp_memory):
        for i in range(5):
            tmp_memory.add_finding(f"Finding #{i}")
        body = tmp_memory.get_section("Open Findings")
        for i in range(5):
            assert f"Finding #{i}" in body


class TestSnapshot:
    def test_snapshot_keys(self, tmp_memory):
        snap = tmp_memory.snapshot()
        assert "status" in snap
        assert "focus" in snap
        assert "findings" in snap
        assert "decisions" in snap
        assert "escalation" in snap

    def test_snapshot_reflects_writes(self, tmp_memory):
        tmp_memory.add_finding("test finding")
        tmp_memory.add_decision("test decision")
        snap = tmp_memory.snapshot()
        assert "test finding" in snap["findings"]
        assert "test decision" in snap["decisions"]


class TestReadWrite:
    def test_write_full_parses_markdown_into_structured_memory(self, tmp_memory):
        new_content = (
            "# CLEAN SLATE\n\n"
            "**Agent Status:** ACTIVE\n\n"
            "## Current Focus\nNew focus\n\n"
            "## Recent Decisions\n- [2026-01-01 00:00:00 UTC] test decision\n"
        )
        tmp_memory.write_full(new_content)
        snapshot = tmp_memory.snapshot()
        assert snapshot["status"] == "ACTIVE"
        assert snapshot["focus"] == "New focus"
        assert "test decision" in snapshot["decisions"]

    def test_read_returns_string(self, tmp_memory):
        content = tmp_memory.read()
        assert isinstance(content, str)
        assert len(content) > 0


class TestMemoryBounds:
    def test_findings_are_capped_to_limit(self, tmp_memory):
        for i in range(60):
            tmp_memory.add_finding(f"Finding #{i}")

        body = tmp_memory.get_section("Open Findings")
        assert "Finding #59" in body
        assert "Finding #0" not in body

    def test_compact_context_stays_small(self, tmp_memory):
        for i in range(40):
            tmp_memory.add_decision(f"Decision #{i} with extra context payload")

        context = tmp_memory.compact_context(max_chars=1500)
        assert len(context) <= 1500


class TestStateBackedMemory:
    def test_from_state_keeps_memory_in_process(self):
        mem = StateBackedMemory.from_state(
            {
                "mem_status": "INVESTIGATING",
                "mem_focus": "Track Iran traffic",
                "mem_findings": [{"timestamp": "2026-01-01T00:00:00+00:00", "text": "finding"}],
                "mem_decisions": [],
                "mem_escalations": [],
            }
        )

        snapshot = mem.snapshot()

        assert snapshot["status"] == "INVESTIGATING"
        assert "Track Iran traffic" in snapshot["focus"]
        assert "finding" in snapshot["findings"]

    def test_to_dict_reflects_in_memory_mutations(self):
        mem = StateBackedMemory()
        mem.set_focus("Investigate port 443")
        mem.add_decision("Used graph state memory")

        state = mem.to_dict()

        assert state["mem_status"] == "INVESTIGATING"
        assert state["mem_focus"] == "Investigate port 443"
        assert state["mem_decisions"][-1]["text"] == "Used graph state memory"


class TestCheckpointBackedMemory:
    def test_checkpoint_memory_persists_across_instances(self, tmp_path):
        db_path = tmp_path / "runtime_memory.db"

        first = CheckpointBackedMemory(path=db_path)
        first.set_focus("Investigate outbound DNS")
        first.add_decision("Started runtime checkpoint memory")
        first.close()

        second = CheckpointBackedMemory(path=db_path)
        snapshot = second.snapshot()
        second.close()

        assert snapshot["status"] == "INVESTIGATING"
        assert "Investigate outbound DNS" in snapshot["focus"]
        assert "Started runtime checkpoint memory" in snapshot["decisions"]
