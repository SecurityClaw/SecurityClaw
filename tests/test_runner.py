"""
tests/test_runner.py — Integration tests for the Runner conductor.

Validates:
  - Skill discovery and wiring
  - Context building (db, llm, memory injected)
  - Manual dispatch via runner
  - Graceful handling of empty skills directory
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from core.memory import CheckpointBackedMemory
from core.runner import Runner
from tests.mock_llm import MockLLMProvider
from tests.mock_opensearch import MockDBConnector


@pytest.fixture
def minimal_skills_dir(tmp_path) -> Path:
    skills = tmp_path / "skills"
    skills.mkdir()

    # Skill A — returns a fixed result
    sa = skills / "skill_alpha"
    sa.mkdir()
    (sa / "instruction.md").write_text("---\nschedule_interval_seconds: 999\n---\n# Alpha")
    (sa / "logic.py").write_text("def run(ctx): return {'skill': 'alpha', 'has_db': ctx['db'] is not None}\n")

    # Skill B — returns the memory status
    sb = skills / "skill_beta"
    sb.mkdir()
    (sb / "instruction.md").write_text("---\nschedule_interval_seconds: 999\n---\n# Beta")
    (sb / "logic.py").write_text(
        "def run(ctx): return {'status': ctx['memory'].snapshot()['status']}\n"
    )
    return skills


class TestRunnerSetup:
    def test_discovers_skills(self, minimal_skills_dir, tmp_path):
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        assert "skill_alpha" in runner._skills
        assert "skill_beta" in runner._skills

    def test_empty_skills_dir_no_crash(self, tmp_path):
        empty_skills = tmp_path / "empty_skills"
        empty_skills.mkdir()
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=empty_skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()  # Should not raise
        assert runner._skills == {}

    def test_first_startup_marker_created(self, tmp_path):
        """Verify startup marker file is created with skill name after first startup."""
        skills = tmp_path / "marker_test_skills"
        skills.mkdir()
        data_dir = tmp_path / "data"
        marker_path = data_dir / ".startup_complete"
        
        # Create a skill that runs on first startup
        startup_skill = skills / "test_startup_skill"
        startup_skill.mkdir()
        (startup_skill / "instruction.md").write_text("# Test Startup Skill")
        (startup_skill / "manifest.yaml").write_text(
            "---\n"
            "name: test_startup_skill\n"
            "run_on_first_startup: true\n"
            "schedule_interval_seconds: 999\n"
        )
        (startup_skill / "logic.py").write_text(
            "def run(ctx):\n"
            "    return {'status': 'initialized'}\n"
        )
        
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        # Override marker path to use our temp directory
        runner._startup_marker_path = marker_path
        
        # Marker should not exist before setup
        assert not marker_path.exists()
        
        # After setup, marker should be created
        runner.setup()
        assert marker_path.exists()
        
        # Marker should contain the skill name
        marker_data = json.loads(marker_path.read_text())
        assert "test_startup_skill" in marker_data

    def test_second_startup_skips_first_startup_skills(self, minimal_skills_dir, tmp_path):
        """Verify that first-startup skills are skipped on subsequent runs."""
        # Create a skill that runs on first startup
        skills = tmp_path / "startup_skills"
        skills.mkdir()
        
        # Counter file to track executions
        counter_path = tmp_path / "execution_counter.txt"
        
        # Skill that runs on first startup
        startup_skill = skills / "startup_skill"
        startup_skill.mkdir()
        (startup_skill / "instruction.md").write_text("# Startup Skill")
        (startup_skill / "manifest.yaml").write_text(
            "---\n"
            "name: startup_skill\n"
            "run_on_first_startup: true\n"
            "schedule_interval_seconds: 999\n"
        )
        (startup_skill / "logic.py").write_text(
            f"def run(ctx):\n"
            f"    # Increment counter file\n"
            f"    path = '{counter_path}'\n"
            f"    count = 0\n"
            f"    try:\n"
            f"        with open(path, 'r') as f:\n"
            f"            count = int(f.read().strip() or '0')\n"
            f"    except:\n"
            f"        pass\n"
            f"    count += 1\n"
            f"    with open(path, 'w') as f:\n"
            f"        f.write(str(count))\n"
            f"    return {{'status': 'initialized'}}\n"
        )
        
        marker_path = tmp_path / "data" / ".startup_complete"
        
        # First startup — skill should execute
        runner1 = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner1._startup_marker_path = marker_path
        runner1.setup()
        
        # Verify counter shows execution happened
        assert counter_path.exists()
        count1 = int(counter_path.read_text().strip())
        assert count1 == 1, f"First startup should execute skill once, got {count1}"
        
        # Verify marker was created with skill name
        assert marker_path.exists()
        marker_content = marker_path.read_text()
        marker_data = json.loads(marker_content)
        assert "startup_skill" in marker_data, f"Marker should contain startup_skill key, got {marker_data}"
        
        # Second startup — skill should NOT execute
        runner2 = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner2._startup_marker_path = marker_path
        runner2.setup()
        
        # Counter should still be 1 (skill didn't run)
        count2 = int(counter_path.read_text().strip())
        assert count2 == 1, f"Second startup should skip skill, but counter is {count2}"
    
    def test_first_startup_for_new_skill_triggers_initialization(self, tmp_path):
        """Verify that adding a new first-startup skill triggers its initialization."""
        skills = tmp_path / "startup_skills2"
        skills.mkdir()
        marker_path = tmp_path / "data" / ".startup_complete"
        
        # Skill A that runs on first startup
        skill_a = skills / "skill_a"
        skill_a.mkdir()
        (skill_a / "instruction.md").write_text("# Skill A")
        (skill_a / "manifest.yaml").write_text(
            "---\n"
            "name: skill_a\n"
            "run_on_first_startup: true\n"
            "schedule_interval_seconds: 999\n"
        )
        (skill_a / "logic.py").write_text(
            "def run(ctx):\n"
            "    return {'status': 'initialized'}\n"
        )
        
        # First startup with skill_a only
        runner1 = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner1._startup_marker_path = marker_path
        runner1.setup()
        
        # Marker should contain skill_a
        marker_data = json.loads(marker_path.read_text())
        assert "skill_a" in marker_data
        assert "skill_b" not in marker_data
        
        # Now add skill_b that also runs on first startup
        execution_log = []
        skill_b = skills / "skill_b"
        skill_b.mkdir()
        (skill_b / "instruction.md").write_text("# Skill B")
        (skill_b / "manifest.yaml").write_text(
            "---\n"
            "name: skill_b\n"
            "run_on_first_startup: true\n"
            "schedule_interval_seconds: 999\n"
        )
        (skill_b / "logic.py").write_text(
            "def run(ctx):\n"
            "    return {'status': 'initialized'}\n"
        )
        
        # Second startup should skip skill_a but run skill_b
        runner2 = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner2._startup_marker_path = marker_path
        runner2.setup()
        
        # Marker should now contain both
        marker_data = json.loads(marker_path.read_text())
        assert "skill_a" in marker_data
        assert "skill_b" in marker_data


class TestRunnerDispatch:
    def test_dispatch_skill_alpha(self, minimal_skills_dir, tmp_path):
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        result = runner.dispatch("skill_alpha")
        assert result["skill"] == "alpha"
        assert result["has_db"] is True

    def test_dispatch_injects_db_and_llm(self, minimal_skills_dir, tmp_path):
        db = MockDBConnector()
        llm = MockLLMProvider()
        runner = Runner(
            db_connector=db,
            llm_provider=llm,
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        result = runner.dispatch("skill_alpha")
        assert result["has_db"] is True

    def test_dispatch_injects_memory(self, minimal_skills_dir, tmp_path):
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        result = runner.dispatch("skill_beta")
        # Memory is initialized — status should be non-empty string
        assert isinstance(result["status"], str)
        assert len(result["status"]) > 0

    def test_runner_uses_checkpoint_backed_memory(self, minimal_skills_dir, tmp_path):
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "runtime_memory.db",
        )

        assert isinstance(runner.memory, CheckpointBackedMemory)

    def test_dispatch_unknown_skill_raises(self, minimal_skills_dir, tmp_path):
        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=minimal_skills_dir,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        with pytest.raises(KeyError):
            runner.dispatch("nonexistent_skill")

    def test_dispatch_with_explicit_context(self, minimal_skills_dir, tmp_path):
        skills = tmp_path / "ctx_skills"
        skills.mkdir()
        sk = skills / "ctx_sk"
        sk.mkdir()
        (sk / "instruction.md").write_text("")
        (sk / "logic.py").write_text("def run(ctx): return ctx.get('extra_key')\n")

        runner = Runner(
            db_connector=MockDBConnector(),
            llm_provider=MockLLMProvider(),
            skills_dir=skills,
            memory_path=tmp_path / "agent_memory.json",
        )
        runner.setup()
        ctx = runner._build_context()
        ctx["extra_key"] = "hello"
        result = runner.dispatch("ctx_sk", context=ctx)
        assert result == "hello"
