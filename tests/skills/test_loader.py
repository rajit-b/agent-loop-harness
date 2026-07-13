"""Skill loading: validation and the lazy body."""

from __future__ import annotations

import pytest
import yaml

from agentloop.skills.loader import load_skill
from agentloop.skills.manager import SkillManager
from agentloop.types import ConfigError

from .conftest import write_skill


class TestLoading:
    def test_valid_skill(self, tmp_path):
        directory = write_skill(
            tmp_path, "code-search", "Search the codebase.",
            patterns=["find", "where is"],
            required_tools=["fs.read_*"],
            budget={"max_tool_calls": 5},
        )
        skill = load_skill(directory)
        assert skill.name == "code-search"
        assert skill.description == "Search the codebase."
        assert skill.config.triggers.patterns == ("find", "where is")
        assert skill.config.budget.max_tool_calls == 5

    def test_missing_config_file(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ConfigError, match="no skill.yaml"):
            load_skill(tmp_path / "empty")

    def test_missing_body_file(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d")
        (directory / "SKILL.md").unlink()
        with pytest.raises(ConfigError, match="no SKILL.md"):
            load_skill(directory)

    def test_unknown_keys_rejected(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d")
        (directory / "skill.yaml").write_text(
            yaml.safe_dump({"name": "s", "description": "d", "toolz": []})
        )
        with pytest.raises(ConfigError, match="invalid skill config"):
            load_skill(directory)

    def test_bad_trigger_pattern_rejected(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d", patterns=["[unclosed"])
        with pytest.raises(ConfigError, match="invalid trigger pattern"):
            load_skill(directory)

    def test_bad_name_rejected(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d")
        (directory / "skill.yaml").write_text(
            yaml.safe_dump({"name": "Bad Name!", "description": "d"})
        )
        with pytest.raises(ConfigError):
            load_skill(directory)


class TestLazyBody:
    def test_body_is_not_read_at_load_time(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d", body="original body")
        skill = load_skill(directory)
        # rewrite the body AFTER loading: if load had read it, we'd see stale
        (directory / "SKILL.md").write_text("rewritten body", encoding="utf-8")
        assert skill.body() == "rewritten body"

    def test_body_cached_after_first_read(self, tmp_path):
        directory = write_skill(tmp_path, "s", "d", body="v1")
        skill = load_skill(directory)
        assert skill.body() == "v1"
        (directory / "SKILL.md").write_text("v2", encoding="utf-8")
        assert skill.body() == "v1"  # stable within the process


class TestManager:
    def test_duplicate_names_rejected(self, make_skill, tmp_path):
        first = make_skill("dup", "one")
        second_dir = write_skill(tmp_path / "other", "dup", "two")
        with pytest.raises(ConfigError, match="duplicate skill name"):
            SkillManager([first, load_skill(second_dir)])

    def test_index_block_lists_descriptions_only(self, make_skill):
        manager = SkillManager(
            [make_skill("a-skill", "Does A.", body="SECRET-BODY-A")]
        )
        index = manager.index_block()
        assert "a-skill: Does A." in index
        assert "use_skill" in index
        assert "SECRET-BODY-A" not in index
