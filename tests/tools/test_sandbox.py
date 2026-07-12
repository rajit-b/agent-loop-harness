"""Path jail: escapes rejected (.., absolute, symlink), result caps."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloop.tools.sandbox import Sandbox
from agentloop.types import ToolCall, ToolSpec

SPEC = ToolSpec(name="fs__read_file", path_hints=("path",))


def call(**arguments) -> ToolCall:
    return ToolCall(id="c1", name="fs__read_file", arguments=arguments)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "jail").mkdir()
    (tmp_path / "jail" / "ok.txt").write_text("fine")
    (tmp_path / "outside.txt").write_text("secret")
    return tmp_path / "jail"


class TestPathJail:
    def test_inside_root_passes(self, root):
        sandbox = Sandbox(roots=[root])
        assert sandbox.check_paths(SPEC, call(path=str(root / "ok.txt"))) is None

    def test_relative_paths_resolve_against_first_root(self, root):
        sandbox = Sandbox(roots=[root])
        assert sandbox.check_paths(SPEC, call(path="ok.txt")) is None
        assert sandbox.check_paths(SPEC, call(path="../outside.txt")) is not None

    def test_dotdot_escape_rejected(self, root):
        sandbox = Sandbox(roots=[root])
        violation = sandbox.check_paths(
            SPEC, call(path=str(root / ".." / "outside.txt"))
        )
        assert violation is not None and "escapes" in violation

    def test_absolute_path_outside_rejected(self, root):
        sandbox = Sandbox(roots=[root])
        assert sandbox.check_paths(SPEC, call(path="/etc/passwd")) is not None

    def test_symlink_escape_rejected(self, root):
        sneaky = root / "sneaky.txt"
        sneaky.symlink_to(root.parent / "outside.txt")
        sandbox = Sandbox(roots=[root])
        assert sandbox.check_paths(SPEC, call(path=str(sneaky))) is not None

    def test_list_valued_arguments_are_checked(self, root):
        spec = ToolSpec(name="fs__read_many", path_hints=("paths",))
        sandbox = Sandbox(roots=[root])
        ok = ToolCall(id="c", name="fs__read_many",
                      arguments={"paths": ["ok.txt"]})
        bad = ToolCall(id="c", name="fs__read_many",
                       arguments={"paths": ["ok.txt", "/etc/passwd"]})
        assert sandbox.check_paths(spec, ok) is None
        assert sandbox.check_paths(spec, bad) is not None

    def test_multiple_roots(self, root, tmp_path):
        second = tmp_path / "jail2"
        second.mkdir()
        (second / "b.txt").write_text("b")
        sandbox = Sandbox(roots=[root, second])
        assert sandbox.check_paths(SPEC, call(path=str(second / "b.txt"))) is None

    def test_unhinted_arguments_are_ignored(self, root):
        sandbox = Sandbox(roots=[root])
        spec = ToolSpec(name="grep", path_hints=())  # pattern is not a path
        probe = ToolCall(id="c", name="grep", arguments={"pattern": "/etc/passwd"})
        assert sandbox.check_paths(spec, probe) is None

    def test_no_roots_means_no_jail(self):
        sandbox = Sandbox()
        assert sandbox.check_paths(SPEC, call(path="/etc/passwd")) is None


class TestResultCap:
    def test_short_results_untouched(self):
        assert Sandbox(max_result_chars=10).cap_result("short") == "short"

    def test_long_results_truncated_with_notice(self):
        capped = Sandbox(max_result_chars=10).cap_result("x" * 100)
        assert capped.startswith("x" * 10)
        assert "truncated 90 of 100 chars" in capped
