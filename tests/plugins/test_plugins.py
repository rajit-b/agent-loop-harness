"""Plugin lifecycle: register one of everything, version gates, rollback."""

from __future__ import annotations

import pytest

from agentloop.hooks.contract import PreToolPayload
from agentloop.plugins.loader import PluginManager
from agentloop.tools.executor import ToolRegistry
from agentloop.hooks.bus import HookBus
from agentloop.types import PluginError, ToolCall, ToolSpec

from .conftest import GOOD_PLUGIN, plugin_config, write_plugin


class TestHappyPath:
    async def test_plugin_registers_tool_hook_and_cli_command(
        self, tmp_path, manager, emitter
    ):
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        report = manager.load_all(
            [plugin_config("jira", source, config={"base_url": "https://x.example"})]
        )
        assert report.loaded == ["jira"] and report.failed == []

        # tool: namespaced wire name, canonical permissions tag, executable
        [spec] = manager.tools.specs()
        assert spec.name == "jira__issue_lookup"
        assert spec.permissions_tag == "jira.issue_lookup"  # matches the manifest allowlist dialect
        assert spec.source == "plugin"
        result = await manager.tools.execute(
            ToolCall(id="c1", name="jira__issue_lookup", arguments={"key": "42"})
        )
        assert "JIRA-42" in result.content
        assert "https://x.example" in result.content  # config reached the plugin

        # hook: present on the bus with the plugin-prefixed name
        [hook] = manager.hooks.hooks_for("pre_tool")
        assert hook.name.startswith("jira:")
        assert hook.priority == 5

        # CLI command captured for later mounting
        assert manager.cli["jira-status"].plugin == "jira"
        assert manager.cli["jira-status"].handler() == "ok"

        kinds = [k for k, _ in emitter.events]
        assert "plugin.loaded" in kinds

    async def test_registered_hook_actually_dispatches(self, tmp_path, manager):
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        manager.load_all([plugin_config("jira", source)])
        outcome = await manager.hooks.dispatch(
            "pre_tool",
            PreToolPayload(call=ToolCall(id="c", name="echo")),
        )
        assert not outcome.vetoed  # the observe-only hook ran without effect

    def test_version_constraint_satisfied(self, tmp_path, manager):
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        report = manager.load_all(
            [plugin_config("jira", source, version=">=0.1,<0.2")]
        )
        assert report.loaded == ["jira"]

    def test_dispose_all_reaches_plugins(self, tmp_path, manager, emitter):
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        manager.load_all([plugin_config("jira", source)])
        manager.dispose_all()
        assert any(k == "plugin.disposed" for k, _ in emitter.events)


class TestVersionGates:
    def test_version_constraint_violation_rejected(self, tmp_path, manager, emitter):
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        report = manager.load_all(
            [plugin_config("jira", source, version=">=0.2,<0.3")]
        )
        assert report.loaded == []
        [(name, error)] = report.failed
        assert name == "jira"
        assert "0.1.5" in error and ">=0.2,<0.3" in error
        assert manager.tools.specs() == ()  # nothing registered
        assert any(k == "plugin.failed" for k, _ in emitter.events)

    def test_api_version_mismatch_rejected(self, tmp_path, manager):
        body = GOOD_PLUGIN.replace("api_version = 1", "api_version = 99")
        source = write_plugin(tmp_path, "futuristic", body)
        report = manager.load_all([plugin_config("futuristic", source)])
        assert report.loaded == []
        assert "api_version 99" in report.failed[0][1]
        assert manager.tools.specs() == ()


ROLLBACK_PLUGIN = """
class HalfBroken:
    name = "half-broken"
    version = "1.0.0"
    api_version = 1

    def __init__(self, config):
        pass

    def register(self, registrar):
        from agentloop.types import ToolSpec

        async def fine(arguments):
            return "fine"

        registrar.add_tool(ToolSpec(name="fine_tool"), fine)
        registrar.add_hook("pre_tool", lambda p, c: None)
        registrar.add_cli_command("fine-cmd", lambda: "ok")
        raise RuntimeError("boom halfway through registration")

    def dispose(self):
        pass


def agentloop_plugin(config):
    return HalfBroken(config)
"""


class TestFailClosed:
    def test_mid_register_failure_rolls_back_everything(
        self, tmp_path, manager, emitter
    ):
        source = write_plugin(tmp_path, "half-broken", ROLLBACK_PLUGIN)
        report = manager.load_all([plugin_config("half-broken", source)])
        assert report.loaded == []
        assert "rolled back" in report.failed[0][1]
        # the tool, hook, and CLI command staged before the crash are all gone
        assert manager.tools.specs() == ()
        assert manager.hooks.hooks_for("pre_tool") == []
        assert manager.cli == {}

    def test_bad_plugin_does_not_block_good_one(self, tmp_path, manager):
        bad = write_plugin(tmp_path, "half-broken", ROLLBACK_PLUGIN)
        good = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        report = manager.load_all(
            [plugin_config("half-broken", bad), plugin_config("jira", good)]
        )
        assert report.loaded == ["jira"]
        assert [name for name, _ in report.failed] == ["half-broken"]
        assert [s.name for s in manager.tools.specs()] == ["jira__issue_lookup"]

    def test_tool_collision_with_existing_registration_rejected(
        self, tmp_path, emitter
    ):
        tools = ToolRegistry()

        async def existing(arguments):
            return "x"

        tools.register(ToolSpec(name="jira__issue_lookup"), existing)
        manager = PluginManager(tools=tools, hooks=HookBus(), emitter=emitter)
        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)
        report = manager.load_all([plugin_config("jira", source)])
        assert report.loaded == []
        assert "already exists" in report.failed[0][1]

    def test_missing_factory_rejected(self, tmp_path, manager):
        source = write_plugin(tmp_path, "empty", "x = 1\n")
        report = manager.load_all([plugin_config("empty", source)])
        assert "agentloop_plugin" in report.failed[0][1]

    def test_import_error_rejected(self, tmp_path, manager):
        source = write_plugin(tmp_path, "broken", "import does_not_exist_xyz\n")
        report = manager.load_all([plugin_config("broken", source)])
        assert report.loaded == []
        assert "import failed" in report.failed[0][1]

    def test_nonexistent_source_rejected(self, manager):
        report = manager.load_all(
            [plugin_config("ghost", "/no/such/path-or-entrypoint")]
        )
        assert "neither a path nor" in report.failed[0][1]

    def test_missing_protocol_attributes_rejected(self, tmp_path, manager):
        body = """
class Nameless:
    version = "1.0"
    api_version = 1
    def register(self, registrar): pass
    def dispose(self): pass

def agentloop_plugin(config):
    return Nameless()
"""
        source = write_plugin(tmp_path, "nameless", body)
        report = manager.load_all([plugin_config("nameless", source)])
        assert "missing attribute 'name'" in report.failed[0][1]

    def test_dispose_failure_is_isolated(self, tmp_path, emitter):
        body = GOOD_PLUGIN.replace(
            "self.disposed = True", "raise RuntimeError('dispose bug')"
        )
        manager = PluginManager(
            tools=ToolRegistry(), hooks=HookBus(), emitter=emitter
        )
        source = write_plugin(tmp_path, "jira", body)
        manager.load_all([plugin_config("jira", source)])
        manager.dispose_all()  # must not raise
        assert any(
            k == "plugin.failed" and "dispose" in p["error"]
            for k, p in emitter.events
        )


class TestEntryPointSource:
    def test_entry_point_resolution(self, tmp_path, manager, monkeypatch):
        import importlib.metadata

        source = write_plugin(tmp_path, "jira", GOOD_PLUGIN)

        class FakeEntryPoint:
            name = "jira-dist"

            def load(self):
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "fake_ep_plugin", source / "plugin.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.agentloop_plugin

        def fake_entry_points(*, group):
            assert group == "agentloop.plugin"
            return [FakeEntryPoint()]

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
        report = manager.load_all([plugin_config("jira", "jira-dist")])
        assert report.loaded == ["jira"]
        assert [s.name for s in manager.tools.specs()] == ["jira__issue_lookup"]


class TestSkillRegistration:
    def test_plugin_can_register_a_skill(self, tmp_path, manager):
        import yaml

        skill_dir = tmp_path / "the-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.yaml").write_text(
            yaml.safe_dump({"name": "issue-triage", "description": "Triage issues."})
        )
        (skill_dir / "SKILL.md").write_text("Triage steps.")

        body = f"""
class SkillPlugin:
    name = "triage"
    version = "1.0.0"
    api_version = 1
    def __init__(self, config): pass
    def register(self, registrar):
        registrar.add_skill(r"{skill_dir}")
    def dispose(self): pass

def agentloop_plugin(config):
    return SkillPlugin(config)
"""
        source = write_plugin(tmp_path, "triage", body)
        report = manager.load_all([plugin_config("triage", source)])
        assert report.loaded == ["triage"]
        assert [s.name for s in manager.skills] == ["issue-triage"]


class TestRegistrarIsTheOnlySurface:
    def test_registrar_exposes_exactly_four_capabilities(self):
        from agentloop.plugins.registrar import Registrar

        public = [n for n in dir(Registrar) if not n.startswith("_") and n != "commit"]
        assert sorted(public) == [
            "add_cli_command", "add_hook", "add_skill", "add_tool",
        ]

    def test_direct_load_one_raises_plugin_error(self, tmp_path, manager):
        source = write_plugin(tmp_path, "half-broken", ROLLBACK_PLUGIN)
        with pytest.raises(PluginError, match="rolled back"):
            manager.load_one(plugin_config("half-broken", source))
