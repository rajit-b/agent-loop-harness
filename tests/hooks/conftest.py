from __future__ import annotations

import pytest

from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry

from ..loop.conftest import RecordingEmitter


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()
