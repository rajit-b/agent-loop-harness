"""The checked-in schema.json must match the models it claims to describe."""

from __future__ import annotations

import json

from agentloop.config.schema import SCHEMA_PATH, manifest_json_schema


def test_schema_file_in_sync():
    on_disk = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == manifest_json_schema(), (
        "schema.json is stale — regenerate with: python -m agentloop.config.schema"
    )
