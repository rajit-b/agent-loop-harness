"""JSON Schema generation for the manifest.

schema.json is checked in (so editors/CI can consume it without Python)
and a test asserts it stays in sync with the models. Regenerate with:

    python -m agentloop.config.schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentloop.config.manifest import Manifest

SCHEMA_PATH = Path(__file__).parent / "schema.json"


def manifest_json_schema() -> dict[str, Any]:
    schema = Manifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "agent.manifest.yaml"
    return schema


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    path.write_text(
        json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    print(f"wrote {write_schema()}")
