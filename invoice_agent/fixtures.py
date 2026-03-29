from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .settings import get_settings


@lru_cache(maxsize=1)
def load_fixture_manifest() -> dict[str, dict]:
    manifest_path = get_settings().fixture_manifest_path
    return json.loads(manifest_path.read_text())


def match_fixture_key(filename: str) -> str | None:
    stem = Path(filename).stem
    manifest = load_fixture_manifest()
    if stem in manifest:
        return stem
    return None
