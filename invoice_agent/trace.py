from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    """Writes reviewable JSONL traces for every run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.trace_path = run_dir / "trace.jsonl"
        self.sse_path = run_dir / "sse.jsonl"

    def write_trace(self, *, kind: str, payload: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "kind": kind,
            **payload,
        }
        self._append(self.trace_path, entry)

    def write_sse(self, *, event: str, data: dict[str, Any]) -> None:
        self._append(
            self.sse_path,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": event,
                "data": data,
            },
        )

    def write_report(self, report: dict[str, Any]) -> None:
        (self.run_dir / "final_report.json").write_text(json.dumps(report, indent=2))

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
