from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import (
    InvoiceAgentConfig,
    config_to_artifact_text,
    flatten_mlflow_params,
)

try:  # pragma: no cover - import guard keeps local tests resilient
    import mlflow
except Exception:  # pragma: no cover - gracefully degrade if optional dep is absent
    mlflow = None

# ---------------------------------------------------------------------------
# Auto-tracing: one line replaces ~400 lines of manual span management.
# Every google.genai call (planner LLM, live extraction, live categorization)
# is automatically traced with inputs, outputs, and token usage.
# ---------------------------------------------------------------------------
if mlflow is not None and hasattr(mlflow, "gemini"):
    mlflow.gemini.autolog()


class TraceWriter:
    """Writes reviewable JSONL traces for every run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.trace_path = run_dir / "trace.jsonl"
        self.sse_path = run_dir / "sse.jsonl"
        self.thought_ledger_path = run_dir / "thought_ledger.json"

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

    def write_thought_ledger(self, entries: list[dict[str, Any]]) -> None:
        self.thought_ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def write_prompt_artifacts(
        self,
        *,
        system_instruction: str,
        request_prompt: str,
    ) -> None:
        prompt_dir = self.run_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "system_instruction.txt").write_text(
            system_instruction,
            encoding="utf-8",
        )
        (prompt_dir / "request_prompt.txt").write_text(
            request_prompt,
            encoding="utf-8",
        )

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")


class MlflowRunRecorder:
    """Logs config, run params, and saved artifacts into MLflow when enabled."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        config_path: Path,
        config: InvoiceAgentConfig,
        prompt: str | None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.config_path = config_path
        self.config = config
        self.prompt = prompt
        self.enabled = mlflow is not None and config.tracing.enabled
        self._run_active = False

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            tracking_uri = self.config.tracing.tracking_uri or self._default_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            if self.config.tracing.tracking_uri:
                mlflow.set_experiment(self.config.tracing.experiment_name)
            else:
                experiment_id = self._ensure_local_experiment(tracking_uri)
                mlflow.set_experiment(experiment_id=experiment_id)
            if (
                self.config.tracing.enable_async_logging
                and hasattr(mlflow, "config")
                and hasattr(mlflow.config, "enable_async_logging")
            ):
                mlflow.config.enable_async_logging()
            mlflow.start_run(run_name=self.run_id)
            self._run_active = True

            for key, value in {
                "run_id": self.run_id,
                "app": self.config.runtime.app_name,
                "planner_mode": self.config.runtime.planner_mode,
                **self.config.tracing.tags,
            }.items():
                mlflow.set_tag(key, str(value))
            mlflow.log_params(flatten_mlflow_params(self.config))

            if self.config.tracing.log_config_artifact:
                config_artifact = self.run_dir / "config" / "invoice_agent.yaml"
                config_artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact_text = (
                    self.config_path.read_text(encoding="utf-8")
                    if self.config_path.exists()
                    else config_to_artifact_text(self.config)
                )
                config_artifact.write_text(artifact_text, encoding="utf-8")
                mlflow.log_artifact(str(config_artifact), artifact_path="config")

            if self.config.tracing.log_prompt_artifacts and self.prompt is not None:
                prompt_dir = self.run_dir / "prompts"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir / "request_prompt.txt"
                if not prompt_path.exists():
                    prompt_path.write_text(self.prompt, encoding="utf-8")
                for artifact in sorted(prompt_dir.glob("*.txt")):
                    mlflow.log_artifact(str(artifact), artifact_path="prompts")
        except Exception:  # pragma: no cover - tracing must not block the run
            self.enabled = False
            self._run_active = False

    def finalize(self, trace_writer: TraceWriter, report: dict[str, Any] | None) -> None:
        if not self.enabled:
            return

        try:
            if self.config.tracing.log_trace_artifacts:
                if trace_writer.trace_path.exists():
                    mlflow.log_artifact(str(trace_writer.trace_path), artifact_path="traces")
                if trace_writer.sse_path.exists():
                    mlflow.log_artifact(str(trace_writer.sse_path), artifact_path="traces")
                if trace_writer.thought_ledger_path.exists():
                    mlflow.log_artifact(
                        str(trace_writer.thought_ledger_path),
                        artifact_path="traces",
                    )

            final_report_path = trace_writer.run_dir / "final_report.json"
            if final_report_path.exists():
                mlflow.log_artifact(str(final_report_path), artifact_path="outputs")

            if report:
                summary = report.get("run_summary", {})
                mlflow.log_metric("invoice_count", float(summary.get("invoice_count", 0)))
                mlflow.log_metric("total_spend", float(summary.get("total_spend", 0.0)))

            if self._run_active:
                mlflow.end_run(status="FINISHED")
                self._run_active = False
        except Exception:  # pragma: no cover - tracing must not block the run
            self.enabled = False
            self._run_active = False

    def finalize_error(self, trace_writer: TraceWriter, error: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            error_path = trace_writer.run_dir / "errors" / "error.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(error, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(error_path), artifact_path="errors")

            if self.config.tracing.log_trace_artifacts:
                if trace_writer.trace_path.exists():
                    mlflow.log_artifact(str(trace_writer.trace_path), artifact_path="traces")
                if trace_writer.sse_path.exists():
                    mlflow.log_artifact(str(trace_writer.sse_path), artifact_path="traces")
                if trace_writer.thought_ledger_path.exists():
                    mlflow.log_artifact(
                        str(trace_writer.thought_ledger_path),
                        artifact_path="traces",
                    )

            if self._run_active:
                mlflow.end_run(status="FAILED")
                self._run_active = False
        except Exception:  # pragma: no cover - tracing must not block the run
            self.enabled = False
            self._run_active = False

    def _default_tracking_uri(self) -> str:
        database_path = (self.config.runtime.mlflow_tracking_dir / "mlflow.db").resolve()
        return f"sqlite:///{database_path}"

    def _ensure_local_experiment(self, tracking_uri: str) -> str:
        artifact_root = (
            self.config.runtime.mlflow_tracking_dir
            / "artifacts"
            / self.config.tracing.experiment_name
        ).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)

        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(self.config.tracing.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        return client.create_experiment(
            self.config.tracing.experiment_name,
            artifact_location=f"file://{artifact_root}",
            tags={key: str(value) for key, value in self.config.tracing.tags.items()},
        )
