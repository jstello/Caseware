from __future__ import annotations

import inspect
import json
from functools import wraps
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext

from .config import (
    InvoiceAgentConfig,
    config_to_artifact_text,
    flatten_mlflow_params,
)

try:  # pragma: no cover - import guard keeps local tests resilient
    import mlflow
except Exception:  # pragma: no cover - gracefully degrade if optional dep is absent
    mlflow = None


def mlflow_trace(*args: Any, **kwargs: Any):
    """Return the MLflow trace decorator when available, otherwise a no-op."""

    if mlflow is None:
        def decorator(fn):
            return fn

        return decorator
    return mlflow.trace(*args, **kwargs)


def _sanitize_trace_input(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_trace_input(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_trace_input(item) for item in value]
    return str(value)


def trace_tool(name: str | None = None):
    """Decorator that traces tool calls with sanitized inputs when MLflow is available."""

    def decorator(func):
        if mlflow is None:
            return func

        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if hasattr(mlflow, "active_run") and mlflow.active_run() is None:
                return func(*args, **kwargs)

            bound = signature.bind_partial(*args, **kwargs)
            sanitized_inputs = {
                key: _sanitize_trace_input(value)
                for key, value in bound.arguments.items()
                if key not in {"self", "tool_context"}
            }

            @mlflow.trace(name=name or func.__name__)
            def invoke(tool_inputs: dict[str, Any]):
                return func(*args, **kwargs)

            return invoke(sanitized_inputs)

        return wrapper

    return decorator


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


class MlflowRunRecorder:
    """Logs config, run params, and saved artifacts into MLflow when enabled."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        config: InvoiceAgentConfig,
        prompt: str | None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.config = config
        self.prompt = prompt
        self.enabled = mlflow is not None and config.tracing.enabled
        self._run_active = False

    def start(self) -> None:
        if not self.enabled:
            return

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
            config_path = self.run_dir / "config" / "invoice_agent.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(config_to_artifact_text(self.config), encoding="utf-8")
            mlflow.log_artifact(str(config_path), artifact_path="config")

        if self.config.tracing.log_prompt_artifacts and self.prompt is not None:
            prompt_path = self.run_dir / "prompts" / "request_prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(self.prompt, encoding="utf-8")
            mlflow.log_artifact(str(prompt_path), artifact_path="prompts")

    def finalize(self, trace_writer: TraceWriter, report: dict[str, Any] | None) -> None:
        if not self.enabled:
            return

        if self.config.tracing.log_trace_artifacts:
            if trace_writer.trace_path.exists():
                mlflow.log_artifact(str(trace_writer.trace_path), artifact_path="traces")
            if trace_writer.sse_path.exists():
                mlflow.log_artifact(str(trace_writer.sse_path), artifact_path="traces")

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

    def finalize_error(self, trace_writer: TraceWriter, error: dict[str, Any]) -> None:
        if not self.enabled:
            return

        error_path = trace_writer.run_dir / "errors" / "error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(error, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(error_path), artifact_path="errors")

        if self.config.tracing.log_trace_artifacts:
            if trace_writer.trace_path.exists():
                mlflow.log_artifact(str(trace_writer.trace_path), artifact_path="traces")
            if trace_writer.sse_path.exists():
                mlflow.log_artifact(str(trace_writer.sse_path), artifact_path="traces")

        if self._run_active:
            mlflow.end_run(status="FAILED")
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
