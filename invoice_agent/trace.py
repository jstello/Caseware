from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
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
            tool_context = bound.arguments.get("tool_context")
            invoice_id = sanitized_inputs.get("invoice_id")
            attributes = {
                "tool.name": name or func.__name__,
                "tool.has_invoice": invoice_id is not None,
            }
            if invoice_id is not None:
                attributes["invoice.id"] = str(invoice_id)
            if isinstance(tool_context, ToolContext):
                stage = tool_context.state.get("trace_current_stage")
                if stage is None:
                    stage = (tool_context.state.get("tool_stages") or {}).get(
                        name or func.__name__
                    )
                if stage is not None:
                    attributes["tool.stage"] = str(stage)
                tool_call_id = (
                    tool_context.state.get("trace_current_tool_call_id")
                    or getattr(tool_context, "function_call_id", None)
                )
                if tool_call_id is not None:
                    attributes["tool.call_id"] = str(tool_call_id)

            with mlflow.start_span(
                name=name or func.__name__,
                span_type="TOOL",
                attributes=attributes,
            ) as span:
                span.set_inputs({"tool_inputs": sanitized_inputs})
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    span.set_status("ERROR")
                    span.set_outputs(
                        {
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        }
                    )
                    raise
                span.set_outputs(_sanitize_trace_input(result))
                return result

        return wrapper

    return decorator


@dataclass
class _ManagedSpan:
    context_manager: Any
    span: Any

    def close(self) -> None:
        self.context_manager.__exit__(None, None, None)


class MlflowTraceSession:
    """Maintains one nested MLflow trace for the full run lifecycle."""

    def __init__(self, *, run_id: str, enabled: bool) -> None:
        self.run_id = run_id
        self.enabled = enabled and mlflow is not None
        self._root_span: _ManagedSpan | None = None
        self._invoice_spans: dict[str, _ManagedSpan] = {}
        self._decision_spans: dict[str, _ManagedSpan] = {}

    def start(self, run_started: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            root = self._open_span(
                name=f"invoice_agent_run:{self.run_id}",
                span_type="AGENT",
                attributes={
                    "run.id": self.run_id,
                    "planner.mode": str(run_started.get("mode")),
                    "input.source_type": str(
                        (run_started.get("input_source") or {}).get("source_type")
                    ),
                    "input.path": str((run_started.get("input_source") or {}).get("path")),
                    "input.has_prompt": run_started.get("prompt") not in (None, ""),
                },
            )
            root.span.set_inputs(_sanitize_trace_input(run_started))
            self._root_span = root
        except Exception:
            self._disable()

    def start_decision_span(
        self,
        *,
        tool_call: dict[str, Any],
        planner_message: str | None,
        agent_name: str | None,
    ) -> None:
        if not self.enabled:
            return

        try:
            invoice_id = tool_call["args"].get("invoice_id")
            if invoice_id is not None:
                self._ensure_invoice_span(str(invoice_id))

            decision = self._open_span(
                name=f"planner:{tool_call['tool_name']}",
                span_type="CHAIN",
                attributes={
                    "run.id": self.run_id,
                    "tool.name": tool_call["tool_name"],
                    "tool.call_id": tool_call["tool_call_id"],
                    "tool.stage": tool_call["stage"],
                    "agent.name": agent_name or "",
                    **(
                        {"invoice.id": str(invoice_id)}
                        if invoice_id is not None
                        else {}
                    ),
                },
            )
            decision.span.set_inputs(
                {
                    "planner_message": planner_message,
                    "tool_call": _sanitize_trace_input(tool_call),
                }
            )
            self._decision_spans[tool_call["tool_call_id"]] = decision
        except Exception:
            self._disable()

    def complete_decision_span(self, tool_result: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            decision = self._decision_spans.pop(tool_result["tool_call_id"], None)
            if decision is None:
                return
            decision.span.set_outputs({"tool_result": _sanitize_trace_input(tool_result)})
            decision.close()
        except Exception:
            self._disable()

    def complete_invoice_span(self, invoice_event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            invoice_id = str(invoice_event["invoice_id"])
            invoice_span = self._invoice_spans.pop(invoice_id, None)
            if invoice_span is None:
                return
            invoice_span.span.set_outputs({"invoice": _sanitize_trace_input(invoice_event["invoice"])})
            invoice_span.close()
        except Exception:
            self._disable()

    def complete(self, final_payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            self._close_all_open_decisions()
            self._close_all_open_invoices()
            if self._root_span is not None:
                report = final_payload.get("report") or {}
                summary = report.get("run_summary") or {}
                self._root_span.span.set_outputs(
                    {
                        "run_summary": _sanitize_trace_input(summary),
                        "report_path": final_payload.get("report_path"),
                        "trace_path": final_payload.get("trace_path"),
                        "sse_path": final_payload.get("sse_path"),
                    }
                )
                self._root_span.close()
                self._root_span = None
        except Exception:
            self._disable()

    def fail(self, error_payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            for decision in list(self._decision_spans.values()):
                decision.span.set_status("ERROR")
                decision.span.set_outputs({"error": _sanitize_trace_input(error_payload)})
                decision.close()
            self._decision_spans.clear()

            for invoice_span in list(self._invoice_spans.values()):
                invoice_span.span.set_status("ERROR")
                invoice_span.span.set_outputs({"error": _sanitize_trace_input(error_payload)})
                invoice_span.close()
            self._invoice_spans.clear()

            if self._root_span is not None:
                self._root_span.span.set_status("ERROR")
                self._root_span.span.set_outputs({"error": _sanitize_trace_input(error_payload)})
                self._root_span.close()
                self._root_span = None
        except Exception:
            self._disable()

    def _ensure_invoice_span(self, invoice_id: str) -> None:
        for active_invoice_id in list(self._invoice_spans):
            if active_invoice_id != invoice_id:
                invoice_span = self._invoice_spans.pop(active_invoice_id)
                invoice_span.close()

        if invoice_id in self._invoice_spans:
            return

        self._invoice_spans[invoice_id] = self._open_span(
            name=f"invoice:{invoice_id}",
            span_type="TASK",
            attributes={
                "run.id": self.run_id,
                "invoice.id": invoice_id,
            },
        )
        self._invoice_spans[invoice_id].span.set_inputs({"invoice_id": invoice_id})

    def _close_all_open_decisions(self) -> None:
        for decision in list(self._decision_spans.values()):
            decision.close()
        self._decision_spans.clear()

    def _close_all_open_invoices(self) -> None:
        for invoice_span in list(self._invoice_spans.values()):
            invoice_span.close()
        self._invoice_spans.clear()

    def _open_span(
        self,
        *,
        name: str,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> _ManagedSpan:
        context_manager = mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=_sanitize_trace_input(attributes or {}),
        )
        return _ManagedSpan(
            context_manager=context_manager,
            span=context_manager.__enter__(),
        )

    def _disable(self) -> None:
        try:
            self._close_all_open_decisions()
            self._close_all_open_invoices()
            if self._root_span is not None:
                self._root_span.close()
        except Exception:
            pass
        self.enabled = False
        self._root_span = None


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
