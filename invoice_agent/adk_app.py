from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .agent import build_root_agent
from .reasoning import build_reasoning_envelope, dump_reasoning_envelope
from .schemas import ThoughtLedgerEntry
from .settings import Settings, get_settings
from .trace import MlflowRunRecorder, MlflowTraceSession, TraceWriter


@dataclass
class _AdkInvocationTraceState:
    run_id: str
    trace_writer: TraceWriter
    mlflow_recorder: MlflowRunRecorder
    mlflow_trace_session: MlflowTraceSession
    final_report: dict[str, Any] | None = None
    thought_ledger_entries: list[dict[str, Any]] = field(default_factory=list)
    finalized: bool = False


class AdkMlflowTracingPlugin(BasePlugin):
    """Captures ADK Web invocations into the same MLflow and local trace sinks."""

    def __init__(self, settings: Settings):
        super().__init__(name="invoice_agent_mlflow_tracing")
        self.settings = settings
        self._runs: dict[str, _AdkInvocationTraceState] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ):
        run_id = invocation_context.invocation_id
        run_dir = self.settings.runtime.traces_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_writer = TraceWriter(run_dir)
        prompt_text = _content_to_text(invocation_context.user_content)
        trace_writer.write_prompt_artifacts(
            system_instruction=_agent_instruction(invocation_context),
            request_prompt=prompt_text or "",
        )
        mlflow_recorder = MlflowRunRecorder(
            run_id=run_id,
            run_dir=run_dir,
            config_path=self.settings.config_path,
            config=self.settings.config,
            prompt=prompt_text,
        )
        mlflow_recorder.start()
        version_tracking = mlflow_recorder.version_tracking_payload()
        mlflow_trace_session = MlflowTraceSession(
            run_id=run_id,
            enabled=mlflow_recorder.enabled,
        )
        input_source = invocation_context.session.state.get("input_source")
        run_started = {
            "run_id": run_id,
            "mode": self.settings.runtime.planner_mode,
            "input_source": input_source if isinstance(input_source, dict) else None,
            "prompt": prompt_text,
            "session_id": invocation_context.session.id,
            "version_tracking": version_tracking,
        }
        trace_writer.write_trace(kind="run_started", payload=run_started)
        mlflow_trace_session.start(run_started, version_tracking=version_tracking)
        self._runs[run_id] = _AdkInvocationTraceState(
            run_id=run_id,
            trace_writer=trace_writer,
            mlflow_recorder=mlflow_recorder,
            mlflow_trace_session=mlflow_trace_session,
        )
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Event | None:
        state = self._runs.get(invocation_context.invocation_id)
        if state is None:
            return None

        state.trace_writer.write_sse(
            event="adk_event",
            data=event.model_dump(mode="json", by_alias=True),
        )

        if not event.content:
            return None

        pending_progress_parts: list[str] = []
        pending_thought_parts = []
        emitted_tool_call = False

        for part in event.content.parts or []:
            if getattr(part, "thought", False):
                pending_thought_parts.append(part)
                continue

            if part.text:
                progress = {
                    "run_id": state.run_id,
                    "message": part.text,
                    "agent": event.author,
                }
                state.trace_writer.write_trace(kind="progress", payload=progress)
                pending_progress_parts.append(part.text)

            if part.function_call:
                emitted_tool_call = True
                tool_name = part.function_call.name
                planner_reasoning = dump_reasoning_envelope(
                    build_reasoning_envelope(
                        source="planner",
                        parts=pending_thought_parts,
                        usage_metadata=event.usage_metadata,
                    )
                )
                progress_text = "\n".join(pending_progress_parts) or None
                tool_call = {
                    "run_id": state.run_id,
                    "tool_name": tool_name,
                    "tool_call_id": part.function_call.id,
                    "stage": self.settings.agent.tool_stages.get(tool_name, "tooling"),
                    "args": dict(part.function_call.args or {}),
                }
                state.trace_writer.write_trace(
                    kind="tool_call",
                    payload={
                        "stage": tool_call["stage"],
                        "tool_name": tool_name,
                        "tool_call_id": part.function_call.id,
                        "args": tool_call["args"],
                        "planner_progress_text": progress_text,
                        "planner_reasoning": planner_reasoning,
                    },
                )
                state.mlflow_trace_session.start_decision_span(
                    tool_call=tool_call,
                    planner_progress_text=progress_text,
                    planner_reasoning=planner_reasoning,
                    agent_name=event.author,
                )
                if planner_reasoning is not None:
                    state.thought_ledger_entries.append(
                        ThoughtLedgerEntry(
                            step_index=len(state.thought_ledger_entries) + 1,
                            source="planner",
                            tool_name=tool_name,
                            tool_call_id=part.function_call.id,
                            progress_text=progress_text,
                            summaries=planner_reasoning["summaries"],
                            summary_count=planner_reasoning["summary_count"],
                            thoughts_token_count=planner_reasoning["thoughts_token_count"],
                            total_token_count=planner_reasoning["total_token_count"],
                            has_thought_signature=planner_reasoning["has_thought_signature"],
                        ).model_dump(mode="json")
                    )
                pending_progress_parts = []
                pending_thought_parts = []

            if part.function_response:
                tool_name = part.function_response.name
                response_payload = dict(part.function_response.response or {})
                tool_result = {
                    "run_id": state.run_id,
                    "tool_name": tool_name,
                    "tool_call_id": part.function_response.id,
                    "stage": self.settings.agent.tool_stages.get(tool_name, "tooling"),
                    "result": response_payload,
                }
                state.trace_writer.write_trace(
                    kind="tool_result",
                    payload={
                        "stage": tool_result["stage"],
                        "tool_name": tool_name,
                        "tool_call_id": part.function_response.id,
                        "result": response_payload,
                    },
                )
                tool_reasoning = response_payload.get("reasoning")
                if (
                    tool_name in {"extract_invoice_fields", "categorize_invoice"}
                    and isinstance(tool_reasoning, dict)
                ):
                    state.thought_ledger_entries.append(
                        ThoughtLedgerEntry(
                            step_index=len(state.thought_ledger_entries) + 1,
                            source=tool_reasoning["source"],
                            tool_name=tool_name,
                            tool_call_id=part.function_response.id,
                            summaries=list(tool_reasoning.get("summaries") or []),
                            summary_count=int(tool_reasoning.get("summary_count") or 0),
                            thoughts_token_count=tool_reasoning.get("thoughts_token_count"),
                            total_token_count=tool_reasoning.get("total_token_count"),
                            has_thought_signature=bool(
                                tool_reasoning.get("has_thought_signature", False)
                            ),
                        ).model_dump(mode="json")
                    )
                state.mlflow_trace_session.complete_decision_span(tool_result)
                if tool_name == "categorize_invoice" and response_payload.get("invoice_result"):
                    invoice_event = {
                        "run_id": state.run_id,
                        "invoice_id": response_payload["invoice_result"]["invoice_id"],
                        "invoice": response_payload["invoice_result"],
                    }
                    state.trace_writer.write_trace(kind="invoice_result", payload=invoice_event)
                    state.mlflow_trace_session.complete_invoice_span(invoice_event)
                if tool_name == "generate_report":
                    state.final_report = response_payload

        if pending_thought_parts and not emitted_tool_call:
            planner_reasoning = dump_reasoning_envelope(
                build_reasoning_envelope(
                    source="planner",
                    parts=pending_thought_parts,
                    usage_metadata=event.usage_metadata,
                )
            )
            if planner_reasoning is not None:
                progress_text = "\n".join(pending_progress_parts) or None
                state.trace_writer.write_trace(
                    kind="planner_reasoning",
                    payload={
                        "run_id": state.run_id,
                        "agent": event.author,
                        "progress_text": progress_text,
                        "planner_reasoning": planner_reasoning,
                    },
                )
                state.thought_ledger_entries.append(
                    ThoughtLedgerEntry(
                        step_index=len(state.thought_ledger_entries) + 1,
                        source="planner",
                        progress_text=progress_text,
                        summaries=planner_reasoning["summaries"],
                        summary_count=planner_reasoning["summary_count"],
                        thoughts_token_count=planner_reasoning["thoughts_token_count"],
                        total_token_count=planner_reasoning["total_token_count"],
                        has_thought_signature=planner_reasoning["has_thought_signature"],
                    ).model_dump(mode="json")
                )
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request,
        error: Exception,
    ):
        self._finalize_error(
            invocation_id=callback_context.invocation_id,
            error_payload={
                "run_id": callback_context.invocation_id,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ):
        self._finalize_error(
            invocation_id=tool_context.invocation_id,
            error_payload={
                "run_id": tool_context.invocation_id,
                "error_type": type(error).__name__,
                "message": str(error),
                "tool_name": tool.name,
                "tool_args": dict(tool_args),
            },
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        state = self._runs.get(invocation_context.invocation_id)
        if state is None or state.finalized:
            return

        if state.final_report is None:
            self._finalize_error(
                invocation_id=invocation_context.invocation_id,
                error_payload={
                    "run_id": state.run_id,
                    "error_type": "RuntimeError",
                    "message": "The ADK run completed without a generated final report.",
                },
            )
            return

        state.trace_writer.write_report(state.final_report)
        if state.thought_ledger_entries:
            state.trace_writer.write_thought_ledger(state.thought_ledger_entries)
        final_payload = {
            "run_id": state.run_id,
            "report": state.final_report,
            "report_path": str(state.trace_writer.run_dir / "final_report.json"),
            "trace_path": str(state.trace_writer.trace_path),
            "sse_path": str(state.trace_writer.sse_path),
        }
        state.trace_writer.write_trace(kind="final_result", payload=final_payload)
        state.mlflow_trace_session.complete(final_payload)
        state.mlflow_recorder.finalize(state.trace_writer, state.final_report)
        state.finalized = True
        self._runs.pop(invocation_context.invocation_id, None)

    def _finalize_error(self, *, invocation_id: str, error_payload: dict[str, Any]) -> None:
        state = self._runs.get(invocation_id)
        if state is None or state.finalized:
            return
        if state.thought_ledger_entries:
            state.trace_writer.write_thought_ledger(state.thought_ledger_entries)
        state.trace_writer.write_trace(kind="error", payload=error_payload)
        state.mlflow_trace_session.fail(error_payload)
        state.mlflow_recorder.finalize_error(state.trace_writer, error_payload)
        state.finalized = True
        self._runs.pop(invocation_id, None)


def build_adk_app(settings: Settings | None = None) -> App:
    effective_settings = settings or get_settings()
    return App(
        name=effective_settings.agent.name,
        root_agent=build_root_agent(effective_settings),
        plugins=[AdkMlflowTracingPlugin(effective_settings)],
    )


def _content_to_text(content) -> str | None:
    if content is None or not getattr(content, "parts", None):
        return None
    texts = [
        part.text.strip()
        for part in content.parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    ]
    if not texts:
        return None
    return "\n".join(texts)


def _agent_instruction(invocation_context: InvocationContext) -> str:
    instruction = getattr(invocation_context.agent, "instruction", None)
    if isinstance(instruction, str):
        return instruction
    return ""


app = build_adk_app()
