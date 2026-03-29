from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse

from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .agent import build_root_agent
from .reasoning import build_reasoning_envelope, dump_reasoning_envelope
from .schemas import ThoughtLedgerEntry
from .settings import Settings, get_settings
from .trace import MlflowRunRecorder, TraceWriter


UPLOAD_ARTIFACT_PATTERN = re.compile(r'\[Uploaded Artifact: "(?P<name>[^"]+)"\]')
UPLOAD_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}


@dataclass
class _AdkInvocationTraceState:
    run_id: str
    trace_writer: TraceWriter
    mlflow_recorder: MlflowRunRecorder
    final_report: dict[str, Any] | None = None
    thought_ledger_entries: list[dict[str, Any]] = field(default_factory=list)
    finalized: bool = False


class AdkMlflowTracingPlugin(BasePlugin):
    """Captures ADK Web invocations into local file traces and MLflow runs.

    MLflow span tracing is handled automatically by ``mlflow.gemini.autolog()``,
    so this plugin only manages: (1) upload materialization + session state sync,
    (2) local TraceWriter file output (JSONL traces, SSE logs), and
    (3) MlflowRunRecorder lifecycle for run/artifact management.
    """

    def __init__(self, settings: Settings):
        super().__init__(name="invoice_agent_mlflow_tracing")
        self.settings = settings
        self._runs: dict[str, _AdkInvocationTraceState] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        await self._sync_run_state_from_message(
            invocation_context=invocation_context,
            user_message=user_message,
            phase="on_user_message",
        )
        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ):
        uploaded_source = await self._sync_run_state_from_message(
            invocation_context=invocation_context,
            user_message=invocation_context.user_content,
            phase="before_run",
        )
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
        input_source = invocation_context.session.state.get("input_source")
        run_started = {
            "run_id": run_id,
            "mode": self.settings.runtime.planner_mode,
            "input_source": input_source if isinstance(input_source, dict) else None,
            "prompt": prompt_text,
            "session_id": invocation_context.session.id,
        }
        trace_writer.write_trace(kind="run_started", payload=run_started)
        self._append_input_source_debug(
            invocation_context=invocation_context,
            user_message=invocation_context.user_content,
            uploaded_source=uploaded_source,
            phase="before_run_trace",
        )
        self._runs[run_id] = _AdkInvocationTraceState(
            run_id=run_id,
            trace_writer=trace_writer,
            mlflow_recorder=mlflow_recorder,
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
                if tool_name == "categorize_invoice" and response_payload.get("invoice_result"):
                    invoice_event = {
                        "run_id": state.run_id,
                        "invoice_id": response_payload["invoice_result"]["invoice_id"],
                        "invoice": response_payload["invoice_result"],
                    }
                    state.trace_writer.write_trace(kind="invoice_result", payload=invoice_event)
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
        state.mlflow_recorder.finalize_error(state.trace_writer, error_payload)
        state.finalized = True
        self._runs.pop(invocation_id, None)

    async def _materialize_uploaded_input_source(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> dict[str, Any] | None:
        upload_dir = (
            self.settings.runtime.traces_dir
            / invocation_context.invocation_id
            / "uploads"
            / "adk_web"
        )
        written_any = False
        placeholder_names: list[str] = []

        for index, part in enumerate(user_message.parts or []):
            if part.inline_data is not None:
                filename = _upload_filename(
                    display_name=part.inline_data.display_name,
                    mime_type=part.inline_data.mime_type,
                    index=index,
                )
                _write_upload_bytes(
                    upload_dir=upload_dir,
                    filename=filename,
                    data=bytes(part.inline_data.data),
                )
                written_any = True
                continue

            if part.file_data is not None:
                local_path = _local_file_uri_to_path(part.file_data.file_uri)
                if local_path is not None and local_path.exists():
                    filename = _upload_filename(
                        display_name=part.file_data.display_name or local_path.name,
                        mime_type=part.file_data.mime_type,
                        index=index,
                    )
                    _write_upload_bytes(
                        upload_dir=upload_dir,
                        filename=filename,
                        data=local_path.read_bytes(),
                    )
                    written_any = True
                continue

            if part.text:
                placeholder_names.extend(_placeholder_artifact_names(part.text))

        if written_any:
            return {
                "source_type": "upload_dir",
                "path": str(upload_dir.resolve()),
            }

        if not placeholder_names or invocation_context.artifact_service is None:
            return None

        for index, artifact_name in enumerate(placeholder_names):
            artifact = await invocation_context.artifact_service.load_artifact(
                app_name=invocation_context.app_name,
                user_id=invocation_context.user_id,
                session_id=invocation_context.session.id,
                filename=artifact_name,
            )
            if artifact is None or artifact.inline_data is None:
                continue
            filename = _upload_filename(
                display_name=artifact_name,
                mime_type=artifact.inline_data.mime_type,
                index=index,
            )
            _write_upload_bytes(
                upload_dir=upload_dir,
                filename=filename,
                data=bytes(artifact.inline_data.data),
            )
            written_any = True

        if not written_any:
            return None

        return {
            "source_type": "upload_dir",
            "path": str(upload_dir.resolve()),
        }

    async def _sync_run_state_from_message(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content | None,
        phase: str,
    ) -> dict[str, Any] | None:
        prompt_text = _content_to_text(user_message)
        invocation_context.session.state["run_prompt"] = prompt_text or ""
        invocation_context.session.state["allowed_categories"] = list(
            self.settings.agent.allowed_categories
        )
        invocation_context.session.state["tool_stages"] = dict(
            self.settings.agent.tool_stages
        )

        uploaded_source = None
        if user_message is not None:
            uploaded_source = await self._materialize_uploaded_input_source(
                invocation_context=invocation_context,
                user_message=user_message,
            )
            if uploaded_source is not None:
                invocation_context.session.state["input_source"] = uploaded_source
        self._append_input_source_debug(
            invocation_context=invocation_context,
            user_message=user_message,
            uploaded_source=uploaded_source,
            phase=phase,
        )
        return uploaded_source

    def _append_input_source_debug(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content | None,
        uploaded_source: dict[str, Any] | None,
        phase: str,
    ) -> None:
        run_dir = self.settings.runtime.traces_dir / invocation_context.invocation_id
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_path = run_dir / "input_source_debug.jsonl"
        debug_payload = {
            "phase": phase,
            "session_id": invocation_context.session.id,
            "invocation_id": invocation_context.invocation_id,
            "prompt_text": _content_to_text(user_message),
            "session_input_source": invocation_context.session.state.get("input_source"),
            "uploaded_source": uploaded_source,
            "parts": [_summarize_part(part) for part in getattr(user_message, "parts", []) or []],
        }
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(debug_payload, sort_keys=True))
            handle.write("\n")


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


def _placeholder_artifact_names(text: str) -> list[str]:
    return [match.group("name") for match in UPLOAD_ARTIFACT_PATTERN.finditer(text)]


def _upload_filename(*, display_name: str | None, mime_type: str | None, index: int) -> str:
    candidate = Path(display_name or "").name
    if candidate in {"", ".", ".."}:
        suffix = UPLOAD_MIME_SUFFIXES.get((mime_type or "").lower(), ".bin")
        return f"uploaded-invoice-{index + 1}{suffix}"
    if Path(candidate).suffix:
        return candidate
    suffix = UPLOAD_MIME_SUFFIXES.get((mime_type or "").lower(), "")
    return f"{candidate}{suffix}"


def _write_upload_bytes(*, upload_dir: Path, filename: str, data: bytes) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename
    if target.exists() and target.read_bytes() == data:
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while target.exists():
        target = upload_dir / f"{stem}-{counter}{suffix}"
        if target.exists() and target.read_bytes() == data:
            return target
        counter += 1
    target.write_bytes(data)
    return target


def _local_file_uri_to_path(file_uri: str | None) -> Path | None:
    if not file_uri:
        return None
    parsed = urlparse(file_uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _summarize_part(part: types.Part) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if part.text is not None:
        summary["text"] = part.text
    if part.inline_data is not None:
        summary["inline_data"] = {
            "display_name": part.inline_data.display_name,
            "mime_type": part.inline_data.mime_type,
            "byte_length": len(part.inline_data.data or b""),
        }
    if part.file_data is not None:
        summary["file_data"] = {
            "display_name": part.file_data.display_name,
            "mime_type": part.file_data.mime_type,
            "file_uri": part.file_data.file_uri,
        }
    return summary


app = build_adk_app()
