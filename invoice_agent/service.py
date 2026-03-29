from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import build_invoice_agent, build_request_prompt
from .reasoning import build_reasoning_envelope, dump_reasoning_envelope
from .schemas import PreparedInputSource, ThoughtLedgerEntry
from .settings import Settings, get_settings
from .tools import InvoiceToolRegistry
from .trace import MlflowRunRecorder, MlflowTraceSession, TraceWriter


class InvoiceAgentService:
    """Orchestrates request parsing, ADK execution, SSE mapping, and trace writing."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def new_run_id(self) -> str:
        return uuid4().hex[:12]

    def create_run_dir(self, run_id: str) -> Path:
        run_dir = self.settings.runtime.traces_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    async def prepare_folder_input(self, folder_path: str) -> PreparedInputSource:
        resolved = Path(folder_path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise HTTPException(status_code=422, detail="folder_path must point to an existing local directory.")
        return PreparedInputSource(source_type="folder", path=str(resolved))

    async def prepare_upload_input(
        self, *, run_dir: Path, uploads: list[UploadFile]
    ) -> PreparedInputSource:
        upload_dir = run_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        written_any = False
        for upload in uploads:
            if not upload.filename:
                continue
            destination = upload_dir / Path(upload.filename).name
            destination.write_bytes(await upload.read())
            written_any = True
        if not written_any:
            raise HTTPException(status_code=422, detail="At least one invoice image is required.")
        return PreparedInputSource(source_type="upload_dir", path=str(upload_dir))

    async def run_stream(
        self,
        *,
        run_id: str,
        run_dir: Path,
        prepared_input: PreparedInputSource,
        prompt: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        effective_request_prompt = build_request_prompt(self.settings, prompt)
        trace_writer = TraceWriter(run_dir)
        trace_writer.write_prompt_artifacts(
            system_instruction=self.settings.agent.system_instruction,
            request_prompt=effective_request_prompt,
        )
        mlflow_recorder = MlflowRunRecorder(
            run_id=run_id,
            run_dir=run_dir,
            config_path=self.settings.config_path,
            config=self.settings.config,
            prompt=effective_request_prompt,
        )
        mlflow_recorder.start()
        version_tracking = mlflow_recorder.version_tracking_payload()
        mlflow_trace_session = MlflowTraceSession(
            run_id=run_id,
            enabled=mlflow_recorder.enabled,
        )
        invoice_tools = InvoiceToolRegistry(self.settings)
        agent = build_invoice_agent(self.settings, invoice_tools)
        session_service = InMemorySessionService()
        runner = Runner(
            app_name=self.settings.app_name,
            agent=agent,
            session_service=session_service,
        )
        initial_state = {
            "input_source": prepared_input.model_dump(),
            "run_prompt": prompt or "",
            "effective_request_prompt": effective_request_prompt,
            "allowed_categories": list(self.settings.agent.allowed_categories),
            "tool_stages": dict(self.settings.agent.tool_stages),
        }
        await session_service.create_session(
            app_name=self.settings.app_name,
            user_id="local-user",
            session_id=run_id,
            state=initial_state,
        )

        run_started = {
            "run_id": run_id,
            "mode": self.settings.runtime.planner_mode,
            "input_source": prepared_input.model_dump(),
            "prompt": prompt,
            "version_tracking": version_tracking,
        }
        trace_writer.write_trace(kind="run_started", payload=run_started)
        mlflow_trace_session.start(run_started, version_tracking=version_tracking)
        yield self._record_sse(trace_writer, "run_started", run_started)

        live_configuration_error = self._live_configuration_error(run_id)
        if live_configuration_error is not None:
            trace_writer.write_trace(kind="error", payload=live_configuration_error)
            yield self._record_sse(trace_writer, "error", live_configuration_error)
            mlflow_trace_session.fail(live_configuration_error)
            mlflow_recorder.finalize_error(trace_writer, live_configuration_error)
            return

        final_report: dict[str, Any] | None = None
        thought_ledger_entries: list[dict[str, Any]] = []
        try:
            async for event in runner.run_async(
                user_id="local-user",
                session_id=run_id,
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=effective_request_prompt
                        )
                    ],
                ),
            ):
                pending_progress_parts: list[str] = []
                pending_thought_parts: list[types.Part] = []
                emitted_tool_call = False
                if event.content:
                    for part in event.content.parts:
                        if getattr(part, "thought", False):
                            pending_thought_parts.append(part)
                            continue
                        if part.text:
                            progress = {
                                "run_id": run_id,
                                "message": part.text,
                                "agent": event.author,
                            }
                            trace_writer.write_trace(kind="progress", payload=progress)
                            pending_progress_parts.append(part.text)
                            yield self._record_sse(trace_writer, "progress", progress)
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
                                "run_id": run_id,
                                "tool_name": tool_name,
                                "tool_call_id": part.function_call.id,
                                "stage": self.settings.agent.tool_stages.get(tool_name, "tooling"),
                                "args": dict(part.function_call.args or {}),
                            }
                            trace_writer.write_trace(
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
                            mlflow_trace_session.start_decision_span(
                                tool_call=tool_call,
                                planner_progress_text=progress_text,
                                planner_reasoning=planner_reasoning,
                                agent_name=event.author,
                            )
                            if planner_reasoning is not None:
                                thought_ledger_entries.append(
                                    ThoughtLedgerEntry(
                                        step_index=len(thought_ledger_entries) + 1,
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
                            yield self._record_sse(trace_writer, "tool_call", tool_call)
                        if part.function_response:
                            tool_name = part.function_response.name
                            response_payload = dict(part.function_response.response or {})
                            tool_result = {
                                "run_id": run_id,
                                "tool_name": tool_name,
                                "tool_call_id": part.function_response.id,
                                "stage": self.settings.agent.tool_stages.get(tool_name, "tooling"),
                                "result": response_payload,
                            }
                            public_result_payload = _strip_internal_reasoning(response_payload)
                            public_tool_result = {
                                **tool_result,
                                "result": public_result_payload,
                            }
                            trace_writer.write_trace(
                                kind="tool_result",
                                payload={
                                    "stage": public_tool_result["stage"],
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
                                thought_ledger_entries.append(
                                    ThoughtLedgerEntry(
                                        step_index=len(thought_ledger_entries) + 1,
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
                            yield self._record_sse(trace_writer, "tool_result", public_tool_result)
                            mlflow_trace_session.complete_decision_span(tool_result)

                            if tool_name == "categorize_invoice" and response_payload.get("invoice_result"):
                                invoice_event = {
                                    "run_id": run_id,
                                    "invoice_id": response_payload["invoice_result"]["invoice_id"],
                                    "invoice": response_payload["invoice_result"],
                                }
                                trace_writer.write_trace(
                                    kind="invoice_result",
                                    payload=invoice_event,
                                )
                                mlflow_trace_session.complete_invoice_span(invoice_event)
                                yield self._record_sse(trace_writer, "invoice_result", invoice_event)
                            if tool_name == "generate_report":
                                final_report = response_payload
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
                            trace_writer.write_trace(
                                kind="planner_reasoning",
                                payload={
                                    "run_id": run_id,
                                    "agent": event.author,
                                    "progress_text": progress_text,
                                    "planner_reasoning": planner_reasoning,
                                },
                            )
                            thought_ledger_entries.append(
                                ThoughtLedgerEntry(
                                    step_index=len(thought_ledger_entries) + 1,
                                    source="planner",
                                    progress_text=progress_text,
                                    summaries=planner_reasoning["summaries"],
                                    summary_count=planner_reasoning["summary_count"],
                                    thoughts_token_count=planner_reasoning["thoughts_token_count"],
                                    total_token_count=planner_reasoning["total_token_count"],
                                    has_thought_signature=planner_reasoning["has_thought_signature"],
                                ).model_dump(mode="json")
                            )
        except Exception as exc:  # pragma: no cover - exercised via manual failure paths
            error_payload = {
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            if thought_ledger_entries:
                trace_writer.write_thought_ledger(thought_ledger_entries)
            trace_writer.write_trace(kind="error", payload=error_payload)
            yield self._record_sse(trace_writer, "error", error_payload)
            mlflow_trace_session.fail(error_payload)
            mlflow_recorder.finalize_error(trace_writer, error_payload)
            return

        if final_report is None:
            error_payload = {
                "run_id": run_id,
                "error_type": "RuntimeError",
                "message": "The run completed without a generated final report.",
            }
            if thought_ledger_entries:
                trace_writer.write_thought_ledger(thought_ledger_entries)
            trace_writer.write_trace(kind="error", payload=error_payload)
            yield self._record_sse(trace_writer, "error", error_payload)
            mlflow_trace_session.fail(error_payload)
            mlflow_recorder.finalize_error(trace_writer, error_payload)
            return

        trace_writer.write_report(final_report)
        if thought_ledger_entries:
            trace_writer.write_thought_ledger(thought_ledger_entries)
        final_payload = {
            "run_id": run_id,
            "report": final_report,
            "trace_path": str(trace_writer.trace_path),
            "sse_path": str(trace_writer.sse_path),
            "report_path": str(run_dir / "final_report.json"),
            "version_tracking": version_tracking,
        }
        trace_writer.write_trace(
            kind="final_result",
            payload={
                "report": final_report,
                "version_tracking": version_tracking,
            },
        )
        yield self._record_sse(trace_writer, "final_result", final_payload)
        mlflow_trace_session.complete(final_payload)
        mlflow_recorder.finalize(trace_writer, final_report)

    def _record_sse(
        self, trace_writer: TraceWriter, event: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        trace_writer.write_sse(event=event, data=data)
        return {"event": event, "data": json.dumps(data)}

    def _live_configuration_error(self, run_id: str) -> dict[str, Any] | None:
        if self.settings.runtime.planner_mode != "live":
            return None

        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
        if use_vertex:
            missing = [
                name
                for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
                if not os.getenv(name)
            ]
            if not missing:
                return None
            return {
                "run_id": run_id,
                "error_type": "LiveConfigurationError",
                "message": (
                    "planner_mode=live with GOOGLE_GENAI_USE_VERTEXAI=true requires "
                    f"{', '.join(missing)} to be set before the run starts."
                ),
            }

        if os.getenv("GOOGLE_API_KEY"):
            return None

        return {
            "run_id": run_id,
            "error_type": "LiveConfigurationError",
            "message": (
                "planner_mode=live requires either GOOGLE_API_KEY for Gemini API access "
                "or GOOGLE_GENAI_USE_VERTEXAI=true together with GOOGLE_CLOUD_PROJECT "
                "and GOOGLE_CLOUD_LOCATION."
            ),
        }


def get_service() -> InvoiceAgentService:
    return InvoiceAgentService(get_settings())


def _strip_internal_reasoning(payload: dict[str, Any]) -> dict[str, Any]:
    return _strip_reasoning_value(payload)


def _strip_reasoning_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_reasoning_value(item)
            for key, item in value.items()
            if key != "reasoning"
        }
    if isinstance(value, list):
        return [_strip_reasoning_value(item) for item in value]
    return value
