from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import build_invoice_agent, build_request_prompt
from .schemas import PreparedInputSource
from .settings import Settings, get_settings
from .tools import InvoiceToolRegistry
from .trace import MlflowRunRecorder, TraceWriter


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
        trace_writer = TraceWriter(run_dir)
        mlflow_recorder = MlflowRunRecorder(
            run_id=run_id,
            run_dir=run_dir,
            config_path=self.settings.config_path,
            config=self.settings.config,
            prompt=prompt,
        )
        mlflow_recorder.start()
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
            "allowed_categories": list(self.settings.agent.allowed_categories),
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
        }
        trace_writer.write_trace(kind="run_started", payload=run_started)
        yield self._record_sse(trace_writer, "run_started", run_started)

        final_report: dict[str, Any] | None = None
        try:
            async for event in runner.run_async(
                user_id="local-user",
                session_id=run_id,
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=build_request_prompt(self.settings, prompt)
                        )
                    ],
                ),
            ):
                if event.content:
                    for part in event.content.parts:
                        if part.text:
                            progress = {
                                "run_id": run_id,
                                "message": part.text,
                                "agent": event.author,
                            }
                            trace_writer.write_trace(kind="progress", payload=progress)
                            yield self._record_sse(trace_writer, "progress", progress)
                        if part.function_call:
                            tool_name = part.function_call.name
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
                                },
                            )
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
                            trace_writer.write_trace(
                                kind="tool_result",
                                payload={
                                    "stage": tool_result["stage"],
                                    "tool_name": tool_name,
                                    "tool_call_id": part.function_response.id,
                                    "result": response_payload,
                                },
                            )
                            yield self._record_sse(trace_writer, "tool_result", tool_result)

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
                                yield self._record_sse(trace_writer, "invoice_result", invoice_event)
                            if tool_name == "generate_report":
                                final_report = response_payload
        except Exception as exc:  # pragma: no cover - exercised via manual failure paths
            error_payload = {
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            trace_writer.write_trace(kind="error", payload=error_payload)
            yield self._record_sse(trace_writer, "error", error_payload)
            mlflow_recorder.finalize_error(trace_writer, error_payload)
            return

        if final_report is None:
            error_payload = {
                "run_id": run_id,
                "error_type": "RuntimeError",
                "message": "The run completed without a generated final report.",
            }
            trace_writer.write_trace(kind="error", payload=error_payload)
            yield self._record_sse(trace_writer, "error", error_payload)
            mlflow_recorder.finalize_error(trace_writer, error_payload)
            return

        trace_writer.write_report(final_report)
        final_payload = {
            "run_id": run_id,
            "report": final_report,
            "trace_path": str(trace_writer.trace_path),
            "sse_path": str(trace_writer.sse_path),
            "report_path": str(run_dir / "final_report.json"),
        }
        trace_writer.write_trace(
            kind="final_result",
            payload={"report": final_report},
        )
        yield self._record_sse(trace_writer, "final_result", final_payload)
        mlflow_recorder.finalize(trace_writer, final_report)

    def _record_sse(
        self, trace_writer: TraceWriter, event: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        trace_writer.write_sse(event=event, data=data)
        return {"event": event, "data": json.dumps(data)}


def get_service() -> InvoiceAgentService:
    return InvoiceAgentService(get_settings())
