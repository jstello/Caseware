from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from invoice_agent.app import app
from invoice_agent.schemas import FinalReport, InvoiceResult
from invoice_agent.settings import get_settings


FIXTURE_DIR = Path("/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices")


def _planner_response(
    *,
    text: str,
    tool_name: str,
    args: dict | None = None,
    call_id: str,
) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text=text),
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id,
                        name=tool_name,
                        args=args or {},
                    )
                ),
            ],
        ),
        partial=False,
    )


class EarlyFinalizePlanner(BaseLlm):
    model: str = "early-finalize-planner"

    async def generate_content_async(self, llm_request, stream: bool = False):
        saw_load_result = False
        for content in llm_request.contents:
            if not content or not content.parts:
                continue
            for part in content.parts:
                if part.function_response and part.function_response.name == "load_images":
                    saw_load_result = True

        if not saw_load_result:
            yield _planner_response(
                text="I will inspect the available invoices first.",
                tool_name="load_images",
                args={},
                call_id="load-images-1",
            )
            return

        yield _planner_response(
            text="I can finalize the run now.",
            tool_name="generate_report",
            args={},
            call_id="generate-report-1",
        )


def _parse_sse_payload(response_text: str) -> list[dict]:
    events: list[dict] = []
    for chunk in re.split(r"\r?\n\r?\n", response_text.strip()):
        if not chunk.strip():
            continue
        event_name = None
        data = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: ").strip())
        if event_name is not None and data is not None:
            events.append({"event": event_name, "data": data})
    return events


def test_folder_path_streams_required_events_and_final_report() -> None:
    client = TestClient(app)

    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(FIXTURE_DIR),
            "prompt": "Use conservative categorization and flag unusual invoices.",
        },
    )

    assert response.status_code == 200
    events = _parse_sse_payload(response.text)
    event_names = [entry["event"] for entry in events]

    assert event_names[0] == "run_started"
    assert event_names[-1] == "final_result"
    assert "progress" in event_names
    assert "tool_call" in event_names
    assert "tool_result" in event_names
    assert "invoice_result" in event_names
    assert "error" not in event_names

    final_payload = events[-1]["data"]
    report = FinalReport.model_validate(final_payload["report"])
    assert report.run_summary.invoice_count == 6
    assert report.run_summary.total_spend == 1508.32
    assert report.run_summary.spend_by_category == {
        "Office Supplies": 84.32,
        "Other": 148.5,
        "Shipping / Postage": 87.4,
        "Software / Subscriptions": 157.0,
        "Travel": 1031.1,
    }

    invoice_results = {invoice.invoice_id: invoice for invoice in report.invoices}
    assert invoice_results["team-bistro-ambiguous"].assigned_category == "Other"
    assert invoice_results["bright-stationery-noisy"].assigned_category == "Office Supplies"
    assert any("retry" in issue.lower() for issue in report.issues_and_assumptions)

    extract_calls = [
        entry
        for entry in events
        if entry["event"] == "tool_call"
        and entry["data"]["tool_name"] == "extract_invoice_fields"
        and entry["data"]["args"].get("invoice_id") == "bright-stationery-noisy"
    ]
    assert len(extract_calls) == 2

    run_dir = Path(final_payload["report_path"]).parent
    request_prompt_path = run_dir / "prompts" / "request_prompt.txt"
    system_instruction_path = run_dir / "prompts" / "system_instruction.txt"
    assert request_prompt_path.exists()
    assert system_instruction_path.exists()
    assert "Allowed final categories:" in request_prompt_path.read_text(encoding="utf-8")
    assert "Use conservative categorization and flag unusual invoices." in request_prompt_path.read_text(
        encoding="utf-8"
    )


def test_multipart_upload_supports_invoice_images() -> None:
    client = TestClient(app)

    acme = FIXTURE_DIR / "acme-air-travel-001.svg"
    team = FIXTURE_DIR / "team-bistro-ambiguous.svg"
    response = client.post(
        "/runs/stream",
        files=[
            ("files", (acme.name, acme.read_bytes(), "image/svg+xml")),
            ("files", (team.name, team.read_bytes(), "image/svg+xml")),
            ("prompt", (None, "Flag unusual invoices.")),
        ],
    )

    assert response.status_code == 200
    events = _parse_sse_payload(response.text)
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "final_result"

    final_payload = events[-1]["data"]
    report = FinalReport.model_validate(final_payload["report"])
    assert report.run_summary.invoice_count == 2
    assert report.run_summary.total_spend == 790.6

    invoice_results = [InvoiceResult.model_validate(invoice.model_dump()) for invoice in report.invoices]
    assert {invoice.invoice_id for invoice in invoice_results} == {
        "acme-air-travel-001",
        "team-bistro-ambiguous",
    }


def test_stream_emits_error_when_planner_tries_to_finalize_before_all_invoices_are_processed(
    monkeypatch,
) -> None:
    def build_unsafe_agent(settings, invoice_tools):
        return LlmAgent(
            name=settings.agent.name,
            model=EarlyFinalizePlanner(),
            description=settings.agent.description,
            instruction=settings.agent.system_instruction,
            tools=invoice_tools.tool_functions(),
        )

    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", build_unsafe_agent)
    client = TestClient(app)

    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(FIXTURE_DIR),
            "prompt": "Finish as quickly as possible.",
        },
    )

    assert response.status_code == 200
    events = _parse_sse_payload(response.text)
    event_names = [entry["event"] for entry in events]

    assert event_names[0] == "run_started"
    assert event_names[-1] == "error"
    assert "final_result" not in event_names
    assert "generate_report" in {
        entry["data"]["tool_name"]
        for entry in events
        if entry["event"] == "tool_call"
    }
    assert "every loaded invoice" in events[-1]["data"]["message"]


def test_live_mode_emits_sse_error_when_provider_configuration_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(FIXTURE_DIR),
            "prompt": "Use the live planner.",
        },
    )

    assert response.status_code == 200
    events = _parse_sse_payload(response.text)
    event_names = [entry["event"] for entry in events]

    assert event_names[0] == "run_started"
    assert event_names[-1] == "error"
    assert "final_result" not in event_names
    assert events[-1]["data"]["error_type"] == "LiveConfigurationError"
    assert "GOOGLE_API_KEY" in events[-1]["data"]["message"]
