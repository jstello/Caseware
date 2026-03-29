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
from invoice_agent.schemas import (
    FinalReport,
    InvoiceResult,
    LiveCategorizationSuggestion,
    LiveExtractionFields,
    ReasoningEnvelope,
)
from invoice_agent.settings import get_settings


FIXTURE_DIR = Path("/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices")


def _force_mock_mode(monkeypatch) -> None:
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "mock")
    get_settings.cache_clear()


def _planner_response(
    *,
    text: str,
    tool_name: str,
    args: dict | None = None,
    call_id: str,
    thought: str | None = None,
) -> LlmResponse:
    parts = []
    if thought is not None:
        parts.append(
            types.Part(
                text=thought,
                thought=True,
                thought_signature=f"{call_id}-signature".encode("utf-8"),
            )
        )
    parts.extend(
        [
            types.Part(text=text),
            types.Part(
                function_call=types.FunctionCall(
                    id=call_id,
                    name=tool_name,
                    args=args or {},
                )
            ),
        ]
    )
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=parts,
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


class ThoughtAwarePlanner(BaseLlm):
    model: str = "thought-aware-planner"
    saw_prior_thought: bool = False
    saw_tool_reasoning: bool = False

    async def generate_content_async(self, llm_request, stream: bool = False):
        responses_by_tool: dict[str, dict] = {}
        saw_prior_thought = False
        for content in llm_request.contents:
            if not content or not content.parts:
                continue
            for part in content.parts:
                if getattr(part, "thought", False):
                    saw_prior_thought = True
                if part.function_response:
                    responses_by_tool[part.function_response.name] = dict(
                        part.function_response.response or {}
                    )

        if saw_prior_thought:
            self.saw_prior_thought = True
        extract_response = responses_by_tool.get("extract_invoice_fields") or {}
        if isinstance(extract_response.get("reasoning"), dict):
            self.saw_tool_reasoning = True

        load_response = responses_by_tool.get("load_images")
        if load_response is None:
            yield _planner_response(
                text="I will inspect the available invoices first.",
                thought="I should load the folder before deciding which invoice to inspect.",
                tool_name="load_images",
                args={},
                call_id="thought-load-images-1",
            )
            return

        invoice_ref = load_response["invoice_refs"][0]
        invoice_id = invoice_ref["invoice_id"]

        if "extract_invoice_fields" not in responses_by_tool:
            yield _planner_response(
                text=f"I will extract the fields for {invoice_ref['filename']}.",
                thought="The next step is extraction so I can inspect the invoice fields and tool history.",
                tool_name="extract_invoice_fields",
                args={"invoice_id": invoice_id},
                call_id="thought-extract-1",
            )
            return

        if "normalize_invoice" not in responses_by_tool:
            if not self.saw_tool_reasoning:
                raise AssertionError("expected extract_invoice_fields reasoning in function response history")
            yield _planner_response(
                text=f"I have enough detail to normalize {invoice_ref['filename']}.",
                thought="The earlier extraction reasoning is visible, so I can continue with normalization.",
                tool_name="normalize_invoice",
                args={"invoice_id": invoice_id},
                call_id="thought-normalize-1",
            )
            return

        if "categorize_invoice" not in responses_by_tool:
            if not self.saw_prior_thought:
                raise AssertionError("expected prior planner thought parts in history")
            yield _planner_response(
                text=f"I will categorize {invoice_ref['filename']} next.",
                thought="My earlier tool-choice thoughts are still present in the conversation history.",
                tool_name="categorize_invoice",
                args={"invoice_id": invoice_id},
                call_id="thought-categorize-1",
            )
            return

        if "aggregate_invoices" not in responses_by_tool:
            yield _planner_response(
                text="Every invoice is categorized, so I will aggregate the totals.",
                thought="I can aggregate now that the per-invoice reasoning path is complete.",
                tool_name="aggregate_invoices",
                args={},
                call_id="thought-aggregate-1",
            )
            return

        if "generate_report" not in responses_by_tool:
            yield _planner_response(
                text="The totals are ready, and I can assemble the report.",
                thought="The final step is to turn the aggregated state into the required report.",
                tool_name="generate_report",
                args={},
                call_id="thought-generate-report-1",
            )
            return

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="The report is complete.")],
            ),
            partial=False,
        )


class _ReasoningLiveAdapter:
    def extract_invoice_fields(self, *, invoice_path: Path, reviewer_prompt: str | None, focus_hint: str | None):
        return LiveExtractionFields(
            vendor="Trace Air",
            invoice_date="2026-02-15",
            invoice_number="TRACE-1007",
            total=642.10,
            currency="USD",
            raw_category_hint="airfare",
            extraction_confidence=0.94,
            notes=["The invoice is legible."],
            reasoning=ReasoningEnvelope(
                summaries=["The airfare invoice clearly exposes vendor and total."],
                summary_count=1,
                thoughts_token_count=6,
                total_token_count=25,
                has_thought_signature=False,
                source="extract_invoice_fields",
            ),
        )

    def categorize_invoice(
        self,
        *,
        normalized_invoice: dict,
        raw_category_hint: str | None,
        reviewer_prompt: str | None,
    ):
        return LiveCategorizationSuggestion(
            category="Travel",
            confidence=0.88,
            notes=["Airfare maps to Travel."],
            reasoning=ReasoningEnvelope(
                summaries=["Travel is the closest assignment category for airfare."],
                summary_count=1,
                thoughts_token_count=5,
                total_token_count=19,
                has_thought_signature=False,
                source="categorize_invoice",
            ),
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


def test_folder_path_streams_required_events_and_final_report(monkeypatch) -> None:
    _force_mock_mode(monkeypatch)
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

    version_tracking = events[0]["data"]["version_tracking"]
    assert version_tracking is None

    final_payload = events[-1]["data"]
    assert final_payload["version_tracking"] is None
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


def test_multipart_upload_supports_invoice_images(monkeypatch) -> None:
    _force_mock_mode(monkeypatch)
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
    _force_mock_mode(monkeypatch)

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


def test_live_mode_keeps_thought_summaries_internal_but_persists_them_to_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    planner = ThoughtAwarePlanner()

    def build_thought_agent(settings, invoice_tools):
        return LlmAgent(
            name=settings.agent.name,
            model=planner,
            description=settings.agent.description,
            instruction=settings.agent.system_instruction,
            tools=invoice_tools.tool_functions(),
        )

    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", build_thought_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: _ReasoningLiveAdapter(),
    )

    source = FIXTURE_DIR / "acme-air-travel-001.svg"
    folder = tmp_path / "live-thoughts"
    folder.mkdir()
    (folder / source.name).write_bytes(source.read_bytes())

    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Use conservative categorization.",
        },
    )

    assert response.status_code == 200
    events = _parse_sse_payload(response.text)
    assert events[-1]["event"] == "final_result"
    assert planner.saw_prior_thought is True
    assert planner.saw_tool_reasoning is True

    progress_messages = [
        entry["data"]["message"]
        for entry in events
        if entry["event"] == "progress"
    ]
    assert all("conversation history" not in message for message in progress_messages)
    assert all("airfare invoice clearly exposes vendor and total" not in message for message in progress_messages)

    public_tool_results = [
        entry["data"]["result"]
        for entry in events
        if entry["event"] == "tool_result"
    ]
    assert all("reasoning" not in result for result in public_tool_results)

    run_dir = Path(events[-1]["data"]["report_path"]).parent
    trace_entries = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    load_trace = next(
        entry
        for entry in trace_entries
        if entry["kind"] == "tool_call" and entry["tool_name"] == "load_images"
    )
    assert load_trace["planner_reasoning"]["summaries"] == [
        "I should load the folder before deciding which invoice to inspect."
    ]

    extract_result_trace = next(
        entry
        for entry in trace_entries
        if entry["kind"] == "tool_result" and entry["tool_name"] == "extract_invoice_fields"
    )
    assert extract_result_trace["result"]["reasoning"]["source"] == "extract_invoice_fields"
    assert extract_result_trace["result"]["reasoning"]["summaries"] == [
        "The airfare invoice clearly exposes vendor and total."
    ]

    thought_ledger = json.loads((run_dir / "thought_ledger.json").read_text(encoding="utf-8"))
    assert any(entry["source"] == "planner" for entry in thought_ledger)
    assert any(entry["source"] == "extract_invoice_fields" for entry in thought_ledger)
