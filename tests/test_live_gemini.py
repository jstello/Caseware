from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.agents import LlmAgent
from google.genai import types

from invoice_agent.app import app
from invoice_agent.live_gemini import GeminiInvoiceToolAdapter
from invoice_agent.mock_planner import MockPlannerLlm
from invoice_agent.schemas import (
    FinalReport,
    LiveCategorizationSuggestion,
    LiveExtractionFields,
    ReasoningEnvelope,
)
from invoice_agent.settings import get_settings


FIXTURE_DIR = Path("/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices")


class _FakeResponse:
    def __init__(
        self,
        text: str,
        *,
        parts: list[types.Part] | None = None,
        usage_metadata: types.GenerateContentResponseUsageMetadata | None = None,
    ) -> None:
        self.text = text
        self.parts = parts or [types.Part(text=text)]
        self.usage_metadata = usage_metadata


class _RecordingModels:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, *, model: str, contents, config=None):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        return self.responses.pop(0)


class _RecordingClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.models = _RecordingModels(responses)


class _FakeLiveAdapter:
    def __init__(
        self,
        *,
        extraction_responses: list[LiveExtractionFields],
        categorization_responses: list[LiveCategorizationSuggestion],
    ) -> None:
        self.extraction_responses = list(extraction_responses)
        self.categorization_responses = list(categorization_responses)
        self.extract_calls: list[dict] = []
        self.categorize_calls: list[dict] = []

    def extract_invoice_fields(self, *, invoice_path: Path, reviewer_prompt: str | None, focus_hint: str | None):
        self.extract_calls.append(
            {
                "invoice_path": invoice_path,
                "reviewer_prompt": reviewer_prompt,
                "focus_hint": focus_hint,
            }
        )
        return self.extraction_responses.pop(0)

    def categorize_invoice(
        self,
        *,
        normalized_invoice: dict,
        raw_category_hint: str | None,
        reviewer_prompt: str | None,
    ):
        self.categorize_calls.append(
            {
                "normalized_invoice": normalized_invoice,
                "raw_category_hint": raw_category_hint,
                "reviewer_prompt": reviewer_prompt,
            }
        )
        return self.categorization_responses.pop(0)


class _FailingLiveAdapter:
    def extract_invoice_fields(self, *, invoice_path: Path, reviewer_prompt: str | None, focus_hint: str | None):
        raise RuntimeError("Gemini transport failed during extraction.")

    def categorize_invoice(
        self,
        *,
        normalized_invoice: dict,
        raw_category_hint: str | None,
        reviewer_prompt: str | None,
    ):
        raise AssertionError("categorize_invoice should not run after extraction fails")


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


def _build_live_test_agent(settings, invoice_tools):
    return LlmAgent(
        name=settings.agent.name,
        model=MockPlannerLlm(),
        description=settings.agent.description,
        instruction=settings.agent.system_instruction,
        tools=invoice_tools.tool_functions(),
    )


def _prepare_single_invoice_folder(tmp_path: Path, filename: str) -> Path:
    source = FIXTURE_DIR / filename
    target = tmp_path / filename
    target.write_bytes(source.read_bytes())
    return tmp_path


def test_live_extraction_adapter_uses_multimodal_input_and_structured_schema() -> None:
    fixture_path = FIXTURE_DIR / "acme-air-travel-001.svg"
    client = _RecordingClient(
        [
            _FakeResponse(
                json.dumps(
                    {
                        "vendor": "Acme Air",
                        "invoice_date": "2026-02-15",
                        "invoice_number": "AA-1007",
                        "total": 642.1,
                        "currency": "USD",
                        "raw_category_hint": "airfare",
                        "extraction_confidence": 0.97,
                        "notes": ["The airline invoice is legible."],
                    }
                )
            )
        ]
    )
    adapter = GeminiInvoiceToolAdapter(
        client=client,
        model="gemini-2.5-flash",
        extraction_prompt_template="Focus hint: {focus_hint}. Reviewer prompt: {reviewer_prompt}. File name: {filename}.",
        categorization_prompt_template="unused",
        allowed_categories=["Travel", "Other"],
    )

    result = adapter.extract_invoice_fields(
        invoice_path=fixture_path,
        reviewer_prompt="Use conservative categorization.",
        focus_hint="vendor, total",
    )

    assert result.vendor == "Acme Air"
    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"][0].inline_data.mime_type == "image/svg+xml"
    assert "Focus hint: vendor, total." in call["contents"][1]
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["thinking_config"].include_thoughts is True
    assert "extraction_confidence" in call["config"]["response_json_schema"]["properties"]


def test_live_categorization_adapter_uses_allowed_categories_and_structured_schema() -> None:
    client = _RecordingClient(
        [
            _FakeResponse(
                json.dumps(
                    {
                        "category": "Travel",
                        "confidence": 0.82,
                        "notes": ["Airfare and hotel phrasing support Travel."],
                    }
                )
            )
        ]
    )
    adapter = GeminiInvoiceToolAdapter(
        client=client,
        model="gemini-2.5-flash",
        extraction_prompt_template="unused",
        categorization_prompt_template=(
            "Allowed categories: {allowed_categories}. Raw hint: {raw_category_hint}. "
            "Invoice: {normalized_invoice}. Reviewer prompt: {reviewer_prompt}."
        ),
        allowed_categories=["Travel", "Other"],
    )

    result = adapter.categorize_invoice(
        normalized_invoice={"vendor": "Acme Air", "total": 642.1, "currency": "USD"},
        raw_category_hint="airfare",
        reviewer_prompt="Be conservative.",
    )

    assert result.category == "Travel"
    call = client.models.calls[0]
    assert "Allowed categories: Travel, Other." in call["contents"]
    assert '"vendor": "Acme Air"' in call["contents"]
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["thinking_config"].include_thoughts is True
    assert "category" in call["config"]["response_json_schema"]["properties"]


def test_live_extraction_adapter_captures_thought_summaries_and_token_counts() -> None:
    client = _RecordingClient(
        [
            _FakeResponse(
                json.dumps(
                    {
                        "vendor": "Acme Air",
                        "invoice_date": "2026-02-15",
                        "invoice_number": "AA-1007",
                        "total": 642.1,
                        "currency": "USD",
                        "raw_category_hint": "airfare",
                        "extraction_confidence": 0.97,
                        "notes": ["The airline invoice is legible."],
                    }
                ),
                parts=[
                    types.Part(
                        text="I should confirm the vendor and total before returning structured fields.",
                        thought=True,
                        thought_signature=b"extract-signature",
                    ),
                    types.Part(
                        text=json.dumps(
                            {
                                "vendor": "Acme Air",
                                "invoice_date": "2026-02-15",
                                "invoice_number": "AA-1007",
                                "total": 642.1,
                                "currency": "USD",
                                "raw_category_hint": "airfare",
                                "extraction_confidence": 0.97,
                                "notes": ["The airline invoice is legible."],
                            }
                        )
                    ),
                ],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    thoughts_token_count=12,
                    total_token_count=74,
                ),
            )
        ]
    )
    adapter = GeminiInvoiceToolAdapter(
        client=client,
        model="gemini-2.5-flash",
        extraction_prompt_template="File name: {filename}. Focus hint: {focus_hint}.",
        categorization_prompt_template="unused",
        allowed_categories=["Travel", "Other"],
    )

    result = adapter.extract_invoice_fields(
        invoice_path=FIXTURE_DIR / "acme-air-travel-001.svg",
        reviewer_prompt=None,
        focus_hint="vendor, total",
    )

    assert result.reasoning == ReasoningEnvelope(
        summaries=["I should confirm the vendor and total before returning structured fields."],
        summary_count=1,
        thoughts_token_count=12,
        total_token_count=74,
        has_thought_signature=True,
        source="extract_invoice_fields",
    )


def test_live_categorization_adapter_captures_thought_summaries_and_token_counts() -> None:
    client = _RecordingClient(
        [
            _FakeResponse(
                json.dumps(
                    {
                        "category": "Travel",
                        "confidence": 0.82,
                        "notes": ["Airfare and hotel phrasing support Travel."],
                    }
                ),
                parts=[
                    types.Part(
                        text="The raw airfare hint aligns with the allowed Travel category.",
                        thought=True,
                    ),
                    types.Part(
                        text=json.dumps(
                            {
                                "category": "Travel",
                                "confidence": 0.82,
                                "notes": ["Airfare and hotel phrasing support Travel."],
                            }
                        )
                    ),
                ],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    thoughts_token_count=8,
                    total_token_count=39,
                ),
            )
        ]
    )
    adapter = GeminiInvoiceToolAdapter(
        client=client,
        model="gemini-2.5-flash",
        extraction_prompt_template="unused",
        categorization_prompt_template="Invoice: {normalized_invoice}.",
        allowed_categories=["Travel", "Other"],
    )

    result = adapter.categorize_invoice(
        normalized_invoice={"vendor": "Acme Air", "total": 642.1, "currency": "USD"},
        raw_category_hint="airfare",
        reviewer_prompt=None,
    )

    assert result.reasoning == ReasoningEnvelope(
        summaries=["The raw airfare hint aligns with the allowed Travel category."],
        summary_count=1,
        thoughts_token_count=8,
        total_token_count=39,
        has_thought_signature=False,
        source="categorize_invoice",
    )


def test_live_mode_run_uses_live_extraction_and_categorization_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    live_adapter = _FakeLiveAdapter(
        extraction_responses=[
            LiveExtractionFields(
                vendor="Live Air",
                invoice_date="2026-02-15",
                invoice_number="LIVE-1007",
                total=777.77,
                currency="USD",
                raw_category_hint="airfare",
                extraction_confidence=0.91,
                notes=["Live extraction recovered all core fields."],
                reasoning=ReasoningEnvelope(
                    summaries=["The invoice clearly shows airfare details."],
                    summary_count=1,
                    thoughts_token_count=4,
                    total_token_count=18,
                    has_thought_signature=False,
                    source="extract_invoice_fields",
                ),
            )
        ],
        categorization_responses=[
            LiveCategorizationSuggestion(
                category="Travel",
                confidence=0.86,
                notes=["Airfare maps to Travel."],
                reasoning=ReasoningEnvelope(
                    summaries=["Travel is the closest allowed category."],
                    summary_count=1,
                    thoughts_token_count=5,
                    total_token_count=17,
                    has_thought_signature=False,
                    source="categorize_invoice",
                ),
            )
        ],
    )
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", _build_live_test_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: live_adapter,
    )

    folder = _prepare_single_invoice_folder(tmp_path, "acme-air-travel-001.svg")
    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Use conservative categorization and flag unusual invoices.",
        },
    )

    events = _parse_sse_payload(response.text)
    assert events[-1]["event"] == "final_result"
    tool_results = [
        entry["data"]["result"]
        for entry in events
        if entry["event"] == "tool_result"
    ]
    assert all("reasoning" not in result for result in tool_results)
    report = FinalReport.model_validate(events[-1]["data"]["report"])
    assert report.run_summary.total_spend == 777.77
    assert report.invoices[0].vendor == "Live Air"
    assert report.invoices[0].assigned_category == "Travel"
    assert live_adapter.extract_calls[0]["invoice_path"].name == "acme-air-travel-001.svg"
    assert live_adapter.categorize_calls[0]["raw_category_hint"] == "airfare"


def test_live_mode_defaults_missing_currency_and_confidence_from_extraction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    live_adapter = _FakeLiveAdapter(
        extraction_responses=[
            LiveExtractionFields(
                vendor="Live Air",
                invoice_date="2026-02-15",
                invoice_number="LIVE-1007",
                total=777.77,
                currency=None,
                raw_category_hint="airfare",
                extraction_confidence=None,
                notes=["Live extraction omitted optional currency and confidence."],
            )
        ],
        categorization_responses=[
            LiveCategorizationSuggestion(
                category="Travel",
                confidence=0.86,
                notes=["Airfare maps to Travel."],
            )
        ],
    )
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", _build_live_test_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: live_adapter,
    )

    folder = _prepare_single_invoice_folder(tmp_path, "acme-air-travel-001.svg")
    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Use conservative categorization.",
        },
    )

    events = _parse_sse_payload(response.text)
    extraction_result = next(
        entry["data"]["result"]
        for entry in events
        if entry["event"] == "tool_result"
        and entry["data"]["tool_name"] == "extract_invoice_fields"
    )
    assert extraction_result["currency"] == "USD"
    assert extraction_result["extraction_confidence"] == 0.5


def test_live_mode_retries_extraction_with_focus_hint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    live_adapter = _FakeLiveAdapter(
        extraction_responses=[
            LiveExtractionFields(
                vendor=None,
                invoice_date="2026-02-23",
                invoice_number="BS-449",
                total=None,
                currency="USD",
                raw_category_hint="office goods",
                extraction_confidence=0.42,
                notes=["The first pass missed vendor and total."],
            ),
            LiveExtractionFields(
                vendor="Bright Stationery",
                invoice_date="2026-02-23",
                invoice_number="BS-449",
                total=84.32,
                currency="USD",
                raw_category_hint="office goods",
                extraction_confidence=0.81,
                notes=["The retry found vendor and total."],
            ),
        ],
        categorization_responses=[
            LiveCategorizationSuggestion(
                category="Office Supplies",
                confidence=0.78,
                notes=["Office goods maps to Office Supplies."],
            )
        ],
    )
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", _build_live_test_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: live_adapter,
    )

    folder = _prepare_single_invoice_folder(tmp_path, "bright-stationery-noisy.svg")
    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Use conservative categorization.",
        },
    )

    events = _parse_sse_payload(response.text)
    assert events[-1]["event"] == "final_result"
    assert len(live_adapter.extract_calls) == 2
    assert live_adapter.extract_calls[1]["focus_hint"] == "vendor, total"
    report = FinalReport.model_validate(events[-1]["data"]["report"])
    assert report.invoices[0].vendor == "Bright Stationery"
    assert report.invoices[0].total == 84.32


def test_live_mode_clamps_invalid_category_to_other(
    monkeypatch,
    tmp_path: Path,
) -> None:
    live_adapter = _FakeLiveAdapter(
        extraction_responses=[
            LiveExtractionFields(
                vendor="Team Bistro",
                invoice_date="2026-02-19",
                invoice_number="TB-204",
                total=148.5,
                currency="USD",
                raw_category_hint="team dinner",
                extraction_confidence=0.88,
                notes=["The invoice appears to be a team meal."],
            )
        ],
        categorization_responses=[
            LiveCategorizationSuggestion(
                category="Team Dinner",
                confidence=0.74,
                notes=["The model suggested a label outside the assignment taxonomy."],
            )
        ],
    )
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", _build_live_test_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: live_adapter,
    )

    folder = _prepare_single_invoice_folder(tmp_path, "team-bistro-ambiguous.svg")
    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Flag unusual invoices.",
        },
    )

    events = _parse_sse_payload(response.text)
    assert events[-1]["event"] == "final_result"
    report = FinalReport.model_validate(events[-1]["data"]["report"])
    assert report.invoices[0].assigned_category == "Other"
    assert any("clamped to Other" in note for note in report.invoices[0].notes)


def test_live_mode_emits_error_when_provider_call_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "live")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr("invoice_agent.service.build_invoice_agent", _build_live_test_agent)
    monkeypatch.setattr(
        "invoice_agent.tools.InvoiceToolRegistry._get_live_adapter",
        lambda self: _FailingLiveAdapter(),
    )

    folder = _prepare_single_invoice_folder(tmp_path, "acme-air-travel-001.svg")
    client = TestClient(app)
    response = client.post(
        "/runs/stream",
        json={
            "folder_path": str(folder),
            "prompt": "Use conservative categorization.",
        },
    )

    events = _parse_sse_payload(response.text)
    event_names = [entry["event"] for entry in events]

    assert event_names[0] == "run_started"
    assert event_names[-1] == "error"
    assert "final_result" not in event_names
    assert events[-1]["data"]["error_type"] == "RuntimeError"
    assert "Gemini transport failed during extraction." in events[-1]["data"]["message"]
