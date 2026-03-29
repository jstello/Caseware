from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Category = Literal[
    "Travel",
    "Meals & Entertainment",
    "Software / Subscriptions",
    "Professional Services",
    "Office Supplies",
    "Shipping / Postage",
    "Utilities",
    "Other",
]


ALLOWED_CATEGORIES: tuple[Category, ...] = (
    "Travel",
    "Meals & Entertainment",
    "Software / Subscriptions",
    "Professional Services",
    "Office Supplies",
    "Shipping / Postage",
    "Utilities",
    "Other",
)


ReasoningSource = Literal[
    "planner",
    "extract_invoice_fields",
    "categorize_invoice",
]


class ReasoningEnvelope(BaseModel):
    summaries: list[str] = Field(default_factory=list)
    summary_count: int = 0
    thoughts_token_count: int | None = None
    total_token_count: int | None = None
    has_thought_signature: bool = False
    source: ReasoningSource


class ThoughtLedgerEntry(BaseModel):
    step_index: int
    source: ReasoningSource
    tool_name: str | None = None
    tool_call_id: str | None = None
    progress_text: str | None = None
    summaries: list[str] = Field(default_factory=list)
    summary_count: int = 0
    thoughts_token_count: int | None = None
    total_token_count: int | None = None
    has_thought_signature: bool = False


class JsonRunRequest(BaseModel):
    folder_path: str
    prompt: str | None = None


class PreparedInputSource(BaseModel):
    source_type: Literal["folder", "upload_dir"]
    path: str


class InvoiceRef(BaseModel):
    invoice_id: str
    filename: str
    path: str
    fixture_key: str | None = None


class ExtractionResult(BaseModel):
    invoice_id: str
    attempt: int
    vendor: str | None = None
    invoice_date: str | None = None
    invoice_number: str | None = None
    total: float | None = None
    currency: str = "USD"
    raw_category_hint: str | None = None
    extraction_confidence: float
    notes: list[str] = Field(default_factory=list)
    missing_critical_fields: list[str] = Field(default_factory=list)
    needs_retry: bool = False
    retry_focus_hint: str | None = None


class NormalizedInvoice(BaseModel):
    invoice_id: str
    vendor: str | None = None
    invoice_date: str | None = None
    invoice_number: str | None = None
    total: float | None = None
    currency: str = "USD"
    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CategorizationDecision(BaseModel):
    invoice_id: str
    category: Category
    confidence: float
    notes: list[str] = Field(default_factory=list)


class LiveExtractionPayload(BaseModel):
    vendor: str | None = Field(default=None, description="Vendor or issuer name.")
    invoice_date: str | None = Field(
        default=None,
        description="Invoice date in YYYY-MM-DD format when visible.",
    )
    invoice_number: str | None = Field(
        default=None,
        description="Invoice number or identifier when visible.",
    )
    total: float | None = Field(
        default=None,
        description="Final invoice total as a number without currency symbols.",
    )
    currency: str | None = Field(
        default="USD",
        description="ISO-like currency code when known, otherwise null or USD.",
    )
    raw_category_hint: str | None = Field(
        default=None,
        description="Short non-binding phrase describing what the expense appears to be.",
    )
    extraction_confidence: float | None = Field(
        default=None,
        description="Confidence between 0 and 1 for the overall extraction.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Brief notes about ambiguity, OCR issues, or visible anomalies.",
    )


class LiveExtractionFields(LiveExtractionPayload):
    reasoning: ReasoningEnvelope | None = None


class LiveCategorizationPayload(BaseModel):
    category: str | None = Field(
        default=None,
        description="One assignment category if possible. Use Other when unsure.",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence between 0 and 1 for the category choice.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Short explanation for the category, especially when using Other.",
    )


class LiveCategorizationSuggestion(LiveCategorizationPayload):
    reasoning: ReasoningEnvelope | None = None


class InvoiceResult(BaseModel):
    invoice_id: str
    filename: str
    vendor: str | None = None
    invoice_date: str | None = None
    invoice_number: str | None = None
    total: float | None = None
    assigned_category: Category
    confidence: float
    notes: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    total_spend: float
    spend_by_category: dict[str, float]
    invoice_count: int


class FinalReport(BaseModel):
    run_summary: RunSummary
    invoices: list[InvoiceResult]
    issues_and_assumptions: list[str]


class SseEventEnvelope(BaseModel):
    event: Literal[
        "run_started",
        "progress",
        "tool_call",
        "tool_result",
        "invoice_result",
        "final_result",
        "error",
    ]
    data: dict[str, Any]
