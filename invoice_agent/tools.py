from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext

from .fixtures import load_fixture_manifest, match_fixture_key
from .live_gemini import GeminiInvoiceToolAdapter
from .reasoning import dump_reasoning_envelope
from .schemas import ALLOWED_CATEGORIES, Category
from .settings import Settings
from .trace import trace_tool


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
CRITICAL_FIELDS = ("vendor", "invoice_date", "total")


def _quantize(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _ensure_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class InvoiceToolRegistry:
    """Constrained invoice-access tool registry."""

    def __init__(
        self,
        settings: Settings,
        live_adapter: GeminiInvoiceToolAdapter | None = None,
    ):
        self.settings = settings
        self.fixture_manifest = load_fixture_manifest()
        self.live_adapter = live_adapter

    def tool_functions(self) -> list:
        return [
            self.load_images,
            self.extract_invoice_fields,
            self.normalize_invoice,
            self.categorize_invoice,
            self.aggregate_invoices,
            self.generate_report,
        ]

    @trace_tool()
    def load_images(self, tool_context: ToolContext) -> dict[str, Any]:
        """Discover the invoice image files available for the current run."""

        source = tool_context.state.get("input_source")
        if not source:
            raise ValueError("No input source was registered for this run.")

        source_path = Path(source["path"])
        if not source_path.is_dir():
            raise ValueError(f"Input source does not exist: {source_path}")

        invoice_refs: list[dict[str, Any]] = []
        working_invoices: dict[str, dict[str, Any]] = {}
        seen_ids: defaultdict[str, int] = defaultdict(int)

        for file_path in sorted(source_path.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            base_id = file_path.stem.replace(" ", "-").replace("_", "-").lower()
            seen_ids[base_id] += 1
            invoice_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
            fixture_key = match_fixture_key(file_path.name)
            invoice_ref = {
                "invoice_id": invoice_id,
                "filename": file_path.name,
                "path": str(file_path),
                "fixture_key": fixture_key,
            }
            invoice_refs.append(invoice_ref)
            working_invoices[invoice_id] = {
                **invoice_ref,
                "attempts": 0,
                "issues": [],
            }

        tool_context.state["invoice_refs"] = invoice_refs
        tool_context.state["invoice_order"] = [invoice["invoice_id"] for invoice in invoice_refs]
        tool_context.state["working_invoices"] = working_invoices

        return {
            "invoice_count": len(invoice_refs),
            "invoice_refs": invoice_refs,
            "notes": (
                ["No invoice images were found in the provided location."]
                if not invoice_refs
                else []
            ),
        }

    @trace_tool()
    def extract_invoice_fields(
        self,
        invoice_id: str,
        focus_hint: str | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Extract invoice fields for one invoice image. Retry with a focus hint when critical fields are missing."""

        invoice = self._get_invoice(tool_context, invoice_id)
        attempts = int(invoice.get("attempts", 0)) + 1
        invoice["attempts"] = attempts

        if self.settings.runtime.planner_mode == "live":
            extraction = self._get_live_adapter().extract_invoice_fields(
                invoice_path=Path(invoice["path"]),
                reviewer_prompt=str(tool_context.state.get("run_prompt") or ""),
                focus_hint=focus_hint,
            )
            payload = extraction.model_dump(mode="json")
            extraction_attempts: list[dict[str, Any]] = []
        else:
            fixture = self.fixture_manifest.get(invoice.get("fixture_key") or "", {})
            extraction_attempts = fixture.get("extraction_attempts") or []
            if extraction_attempts:
                payload = extraction_attempts[min(attempts - 1, len(extraction_attempts) - 1)].copy()
            else:
                payload = {
                    "vendor": invoice["filename"].rsplit(".", 1)[0].replace("-", " ").title(),
                    "invoice_date": None,
                    "invoice_number": None,
                    "total": None,
                    "currency": "USD",
                    "raw_category_hint": None,
                    "extraction_confidence": 0.42,
                    "notes": [
                        "This file does not match a curated synthetic fixture, so mock extraction returned a low-confidence placeholder."
                    ],
                }

        notes = _ensure_list(payload.get("notes"))
        missing_fields = [
            field for field in CRITICAL_FIELDS if payload.get(field) in (None, "", [])
        ]
        max_attempts = max(self.settings.runtime.max_extraction_attempts, len(extraction_attempts))
        needs_retry = bool(missing_fields) and attempts < max_attempts
        retry_focus_hint = None
        if needs_retry:
            retry_focus_hint = ", ".join(missing_fields)
            notes.append(
                f"Critical fields are still missing after attempt {attempts}; a targeted retry should focus on {retry_focus_hint}."
            )
        elif missing_fields:
            notes.append(
                "The agent exhausted the configured extraction retries and will continue with normalization using assumptions."
            )

        if focus_hint:
            notes.append(f"Focused retry hint: {focus_hint}.")

        extraction_confidence = payload.get("extraction_confidence")
        result = {
            "invoice_id": invoice_id,
            "attempt": attempts,
            "vendor": payload.get("vendor"),
            "invoice_date": payload.get("invoice_date"),
            "invoice_number": payload.get("invoice_number"),
            "total": payload.get("total"),
            "currency": payload.get("currency") or "USD",
            "raw_category_hint": payload.get("raw_category_hint"),
            "extraction_confidence": (
                float(extraction_confidence) if extraction_confidence is not None else 0.5
            ),
            "notes": _dedupe(notes),
            "missing_critical_fields": missing_fields,
            "needs_retry": needs_retry,
            "retry_focus_hint": retry_focus_hint,
        }
        if payload.get("reasoning") is not None:
            result["reasoning"] = payload["reasoning"]
        self._clear_invoice_outputs(invoice, clear_normalized=True)
        self._clear_run_outputs(tool_context)
        invoice["extracted"] = result
        invoice["issues"] = _dedupe(invoice.get("issues", []) + result["notes"])
        tool_context.state["working_invoices"] = tool_context.state["working_invoices"]
        return result

    @trace_tool()
    def normalize_invoice(
        self,
        invoice_id: str,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Normalize the extracted invoice fields into one consistent structured record."""

        invoice = self._get_invoice(tool_context, invoice_id)
        extracted = invoice.get("extracted")
        if not extracted:
            raise ValueError(f"Invoice {invoice_id} has not been extracted yet.")

        if self.settings.runtime.planner_mode == "live":
            normalized = {
                "vendor": extracted.get("vendor"),
                "invoice_date": extracted.get("invoice_date"),
                "invoice_number": extracted.get("invoice_number"),
                "total": extracted.get("total"),
                "currency": extracted.get("currency") or "USD",
            }
            assumptions = []
            notes = _ensure_list(extracted.get("notes"))
            if normalized["total"] is None:
                assumptions.append(
                    "The live extraction could not recover a total, so categorization must rely on context and may fall back to Other."
                )
            if normalized["vendor"] is None:
                assumptions.append(
                    "Vendor information is incomplete, so the filename is the only stable identifier for this live extraction."
                )
        else:
            fixture = self.fixture_manifest.get(invoice.get("fixture_key") or "", {})
            normalized = (fixture.get("normalized") or {}).copy()
            assumptions = _ensure_list(normalized.get("assumptions"))
            notes = _ensure_list(normalized.get("notes")) + _ensure_list(extracted.get("notes"))

            if not normalized:
                normalized = {
                    "vendor": extracted.get("vendor"),
                    "invoice_date": extracted.get("invoice_date"),
                    "invoice_number": extracted.get("invoice_number"),
                    "total": extracted.get("total"),
                    "currency": extracted.get("currency") or "USD",
                }
                if normalized["total"] is None:
                    assumptions.append(
                        "No total was available in mock mode, so the invoice stays unresolved and must be categorized as Other."
                    )
                if normalized["vendor"] is None:
                    assumptions.append(
                        "Vendor information is incomplete, so the filename is the only stable identifier in mock mode."
                    )

        result = {
            "invoice_id": invoice_id,
            "vendor": normalized.get("vendor"),
            "invoice_date": normalized.get("invoice_date"),
            "invoice_number": normalized.get("invoice_number"),
            "total": normalized.get("total"),
            "currency": normalized.get("currency") or "USD",
            "notes": _dedupe(notes),
            "assumptions": _dedupe(assumptions),
        }
        self._clear_invoice_outputs(invoice, clear_normalized=False)
        self._clear_run_outputs(tool_context)
        invoice["normalized"] = result
        invoice["issues"] = _dedupe(invoice.get("issues", []) + result["notes"] + result["assumptions"])
        tool_context.state["working_invoices"] = tool_context.state["working_invoices"]
        return result

    @trace_tool()
    def categorize_invoice(
        self,
        invoice_id: str,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Assign one allowed expense category to a normalized invoice and include confidence or explanatory notes."""

        invoice = self._get_invoice(tool_context, invoice_id)
        normalized = invoice.get("normalized")
        if not normalized:
            raise ValueError(f"Invoice {invoice_id} has not been normalized yet.")

        prompt = str(tool_context.state.get("run_prompt") or "").lower()
        assumptions = _ensure_list(normalized.get("assumptions"))
        extracted = invoice.get("extracted") or {}

        if self.settings.runtime.planner_mode == "live":
            suggestion = self._get_live_adapter().categorize_invoice(
                normalized_invoice={
                    "vendor": normalized.get("vendor"),
                    "invoice_date": normalized.get("invoice_date"),
                    "invoice_number": normalized.get("invoice_number"),
                    "total": normalized.get("total"),
                    "currency": normalized.get("currency") or "USD",
                },
                raw_category_hint=extracted.get("raw_category_hint"),
                reviewer_prompt=str(tool_context.state.get("run_prompt") or ""),
            )
            notes = _ensure_list(suggestion.notes) + _ensure_list(normalized.get("notes"))
            category = str(suggestion.category or "Other")
            confidence = float(suggestion.confidence or 0.5)
            reasoning_payload = dump_reasoning_envelope(suggestion.reasoning)
            if category not in ALLOWED_CATEGORIES:
                notes.append(
                    f"The live categorizer returned unsupported category '{category}', so it was clamped to Other."
                )
                category = "Other"
            if category == "Other" and not any("Other" in note or "other" in note for note in notes):
                notes.append(
                    "Other was selected because the invoice did not map cleanly to one of the assignment-defined categories."
                )
        else:
            fixture = self.fixture_manifest.get(invoice.get("fixture_key") or "", {})
            categorization = (fixture.get("categorization") or {}).copy()
            notes = _ensure_list(categorization.get("notes")) + _ensure_list(normalized.get("notes"))
            reasoning_payload = None

            category = categorization.get("category", "Other")
            confidence = float(categorization.get("confidence", 0.5))

            if "conservative" in prompt and confidence < 0.75:
                notes.append(
                    "Conservative mode moved this invoice to Other because the categorization confidence stayed below 0.75."
                )
                category = "Other"
                confidence = min(confidence, 0.7)
            if category == "Other" and not any("Other" in note or "other" in note for note in notes):
                notes.append(
                    "Other was selected because the invoice did not map cleanly to one of the assignment-defined categories."
                )
            if "flag unusual" in prompt and fixture.get("unusual"):
                notes.append(str(fixture["unusual"]))

            if category not in ALLOWED_CATEGORIES:
                category = "Other"
                notes.append("The mock categorizer produced an unsupported category and it was remapped to Other.")

        invoice_result = {
            "invoice_id": invoice_id,
            "filename": invoice["filename"],
            "vendor": normalized.get("vendor"),
            "invoice_date": normalized.get("invoice_date"),
            "invoice_number": normalized.get("invoice_number"),
            "total": normalized.get("total"),
            "assigned_category": category,
            "confidence": confidence,
            "notes": _dedupe(notes + assumptions),
        }
        invoice["categorized"] = {
            "invoice_id": invoice_id,
            "category": category,
            "confidence": confidence,
            "notes": _dedupe(notes),
        }
        if reasoning_payload is not None:
            invoice["categorized"]["reasoning"] = reasoning_payload
        invoice["invoice_result"] = invoice_result
        self._clear_run_outputs(tool_context)
        invoice["issues"] = _dedupe(invoice.get("issues", []) + invoice_result["notes"])
        tool_context.state["working_invoices"] = tool_context.state["working_invoices"]
        response = {
            "invoice_id": invoice_id,
            "decision": invoice["categorized"],
            "invoice_result": invoice_result,
        }
        if reasoning_payload is not None:
            response["reasoning"] = reasoning_payload
        return response

    @trace_tool()
    def aggregate_invoices(self, tool_context: ToolContext) -> dict[str, Any]:
        """Aggregate the processed invoices into run-level totals and category totals."""

        self._ensure_all_invoices_categorized(tool_context)
        invoice_results = self._ordered_invoice_results(tool_context)
        totals: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0.00"))
        grand_total = Decimal("0.00")
        for invoice in invoice_results:
            amount = Decimal(str(invoice.get("total") or 0))
            grand_total += amount
            totals[invoice["assigned_category"]] += amount

        issues = self._collect_issues(tool_context)
        summary = {
            "total_spend": _quantize(grand_total),
            "spend_by_category": {
                category: _quantize(amount) for category, amount in sorted(totals.items())
            },
            "invoice_count": len(invoice_results),
            "issues_and_assumptions": issues,
        }
        tool_context.state["run_summary"] = summary
        return summary

    @trace_tool()
    def generate_report(self, tool_context: ToolContext) -> dict[str, Any]:
        """Generate the final structured run output from the aggregated results and the per-invoice records."""

        self._ensure_all_invoices_categorized(tool_context)
        summary = tool_context.state.get("run_summary")
        if not summary:
            summary = self.aggregate_invoices(tool_context)

        report = {
            "run_summary": {
                "total_spend": summary["total_spend"],
                "spend_by_category": summary["spend_by_category"],
                "invoice_count": summary["invoice_count"],
            },
            "invoices": self._ordered_invoice_results(tool_context),
            "issues_and_assumptions": summary["issues_and_assumptions"],
        }
        tool_context.state["final_report"] = report
        return report

    def _get_invoice(self, tool_context: ToolContext, invoice_id: str) -> dict[str, Any]:
        working_invoices = tool_context.state.get("working_invoices") or {}
        if invoice_id not in working_invoices:
            raise ValueError(f"Unknown invoice id: {invoice_id}")
        return working_invoices[invoice_id]

    def _ordered_invoice_results(self, tool_context: ToolContext) -> list[dict[str, Any]]:
        working_invoices = tool_context.state.get("working_invoices") or {}
        ordered_ids = tool_context.state.get("invoice_order") or []
        results: list[dict[str, Any]] = []
        for invoice_id in ordered_ids:
            invoice = working_invoices.get(invoice_id, {})
            if invoice.get("invoice_result"):
                results.append(invoice["invoice_result"])
        return results

    def _collect_issues(self, tool_context: ToolContext) -> list[str]:
        working_invoices = tool_context.state.get("working_invoices") or {}
        issues: list[str] = []
        for invoice_id in tool_context.state.get("invoice_order") or []:
            invoice = working_invoices.get(invoice_id, {})
            for issue in _ensure_list(invoice.get("issues")):
                issues.append(f"{invoice.get('filename', invoice_id)}: {issue}")
        return _dedupe(issues)

    def _ensure_all_invoices_categorized(self, tool_context: ToolContext) -> None:
        if "invoice_refs" not in tool_context.state:
            raise ValueError(
                "The planner must call load_images before aggregate_invoices or generate_report."
            )

        pending_invoice_ids = self._pending_invoice_ids(tool_context)
        if pending_invoice_ids:
            pending_list = ", ".join(pending_invoice_ids)
            raise ValueError(
                "The planner must finish extract, normalize, and categorize for every loaded invoice "
                f"before aggregate_invoices or generate_report. Pending invoice_ids: {pending_list}."
            )

    def _pending_invoice_ids(self, tool_context: ToolContext) -> list[str]:
        working_invoices = tool_context.state.get("working_invoices") or {}
        pending: list[str] = []
        for invoice_id in tool_context.state.get("invoice_order") or []:
            invoice = working_invoices.get(invoice_id, {})
            if not invoice.get("invoice_result"):
                pending.append(invoice_id)
        return pending

    def _clear_invoice_outputs(
        self,
        invoice: dict[str, Any],
        *,
        clear_normalized: bool,
    ) -> None:
        if clear_normalized:
            invoice.pop("normalized", None)
        invoice.pop("categorized", None)
        invoice.pop("invoice_result", None)

    def _clear_run_outputs(self, tool_context: ToolContext) -> None:
        tool_context.state["run_summary"] = None
        tool_context.state["final_report"] = None

    def _get_live_adapter(self) -> GeminiInvoiceToolAdapter:
        if self.live_adapter is None:
            self.live_adapter = GeminiInvoiceToolAdapter.from_settings(self.settings)
        return self.live_adapter
