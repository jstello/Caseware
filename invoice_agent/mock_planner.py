from __future__ import annotations

from collections import defaultdict
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def _function_call_part(tool_name: str, args: dict[str, Any], call_id: str) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name=tool_name,
            args=args,
        )
    )


def _response(text: str, tool_name: str | None = None, args: dict[str, Any] | None = None) -> LlmResponse:
    parts = [types.Part(text=text)]
    if tool_name is not None:
        parts.append(
            _function_call_part(
                tool_name=tool_name,
                args=args or {},
                call_id=f"{tool_name}-{abs(hash((tool_name, repr(args or {})))) % 100000}",
            )
        )
    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
    )


class MockPlannerLlm(BaseLlm):
    """Deterministic mock planner that still makes tool choices from intermediate results."""

    model: str = "mock-invoice-planner"

    async def generate_content_async(self, llm_request, stream: bool = False):
        responses_by_tool: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for content in llm_request.contents:
            if not content or not content.parts:
                continue
            for part in content.parts:
                if part.function_response:
                    responses_by_tool[part.function_response.name].append(
                        dict(part.function_response.response or {})
                    )

        load_response = responses_by_tool["load_images"][-1] if responses_by_tool["load_images"] else None
        report_response = (
            responses_by_tool["generate_report"][-1]
            if responses_by_tool["generate_report"]
            else None
        )
        aggregate_response = (
            responses_by_tool["aggregate_invoices"][-1]
            if responses_by_tool["aggregate_invoices"]
            else None
        )

        if report_response:
            invoice_count = len(report_response.get("invoices", []))
            total = report_response.get("run_summary", {}).get("total_spend")
            yield _response(
                f"I completed the run across {invoice_count} invoices. The final report is ready and totals {total}."
            )
            return

        if not load_response:
            yield _response(
                "I need to discover the local invoice images before I can inspect any invoice details.",
                "load_images",
                {},
            )
            return

        invoice_refs = load_response.get("invoice_refs", [])
        if not invoice_refs:
            if not aggregate_response:
                yield _response(
                    "There are no invoice files in this run, so I will aggregate an empty result set.",
                    "aggregate_invoices",
                    {},
                )
                return
            yield _response(
                "With the empty aggregate prepared, I can still generate the final report.",
                "generate_report",
                {},
            )
            return

        extracted_by_invoice = {
            response["invoice_id"]: response
            for response in responses_by_tool["extract_invoice_fields"]
            if response.get("invoice_id")
        }
        normalized_by_invoice = {
            response["invoice_id"]: response
            for response in responses_by_tool["normalize_invoice"]
            if response.get("invoice_id")
        }
        categorized_by_invoice = {}
        for response in responses_by_tool["categorize_invoice"]:
            invoice_result = response.get("invoice_result") or {}
            invoice_id = response.get("invoice_id") or invoice_result.get("invoice_id")
            if invoice_id:
                categorized_by_invoice[invoice_id] = response

        for invoice_ref in invoice_refs:
            invoice_id = invoice_ref["invoice_id"]
            extracted = extracted_by_invoice.get(invoice_id)
            normalized = normalized_by_invoice.get(invoice_id)
            categorized = categorized_by_invoice.get(invoice_id)

            if extracted is None:
                yield _response(
                    f"I have loaded {invoice_ref['filename']}. I will extract its fields before making any assumptions.",
                    "extract_invoice_fields",
                    {"invoice_id": invoice_id},
                )
                return

            if extracted.get("needs_retry") and normalized is None:
                yield _response(
                    f"{invoice_ref['filename']} is still missing critical fields, so I will retry extraction with a focused hint.",
                    "extract_invoice_fields",
                    {
                        "invoice_id": invoice_id,
                        "focus_hint": extracted.get("retry_focus_hint") or "missing critical fields",
                    },
                )
                return

            if normalized is None:
                yield _response(
                    f"I have enough information about {invoice_ref['filename']} to normalize the invoice record next.",
                    "normalize_invoice",
                    {"invoice_id": invoice_id},
                )
                return

            if categorized is None:
                yield _response(
                    f"The normalized record for {invoice_ref['filename']} is ready, so I will categorize it against the allowed expense taxonomy.",
                    "categorize_invoice",
                    {"invoice_id": invoice_id},
                )
                return

        if aggregate_response is None:
            yield _response(
                "Every invoice now has a category assignment, so I can aggregate the run-level totals.",
                "aggregate_invoices",
                {},
            )
            return

        yield _response(
            "The totals are ready, and the last step is to assemble the final structured report.",
            "generate_report",
            {},
        )
