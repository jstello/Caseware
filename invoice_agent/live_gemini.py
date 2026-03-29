from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .reasoning import build_reasoning_envelope
from .schemas import (
    LiveCategorizationPayload,
    LiveCategorizationSuggestion,
    LiveExtractionFields,
    LiveExtractionPayload,
)
from .settings import Settings


IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _response_to_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    raise ValueError("Gemini returned no text payload for a structured tool response.")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in IMAGE_MIME_TYPES:
        raise ValueError(f"Unsupported invoice image type for live extraction: {path.suffix}")
    return IMAGE_MIME_TYPES[suffix]


def _build_extraction_contents(invoice_path: Path, prompt: str) -> list[Any]:
    if invoice_path.suffix.lower() == ".svg":
        svg_source = invoice_path.read_text(encoding="utf-8", errors="replace")
        return [
            prompt,
            (
                "The invoice file is SVG markup. Gemini does not accept image/svg+xml directly, "
                "so parse this SVG source as the invoice document.\n"
                f"```svg\n{svg_source}\n```"
            ),
        ]
    return [
        types.Part.from_bytes(
            data=invoice_path.read_bytes(),
            mime_type=_guess_mime_type(invoice_path),
        ),
        prompt,
    ]


def _coerce_float(value: float | None, *, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


class GeminiInvoiceToolAdapter:
    """Live Gemini-backed helpers for extraction and categorization tools."""

    def __init__(
        self,
        *,
        client: genai.Client,
        model: str,
        extraction_prompt_template: str,
        categorization_prompt_template: str,
        allowed_categories: list[str],
    ) -> None:
        self.client = client
        self.model = model
        self.extraction_prompt_template = extraction_prompt_template
        self.categorization_prompt_template = categorization_prompt_template
        self.allowed_categories = allowed_categories

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiInvoiceToolAdapter":
        return cls(
            client=_build_client(),
            model=settings.runtime.live_model,
            extraction_prompt_template=settings.agent.live_extraction_prompt_template,
            categorization_prompt_template=settings.agent.live_categorization_prompt_template,
            allowed_categories=list(settings.agent.allowed_categories),
        )

    def extract_invoice_fields(
        self,
        *,
        invoice_path: Path,
        reviewer_prompt: str | None,
        focus_hint: str | None,
    ) -> LiveExtractionFields:
        prompt = self.extraction_prompt_template.format(
            reviewer_prompt=reviewer_prompt or "none",
            focus_hint=focus_hint or "none",
            filename=invoice_path.name,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=_build_extraction_contents(invoice_path, prompt),
            config={
                "temperature": 0.0,
                "thinking_config": types.ThinkingConfig(include_thoughts=True),
                "response_mime_type": "application/json",
                "response_json_schema": LiveExtractionPayload.model_json_schema(),
            },
        )
        parsed = LiveExtractionPayload.model_validate_json(
            _strip_code_fences(_response_to_text(response))
        )
        reasoning = build_reasoning_envelope(
            source="extract_invoice_fields",
            parts=getattr(response, "parts", None),
            usage_metadata=getattr(response, "usage_metadata", None),
        )
        return LiveExtractionFields.model_validate(
            {
                **parsed.model_dump(mode="json"),
                "reasoning": reasoning.model_dump(mode="json") if reasoning is not None else None,
            }
        )

    def categorize_invoice(
        self,
        *,
        normalized_invoice: dict[str, Any],
        raw_category_hint: str | None,
        reviewer_prompt: str | None,
    ) -> LiveCategorizationSuggestion:
        prompt = self.categorization_prompt_template.format(
            reviewer_prompt=reviewer_prompt or "none",
            allowed_categories=", ".join(self.allowed_categories),
            raw_category_hint=raw_category_hint or "none",
            normalized_invoice=json.dumps(normalized_invoice, indent=2, sort_keys=True),
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.0,
                "thinking_config": types.ThinkingConfig(include_thoughts=True),
                "response_mime_type": "application/json",
                "response_json_schema": LiveCategorizationPayload.model_json_schema(),
            },
        )
        parsed = LiveCategorizationPayload.model_validate_json(
            _strip_code_fences(_response_to_text(response))
        )
        reasoning = build_reasoning_envelope(
            source="categorize_invoice",
            parts=getattr(response, "parts", None),
            usage_metadata=getattr(response, "usage_metadata", None),
        )
        suggestion = LiveCategorizationSuggestion.model_validate(
            {
                **parsed.model_dump(mode="json"),
                "reasoning": reasoning.model_dump(mode="json") if reasoning is not None else None,
            }
        )
        suggestion.confidence = _coerce_float(suggestion.confidence, default=0.5)
        suggestion.notes = [str(note) for note in suggestion.notes]
        return suggestion


def _build_client() -> genai.Client:
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
