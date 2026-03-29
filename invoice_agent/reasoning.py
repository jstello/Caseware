from __future__ import annotations

from typing import Any, Iterable

from .schemas import ReasoningEnvelope, ReasoningSource


def build_reasoning_envelope(
    *,
    source: ReasoningSource,
    parts: Iterable[Any] | None,
    usage_metadata: Any | None = None,
) -> ReasoningEnvelope | None:
    """Extract SDK-exposed thought summaries and token counts from a model response."""

    summaries: list[str] = []
    has_thought_signature = False

    for part in parts or []:
        if getattr(part, "thought", False):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                summaries.append(text.strip())
        signature = getattr(part, "thought_signature", None)
        if signature not in (None, b""):
            has_thought_signature = True

    thoughts_token_count = _coerce_optional_int(
        getattr(usage_metadata, "thoughts_token_count", None)
    )
    total_token_count = _coerce_optional_int(
        getattr(usage_metadata, "total_token_count", None)
    )

    if (
        not summaries
        and thoughts_token_count is None
        and total_token_count is None
        and not has_thought_signature
    ):
        return None

    return ReasoningEnvelope(
        summaries=summaries,
        summary_count=len(summaries),
        thoughts_token_count=thoughts_token_count,
        total_token_count=total_token_count,
        has_thought_signature=has_thought_signature,
        source=source,
    )


def dump_reasoning_envelope(reasoning: ReasoningEnvelope | None) -> dict[str, Any] | None:
    if reasoning is None:
        return None
    return reasoning.model_dump(mode="json")


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
