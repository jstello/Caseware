from __future__ import annotations

from dataclasses import dataclass

from google.adk.agents import LlmAgent

from .mock_planner import MockPlannerLlm
from .settings import Settings
from .tools import InvoiceToolRegistry


@dataclass(frozen=True)
class AgentPatternSummary:
    """Short, testable description of the invoice agent layout."""

    orchestrator: str
    sequential: str
    parallel: str
    loop: str


INVOICE_AGENT_PATTERN = AgentPatternSummary(
    orchestrator="Single planner-driven LlmAgent",
    sequential=(
        "SequentialAgent is reserved for future coarse phases only; the current run keeps the real decision loop in the planner."
    ),
    parallel=(
        "ParallelAgent is intentionally deferred because invoice processing shares run state and reviewer-facing trace ordering matters."
    ),
    loop=(
        "Loop-style behavior fits bounded retry, but the root planner remains the decision-maker so the runtime does not collapse into a fixed graph."
    ),
)


def build_invoice_agent(
    settings: Settings,
    invoice_tools: InvoiceToolRegistry,
) -> LlmAgent:
    """Build the root invoice agent from the effective YAML-backed config."""

    model = (
        settings.runtime.live_model
        if settings.runtime.planner_mode == "live"
        else MockPlannerLlm()
    )
    return LlmAgent(
        name=settings.agent.name,
        model=model,
        description=settings.agent.description,
        instruction=settings.agent.system_instruction,
        tools=invoice_tools.tool_functions(),
    )


def build_request_prompt(settings: Settings, prompt: str | None) -> str:
    reviewer_prompt = prompt or "none"
    return settings.agent.request_prompt_template.format(prompt=reviewer_prompt)


def describe_invoice_agent_pattern() -> AgentPatternSummary:
    """Return the recommended layout summary for docs and tests."""

    return INVOICE_AGENT_PATTERN
