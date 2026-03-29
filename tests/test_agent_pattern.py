from __future__ import annotations

from invoice_agent.agent import (
    build_invoice_agent,
    describe_invoice_agent_pattern,
)
from invoice_agent.settings import Settings
from invoice_agent.tools import InvoiceToolRegistry


def test_invoice_agent_builder_uses_yaml_backed_prompt_configuration() -> None:
    settings = Settings(planner_mode="mock")
    tools = InvoiceToolRegistry(settings)

    agent = build_invoice_agent(settings, tools)

    assert agent.name == settings.agent.name
    assert agent.description == settings.agent.description
    assert agent.instruction == settings.agent.system_instruction
    assert [
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in agent.tools
    ] == [
        "load_images",
        "extract_invoice_fields",
        "normalize_invoice",
        "categorize_invoice",
        "aggregate_invoices",
        "generate_report",
    ]


def test_invoice_agent_pattern_prefers_planner_over_parallel_fanout() -> None:
    summary = describe_invoice_agent_pattern()

    assert summary.orchestrator == "Single planner-driven LlmAgent"
    assert "SequentialAgent" in summary.sequential
    assert "ParallelAgent" in summary.parallel
    assert "retry" in summary.loop.lower() or "planner" in summary.loop.lower()
