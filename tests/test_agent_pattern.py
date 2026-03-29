from __future__ import annotations

from google.adk.models import Gemini

from invoice_agent.agent import (
    build_invoice_agent,
    build_root_agent,
    build_request_prompt,
    describe_invoice_agent_pattern,
    root_agent,
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


def test_request_prompt_includes_allowed_categories_and_completion_guardrail() -> None:
    settings = Settings(planner_mode="mock")

    prompt = build_request_prompt(settings, "Use conservative categorization.")

    assert "Allowed final categories:" in prompt
    assert "Travel" in prompt
    assert "generate_report" in prompt
    assert "Do not finalize early." in prompt
    assert "Use conservative categorization." in prompt


def test_live_invoice_agent_uses_gemini_with_low_temperature() -> None:
    settings = Settings(planner_mode="live")
    tools = InvoiceToolRegistry(settings)

    agent = build_invoice_agent(settings, tools)

    assert isinstance(agent.model, Gemini)
    assert agent.model.model == settings.runtime.live_model
    assert agent.generate_content_config is not None
    assert agent.generate_content_config.temperature == 0.0
    assert agent.generate_content_config.thinking_config is not None
    assert agent.generate_content_config.thinking_config.include_thoughts is True


def test_root_agent_is_exported_for_adk_web_loading() -> None:
    built_root = build_root_agent(Settings(planner_mode="mock"))

    assert root_agent.name == "invoice_agent"
    assert built_root.name == "invoice_agent"
    assert [
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in root_agent.tools
    ] == [
        "load_images",
        "extract_invoice_fields",
        "normalize_invoice",
        "categorize_invoice",
        "aggregate_invoices",
        "generate_report",
    ]
