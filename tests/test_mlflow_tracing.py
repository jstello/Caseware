from __future__ import annotations

import asyncio
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from invoice_agent.config import (
    ROOT_DIR,
    AgentConfig,
    InvoiceAgentConfig,
    RuntimeConfig,
    TracingConfig,
    config_to_artifact_text,
    load_invoice_agent_config,
)
from invoice_agent.service import InvoiceAgentService
from invoice_agent.settings import Settings
from invoice_agent.trace import MlflowRunRecorder, TraceWriter


FIXTURE_DIR = Path("/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices")


def test_mlflow_recorder_logs_to_local_sqlite_store(tmp_path: Path) -> None:
    config_path = tmp_path / "invoice_agent.yaml"
    config_path.write_text(
        """
runtime:
  app_name: invoice-agent
  planner_mode: mock
  live_model: gemini-2.5-flash
  max_extraction_attempts: 2
  traces_dir: artifacts/runs
  mlflow_tracking_dir: artifacts/mlflow
  fixture_manifest_path: fixtures/invoices/manifest.json
agent:
  name: invoice_agent
  description: Processes invoice images using a constrained tool registry.
  system_instruction: |
    You are a local invoice-processing agent.
  request_prompt_template: |
    Optional reviewer prompt: {prompt}.
  allowed_categories:
    - Travel
    - Other
  tool_stages:
    load_images: loading
    extract_invoice_fields: extraction
    normalize_invoice: normalization
    categorize_invoice: categorization
    aggregate_invoices: aggregation
    generate_report: reporting
tracing:
  enabled: true
  experiment_name: invoice-agent-test
  tracking_uri: null
  enable_async_logging: false
  log_config_artifact: true
  log_trace_artifacts: true
  log_prompt_artifacts: true
  tags:
    app: invoice-agent
""",
        encoding="utf-8",
    )
    config = InvoiceAgentConfig(
        runtime=RuntimeConfig(
            app_name="invoice-agent",
            planner_mode="mock",
            live_model="gemini-2.5-flash",
            max_extraction_attempts=2,
            traces_dir=tmp_path / "runs",
            mlflow_tracking_dir=tmp_path / "mlflow",
            fixture_manifest_path=tmp_path / "fixtures" / "manifest.json",
        ),
        agent=AgentConfig(
            name="invoice_agent",
            description="Processes invoice images using a constrained tool registry.",
            system_instruction="You are a local invoice-processing agent.",
            request_prompt_template="Optional reviewer prompt: {prompt}.",
            allowed_categories=["Travel", "Other"],
            tool_stages={
                "load_images": "loading",
                "extract_invoice_fields": "extraction",
                "normalize_invoice": "normalization",
                "categorize_invoice": "categorization",
                "aggregate_invoices": "aggregation",
                "generate_report": "reporting",
            },
        ),
        tracing=TracingConfig(
            enabled=True,
            experiment_name="invoice-agent-test",
            tracking_uri=None,
            enable_async_logging=False,
            log_config_artifact=True,
            log_trace_artifacts=True,
            log_prompt_artifacts=True,
            tags={"app": "invoice-agent"},
        ),
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    trace_writer = TraceWriter(run_dir)
    trace_writer.write_trace(kind="run_started", payload={"run_id": "run-123"})
    trace_writer.write_sse(event="run_started", data={"run_id": "run-123"})
    trace_writer.write_report(
        {
            "run_summary": {
                "total_spend": 12.5,
                "spend_by_category": {"Other": 12.5},
                "invoice_count": 1,
            },
            "invoices": [],
            "issues_and_assumptions": [],
        }
    )

    recorder = MlflowRunRecorder(
        run_id="run-123",
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        prompt="Use conservative categorization.",
    )
    recorder.start()
    recorder.finalize(
        trace_writer,
        {
            "run_summary": {
                "total_spend": 12.5,
                "spend_by_category": {"Other": 12.5},
                "invoice_count": 1,
            },
            "invoices": [],
            "issues_and_assumptions": [],
        },
    )

    client = MlflowClient(
        tracking_uri=f"sqlite:///{config.runtime.mlflow_tracking_dir / 'mlflow.db'}"
    )
    experiment = client.get_experiment_by_name("invoice-agent-test")
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    assert runs, "expected an MLflow run to be recorded"
    run = runs[0]
    assert run.data.params["runtime.app_name"] == "invoice-agent"
    assert run.data.params["agent.name"] == "invoice_agent"
    assert run.data.tags["run_id"] == "run-123"
    assert client.list_artifacts(run.info.run_id, "config")
    assert client.list_artifacts(run.info.run_id, "traces")


def test_mlflow_groups_invoice_execution_under_one_nested_trace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INVOICE_AGENT_MLFLOW_ENABLED", "true")
    base_config = load_invoice_agent_config(ROOT_DIR / "config" / "invoice_agent.yaml")
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(
                update={
                    "traces_dir": tmp_path / "runs",
                    "mlflow_tracking_dir": tmp_path / "mlflow",
                    "fixture_manifest_path": FIXTURE_DIR / "manifest.json",
                }
            ),
            "tracing": base_config.tracing.model_copy(
                update={
                    "experiment_name": "invoice-agent-nested-trace-test",
                    "enable_async_logging": False,
                }
            ),
        }
    )
    config_path = tmp_path / "invoice_agent.yaml"
    config_path.write_text(config_to_artifact_text(config), encoding="utf-8")

    settings = Settings(config_path=config_path)
    service = InvoiceAgentService(settings)

    async def run_stream_once() -> tuple[str, list[dict[str, str]]]:
        run_id = service.new_run_id()
        run_dir = service.create_run_dir(run_id)
        prepared_input = await service.prepare_folder_input(str(FIXTURE_DIR))
        events: list[dict[str, str]] = []
        async for event in service.run_stream(
            run_id=run_id,
            run_dir=run_dir,
            prepared_input=prepared_input,
            prompt="Use conservative categorization and flag unusual invoices.",
        ):
            events.append(event)
        return run_id, events

    run_id, events = asyncio.run(run_stream_once())

    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "final_result"

    tracking_uri = f"sqlite:///{config.runtime.mlflow_tracking_dir / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("invoice-agent-nested-trace-test")
    assert experiment is not None

    traces = client.search_traces(experiment_ids=[experiment.experiment_id], max_results=10)
    assert len(traces) == 1

    trace = mlflow.get_trace(traces[0].info.trace_id)
    spans_by_name = {span.name: span for span in trace.data.spans}

    root_name = f"invoice_agent_run:{run_id}"
    assert root_name in spans_by_name
    assert spans_by_name[root_name].parent_id is None
    assert spans_by_name[root_name].span_type == "AGENT"

    bright_invoice = spans_by_name["invoice:bright-stationery-noisy"]
    assert bright_invoice.parent_id == spans_by_name[root_name].span_id
    assert bright_invoice.span_type == "TASK"

    extract_decisions = [
        span
        for span in trace.data.spans
        if span.name == "planner:extract_invoice_fields"
        and span.attributes.get("invoice.id") == "bright-stationery-noisy"
    ]
    assert len(extract_decisions) == 2
    assert all(span.parent_id == bright_invoice.span_id for span in extract_decisions)

    extract_tools = [
        span
        for span in trace.data.spans
        if span.name == "extract_invoice_fields"
        and span.attributes.get("invoice.id") == "bright-stationery-noisy"
    ]
    assert len(extract_tools) == 2
    assert {span.parent_id for span in extract_tools} == {
        span.span_id for span in extract_decisions
    }
    assert all(span.attributes.get("tool.stage") == "extraction" for span in extract_tools)
    assert all(span.attributes.get("tool.call_id") for span in extract_tools)

    report_decision = spans_by_name["planner:generate_report"]
    report_tool = spans_by_name["generate_report"]
    assert report_decision.parent_id == spans_by_name[root_name].span_id
    assert report_tool.parent_id == report_decision.span_id
    assert report_tool.attributes.get("tool.stage") == "reporting"
    assert report_tool.attributes.get("tool.call_id")
