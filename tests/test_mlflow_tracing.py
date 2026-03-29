from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from invoice_agent.config import AgentConfig, InvoiceAgentConfig, RuntimeConfig, TracingConfig
from invoice_agent.trace import MlflowRunRecorder, TraceWriter


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
