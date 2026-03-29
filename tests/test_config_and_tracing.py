from __future__ import annotations

from pathlib import Path

from invoice_agent.config import ROOT_DIR, flatten_mlflow_params, load_invoice_agent_config
from invoice_agent.settings import Settings
from invoice_agent.trace import MlflowRunRecorder, TraceWriter


class FakeMlflowConfig:
    def __init__(self, calls: list[tuple]):
        self.calls = calls

    def enable_async_logging(self) -> None:
        self.calls.append(("enable_async_logging",))


class FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.config = FakeMlflowConfig(self.calls)

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name: str | None = None) -> None:
        self.calls.append(("start_run", run_name))

    def set_tag(self, key: str, value: str) -> None:
        self.calls.append(("set_tag", key, value))

    def log_params(self, params: dict[str, str]) -> None:
        self.calls.append(("log_params", params))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifact", Path(path).name, artifact_path))

    def log_metric(self, key: str, value: float) -> None:
        self.calls.append(("log_metric", key, value))

    def end_run(self, status: str | None = None) -> None:
        self.calls.append(("end_run", status))


def test_settings_merge_yaml_and_explicit_overrides(tmp_path: Path) -> None:
    settings = Settings(
        config_path=ROOT_DIR / "config" / "invoice_agent.yaml",
        planner_mode="live",
        mlflow_enabled=False,
        traces_dir=tmp_path / "runs",
        mlflow_tracking_dir=tmp_path / "mlflow",
    )

    assert settings.runtime.planner_mode == "live"
    assert settings.tracing.enabled is False
    assert settings.runtime.traces_dir == (tmp_path / "runs").resolve()
    assert settings.runtime.mlflow_tracking_dir == (tmp_path / "mlflow").resolve()
    assert settings.agent.allowed_categories[0] == "Travel"


def test_flatten_mlflow_params_includes_prompt_and_runtime_fields() -> None:
    config = load_invoice_agent_config(ROOT_DIR / "config" / "invoice_agent.yaml")

    params = flatten_mlflow_params(config)

    assert params["runtime.app_name"] == "invoice-agent"
    assert params["agent.tool_stages.load_images"] == "loading"
    assert "Always begin by calling load_images." in params["agent.system_instruction"]


def test_mlflow_run_recorder_logs_config_prompt_and_outputs(monkeypatch, tmp_path: Path) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr("invoice_agent.trace.mlflow", fake_mlflow)

    config = load_invoice_agent_config(ROOT_DIR / "config" / "invoice_agent.yaml").model_copy(
        update={
            "runtime": load_invoice_agent_config(
                ROOT_DIR / "config" / "invoice_agent.yaml"
            ).runtime.model_copy(
                update={
                    "traces_dir": tmp_path / "runs",
                    "mlflow_tracking_dir": tmp_path / "mlflow",
                }
            ),
            "tracing": load_invoice_agent_config(
                ROOT_DIR / "config" / "invoice_agent.yaml"
            ).tracing.model_copy(update={"tracking_uri": "http://mlflow.test"}),
        }
    )
    run_dir = tmp_path / "runs" / "run-123"
    trace_writer = TraceWriter(run_dir)
    trace_writer.write_trace(kind="run_started", payload={"run_id": "run-123"})
    trace_writer.write_sse(event="run_started", data={"run_id": "run-123"})
    trace_writer.write_report(
        {
            "run_summary": {"invoice_count": 2, "total_spend": 42.5},
            "invoices": [],
            "issues_and_assumptions": [],
        }
    )

    recorder = MlflowRunRecorder(
        run_id="run-123",
        run_dir=run_dir,
        config=config,
        prompt="Use conservative categorization.",
    )
    recorder.start()
    recorder.finalize(
        trace_writer,
        {
            "run_summary": {"invoice_count": 2, "total_spend": 42.5},
            "invoices": [],
            "issues_and_assumptions": [],
        },
    )

    assert ("enable_async_logging",) in fake_mlflow.calls
    assert ("set_experiment", "invoice-agent") in fake_mlflow.calls
    assert ("start_run", "run-123") in fake_mlflow.calls
    assert any(call[0] == "log_params" for call in fake_mlflow.calls)
    assert ("log_metric", "invoice_count", 2.0) in fake_mlflow.calls
    assert ("log_metric", "total_spend", 42.5) in fake_mlflow.calls
    assert ("end_run", "FINISHED") in fake_mlflow.calls
