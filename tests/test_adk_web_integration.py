from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
from mlflow.tracking import MlflowClient

from invoice_agent.settings import get_settings


ROOT_DIR = Path(__file__).resolve().parents[1]


def _reload_invoice_agent_for_mock_mode(monkeypatch) -> None:
    monkeypatch.setenv("INVOICE_AGENT_PLANNER_MODE", "mock")
    get_settings.cache_clear()

    import invoice_agent as invoice_agent_package
    import invoice_agent.adk_app as invoice_agent_adk_app
    import invoice_agent.agent as invoice_agent_agent

    importlib.reload(invoice_agent_agent)
    importlib.reload(invoice_agent_adk_app)
    importlib.reload(invoice_agent_package)


def test_adk_web_lists_invoice_agent_and_runs_mock_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(ROOT_DIR)
    monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "1")
    monkeypatch.setenv("INVOICE_AGENT_MLFLOW_ENABLED", "true")
    monkeypatch.setenv("INVOICE_AGENT_TRACES_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("INVOICE_AGENT_MLFLOW_TRACKING_DIR", str(tmp_path / "mlflow"))
    _reload_invoice_agent_for_mock_mode(monkeypatch)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    app = get_fast_api_app(agents_dir=".", web=True)
    with TestClient(app) as client:
        list_response = client.get("/list-apps?detailed=true")
        assert list_response.status_code == 200
        apps = list_response.json()["apps"]
        invoice_app = next(app for app in apps if app["name"] == "invoice_agent")
        assert invoice_app["rootAgentName"] == "invoice_agent"

        session_response = client.post(
            "/apps/invoice_agent/users/user/sessions",
            json={"state": {}},
        )
        assert session_response.status_code == 200
        session_id = session_response.json()["id"]

        run_response = client.post(
            "/run_sse",
            json={
                "app_name": "invoice_agent",
                "user_id": "user",
                "session_id": session_id,
                "streaming": True,
                "new_message": {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Use conservative categorization and flag unusual invoices.",
                        }
                    ],
                },
            },
        )
        assert run_response.status_code == 200

        sse_events = [
            json.loads(line.removeprefix("data: "))
            for line in run_response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert sse_events
        assert not any("error" in event for event in sse_events)
        assert "load_images" in run_response.text
        assert "generate_report" in run_response.text
        assert "I completed the run across 6 invoices." in run_response.text

    trace_runs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert trace_runs
    latest_run = max(trace_runs, key=lambda path: path.stat().st_mtime)
    assert (latest_run / "trace.jsonl").exists()
    assert (latest_run / "sse.jsonl").exists()
    assert (latest_run / "final_report.json").exists()

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').resolve()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("invoice-agent")
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    assert runs
    trace_artifacts = client.list_artifacts(runs[0].info.run_id, "traces")
    assert {artifact.path for artifact in trace_artifacts} >= {
        "traces/sse.jsonl",
        "traces/trace.jsonl",
    }
    output_artifacts = client.list_artifacts(runs[0].info.run_id, "outputs")
    assert {artifact.path for artifact in output_artifacts} >= {
        "outputs/final_report.json",
    }
