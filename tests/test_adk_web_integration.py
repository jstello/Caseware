from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.apps.app import App
from mlflow.tracking import MlflowClient

from invoice_agent.settings import get_settings


ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT_DIR / "fixtures" / "invoices"


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


def test_agent_loader_prefers_invoice_agent_app_with_plugins(monkeypatch) -> None:
    monkeypatch.chdir(ROOT_DIR)
    monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "1")
    _reload_invoice_agent_for_mock_mode(monkeypatch)

    loaded = AgentLoader(".").load_agent("invoice_agent")

    assert isinstance(loaded, App)
    assert loaded.name == "invoice_agent"
    assert loaded.plugins


def test_adk_web_inline_upload_uses_uploaded_invoice_instead_of_fixture_fallback(
    monkeypatch, tmp_path: Path
) -> None:
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

    uploaded_invoice = FIXTURE_DIR / "acme-air-travel-001.svg"
    inline_data = base64.b64encode(uploaded_invoice.read_bytes()).decode("ascii")

    app = get_fast_api_app(agents_dir=".", web=True)
    with TestClient(app) as client:
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
                            "text": "Please process this uploaded invoice conservatively.",
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/svg+xml",
                                "displayName": "acme-air-travel-001.svg",
                                "data": inline_data,
                            }
                        },
                    ],
                },
            },
        )
        assert run_response.status_code == 200
        assert "I completed the run across 1 invoices." in run_response.text

    trace_runs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert trace_runs
    latest_run = max(trace_runs, key=lambda path: path.stat().st_mtime)
    assert (latest_run / "uploads" / "adk_web" / "acme-air-travel-001.svg").exists()

    trace_entries = [
        json.loads(line)
        for line in (latest_run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_started = next(entry for entry in trace_entries if entry["kind"] == "run_started")
    assert run_started["input_source"]["source_type"] == "upload_dir"
    load_images_trace = next(
        entry
        for entry in trace_entries
        if entry["kind"] == "tool_result" and entry["tool_name"] == "load_images"
    )
    assert load_images_trace["result"]["invoice_count"] == 1
    assert any(
        "uploaded invoice files from the current session" in note.lower()
        for note in load_images_trace["result"]["notes"]
    )
    assert not any(
        "bundled fixture directory" in note.lower()
        for note in load_images_trace["result"]["notes"]
    )
    assert (latest_run / "input_source_debug.jsonl").exists()
    assert (latest_run / "load_images_debug.json").exists()


def test_adk_web_inline_upload_overrides_stale_session_input_source(
    monkeypatch, tmp_path: Path
) -> None:
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

    uploaded_invoice = FIXTURE_DIR / "acme-air-travel-001.svg"
    inline_data = base64.b64encode(uploaded_invoice.read_bytes()).decode("ascii")

    app = get_fast_api_app(agents_dir=".", web=True)
    with TestClient(app) as client:
        session_response = client.post(
            "/apps/invoice_agent/users/user/sessions",
            json={
                "state": {
                    "input_source": {
                        "source_type": "folder",
                        "path": str(FIXTURE_DIR),
                    }
                }
            },
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
                            "text": "Please process only this uploaded invoice.",
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/svg+xml",
                                "displayName": "acme-air-travel-001.svg",
                                "data": inline_data,
                            }
                        },
                    ],
                },
            },
        )
        assert run_response.status_code == 200
        assert "I completed the run across 1 invoices." in run_response.text

    trace_runs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert trace_runs
    latest_run = max(trace_runs, key=lambda path: path.stat().st_mtime)
    trace_entries = [
        json.loads(line)
        for line in (latest_run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_started = next(entry for entry in trace_entries if entry["kind"] == "run_started")
    assert run_started["input_source"]["source_type"] == "upload_dir"
    assert run_started["input_source"]["path"].endswith("/uploads/adk_web")

    debug_entries = [
        json.loads(line)
        for line in (latest_run / "input_source_debug.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["phase"] == "before_run" for entry in debug_entries)
    before_run_entry = next(entry for entry in debug_entries if entry["phase"] == "before_run")
    assert before_run_entry["uploaded_source"]["source_type"] == "upload_dir"
