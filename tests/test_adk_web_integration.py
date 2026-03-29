from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from google.adk.cli.fast_api import get_fast_api_app


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_adk_web_lists_invoice_agent_and_runs_mock_session(monkeypatch) -> None:
    monkeypatch.chdir(ROOT_DIR)
    monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "1")
    monkeypatch.delenv("INVOICE_AGENT_PLANNER_MODE", raising=False)
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
