from __future__ import annotations

import pytest

from invoice_agent.fixtures import load_fixture_manifest
from invoice_agent.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_test_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("INVOICE_AGENT_MLFLOW_ENABLED", "false")
    monkeypatch.setenv("INVOICE_AGENT_TRACES_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("INVOICE_AGENT_MLFLOW_TRACKING_DIR", str(tmp_path / "mlflow"))
    get_settings.cache_clear()
    load_fixture_manifest.cache_clear()
    yield
    get_settings.cache_clear()
    load_fixture_manifest.cache_clear()
