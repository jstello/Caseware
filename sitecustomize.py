from __future__ import annotations

import os
from pathlib import Path


def _default_mlflow_sqlite_uri() -> str:
    root_dir = Path(__file__).resolve().parent
    database_path = (root_dir / "artifacts" / "mlflow" / "mlflow.db").resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path}"


_DEFAULT_TRACKING_URI = _default_mlflow_sqlite_uri()

# Keep plain `uv run mlflow ui` aligned with the app's SQLite tracking backend.
os.environ.setdefault("MLFLOW_BACKEND_STORE_URI", _DEFAULT_TRACKING_URI)
os.environ.setdefault("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
