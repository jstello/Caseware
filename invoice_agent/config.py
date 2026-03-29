from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parents[1]


class RuntimeConfig(BaseModel):
    app_name: str = "invoice-agent"
    planner_mode: str = "mock"
    live_model: str = "gemini-2.5-flash"
    max_extraction_attempts: int = 2
    traces_dir: Path = Field(default=ROOT_DIR / "artifacts" / "runs")
    mlflow_tracking_dir: Path = Field(default=ROOT_DIR / "artifacts" / "mlflow")
    fixture_manifest_path: Path = Field(
        default=ROOT_DIR / "fixtures" / "invoices" / "manifest.json"
    )


class AgentConfig(BaseModel):
    name: str = "invoice_agent"
    description: str = "Processes invoice images using a constrained tool registry."
    system_instruction: str
    request_prompt_template: str
    live_extraction_prompt_template: str
    live_categorization_prompt_template: str
    allowed_categories: list[str]
    tool_stages: dict[str, str]


class TracingConfig(BaseModel):
    enabled: bool = True
    experiment_name: str = "invoice-agent"
    run_name_prefix: str | None = None
    tracking_uri: str | None = None
    enable_async_logging: bool = True
    log_config_artifact: bool = True
    log_trace_artifacts: bool = True
    log_prompt_artifacts: bool = True
    tags: dict[str, str] = Field(default_factory=dict)


class InvoiceAgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    runtime: RuntimeConfig
    agent: AgentConfig
    tracing: TracingConfig = Field(default_factory=TracingConfig)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def _resolve_model_paths(config: InvoiceAgentConfig) -> InvoiceAgentConfig:
    runtime = config.runtime.model_copy(
        update={
            "traces_dir": _resolve_path(config.runtime.traces_dir),
            "mlflow_tracking_dir": _resolve_path(config.runtime.mlflow_tracking_dir),
            "fixture_manifest_path": _resolve_path(config.runtime.fixture_manifest_path),
        }
    )
    return config.model_copy(update={"runtime": runtime})


@lru_cache(maxsize=4)
def load_invoice_agent_config(config_path: str | Path) -> InvoiceAgentConfig:
    resolved_path = _resolve_path(config_path)
    raw: dict[str, Any] = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    config = InvoiceAgentConfig.model_validate(raw)
    return _resolve_model_paths(config)


def _normalize_mlflow_param(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = yaml.safe_dump(value, sort_keys=False)
    else:
        text = str(value)

    if len(text) <= 240:
        return text

    digest = sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text[:220]}... [sha256:{digest}]"


def flatten_mlflow_params(config: InvoiceAgentConfig) -> dict[str, str]:
    """Flatten nested config into MLflow-friendly string parameters."""

    flattened: dict[str, str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else key, nested)
            return
        flattened[prefix] = _normalize_mlflow_param(value)

    visit("", config.model_dump(mode="json"))
    return flattened


def config_to_artifact_text(config: InvoiceAgentConfig) -> str:
    """Serialize the effective config as stable YAML for artifact logging."""

    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
