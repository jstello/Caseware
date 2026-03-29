from __future__ import annotations

from functools import lru_cache
from functools import cached_property
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import (
    AgentConfig,
    InvoiceAgentConfig,
    RuntimeConfig,
    TracingConfig,
    load_invoice_agent_config,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings combining env overrides with YAML-backed config."""

    model_config = SettingsConfigDict(
        env_prefix="INVOICE_AGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    config_path: Path = Field(default=ROOT_DIR / "config" / "invoice_agent.yaml")
    planner_mode: str | None = None
    live_model: str | None = None
    max_extraction_attempts: int | None = None
    traces_dir: Path | None = None
    mlflow_tracking_dir: Path | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_enabled: bool | None = None

    @cached_property
    def config(self) -> InvoiceAgentConfig:
        config = load_invoice_agent_config(self.config_path)
        runtime_updates: dict[str, object] = {}
        tracing_updates: dict[str, object] = {}

        if self.planner_mode is not None:
            runtime_updates["planner_mode"] = self.planner_mode
        if self.live_model is not None:
            runtime_updates["live_model"] = self.live_model
        if self.max_extraction_attempts is not None:
            runtime_updates["max_extraction_attempts"] = self.max_extraction_attempts
        if self.traces_dir is not None:
            runtime_updates["traces_dir"] = self.traces_dir
        if self.mlflow_tracking_dir is not None:
            runtime_updates["mlflow_tracking_dir"] = self.mlflow_tracking_dir
        if self.mlflow_tracking_uri is not None:
            tracing_updates["tracking_uri"] = self.mlflow_tracking_uri
        if self.mlflow_experiment_name is not None:
            tracing_updates["experiment_name"] = self.mlflow_experiment_name
        if self.mlflow_enabled is not None:
            tracing_updates["enabled"] = self.mlflow_enabled

        if runtime_updates:
            config = config.model_copy(
                update={"runtime": config.runtime.model_copy(update=runtime_updates)}
            )
        if tracing_updates:
            config = config.model_copy(
                update={"tracing": config.tracing.model_copy(update=tracing_updates)}
            )
        return config

    @property
    def app_name(self) -> str:
        return self.config.runtime.app_name

    @property
    def fixture_manifest_path(self) -> Path:
        return self.config.runtime.fixture_manifest_path

    @property
    def runtime(self) -> RuntimeConfig:
        return self.config.runtime

    @property
    def agent(self) -> AgentConfig:
        return self.config.agent

    @property
    def tracing(self) -> TracingConfig:
        return self.config.tracing

    @property
    def fixture_dir(self) -> Path:
        return self.fixture_manifest_path.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.runtime.traces_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime.mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
    return settings
