from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables only."""

    model_config = SettingsConfigDict(
        env_prefix="INVOICE_AGENT_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "invoice-agent"
    planner_mode: str = "mock"
    live_model: str = "gemini-2.5-flash"
    max_extraction_attempts: int = 2
    traces_dir: Path = Field(default=ROOT_DIR / "artifacts" / "runs")
    fixture_manifest_path: Path = Field(
        default=ROOT_DIR / "fixtures" / "invoices" / "manifest.json"
    )

    @property
    def fixture_dir(self) -> Path:
        return self.fixture_manifest_path.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    return settings
