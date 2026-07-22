"""
Project Kalachakra — Configuration Management

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global project configuration."""

    model_config = SettingsConfigDict(
        env_prefix="KALACHAKRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Project ---
    project_name: str = "Project Kalachakra"
    version: str = "0.1.0"
    debug: bool = False

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    experiments_dir: Path = Field(default=Path("experiments"))
    configs_dir: Path = Field(default=Path("configs"))

    # --- Astronomy ---
    ephemeris_path: Path | None = None
    ayanamsha: str = "lahiri"
    default_house_system: str = "P"  # Placidus

    # --- ML ---
    random_seed: int = 42
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100

    # --- Experiment Tracking ---
    mlflow_tracking_uri: str = "mlruns"
    wandb_project: str = "kalachakra"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Databases ---
    postgres_url: str = "postgresql://localhost:5432/kalachakra"
    neo4j_url: str = "bolt://localhost:7687"
    redis_url: str = "redis://localhost:6379"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        for d in [self.data_dir, self.models_dir, self.experiments_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Global singleton
settings = Settings()
