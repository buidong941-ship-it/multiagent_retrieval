"""
Base configuration module using Pydantic Settings for validation
and environment variable support.

Design Decision:
    - Use Pydantic BaseSettings so every config can be overridden via
      environment variables or a .env file.
    - Use dataclass-like frozen models to prevent accidental mutation.
    - All configs inherit from BaseConfig for consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class BaseConfig(BaseSettings):
    """Root configuration class.

    All module configs should inherit from this class to get
    automatic environment variable loading and validation.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
