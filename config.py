from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model id (see https://openrouter.ai/models)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible API base URL",
    )

    @field_validator("openrouter_model", "openrouter_base_url", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — re-read env only after process restart."""
    return Settings()
