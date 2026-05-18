from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SearchProvider = Literal["tavily", "mock", "ddg"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-sonnet-4-6")

    search_provider: SearchProvider = Field(default="mock")
    tavily_api_key: str = Field(default="")

    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="research-agent")

    database_url: str = Field(default="postgresql://agent:agent@postgres:5432/agent")

    max_sub_questions: int = Field(default=4, ge=1, le=10)
    max_iterations: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    search_results_per_query: int = Field(default=5, ge=1, le=20)
    fetch_top_n: int = Field(default=3, ge=1, le=10)
    circuit_breaker_failures: int = Field(default=3, ge=1, le=20)

    log_level: str = Field(default="INFO")

    @property
    def effective_search_provider(self) -> SearchProvider:
        """Fall back to mock if user picked tavily but didn't supply a key."""
        if self.search_provider == "tavily" and not self.tavily_api_key:
            return "mock"
        return self.search_provider

    @property
    def langsmith_enabled(self) -> bool:
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
