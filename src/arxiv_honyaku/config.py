"""configを読み込む."""
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

BASE_CONFIG_PATH = Path(__file__).with_name("config.base.toml")
OVERRIDE_CONFIG_PATH = Path("config.toml")


class LLMConfig(BaseModel):
    """LLMの設定."""

    client: dict[str, Any]
    generation: dict[str, Any]
    max_concurrency: int = Field(default=1, ge=1)
    translation_logic: str = "general_chat"


class RunConfig(BaseModel):
    """実行時の設定."""

    runs_dir: Path
    texlive_versions: list[int]
    japanese_font_modes: list[Literal["compat", "paper-like"]] = Field(
        default_factory=lambda: ["paper-like"],
        min_length=1,
    )
    japanese_layout_modes: list[Literal["preserve", "adaptive", "safe"]] = Field(
        default_factory=lambda: ["safe"],
        min_length=1,
    )


class Config(BaseSettings):
    """アプリケーション設定."""

    llm: LLMConfig
    run: RunConfig

    model_config = SettingsConfigDict(
        toml_file=[BASE_CONFIG_PATH, OVERRIDE_CONFIG_PATH],
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """TOML設定だけを読み込み, 後のファイルで前のファイルを上書きする."""
        return (TomlConfigSettingsSource(settings_cls, deep_merge=True),)


def load_config() -> Config:
    """configを読み込む."""
    return Config()
