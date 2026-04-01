"""翻訳プロンプトカタログの型定義とロード処理を提供するモジュール."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import re
import tomllib
from typing import Literal

DEFAULT_PROMPT_CATALOG_PATH = Path("prompts/translation-prompts.toml")
DEFAULT_PROMPT_PROFILE = "ja_default"
CHUNK_PLACEHOLDER = "{{CHUNK_TEXT}}"
PromptMessageStyle = Literal["system_user", "user_only"]
PromptOutputStyle = Literal["tagged", "raw_text"]


@dataclass(slots=True, frozen=True)
class PromptRule:
    """モデル条件に応じたsystem prompt上書きルール.

    Attributes:
        pattern: ``provider:model`` または ``model`` に対するマッチパターン.
        system_prompt: 条件一致時に使うsystem prompt本文.
    """

    pattern: str
    system_prompt: str


@dataclass(slots=True, frozen=True)
class PromptProfile:
    """翻訳時に利用するプロンプト群.

    Attributes:
        name: プロファイル名.
        user_prompt_template: ``{{CHUNK_TEXT}}`` を含むuser promptテンプレート.
        default_system_prompt: 既定のsystem prompt本文.
        system_rules: モデル条件に応じたsystem prompt上書きルール.
        message_style: Chat Completionsへ載せるメッセージ構成.
        output_open_tag: 生成文抽出に使う開始タグ.
        output_close_tag: 生成文抽出に使う終了タグ.
        output_style: モデル出力から翻訳本文を取り出す方法.
        description: 任意の説明文.
    """

    name: str
    user_prompt_template: str
    default_system_prompt: str
    system_rules: tuple[PromptRule, ...]
    output_open_tag: str
    output_close_tag: str
    message_style: PromptMessageStyle = "system_user"
    output_style: PromptOutputStyle = "tagged"
    description: str | None = None

    def resolve_system_prompt(self, *, provider: str, model: str) -> str:
        """プロバイダとモデル名から利用するsystem promptを解決する.

        Args:
            provider: プロバイダ名.
            model: モデル名.

        Returns:
            str: ルール適用後のsystem prompt本文.
        """
        target = f"{provider}:{model}"
        for rule in self.system_rules:
            if fnmatch(target, rule.pattern) or fnmatch(model, rule.pattern):
                return rule.system_prompt
        return self.default_system_prompt

    def build_messages(
        self,
        *,
        provider: str,
        model: str,
        chunk_text: str,
    ) -> list[dict[str, str]]:
        """モデル呼び出しに渡すメッセージ列を組み立てる.

        Args:
            provider: プロバイダ名.
            model: モデル名.
            chunk_text: 翻訳対象のTeXチャンク本文.

        Returns:
            list[dict[str, str]]: Chat Completions互換のメッセージ列.
        """
        user_prompt = self.render_user_prompt(chunk_text=chunk_text)
        if self.message_style == "user_only":
            return [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        return [
            {
                "role": "system",
                "content": self.resolve_system_prompt(
                    provider=provider,
                    model=model,
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def render_user_prompt(self, *, chunk_text: str) -> str:
        """チャンク本文を埋め込んだuser promptを生成する.

        Args:
            chunk_text: 翻訳対象のTeXチャンク本文.

        Returns:
            str: ``{{CHUNK_TEXT}}`` を差し込んだ完成user prompt.

        Raises:
            ValueError: テンプレートに ``{{CHUNK_TEXT}}`` が含まれない場合.
        """
        if CHUNK_PLACEHOLDER not in self.user_prompt_template:
            raise ValueError(
                f"Prompt profile '{self.name}' does not contain {CHUNK_PLACEHOLDER}"
            )
        return self.user_prompt_template.replace(CHUNK_PLACEHOLDER, chunk_text)

    def extract_translated_text(self, content: str) -> str:
        """モデル出力から翻訳本文を抽出する.

        Args:
            content: モデルの生出力文字列.

        Returns:
            str: 翻訳本文.

        Raises:
            ValueError: 生出力が空で, 本文抽出に失敗した場合.
        """
        stripped = content.strip()
        if self.output_style == "raw_text":
            if not stripped:
                raise ValueError("Translated response is empty")
            return stripped

        pattern = re.compile(
            f"{re.escape(self.output_open_tag)}\\s*(.*?)\\s*{re.escape(self.output_close_tag)}",
            re.DOTALL,
        )
        match = pattern.search(stripped)
        if match is None:
            if not stripped:
                raise ValueError(
                    f"Model output does not contain expected block: "
                    f"{self.output_open_tag}...{self.output_close_tag}"
                )
            return stripped
        translated = match.group(1).strip()
        if not translated:
            raise ValueError("Translated block is empty")
        return translated


def load_prompt_profile(catalog_path: Path, *, profile: str) -> PromptProfile:
    """TOMLカタログから指定プロファイルを読み込む.

    Args:
        catalog_path: プロンプトカタログTOMLへのパス.
        profile: 読み込むプロファイル名.

    Returns:
        PromptProfile: 検証済みプロファイルオブジェクト.

    Raises:
        FileNotFoundError: カタログファイルが存在しない場合.
        ValueError: プロファイル定義が不正, または必須項目が不足している場合.
    """
    resolved_path = catalog_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Prompt catalog not found: {resolved_path}")

    with resolved_path.open("rb") as fh:
        data = tomllib.load(fh)

    root_profiles = _dict_section(data, "profiles", context="root")
    raw_profile = root_profiles.get(profile)
    if raw_profile is None:
        raise ValueError(f"Profile '{profile}' not found in {resolved_path}")
    if not isinstance(raw_profile, dict):
        raise ValueError(f"profiles.{profile} must be a table")

    description = _maybe_non_empty_str(raw_profile.get("description"))
    message_style = _parse_message_style(
        raw_profile.get("message_style", "system_user"),
        name=f"profiles.{profile}.message_style",
    )
    user_prompt_template = _require_non_empty_str(
        raw_profile.get("user_prompt_template"),
        name=f"profiles.{profile}.user_prompt_template",
    )
    if CHUNK_PLACEHOLDER not in user_prompt_template:
        raise ValueError(
            f"profiles.{profile}.user_prompt_template must include {CHUNK_PLACEHOLDER}"
        )
    if message_style == "system_user":
        default_system_prompt = _require_non_empty_str(
            raw_profile.get("default_system_prompt"),
            name=f"profiles.{profile}.default_system_prompt",
        )
    else:
        default_system_prompt = (
            _maybe_non_empty_str(raw_profile.get("default_system_prompt")) or ""
        )

    raw_rules = raw_profile.get("system_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"profiles.{profile}.system_rules must be an array of tables")

    rules: list[PromptRule] = []
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ValueError(
                f"profiles.{profile}.system_rules[{index}] must be a table"
            )
        pattern = _require_non_empty_str(
            raw_rule.get("pattern"),
            name=f"profiles.{profile}.system_rules[{index}].pattern",
        )
        system_prompt = _require_non_empty_str(
            raw_rule.get("system_prompt"),
            name=f"profiles.{profile}.system_rules[{index}].system_prompt",
        )
        rules.append(PromptRule(pattern=pattern, system_prompt=system_prompt))

    output_style = _parse_output_style(
        raw_profile.get("output_style", "tagged"),
        name=f"profiles.{profile}.output_style",
    )
    if output_style == "tagged":
        output_open_tag = _require_non_empty_str(
            raw_profile.get("output_open_tag", "<translated>"),
            name=f"profiles.{profile}.output_open_tag",
        )
        output_close_tag = _require_non_empty_str(
            raw_profile.get("output_close_tag", "</translated>"),
            name=f"profiles.{profile}.output_close_tag",
        )
    else:
        output_open_tag = _maybe_non_empty_str(raw_profile.get("output_open_tag")) or ""
        output_close_tag = (
            _maybe_non_empty_str(raw_profile.get("output_close_tag")) or ""
        )

    return PromptProfile(
        name=profile,
        description=description,
        user_prompt_template=user_prompt_template,
        default_system_prompt=default_system_prompt,
        system_rules=tuple(rules),
        message_style=message_style,
        output_open_tag=output_open_tag,
        output_close_tag=output_close_tag,
        output_style=output_style,
    )


def _dict_section(root: object, key: str, *, context: str) -> dict[str, object]:
    """辞書から指定キーの辞書セクションを取り出す.

    Args:
        root: ルートオブジェクト.
        key: 取り出すキー名.
        context: エラーメッセージに含める文脈名.

    Returns:
        dict[str, object]: 取り出したセクション辞書.

    Raises:
        ValueError: ルートまたは対象値が辞書でない場合.
    """
    if not isinstance(root, dict):
        raise ValueError(f"{context} must be a table")
    section = root.get(key)
    if section is None:
        raise ValueError(f"Missing required table: {key}")
    if not isinstance(section, dict):
        raise ValueError(f"{key} must be a table")
    return section


def _require_non_empty_str(value: object, *, name: str) -> str:
    """空でない文字列を要求し, 文字列を返す.

    Args:
        value: 検証対象値.
        name: エラーメッセージ用の項目名.

    Returns:
        str: 空でない文字列.

    Raises:
        ValueError: 空文字, または ``None`` の場合.
    """
    text = _maybe_non_empty_str(value)
    if text is None:
        raise ValueError(f"{name} must not be empty")
    return text


def _maybe_non_empty_str(value: object) -> str | None:
    """値を文字列化し, 空でなければ返す.

    Args:
        value: 文字列化対象値.

    Returns:
        str | None: 空でない文字列, または ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text:
        return text
    return None


def _parse_message_style(value: object, *, name: str) -> PromptMessageStyle:
    """メッセージ構成種別を検証し, 許可値として返す.

    Args:
        value: 検証対象値.
        name: エラーメッセージ用の項目名.

    Returns:
        PromptMessageStyle: 許可済みメッセージ構成種別.

    Raises:
        ValueError: 許可外の値が指定された場合.
    """
    text = _require_non_empty_str(value, name=name)
    if text not in {"system_user", "user_only"}:
        raise ValueError(f"{name} must be one of: system_user, user_only")
    return text


def _parse_output_style(value: object, *, name: str) -> PromptOutputStyle:
    """翻訳本文の抽出方式を検証し, 許可値として返す.

    Args:
        value: 検証対象値.
        name: エラーメッセージ用の項目名.

    Returns:
        PromptOutputStyle: 許可済み出力抽出方式.

    Raises:
        ValueError: 許可外の値が指定された場合.
    """
    text = _require_non_empty_str(value, name=name)
    if text not in {"tagged", "raw_text"}:
        raise ValueError(f"{name} must be one of: tagged, raw_text")
    return text
