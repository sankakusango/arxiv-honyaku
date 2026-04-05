"""アプリケーション設定を読み込み, 検証し, 実行可能な型へ正規化するモジュール."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import tomllib

from arxiv_honyaku.prompts import DEFAULT_PROMPT_CATALOG_PATH, DEFAULT_PROMPT_PROFILE

DEFAULT_CONFIG_PATH = Path("arxiv-honyaku.toml")
ProviderName = Literal["openai", "deepseek", "vllm", "ollama"]
JapaneseLayoutMode = Literal["preserve", "adaptive", "safe"]
JapaneseFontMode = Literal["compat", "paper-like"]


@dataclass(slots=True, frozen=True)
class LLMSettings:
    """LLM接続設定を保持するデータクラス.

    Attributes:
        provider: 利用するLLMプロバイダ名.
        model: 推論に使うモデル名.
        base_url: OpenAI互換APIのベースURL, 既定URL利用時は ``None``.
        api_key: APIキー, 未設定時は ``None``.
        retry_temperatures: チャンク再試行ごとに使う temperature 列.
    """

    provider: ProviderName
    model: str
    base_url: str | None
    api_key: str | None
    retry_temperatures: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class RunSettings:
    """翻訳実行パラメータを保持するデータクラス.

    Attributes:
        workspace_root: 実行結果を保存する作業ルート.
        max_chunk_chars: TeX分割時の1チャンク最大文字数.
        concurrency: チャンク翻訳の並列数.
        max_retries: 失敗チャンクの最大再試行回数.
        translate_section_titles: ``False`` の場合, ``\\section`` 系タイトルを原文のまま残す.
        japanese_layout_mode: 日本語組版で崩れやすいレイアウトをどこまで補正するかの方針.
        japanese_font_mode: 日本語フォントをどの程度 paper-like に寄せるかの方針.
        allow_shell_escape: ``True`` の場合, LaTeX コンパイルで ``-shell-escape`` を有効化する.
        texlive_versions: コンパイル時に順番に試す TeX Live バージョン列.
            ``None`` の場合は ``/opt/texlive`` 配下の年ディレクトリを自動検出して使う.
    """

    workspace_root: Path
    max_chunk_chars: int = 2200
    concurrency: int = 10
    max_retries: int = 3
    translate_section_titles: bool = False
    japanese_layout_mode: JapaneseLayoutMode = "preserve"
    japanese_font_mode: JapaneseFontMode = "paper-like"
    allow_shell_escape: bool = False
    texlive_versions: tuple[str, ...] | None = None


@dataclass(slots=True, frozen=True)
class PromptCatalogSettings:
    """プロンプトカタログ参照設定を保持するデータクラス.

    Attributes:
        catalog_path: プロンプト定義TOMLへの絶対パス.
        profile: カタログ内で利用するプロファイル名.
    """

    catalog_path: Path
    profile: str


@dataclass(slots=True, frozen=True)
class AppSettings:
    """アプリ全体で利用する統合設定を保持するデータクラス.

    Attributes:
        llm: LLM接続設定.
        run: 実行パラメータ設定.
        prompts: プロンプトカタログ参照設定.
    """

    llm: LLMSettings
    run: RunSettings
    prompts: PromptCatalogSettings

    @staticmethod
    def load(config_path: Path | None = None) -> "AppSettings":
        """TOMLと環境変数から設定を読み込み, 統合設定を返す.

        Args:
            config_path: 設定ファイルパス, ``None`` の場合は既定パスを使う.

        Returns:
            AppSettings: 検証済みの統合設定.
        """
        settings = AppSettings.defaults()
        resolved_path = (config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
        _load_env_file(resolved_path.parent / ".env")
        if resolved_path.exists():
            settings = _load_from_toml(resolved_path, base=settings)
        return _apply_standard_env_fallback(settings)

    @staticmethod
    def defaults() -> "AppSettings":
        """ソースコード定義の既定設定を返す.

        Returns:
            AppSettings: 既定値のみで構成された統合設定.
        """
        return AppSettings(
            llm=LLMSettings(
                provider="openai",
                model="gpt-4.1-mini",
                base_url=None,
                api_key=None,
                retry_temperatures=(0.0, 0.2, 0.4),
            ),
            run=RunSettings(
                workspace_root=Path("./runs").resolve(),
                max_chunk_chars=2200,
                concurrency=10,
                max_retries=3,
                translate_section_titles=False,
                japanese_layout_mode="preserve",
                japanese_font_mode="paper-like",
                allow_shell_escape=False,
                texlive_versions=None,
            ),
            prompts=PromptCatalogSettings(
                catalog_path=DEFAULT_PROMPT_CATALOG_PATH.resolve(),
                profile=DEFAULT_PROMPT_PROFILE,
            ),
        )


def _load_from_toml(path: Path, *, base: AppSettings) -> AppSettings:
    """TOML設定で既定値を上書きし, 新しい統合設定を返す.

    Args:
        path: 設定ファイルの実パス.
        base: TOML適用前の基準設定.

    Returns:
        AppSettings: TOML内容を反映した統合設定.

    Raises:
        ValueError: TOML値が期待型や制約を満たさない場合.
    """
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    llm_table = _dict_section(data, "llm")
    run_table = _dict_section(data, "run")
    prompts_table = _dict_section(data, "prompts")

    provider = _validate_provider(str(llm_table.get("provider", base.llm.provider)))
    model = str(llm_table.get("model", base.llm.model))
    base_url = _maybe_str(llm_table.get("base_url", base.llm.base_url))
    api_key = _maybe_str(llm_table.get("api_key", base.llm.api_key))
    retry_temperatures = _parse_retry_temperatures(
        llm_table.get("retry_temperatures", list(base.llm.retry_temperatures)),
        name="llm.retry_temperatures",
    )

    workspace_raw = run_table.get("workspace_root", str(base.run.workspace_root))
    workspace_root = Path(str(workspace_raw)).expanduser().resolve()
    max_chunk_chars = _parse_positive_int_from_value(
        run_table.get("max_chunk_chars", base.run.max_chunk_chars),
        name="run.max_chunk_chars",
    )
    concurrency = _parse_positive_int_from_value(
        run_table.get("concurrency", base.run.concurrency),
        name="run.concurrency",
    )
    max_retries = _parse_positive_int_from_value(
        run_table.get("max_retries", base.run.max_retries),
        name="run.max_retries",
    )
    translate_section_titles = _parse_bool_from_value(
        run_table.get(
            "translate_section_titles",
            base.run.translate_section_titles,
        ),
        name="run.translate_section_titles",
    )
    japanese_layout_mode = _validate_japanese_layout_mode(
        str(
            run_table.get(
                "japanese_layout_mode",
                base.run.japanese_layout_mode,
            )
        )
    )
    japanese_font_mode = _validate_japanese_font_mode(
        str(
            run_table.get(
                "japanese_font_mode",
                base.run.japanese_font_mode,
            )
        )
    )
    allow_shell_escape = _parse_bool_from_value(
        run_table.get("allow_shell_escape", base.run.allow_shell_escape),
        name="run.allow_shell_escape",
    )
    texlive_versions = _parse_texlive_versions(
        run_table.get("texlive_versions", base.run.texlive_versions),
        name="run.texlive_versions",
    )

    catalog_raw = prompts_table.get("catalog_path", str(base.prompts.catalog_path))
    profile = _require_non_empty_str(
        prompts_table.get("profile", base.prompts.profile),
        name="prompts.profile",
    )
    catalog_path = _resolve_path_near(path.parent, Path(str(catalog_raw)))

    return AppSettings(
        llm=LLMSettings(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            retry_temperatures=retry_temperatures,
        ),
        run=RunSettings(
            workspace_root=workspace_root,
            max_chunk_chars=max_chunk_chars,
            concurrency=concurrency,
            max_retries=max_retries,
            translate_section_titles=translate_section_titles,
            japanese_layout_mode=japanese_layout_mode,
            japanese_font_mode=japanese_font_mode,
            allow_shell_escape=allow_shell_escape,
            texlive_versions=texlive_versions,
        ),
        prompts=PromptCatalogSettings(
            catalog_path=catalog_path,
            profile=profile,
        ),
    )


def _apply_standard_env_fallback(settings: AppSettings) -> AppSettings:
    """標準環境変数で不足設定を補完する.

    Args:
        settings: TOML適用後の統合設定.

    Returns:
        AppSettings: 環境変数補完を反映した統合設定.
    """
    model = os.getenv("ARXIV_HONYAKU_LLM_MODEL") or settings.llm.model

    api_key = settings.llm.api_key
    if api_key is None and settings.llm.provider == "deepseek":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    elif api_key is None and settings.llm.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

    if model == settings.llm.model and api_key == settings.llm.api_key:
        return settings

    return AppSettings(
        llm=LLMSettings(
            provider=settings.llm.provider,
            model=model,
            base_url=settings.llm.base_url,
            api_key=api_key,
            retry_temperatures=settings.llm.retry_temperatures,
        ),
        run=settings.run,
        prompts=settings.prompts,
    )


def _validate_provider(raw: str) -> ProviderName:
    """LLMプロバイダ名を検証し, 許可値として返す.

    Args:
        raw: 検証対象のプロバイダ文字列.

    Returns:
        ProviderName: 許可済みプロバイダ名.

    Raises:
        ValueError: 許可外プロバイダ名の場合.
    """
    if raw not in {"openai", "deepseek", "vllm", "ollama"}:
        raise ValueError(f"Unsupported provider: {raw}")
    return raw


def _validate_japanese_layout_mode(raw: str) -> JapaneseLayoutMode:
    """日本語レイアウト補正モード名を検証し, 許可値として返す.

    Args:
        raw: 検証対象のモード文字列.

    Returns:
        JapaneseLayoutMode: 許可済みモード名.

    Raises:
        ValueError: 許可外モード名の場合.
    """
    if raw not in {"preserve", "adaptive", "safe"}:
        raise ValueError(f"Unsupported japanese_layout_mode: {raw}")
    return raw


def _validate_japanese_font_mode(raw: str) -> JapaneseFontMode:
    """日本語フォント注入モード名を検証し, 許可値として返す.

    Args:
        raw: 検証対象のモード文字列.

    Returns:
        JapaneseFontMode: 許可済みモード名.

    Raises:
        ValueError: 許可外モード名の場合.
    """
    if raw not in {"compat", "paper-like"}:
        raise ValueError(f"Unsupported japanese_font_mode: {raw}")
    return raw


def _load_env_file(path: Path) -> None:
    """設定ファイルと同じディレクトリの ``.env`` を読み込み, 未設定環境変数を補完する.

    Args:
        path: 読み込み対象の ``.env`` ファイルパス.

    Returns:
        None: 常に ``None``.
    """
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        env_name = key.strip()
        if not env_name or env_name in os.environ:
            continue
        os.environ[env_name] = _strip_optional_quotes(raw_value.strip())


def _strip_optional_quotes(value: str) -> str:
    """単純なクオートで囲まれた値を展開する.

    Args:
        value: ``.env`` から読み出した生文字列.

    Returns:
        str: 先頭末尾の対応する引用符を外した文字列.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _dict_section(root: object, key: str) -> dict[str, object]:
    """辞書から指定キーのテーブルを安全に取り出す.

    Args:
        root: TOMLロード結果のルートオブジェクト.
        key: 取り出すセクション名.

    Returns:
        dict[str, object]: セクション辞書, 不在時は空辞書.

    Raises:
        ValueError: 指定キーが辞書でない場合.
    """
    if not isinstance(root, dict):
        return {}
    section = root.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"[{key}] must be a table")
    return section


def _parse_positive_int_from_value(value: object, *, name: str) -> int:
    """値を正の整数として検証し, 変換結果を返す.

    Args:
        value: 整数へ変換する入力値.
        name: エラーメッセージに表示する設定名.

    Returns:
        int: 検証済み正整数.

    Raises:
        ValueError: 0以下の値に変換された場合.
    """
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _parse_bool_from_value(value: object, *, name: str) -> bool:
    """値を真偽値として検証し, そのまま返す.

    Args:
        value: 真偽値として検証する入力値.
        name: エラーメッセージに表示する設定名.

    Returns:
        bool: 検証済み真偽値.

    Raises:
        ValueError: 真偽値でない値が渡された場合.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _parse_retry_temperatures(value: object, *, name: str) -> tuple[float, ...]:
    """temperature リトライ列を検証し, タプルへ正規化する.

    Args:
        value: TOMLから得た配列候補.
        name: エラーメッセージに表示する設定名.

    Returns:
        tuple[float, ...]: 検証済み temperature 列.

    Raises:
        ValueError: 配列でない, 空配列, もしくは各要素が 0 以上 2 以下の数値でない場合.
    """
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if not value:
        raise ValueError(f"{name} must not be empty")

    parsed_values: list[float] = []
    for index, item in enumerate(value):
        parsed = float(item)
        if not 0 <= parsed <= 2:
            raise ValueError(f"{name}[{index}] must be between 0 and 2")
        parsed_values.append(parsed)
    return tuple(parsed_values)


def _parse_texlive_versions(
    value: object,
    *,
    name: str,
) -> tuple[str, ...] | None:
    """TeX Live バージョン指定を検証し, タプルへ正規化する.

    Args:
        value: TOML から得た ``run.texlive_versions`` 値.
        name: エラーメッセージに表示する設定名.

    Returns:
        tuple[str, ...] | None:
            ``None`` の場合は自動検出, それ以外は指定順のバージョン列.

    Raises:
        ValueError: 型が不正, 空配列, 空文字, または 0 以下の整数を含む場合.
    """
    if value is None:
        return None

    raw_items: list[object]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be empty")
        raw_items = list(value)
    else:
        raw_items = [value]

    parsed_items: list[str] = []
    for index, item in enumerate(raw_items):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be a string or integer")
        if isinstance(item, int):
            if item <= 0:
                raise ValueError(f"{name}[{index}] must be > 0")
            parsed_items.append(str(item))
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                raise ValueError(f"{name}[{index}] must not be empty")
            parsed_items.append(text)
            continue
        raise ValueError(f"{name}[{index}] must be a string or integer")
    return tuple(parsed_items)


def _maybe_str(value: object) -> str | None:
    """値を文字列へ変換し, 空文字なら ``None`` を返す.

    Args:
        value: 文字列化対象値.

    Returns:
        str | None: 空でない文字列, または ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_non_empty_str(value: object, *, name: str) -> str:
    """空でない文字列を要求し, 文字列を返す.

    Args:
        value: 検証対象値.
        name: エラーメッセージに表示する設定名.

    Returns:
        str: 空でない文字列.

    Raises:
        ValueError: ``None`` または空文字の場合.
    """
    if value is None:
        raise ValueError(f"{name} must not be empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _resolve_path_near(base_dir: Path, raw_path: Path) -> Path:
    """基準ディレクトリ基準でパスを絶対化する.

    Args:
        base_dir: 相対パス解決に使う基準ディレクトリ.
        raw_path: 絶対または相対の入力パス.

    Returns:
        Path: 展開・解決済みの絶対パス.
    """
    if raw_path.is_absolute():
        return raw_path.expanduser().resolve()
    return (base_dir / raw_path).expanduser().resolve()
