"""設定から依存関係を組み立てる, コンポジションルートモジュール."""

from __future__ import annotations

from arxiv_honyaku.adapters.arxiv_http import HttpArxivDownloader
from arxiv_honyaku.adapters.http_llm import OpenAICompatibleTranslator
from arxiv_honyaku.adapters.latexmk import LatexmkCompiler
from arxiv_honyaku.adapters.state_file import JsonCheckpointRepository
from arxiv_honyaku.adapters.tex_project import TarArchiveExtractor, TeXMainFileLocator
from arxiv_honyaku.config import AppSettings
from arxiv_honyaku.core.progress import NullProgressReporter
from arxiv_honyaku.core.ports import ProgressReporter
from arxiv_honyaku.core.service import ArxivHonyakuService
from arxiv_honyaku.core.translation_runner import ChunkTranslationRunner
from arxiv_honyaku.prompts import load_prompt_profile


def build_service(
    settings: AppSettings,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> ArxivHonyakuService:
    """設定を解釈し, 翻訳ユースケースサービスを構築する.

    Args:
        settings: LLM接続, 実行, プロンプトカタログの統合設定.
        progress_reporter: 進捗通知ポート実装, ``None`` の場合は無通知.

    Returns:
        ArxivHonyakuService: 外部I/Oアダプタを注入済みのサービスインスタンス.
    """
    llm = _build_llm(settings)
    translator = ChunkTranslationRunner(
        llm=llm,
        checkpoints=JsonCheckpointRepository(),
        progress_reporter=progress_reporter,
        concurrency=settings.run.concurrency,
        max_retries=settings.run.max_retries,
        retry_temperatures=settings.llm.retry_temperatures,
    )
    return ArxivHonyakuService(
        downloader=HttpArxivDownloader(),
        extractor=TarArchiveExtractor(),
        tex_locator=TeXMainFileLocator(),
        translator=translator,
        compiler=LatexmkCompiler(),
        progress_reporter=progress_reporter or NullProgressReporter(),
    )


def _build_llm(settings: AppSettings):
    """設定に応じたLLMトランスレータを生成する.

    Args:
        settings: LLMプロバイダ, モデル, API接続先, プロンプト設定を含む統合設定.

    Returns:
        OpenAICompatibleTranslator: OpenAI互換API向けの翻訳アダプタ.

    Raises:
        ValueError: OpenAI利用時にAPIキーが解決できない場合.
    """
    llm_settings = settings.llm
    prompt_profile = load_prompt_profile(
        settings.prompts.catalog_path,
        profile=settings.prompts.profile,
    )

    if llm_settings.provider == "ollama":
        return OpenAICompatibleTranslator(
            provider="ollama",
            model=llm_settings.model,
            prompt_profile=prompt_profile,
            base_url=llm_settings.base_url or "http://localhost:11434/v1",
            api_key=llm_settings.api_key or "ollama",
        )

    if llm_settings.provider == "vllm":
        return OpenAICompatibleTranslator(
            provider="vllm",
            model=llm_settings.model,
            prompt_profile=prompt_profile,
            base_url=llm_settings.base_url or "http://localhost:8000/v1",
            api_key=llm_settings.api_key or "EMPTY",
        )

    if llm_settings.provider == "deepseek":
        if llm_settings.api_key is None:
            raise ValueError(
                "DeepSeek API key is not set. "
                "Set OPENAI_API_KEY, DEEPSEEK_API_KEY, "
                "or provide llm.api_key in arxiv-honyaku.toml."
            )
        return OpenAICompatibleTranslator(
            provider="deepseek",
            model=llm_settings.model,
            prompt_profile=prompt_profile,
            base_url=llm_settings.base_url or "https://api.deepseek.com",
            api_key=llm_settings.api_key,
        )

    if llm_settings.api_key is None and llm_settings.base_url is None:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or provide llm.api_key in arxiv-honyaku.toml."
        )
    return OpenAICompatibleTranslator(
        provider="openai",
        model=llm_settings.model,
        prompt_profile=prompt_profile,
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key,
    )
