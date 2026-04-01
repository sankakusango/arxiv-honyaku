"""ライブラリ利用向けに, 翻訳パイプラインを実行する公開APIを提供するモジュール."""

from __future__ import annotations

import asyncio

from arxiv_honyaku.bootstrap import build_service
from arxiv_honyaku.config import AppSettings
from arxiv_honyaku.core.models import TranslateRequest, TranslateResult
from arxiv_honyaku.core.ports import ProgressReporter


async def translate_arxiv_async(
    arxiv_ref: str,
    *,
    settings: AppSettings | None = None,
    progress_reporter: ProgressReporter | None = None,
    force: bool = False,
) -> TranslateResult:
    """arXiv論文の翻訳処理を非同期で実行し, 生成物情報を返す.

    Args:
        arxiv_ref: arXiv ID, または arXiv URL.
        settings: 実行設定, 未指定時は ``AppSettings.load()`` の結果を使う.
        progress_reporter: 進捗通知ポート実装, ``None`` の場合は無通知.
        force: ``True`` の場合, 既存成果物とチェックポイントを破棄して再実行する.

    Returns:
        TranslateResult: 翻訳済みTeX, 生成PDF, 作業ディレクトリを含む結果オブジェクト.
    """
    resolved = settings or AppSettings.load()
    service = build_service(resolved, progress_reporter=progress_reporter)
    request = TranslateRequest(
        arxiv_ref=arxiv_ref,
        workspace_root=resolved.run.workspace_root,
        max_chunk_chars=resolved.run.max_chunk_chars,
        force=force,
        translate_section_titles=resolved.run.translate_section_titles,
        japanese_layout_mode=resolved.run.japanese_layout_mode,
        japanese_font_mode=resolved.run.japanese_font_mode,
    )
    return await service.translate(request)


def translate_arxiv(
    arxiv_ref: str,
    *,
    settings: AppSettings | None = None,
    progress_reporter: ProgressReporter | None = None,
    force: bool = False,
) -> TranslateResult:
    """arXiv論文の翻訳処理を同期で実行する.

    Args:
        arxiv_ref: arXiv ID, または arXiv URL.
        settings: 実行設定, 未指定時は ``AppSettings.load()`` の結果を使う.
        progress_reporter: 進捗通知ポート実装, ``None`` の場合は無通知.
        force: ``True`` の場合, 既存成果物とチェックポイントを破棄して再実行する.

    Returns:
        TranslateResult: 翻訳済みTeX, 生成PDF, 作業ディレクトリを含む結果オブジェクト.
    """
    return asyncio.run(
        translate_arxiv_async(
            arxiv_ref,
            settings=settings,
            progress_reporter=progress_reporter,
            force=force,
        )
    )
