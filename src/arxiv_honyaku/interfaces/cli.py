"""CLIインターフェース実装モジュール."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from arxiv_honyaku.api import translate_arxiv_async
from arxiv_honyaku.config import AppSettings
from arxiv_honyaku.interfaces.cli_progress import TqdmProgressReporter

LOGGER = logging.getLogger("arxiv_honyaku.cli")


def main() -> None:
    """CLIエントリポイントを実行する.

    Returns:
        None: 常に ``None`` を返す, 失敗時は ``SystemExit`` を送出する.
    """
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()
    settings = _resolve_settings(args)
    progress_reporter = TqdmProgressReporter()

    try:
        result = asyncio.run(
            translate_arxiv_async(
                args.arxiv_ref,
                settings=settings,
                progress_reporter=progress_reporter,
                force=args.force,
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Translation failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("arXiv ID: %s", result.arxiv_id)
    LOGGER.info("Run Dir: %s", result.run_dir)
    LOGGER.info("Translated TeX: %s", result.translated_main_tex)
    LOGGER.info("PDF: %s", result.pdf_path)
    LOGGER.info("Chunks: %s", result.total_chunks)


def _build_parser() -> argparse.ArgumentParser:
    """CLI引数パーサを構築する.

    Returns:
        argparse.ArgumentParser: 設定済み引数パーサ.
    """
    parser = argparse.ArgumentParser(
        prog="arxiv-honyaku",
        description="Download arXiv TeX, translate it in chunks, and compile PDF.",
    )
    parser.add_argument("arxiv_ref", help="arXiv ID or URL")
    parser.add_argument(
        "--config",
        default="arxiv-honyaku.toml",
        help="Path to TOML config file (default: arxiv-honyaku.toml)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Ignore previous successful outputs and rerun translation from scratch.",
    )
    return parser


def _resolve_settings(args: argparse.Namespace) -> AppSettings:
    """CLI引数からアプリ設定を読み込む.

    Args:
        args: 解析済みCLI引数.

    Returns:
        AppSettings: 読み込み済み統合設定.
    """
    return AppSettings.load(Path(args.config))


def _configure_logging() -> None:
    """既定ロギング設定を初期化する.

    Returns:
        None: 常に ``None``.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
