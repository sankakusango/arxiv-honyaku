"""コア層が依存する外部I/Oポートを定義するモジュール."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from arxiv_honyaku.core.models import ChunkIssue, TranslationCheckpoint


class SourceDownloader(Protocol):
    """arXivソースダウンロードのポート."""

    def download(self, arxiv_id: str, destination: Path) -> Path:
        """arXivソースアーカイブをダウンロードする.

        Args:
            arxiv_id: 正規化済みarXiv ID.
            destination: 保存先ファイルパス.

        Returns:
            Path: 保存済みアーカイブのファイルパス.
        """
        ...


class ArchiveExtractor(Protocol):
    """ソースアーカイブ展開のポート."""

    def extract(self, archive_path: Path, destination: Path) -> None:
        """アーカイブを指定ディレクトリへ展開する.

        Args:
            archive_path: 展開対象アーカイブパス.
            destination: 展開先ディレクトリ.
        """
        ...


class MainTexLocator(Protocol):
    """メインTeXファイル特定のポート."""

    def find(self, source_dir: Path) -> Path:
        """ソースツリーからメインTeXファイルを特定する.

        Args:
            source_dir: 展開済みソースルートディレクトリ.

        Returns:
            Path: メインTeXファイルパス.
        """
        ...


class LLMTranslator(Protocol):
    """LLM翻訳処理のポート."""

    async def translate(self, text: str, *, temperature: float) -> str:
        """1チャンク分のTeX本文を翻訳する.

        Args:
            text: 翻訳対象のTeXチャンク本文.
            temperature: 当該試行で使う推論 temperature.

        Returns:
            str: 翻訳済みTeX本文.
        """
        ...


class ProgressReporter(Protocol):
    """翻訳進捗を外部へ通知するポート."""

    def on_translation_started(self, *, total_files: int, total_chunks: int) -> None:
        """翻訳全体の開始を通知する.

        Args:
            total_files: 今回処理するTeXファイル総数.
            total_chunks: 翻訳対象チャンク総数.

        Returns:
            None: 常に ``None``.
        """
        ...

    def on_file_started(self, *, relative_path: str, total_chunks: int) -> None:
        """個別TeXファイルの処理開始を通知する.

        Args:
            relative_path: ソースルートからの相対パス.
            total_chunks: 当該ファイル内の翻訳対象チャンク数.

        Returns:
            None: 常に ``None``.
        """
        ...

    def on_chunk_finished(
        self,
        *,
        relative_path: str,
        chunk_index: int,
        successful: bool,
        cached: bool,
    ) -> None:
        """1チャンクの終端状態到達を通知する.

        Args:
            relative_path: ソースルートからの相対パス.
            chunk_index: ファイル内チャンク番号ではなく, 文書処理順のチャンク番号.
            successful: 翻訳成功時は ``True``.
            cached: 既存チェックポイント再利用時は ``True``.

        Returns:
            None: 常に ``None``.
        """
        ...

    def on_chunk_issue(self, *, issue: ChunkIssue) -> None:
        """翻訳不能, または原文フォールバックになったチャンク情報を通知する.

        Args:
            issue: 解析用に整形済みのチャンク問題情報.

        Returns:
            None: 常に ``None``.
        """
        ...

    def on_translation_finished(self, *, successful: bool) -> None:
        """翻訳全体の終了を通知する.

        Args:
            successful: 翻訳と後続処理が成功した場合は ``True``.

        Returns:
            None: 常に ``None``.
        """
        ...


class CheckpointRepository(Protocol):
    """翻訳進捗永続化のポート."""

    def load(self, checkpoint_path: Path) -> TranslationCheckpoint | None:
        """チェックポイントを読み込む.

        Args:
            checkpoint_path: 読み込み対象ファイルパス.

        Returns:
            TranslationCheckpoint | None: 読み込み成功時はチェックポイント, 未存在時は ``None``.
        """
        ...

    def save(self, checkpoint_path: Path, checkpoint: TranslationCheckpoint) -> None:
        """チェックポイントを保存する.

        Args:
            checkpoint_path: 保存先ファイルパス.
            checkpoint: 保存するチェックポイントオブジェクト.
        """
        ...


class PdfCompiler(Protocol):
    """TeXコンパイルのポート."""

    def compile(self, tex_file: Path, build_dir: Path) -> Path:
        """TeXをコンパイルし, 生成PDFパスを返す.

        Args:
            tex_file: コンパイル対象のメインTeXファイル.
            build_dir: ビルド成果物出力ディレクトリ.

        Returns:
            Path: 生成されたPDFファイルパス.
        """
        ...
