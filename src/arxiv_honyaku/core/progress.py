"""翻訳進捗通知まわりの補助実装を定義するモジュール."""

from __future__ import annotations

from arxiv_honyaku.core.models import ChunkIssue
from arxiv_honyaku.core.ports import ProgressReporter


class NullProgressReporter(ProgressReporter):
    """何も通知しない既定の進捗レポータ."""

    def on_translation_started(self, *, total_files: int, total_chunks: int) -> None:
        """翻訳開始通知を無視する.

        Args:
            total_files: 今回処理するTeXファイル総数.
            total_chunks: 翻訳対象チャンク総数.

        Returns:
            None: 常に ``None``.
        """
        del total_files, total_chunks
        return

    def on_file_started(self, *, relative_path: str, total_chunks: int) -> None:
        """ファイル開始通知を無視する.

        Args:
            relative_path: ソースルートからの相対パス.
            total_chunks: 当該ファイル内の翻訳対象チャンク数.

        Returns:
            None: 常に ``None``.
        """
        del relative_path, total_chunks
        return

    def on_chunk_finished(
        self,
        *,
        relative_path: str,
        chunk_index: int,
        successful: bool,
        cached: bool,
    ) -> None:
        """チャンク終了通知を無視する.

        Args:
            relative_path: ソースルートからの相対パス.
            chunk_index: 文書処理順のチャンク番号.
            successful: 翻訳成功時は ``True``.
            cached: チェックポイント再利用時は ``True``.

        Returns:
            None: 常に ``None``.
        """
        del relative_path, chunk_index, successful, cached
        return

    def on_chunk_issue(self, *, issue: ChunkIssue) -> None:
        """チャンク問題通知を無視する.

        Args:
            issue: 解析用に整形済みのチャンク問題情報.

        Returns:
            None: 常に ``None``.
        """
        del issue
        return

    def on_translation_finished(self, *, successful: bool) -> None:
        """翻訳終了通知を無視する.

        Args:
            successful: 翻訳と後続処理が成功した場合は ``True``.

        Returns:
            None: 常に ``None``.
        """
        del successful
        return
