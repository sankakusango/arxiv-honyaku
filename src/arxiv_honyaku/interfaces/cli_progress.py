"""CLI向けのtqdm進捗表示実装モジュール."""

from __future__ import annotations

from arxiv_honyaku.core.models import ChunkIssue
from arxiv_honyaku.core.ports import ProgressReporter

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError as import_error:
    tqdm = None
    _TQDM_IMPORT_ERROR = import_error
else:
    _TQDM_IMPORT_ERROR = None


class TqdmProgressReporter(ProgressReporter):
    """CLIで進捗バーを表示する ``tqdm`` ベース実装."""

    def __init__(self) -> None:
        """依存ライブラリを検証し, 進捗表示実装を初期化する.

        Returns:
            None: 常に ``None``.

        Raises:
            RuntimeError: ``tqdm`` が未導入の場合.
        """
        if tqdm is None:
            raise RuntimeError(
                "tqdm package is required for CLI progress display."
            ) from _TQDM_IMPORT_ERROR
        self._bar = None
        self._current_file = ""
        self._successful_chunks = 0
        self._failed_chunks = 0
        self._issues: list[ChunkIssue] = []

    def on_translation_started(self, *, total_files: int, total_chunks: int) -> None:
        """翻訳全体の開始時に進捗バーを初期化する.

        Args:
            total_files: 今回処理するTeXファイル総数.
            total_chunks: 翻訳対象チャンク総数.

        Returns:
            None: 常に ``None``.
        """
        self._bar = tqdm(
            total=total_chunks,
            desc="Translating",
            unit="chunk",
            dynamic_ncols=True,
        )
        self._bar.set_postfix_str(f"files={total_files}")

    def on_file_started(self, *, relative_path: str, total_chunks: int) -> None:
        """現在処理中のファイル名を進捗バーへ反映する.

        Args:
            relative_path: ソースルートからの相対パス.
            total_chunks: 当該ファイル内の翻訳対象チャンク数.

        Returns:
            None: 常に ``None``.
        """
        self._current_file = relative_path
        if self._bar is None:
            return
        self._bar.set_postfix_str(
            f"file={relative_path} chunks={total_chunks} ok={self._successful_chunks} failed={self._failed_chunks}"
        )

    def on_chunk_finished(
        self,
        *,
        relative_path: str,
        chunk_index: int,
        successful: bool,
        cached: bool,
    ) -> None:
        """終端状態に到達したチャンク分だけ進捗バーを進める.

        Args:
            relative_path: ソースルートからの相対パス.
            chunk_index: 文書処理順のチャンク番号.
            successful: 翻訳成功時は ``True``.
            cached: チェックポイント再利用時は ``True``.

        Returns:
            None: 常に ``None``.
        """
        del chunk_index
        self._current_file = relative_path
        if successful:
            self._successful_chunks += 1
        else:
            self._failed_chunks += 1
        if self._bar is None:
            return
        self._bar.update(1)
        cache_suffix = " cached" if cached else ""
        self._bar.set_postfix_str(
            f"file={relative_path}{cache_suffix} ok={self._successful_chunks} failed={self._failed_chunks}"
        )

    def on_chunk_issue(self, *, issue: ChunkIssue) -> None:
        """問題チャンクを収集し, 終了時サマリー用に保持する.

        Args:
            issue: 解析用に整形済みのチャンク問題情報.

        Returns:
            None: 常に ``None``.
        """
        self._issues.append(issue)

    def on_translation_finished(self, *, successful: bool) -> None:
        """翻訳全体終了時に進捗バーを閉じる.

        Args:
            successful: 翻訳と後続処理が成功した場合は ``True``.

        Returns:
            None: 常に ``None``.
        """
        if self._bar is None:
            return
        status_label = "done" if successful else "failed"
        self._bar.set_postfix_str(
            f"file={self._current_file} ok={self._successful_chunks} failed={self._failed_chunks} status={status_label}"
        )
        self._bar.close()
        self._bar = None
        self._print_issue_summary()

    def _print_issue_summary(self) -> None:
        """収集済みの問題チャンク一覧を, CLI終端サマリーとして出力する.

        Returns:
            None: 常に ``None``.
        """
        if not self._issues:
            return

        self._write_line(
            f"Translation issue summary: {len(self._issues)} chunk(s) used fallback or cached fallback."
        )
        for issue in self._issues:
            cached_label = " cached" if issue.cached else ""
            self._write_line(
                f"- {issue.relative_path}:{issue.start_line} chunk={issue.chunk_index} kind={issue.chunk_kind} status={issue.status}{cached_label} attempts={issue.attempts}"
            )
            self._write_line(f"  error: {issue.error or 'unknown error'}")
            self._write_line(f"  source: {issue.source_preview}")

    def _write_line(self, message: str) -> None:
        """進捗バーと競合しない形で1行メッセージを出力する.

        Args:
            message: 出力する1行メッセージ.

        Returns:
            None: 常に ``None``.
        """
        if tqdm is None:
            print(message)
            return
        tqdm.write(message)
