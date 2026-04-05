"""チャンク翻訳の並列実行, 再試行, fallback, チェックポイント管理を行うモジュール.

このモジュールは, chunker が切り出した ``TexChunk`` 群に対して
LLM 呼び出しを並列実行し, 構造破壊を検知した場合は段階的に救済経路へ切り替える.
さらに, 実行途中と最終結果をチェックポイントへ保存し, 次回実行時には
成功結果や fallback 結果を再利用する.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

from arxiv_honyaku.core.models import (
    ChunkIssue,
    ChunkProgress,
    TexChunk,
    TranslationCheckpoint,
)
from arxiv_honyaku.core.progress import NullProgressReporter
from arxiv_honyaku.core.ports import (
    CheckpointRepository,
    LLMTranslator,
    ProgressReporter,
)
from arxiv_honyaku.core.translation_preparation import (
    PreparedTranslationInput,
    ProtectedPlaceholderError,
    prepare_translation_input,
    restore_translation_output,
    split_prepared_input_into_segments,
)
from arxiv_honyaku.core.translation_validation import validate_translated_chunk

LOGGER = logging.getLogger("arxiv_honyaku.core.translation_runner")

TranslationStrategy = Literal["direct", "segment", "strict_segment"]


@dataclass(slots=True, frozen=True)
class _ChunkTranslationResult:
    """1 チャンク翻訳の成功結果と採用した翻訳経路.

    Attributes:
        text: 検証済みの翻訳結果.
        strategy: 採用した翻訳経路.
        reason: 救済経路へ入ったきっかけのエラー文字列.
    """

    text: str
    strategy: TranslationStrategy
    reason: str | None = None


class ChunkTranslationRunner:
    """翻訳チャンクを並列実行し, 再試行, fallback, 永続化を担うランナー.

    Attributes:
        _llm: 1チャンク翻訳を実行するポート実装.
        _checkpoints: 進捗永続化ポート実装.
        _concurrency: 同時実行数.
        _max_retries: チャンクごとの最大再試行回数.
        _retry_temperatures: 試行回数に応じて使う temperature 列.
    """

    def __init__(
        self,
        *,
        llm: LLMTranslator,
        checkpoints: CheckpointRepository,
        progress_reporter: ProgressReporter | None,
        concurrency: int,
        max_retries: int,
        retry_temperatures: tuple[float, ...],
    ) -> None:
        """翻訳実行パラメータと依存実装を受け取り初期化する.

        Args:
            llm: LLM翻訳ポート実装.
            checkpoints: チェックポイント永続化ポート実装.
            progress_reporter: 進捗通知ポート実装, ``None`` の場合は無通知.
            concurrency: 同時実行数.
            max_retries: チャンクごとの最大再試行回数.
            retry_temperatures: 試行回数に応じて使う temperature 列.

        Raises:
            ValueError: ``concurrency``, ``max_retries`` が0以下, または ``retry_temperatures`` が空の場合.
        """
        if concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if max_retries <= 0:
            raise ValueError("max_retries must be > 0")
        if not retry_temperatures:
            raise ValueError("retry_temperatures must not be empty")
        self._llm = llm
        self._checkpoints = checkpoints
        self._progress_reporter = progress_reporter or NullProgressReporter()
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._retry_temperatures = retry_temperatures

    async def translate_all(
        self,
        *,
        chunks: list[TexChunk],
        source_digest: str,
        checkpoint_path: Path,
        relative_path: str,
    ) -> list[str]:
        """全チャンクを翻訳し, 入力順の翻訳文字列リストを返す.

        最大再試行後も翻訳できないチャンクは, 直近エラーを保持したまま
        原文を採用して処理を継続する. すでに成功済み, もしくは fallback 済みの
        チェックポイントがあれば再利用し, ただし現在の validator に通らない
        既存結果は破棄して再試行する.

        Args:
            chunks: 翻訳対象チャンク配列.
            source_digest: 元全文のダイジェスト.
            checkpoint_path: 進捗保存先パス.
            relative_path: ソースルートから見た対象TeXファイルの相対パス.

        Returns:
            list[str]: チャンク順の翻訳済み本文配列.
        """
        checkpoint = self._prepare_checkpoint(
            chunks=chunks,
            source_digest=source_digest,
            checkpoint_path=checkpoint_path,
        )
        semaphore = asyncio.Semaphore(self._concurrency)

        async def worker(chunk: TexChunk) -> None:
            if not chunk.translate:
                return

            progress = checkpoint.chunks[chunk.index]
            if (
                progress.status in {"success", "fallback"}
                and progress.translated_text is not None
            ):
                try:
                    validate_translated_chunk(chunk, progress.translated_text)
                except ValueError as exc:
                    progress.status = "pending"
                    progress.attempts = 0
                    progress.translated_text = None
                    progress.last_error = str(exc)
                    self._checkpoints.save(checkpoint_path, checkpoint)
                else:
                    if progress.status == "fallback":
                        self._report_chunk_issue(
                            chunk=chunk,
                            progress=progress,
                            relative_path=relative_path,
                            cached=True,
                        )
                    self._progress_reporter.on_chunk_finished(
                        relative_path=relative_path,
                        chunk_index=chunk.index,
                        successful=progress.status == "success",
                        cached=True,
                    )
                    return

            while progress.attempts < self._max_retries:
                progress.attempts += 1
                try:
                    temperature = self._temperature_for_attempt(progress.attempts)
                    result = await self._translate_chunk_once(
                        chunk=chunk,
                        semaphore=semaphore,
                        temperature=temperature,
                    )
                    progress.translated_text = result.text
                    progress.status = "success"
                    progress.last_error = None
                    self._checkpoints.save(checkpoint_path, checkpoint)
                    self._log_successful_recovery(
                        chunk=chunk,
                        relative_path=relative_path,
                        strategy=result.strategy,
                        reason=result.reason,
                    )
                    self._progress_reporter.on_chunk_finished(
                        relative_path=relative_path,
                        chunk_index=chunk.index,
                        successful=True,
                        cached=False,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    progress.status = "failed"
                    progress.last_error = str(exc)
                    self._checkpoints.save(checkpoint_path, checkpoint)
                    if progress.attempts >= self._max_retries:
                        self._progress_reporter.on_chunk_finished(
                            relative_path=relative_path,
                            chunk_index=chunk.index,
                            successful=False,
                            cached=False,
                        )

        await asyncio.gather(*(worker(chunk) for chunk in chunks))

        self._fallback_failed_chunks(
            chunks=chunks,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            relative_path=relative_path,
        )

        return [
            chunk.text
            if not chunk.translate
            else checkpoint.chunks[chunk.index].translated_text or chunk.text
            for chunk in chunks
        ]

    def _prepare_checkpoint(
        self,
        *,
        chunks: list[TexChunk],
        source_digest: str,
        checkpoint_path: Path,
    ) -> TranslationCheckpoint:
        """入力チャンクに整合するチェックポイント状態を準備する.

        source digest, chunk digest, validator の 3 点から既存 checkpoint を見直し,
        再利用してよい状態だけを残す. これにより, 以前の実装で保存された壊れた
        成功キャッシュも自動で取り除ける.

        Args:
            chunks: 今回処理するチャンク配列.
            source_digest: 元全文のダイジェスト.
            checkpoint_path: 進捗保存先パス.

        Returns:
            TranslationCheckpoint: 処理対象に整合したチェックポイント.
        """
        loaded = self._checkpoints.load(checkpoint_path)
        if loaded is None or loaded.source_digest != source_digest:
            checkpoint = TranslationCheckpoint(source_digest=source_digest, chunks={})
        else:
            checkpoint = loaded

        indices = {chunk.index for chunk in chunks if chunk.translate}
        checkpoint.chunks = {
            index: progress
            for index, progress in checkpoint.chunks.items()
            if index in indices
        }

        for chunk in chunks:
            if not chunk.translate:
                continue
            progress = checkpoint.chunks.get(chunk.index)
            if progress is None or progress.digest != chunk.digest:
                checkpoint.chunks[chunk.index] = ChunkProgress(digest=chunk.digest)
                continue
            if progress.status not in {"success", "fallback"}:
                checkpoint.chunks[chunk.index] = ChunkProgress(digest=chunk.digest)
                continue
            if progress.translated_text is None:
                checkpoint.chunks[chunk.index] = ChunkProgress(digest=chunk.digest)
                continue
            try:
                validate_translated_chunk(chunk, progress.translated_text)
            except ValueError as exc:
                checkpoint.chunks[chunk.index] = ChunkProgress(
                    digest=chunk.digest,
                    last_error=str(exc),
                )
        self._checkpoints.save(checkpoint_path, checkpoint)
        return checkpoint

    def _fallback_failed_chunks(
        self,
        *,
        chunks: list[TexChunk],
        checkpoint: TranslationCheckpoint,
        checkpoint_path: Path,
        relative_path: str,
    ) -> None:
        """最終的に失敗したチャンクを原文フォールバックへ切り替える.

        ここで fallback されたチャンクは, 例外で全体停止させずに PDF 生成へ進める
        ための最終手段である. 状態は checkpoint にも保存し, 次回以降の再実行でも
        問題サマリーへ出せるようにする.

        Args:
            chunks: 今回処理したチャンク配列.
            checkpoint: 更新対象のチェックポイント.
            checkpoint_path: 進捗保存先パス.
            relative_path: ソースルートから見た対象TeXファイルの相対パス.

        Returns:
            None: 常に ``None``.
        """
        changed = False
        for chunk in chunks:
            if not chunk.translate:
                continue
            progress = checkpoint.chunks[chunk.index]
            if progress.status != "failed":
                continue
            progress.status = "fallback"
            progress.translated_text = chunk.text
            changed = True
            self._report_chunk_issue(
                chunk=chunk,
                progress=progress,
                relative_path=relative_path,
                cached=False,
            )
        if changed:
            self._checkpoints.save(checkpoint_path, checkpoint)

    def _report_chunk_issue(
        self,
        *,
        chunk: TexChunk,
        progress: ChunkProgress,
        relative_path: str,
        cached: bool,
    ) -> None:
        """問題チャンクの詳細をログと進捗レポータへ通知する.

        Args:
            chunk: 問題が確定した対象チャンク.
            progress: 当該チャンクの進捗状態.
            relative_path: ソースルートから見た対象TeXファイルの相対パス.
            cached: 既存チェックポイント由来の問題状態を再利用した場合は ``True``.

        Returns:
            None: 常に ``None``.
        """
        issue = ChunkIssue(
            relative_path=relative_path,
            chunk_index=chunk.index,
            chunk_kind=chunk.kind,
            start_line=chunk.start_line,
            status=progress.status,
            attempts=progress.attempts,
            cached=cached,
            error=progress.last_error,
            source_preview=_preview_text(chunk.text),
        )
        LOGGER.warning(
            "Using original text for untranslated chunk. file=%s line=%s chunk=%s kind=%s cached=%s error=%s preview=%s",
            issue.relative_path,
            issue.start_line,
            issue.chunk_index,
            issue.chunk_kind,
            issue.cached,
            issue.error or "unknown error",
            issue.source_preview,
        )
        self._progress_reporter.on_chunk_issue(issue=issue)

    def _log_successful_recovery(
        self,
        *,
        chunk: TexChunk,
        relative_path: str,
        strategy: TranslationStrategy,
        reason: str | None,
    ) -> None:
        """成功した救済経路の利用をログへ明示する.

        現在は, 最も保守的な ``strict_segment`` が使われたとき debug を出す.
        direct や通常 segment は通常の成功として扱う.

        Args:
            chunk: 救済経路が使われた対象チャンク.
            relative_path: ソースルートから見た対象TeXファイルの相対パス.
            strategy: 採用した翻訳経路.
            reason: 救済経路へ入ったきっかけのエラー文字列.

        Returns:
            None: 常に ``None``.
        """
        if strategy != "strict_segment":
            return
        LOGGER.debug(
            "Used strict segment fallback for translated chunk. file=%s line=%s chunk=%s kind=%s reason=%s preview=%s",
            relative_path,
            chunk.start_line,
            chunk.index,
            chunk.kind,
            reason or "unknown error",
            _preview_text(chunk.text),
        )

    async def _translate_chunk_once(
        self,
        *,
        chunk: TexChunk,
        semaphore: asyncio.Semaphore,
        temperature: float,
    ) -> _ChunkTranslationResult:
        """1 回分の翻訳試行を実行し, 必要なら代替経路へ段階的に切り替える.

        経路は次の順に試す.
        1. placeholder を含む全文をそのまま訳す ``direct``.
        2. command を固定し, math placeholder は文脈内へ残す ``segment``.
        3. 保護断片をすべて固定し, plain text だけを訳す ``strict_segment``.
        どの経路でも validator を通った時点で成功とし, すべて失敗した場合だけ
        各経路の失敗理由を束ねて例外化する.

        Args:
            chunk: 翻訳対象チャンク.
            semaphore: 同時実行数制御用セマフォ.
            temperature: 当該試行で使う推論 temperature.

        Returns:
            _ChunkTranslationResult: 検証済み本文と採用した翻訳経路.
        """
        prepared_input = prepare_translation_input(chunk)
        translated_with_restoration = ""
        initial_error: ValueError | None = None
        segment_error: ValueError | None = None
        strict_segment_error: ValueError | None = None
        async with semaphore:
            translated = await self._llm.translate(
                prepared_input.prompt_text,
                temperature=temperature,
            )
        try:
            translated_with_restoration = restore_translation_output(
                prepared_input,
                translated,
            )
            return _ChunkTranslationResult(
                text=validate_translated_chunk(chunk, translated_with_restoration),
                strategy="direct",
            )
        except (ProtectedPlaceholderError, ValueError) as exc:
            initial_error = exc

        if not prepared_input.protected_spans:
            assert initial_error is not None
            raise initial_error

        try:
            translated_by_segments = await self._translate_chunk_segments(
                prepared_input=prepared_input,
                semaphore=semaphore,
                temperature=temperature,
                preserve_math_context=True,
            )
            return _ChunkTranslationResult(
                text=validate_translated_chunk(chunk, translated_by_segments),
                strategy="segment",
                reason=str(initial_error),
            )
        except ValueError as exc:
            segment_error = exc

        try:
            translated_by_strict_segments = await self._translate_chunk_segments(
                prepared_input=prepared_input,
                semaphore=semaphore,
                temperature=temperature,
                preserve_math_context=False,
            )
            return _ChunkTranslationResult(
                text=validate_translated_chunk(chunk, translated_by_strict_segments),
                strategy="strict_segment",
                reason=str(initial_error),
            )
        except ValueError as exc:
            strict_segment_error = exc
            assert initial_error is not None
            raise _build_strategy_failure_error(
                initial_error=initial_error,
                segment_error=segment_error,
                strict_segment_error=strict_segment_error,
            )

    async def _translate_chunk_segments(
        self,
        *,
        prepared_input: PreparedTranslationInput,
        semaphore: asyncio.Semaphore,
        temperature: float,
        preserve_math_context: bool,
    ) -> str:
        """保護 placeholder が崩れた場合に, 断片ごとの翻訳で再構成する.

        ``preserve_math_context=True`` では数式 placeholder を翻訳対象の文脈へ残し,
        文章と式のつながりを優先する. ``False`` では保護断片をすべて固定し,
        plain text だけを訳す厳格 fallback として振る舞う.

        Args:
            prepared_input: 事前退避済みの翻訳入力情報.
            semaphore: 同時実行数制御用セマフォ.
            temperature: 当該試行で使う推論 temperature.
            preserve_math_context: 数式 placeholder を翻訳対象断片内へ残す場合は ``True``.

        Returns:
            str: 断片再構成後の翻訳済み本文.
        """
        translated_parts: list[str] = []
        for segment in split_prepared_input_into_segments(
            prepared_input,
            preserve_math_context=preserve_math_context,
        ):
            if not segment.translate:
                translated_parts.append(segment.text)
                continue
            translated_parts.append(
                await self._translate_preserving_outer_whitespace(
                    text=segment.text,
                    semaphore=semaphore,
                    temperature=temperature,
                )
            )
        joined_text = "".join(translated_parts)
        if not preserve_math_context:
            return joined_text
        return restore_translation_output(
            prepared_input,
            joined_text,
            restore_kinds=frozenset({"math"}),
        )

    async def _translate_preserving_outer_whitespace(
        self,
        *,
        text: str,
        semaphore: asyncio.Semaphore,
        temperature: float,
    ) -> str:
        """先頭末尾の空白を保持しつつ, 中核部分だけを翻訳する.

        segment fallback では, 固定断片の前後に空白や改行が残ることが多い.
        そのため, この関数は外側空白だけを元のまま保持し, 翻訳器には core 部分だけを
        渡して再構成しやすい形を保つ.

        Args:
            text: 翻訳対象断片.
            semaphore: 同時実行数制御用セマフォ.
            temperature: 当該試行で使う推論 temperature.

        Returns:
            str: 空白を復元した翻訳済み断片.
        """
        if not text or not text.strip():
            return text

        leading_length = len(text) - len(text.lstrip())
        trailing_length = len(text) - len(text.rstrip())
        trailing_start = len(text) - trailing_length if trailing_length else len(text)
        leading_text = text[:leading_length]
        core_text = text[leading_length:trailing_start]
        trailing_text = text[trailing_start:]
        if not core_text:
            return text

        async with semaphore:
            translated_core = await self._llm.translate(
                core_text,
                temperature=temperature,
            )
        return f"{leading_text}{translated_core}{trailing_text}"

    def _temperature_for_attempt(self, attempt_number: int) -> float:
        """試行回数に対応する temperature を返す.

        Args:
            attempt_number: 1始まりの試行回数.

        Returns:
            float: 当該試行で使う temperature.
        """
        schedule_index = min(
            max(attempt_number - 1, 0),
            len(self._retry_temperatures) - 1,
        )
        return self._retry_temperatures[schedule_index]


def _preview_text(text: str, *, max_length: int = 140) -> str:
    """長いチャンク本文を, ログ向けの短い1行プレビューへ整形する.

    Args:
        text: 整形対象の元チャンク本文.
        max_length: 出力する最大文字数.

    Returns:
        str: 余分な空白を潰して切り詰めた1行プレビュー.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= max_length:
        return collapsed
    return f"{collapsed[: max_length - 3]}..."


def _build_strategy_failure_error(
    *,
    initial_error: ValueError,
    segment_error: ValueError | None,
    strict_segment_error: ValueError | None,
) -> ValueError:
    """複数の翻訳経路で起きた失敗理由を1つの例外へ束ねる.

    Args:
        initial_error: direct 経路の失敗理由.
        segment_error: segment 経路の失敗理由.
        strict_segment_error: strict_segment 経路の失敗理由.

    Returns:
        ValueError: 失敗経路ごとの理由をまとめた例外.
    """
    error_parts = [f"direct={initial_error}"]
    if segment_error is not None:
        error_parts.append(f"segment={segment_error}")
    if strict_segment_error is not None:
        error_parts.append(f"strict_segment={strict_segment_error}")
    return ValueError("All translation strategies failed. " + " | ".join(error_parts))
