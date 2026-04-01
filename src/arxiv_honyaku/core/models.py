"""コア層で利用する, 翻訳パイプライン向けデータモデル定義."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ChunkStatus = Literal["pending", "success", "failed", "fallback"]
JapaneseLayoutMode = Literal["preserve", "adaptive", "safe"]
JapaneseFontMode = Literal["compat", "paper-like"]
ChunkKind = Literal[
    "body",
    "section_title",
    "paragraph_title",
    "caption",
    "list_item",
]


@dataclass(slots=True, frozen=True)
class TranslateRequest:
    """翻訳実行リクエスト.

    Attributes:
        arxiv_ref: 入力されたarXiv ID, または URL.
        workspace_root: 実行用ワークスペースルート.
        max_chunk_chars: チャンク分割時の最大文字数.
        force: ``True`` の場合, 既存成果物とチェックポイントを破棄して最初から再実行する.
        translate_section_titles: ``False`` の場合, ``\\section`` 系タイトルは原文のまま残す.
        japanese_layout_mode: 日本語PDF向けのレイアウト補正方針.
        japanese_font_mode: 日本語PDF向けのフォント注入方針.
    """

    arxiv_ref: str
    workspace_root: Path
    max_chunk_chars: int
    force: bool = False
    translate_section_titles: bool = False
    japanese_layout_mode: JapaneseLayoutMode = "safe"
    japanese_font_mode: JapaneseFontMode = "compat"


@dataclass(slots=True, frozen=True)
class TranslateResult:
    """翻訳実行結果.

    Attributes:
        arxiv_id: 正規化済みarXiv ID.
        run_dir: 実行単位の作業ディレクトリ.
        translated_main_tex: 翻訳後メインTeXファイルパス.
        pdf_path: コンパイル済みPDFファイルパス.
        total_chunks: 翻訳対象になったチャンク総数.
    """

    arxiv_id: str
    run_dir: Path
    translated_main_tex: Path
    pdf_path: Path
    total_chunks: int


@dataclass(slots=True, frozen=True)
class TexChunk:
    """翻訳対象TeXチャンク.

    Attributes:
        index: 元文書中のチャンク順序.
        text: チャンク本文.
        digest: チャンク本文の整合性確認用ダイジェスト.
        translate: ``True`` の場合のみLLM翻訳対象として扱う.
        kind: 翻訳対象本文の文脈種別.
        start_line: 元文書内でこのチャンクが始まる1始まり行番号.
    """

    index: int
    text: str
    digest: str
    translate: bool = True
    kind: ChunkKind = "body"
    start_line: int = 1


@dataclass(slots=True, frozen=True)
class ChunkIssue:
    """翻訳中に解析対象として記録するチャンク問題情報.

    Attributes:
        relative_path: ソースルートから見た対象TeXファイルの相対パス.
        chunk_index: 文書処理順のチャンク番号.
        chunk_kind: チャンク種別.
        start_line: 元文書内での開始行番号.
        status: 問題確定時のチャンク状態.
        attempts: 当該チャンクの翻訳試行回数.
        cached: 既存チェックポイント由来の問題状態を再利用した場合は ``True``.
        error: 問題発生時のエラー文字列.
        source_preview: 元チャンク本文の短いプレビュー.
    """

    relative_path: str
    chunk_index: int
    chunk_kind: ChunkKind
    start_line: int
    status: ChunkStatus
    attempts: int
    cached: bool
    error: str | None
    source_preview: str


@dataclass(slots=True)
class ChunkProgress:
    """1チャンク分の翻訳進捗.

    Attributes:
        digest: 入力チャンクのダイジェスト.
        status: ``pending``, ``success``, ``failed``, ``fallback`` の状態.
        attempts: 翻訳試行回数.
        translated_text: 翻訳成功時の出力本文.
        last_error: 直近失敗時のエラー文字列.
    """

    digest: str
    status: ChunkStatus = "pending"
    attempts: int = 0
    translated_text: str | None = None
    last_error: str | None = None


@dataclass(slots=True)
class TranslationCheckpoint:
    """翻訳途中経過を永続化するモデル.

    Attributes:
        source_digest: 入力全文のダイジェスト.
        chunks: チャンク番号をキーにした進捗辞書.
    """

    source_digest: str
    chunks: dict[int, ChunkProgress] = field(default_factory=dict)


class ChunkTranslationFailedError(RuntimeError):
    """再試行後も翻訳に失敗したチャンクが残る場合の例外."""

    def __init__(
        self,
        failed_indices: list[int],
        *,
        failed_errors: dict[int, str] | None = None,
    ):
        """失敗チャンク番号を保持し, 例外メッセージを構築する.

        Args:
            failed_indices: 翻訳失敗のまま残ったチャンク番号配列.
            failed_errors: 失敗チャンクごとの最終エラー文字列.
        """
        self.failed_indices = failed_indices
        self.failed_errors = failed_errors or {}
        joined = ",".join(str(index) for index in failed_indices)
        if self.failed_errors:
            error_summary = "; ".join(
                f"{index}: {self.failed_errors[index]}"
                for index in failed_indices
                if index in self.failed_errors
            )
            super().__init__(f"Failed to translate chunks: {joined}. {error_summary}")
            return
        super().__init__(f"Failed to translate chunks: {joined}")
