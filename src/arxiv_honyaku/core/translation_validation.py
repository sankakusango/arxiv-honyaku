"""翻訳後チャンクの構造整合性を検査するモジュール.

このモジュールは, LLM が返した日本語文をそのまま採用してよいかを,
LaTeX 構造保全の観点から判定する. 制御綴列, 区切り記号個数, compact chunk の
改行や Markdown 混入, 見出しの不自然な文末句読点などをここで検査する.
"""

from __future__ import annotations

from collections import Counter
import logging
import re

from arxiv_honyaku.core.models import ChunkKind, TexChunk
from arxiv_honyaku.core.translation_preparation import (
    replace_math_ranges_with_placeholders,
)

_CONTROL_SEQUENCE_RE = re.compile(r"\\(?:[A-Za-z]+[*]?|.)")
_CITE_WITH_ARGS_RE = re.compile(
    r"(?P<full>(?P<cmd>\\cite[A-Za-z*]*)"
    r"(?:\s*\[[^\]]*\])*\s*(?:\{[^{}]*\})+)"
)
_SENTENCE_TERMINATOR_RE = re.compile(r"[。．!?！？]")
_MARKDOWN_MARKER_RE = re.compile(r"^\s*(?:[#*]+|-)(?=\S)")
_SOFT_FORMATTING_COMMANDS = frozenset(
    {
        "emph",
        "textbf",
        "textit",
        "textrm",
        "textsf",
        "texttt",
        "underline",
    }
)
_COMPACT_CHUNK_KINDS: frozenset[ChunkKind] = frozenset(
    {"section_title", "paragraph_title", "caption", "list_item"}
)

LOGGER = logging.getLogger("arxiv_honyaku.core.translation_validation")


def validate_translated_chunk(chunk: TexChunk, translated_text: str) -> str:
    """翻訳済みチャンクが, 元チャンクの TeX 構造を壊していないか検査する.

    まず機械的に適用できる正規化を行い, その後で LaTeX 制御綴, 区切り記号,
    compact chunk 特有の制約を順に検証する. 返り値は validator を通過した
    正規化済み文字列であり, 呼び出し元はこれを checkpoint や TeX 出力へ保存する.

    Args:
        chunk: 元の翻訳対象チャンク.
        translated_text: LLMが返した翻訳文字列.

    Returns:
        str: 検査を通過した翻訳文字列.

    Raises:
        ValueError: TeX制御綴, 区切り記号, 断片種別に対する制約のいずれかを破った場合.
    """
    normalized_text = _normalize_translated_text(
        chunk=chunk,
        translated_text=translated_text,
    )
    normalized_text = _remove_unexpected_citation_commands(
        chunk=chunk,
        source_text=chunk.text,
        translated_text=normalized_text,
    )
    if not normalized_text.strip():
        raise ValueError("Translated chunk is empty")

    _validate_control_sequences_unchanged(
        source_text=chunk.text,
        translated_text=normalized_text,
    )
    _validate_delimiters_unchanged(
        source_text=chunk.text,
        translated_text=normalized_text,
    )
    _validate_chunk_kind_constraints(chunk=chunk, translated_text=normalized_text)
    return normalized_text


def _remove_unexpected_citation_commands(
    *,
    chunk: TexChunk,
    source_text: str,
    translated_text: str,
) -> str:
    """翻訳で不正に増えた ``\\cite...`` だけを除去する.

    ``\\cite`` 系コマンドは, LLM が原文に無い参照を幻覚として挿入しやすい.
    この関数は制御綴多重集合の差分から余剰 ``\\cite`` 系だけを取り除き,
    取り除いた事実を warning ログへ残す.

    Args:
        chunk: 対象チャンク.
        source_text: 原文チャンク.
        translated_text: 正規化済み翻訳文字列.

    Returns:
        str: 余剰 ``\\cite`` 系を除去した文字列.
    """
    source_sequences = _CONTROL_SEQUENCE_RE.findall(
        replace_math_ranges_with_placeholders(
            _strip_soft_formatting_commands(source_text)
        )
    )
    translated_sequences = _CONTROL_SEQUENCE_RE.findall(
        replace_math_ranges_with_placeholders(
            _strip_soft_formatting_commands(translated_text)
        )
    )
    source_cite_counter = Counter(
        sequence for sequence in source_sequences if _is_citation_command(sequence)
    )
    translated_cite_counter = Counter(
        sequence for sequence in translated_sequences if _is_citation_command(sequence)
    )
    extra_cites = translated_cite_counter - source_cite_counter
    if not extra_cites:
        return translated_text

    rebuilt_parts: list[str] = []
    cursor = 0
    removed_commands: list[str] = []
    for match in _CITE_WITH_ARGS_RE.finditer(translated_text):
        command = match.group("cmd")
        if extra_cites.get(command, 0) <= 0:
            continue
        rebuilt_parts.append(translated_text[cursor : match.start()])
        cursor = match.end()
        extra_cites[command] -= 1
        removed_commands.append(match.group("full"))

    if not removed_commands:
        return translated_text

    rebuilt_parts.append(translated_text[cursor:])
    repaired_text = "".join(rebuilt_parts)
    repaired_text = re.sub(r"[ \t]{2,}", " ", repaired_text)
    repaired_text = re.sub(r"[ \t]+([.,;:!?。．！？])", r"\1", repaired_text)
    LOGGER.warning(
        "Removed unexpected citation commands from translated chunk. "
        "line=%s chunk=%s kind=%s removed=%s preview=%s",
        chunk.start_line,
        chunk.index,
        chunk.kind,
        len(removed_commands),
        _preview_text(" ".join(removed_commands)),
    )
    return repaired_text


def _normalize_translated_text(
    *,
    chunk: TexChunk,
    translated_text: str,
) -> str:
    """翻訳後文字列へ機械的に適用できる安全な正規化を行う.

    ここで行うのは, 元の外側空白の復元, compact chunk の改行畳み込み,
    見出しの不要な末尾句読点除去のように, 意味を変えずに構造整合性を上げる
    正規化だけである.

    Args:
        chunk: 元の翻訳対象チャンク.
        translated_text: LLMが返した翻訳文字列.

    Returns:
        str: 外側空白と compact chunk の改行を整えた文字列.
    """
    normalized = _preserve_outer_whitespace_like_source(
        source_text=chunk.text,
        translated_text=translated_text,
    )
    if chunk.kind in _COMPACT_CHUNK_KINDS:
        normalized = _collapse_compact_chunk_whitespace(
            source_text=chunk.text,
            translated_text=normalized,
        )
    if chunk.kind in {"section_title", "paragraph_title"}:
        normalized = _strip_unexpected_heading_punctuation(
            source_text=chunk.text,
            translated_text=normalized,
        )
    return normalized


def _preserve_outer_whitespace_like_source(
    *,
    source_text: str,
    translated_text: str,
) -> str:
    """元文字列の先頭末尾空白を保ったまま翻訳文字列を整える.

    Args:
        source_text: 元チャンク本文.
        translated_text: 翻訳後本文.

    Returns:
        str: 元本文の外側空白を復元した文字列.
    """
    if not translated_text.strip():
        return translated_text

    source_leading_length = len(source_text) - len(source_text.lstrip())
    source_trailing_length = len(source_text) - len(source_text.rstrip())
    source_trailing_start = (
        len(source_text) - source_trailing_length
        if source_trailing_length
        else len(source_text)
    )
    core_text = translated_text.strip()
    return (
        f"{source_text[:source_leading_length]}"
        f"{core_text}"
        f"{source_text[source_trailing_start:]}"
    )


def _collapse_compact_chunk_whitespace(
    *,
    source_text: str,
    translated_text: str,
) -> str:
    """compact chunk の内部改行を1行へ畳み込む.

    Args:
        source_text: 元チャンク本文.
        translated_text: 翻訳後本文.

    Returns:
        str: 内部改行を空白へ畳み込んだ文字列.
    """
    source_leading_length = len(source_text) - len(source_text.lstrip())
    source_trailing_length = len(source_text) - len(source_text.rstrip())
    source_trailing_start = (
        len(source_text) - source_trailing_length
        if source_trailing_length
        else len(source_text)
    )
    core_text = translated_text[source_leading_length:source_trailing_start]
    collapsed_core = re.sub(r"[ \t\f\v]*[\r\n]+[ \t\f\v]*", " ", core_text)
    collapsed_core = re.sub(r"[ \t]{2,}", " ", collapsed_core).strip()
    return (
        f"{source_text[:source_leading_length]}"
        f"{collapsed_core}"
        f"{source_text[source_trailing_start:]}"
    )


def _strip_unexpected_heading_punctuation(
    *,
    source_text: str,
    translated_text: str,
) -> str:
    """原文見出しに無い文末句読点を機械的に取り除く.

    Args:
        source_text: 元チャンク本文.
        translated_text: 翻訳後本文.

    Returns:
        str: 不要な末尾句読点を落とした文字列.
    """
    source_stripped = source_text.strip()
    if _SENTENCE_TERMINATOR_RE.search(source_stripped):
        return translated_text

    source_leading_length = len(source_text) - len(source_text.lstrip())
    source_trailing_length = len(source_text) - len(source_text.rstrip())
    source_trailing_start = (
        len(source_text) - source_trailing_length
        if source_trailing_length
        else len(source_text)
    )
    core_text = translated_text[source_leading_length:source_trailing_start]
    stripped_core = core_text.rstrip()
    trimmed_core = re.sub(r"[。．!?！？]+\Z", "", stripped_core).rstrip()
    trailing_whitespace = core_text[len(stripped_core) :]
    return (
        f"{source_text[:source_leading_length]}"
        f"{trimmed_core}"
        f"{trailing_whitespace}"
        f"{source_text[source_trailing_start:]}"
    )


def _validate_control_sequences_unchanged(
    *,
    source_text: str,
    translated_text: str,
) -> None:
    """LaTeX制御綴の多重集合が変化していないことを検査する.

    Args:
        source_text: 元チャンク本文.
        translated_text: 翻訳後本文.

    Returns:
        None: 常に ``None``.

    Raises:
        ValueError: 制御綴の種類または個数が変化している場合.
    """
    source_sequences = _CONTROL_SEQUENCE_RE.findall(
        replace_math_ranges_with_placeholders(
            _strip_soft_formatting_commands(source_text)
        )
    )
    translated_sequences = _CONTROL_SEQUENCE_RE.findall(
        replace_math_ranges_with_placeholders(
            _strip_soft_formatting_commands(translated_text)
        )
    )
    if Counter(source_sequences) != Counter(translated_sequences):
        raise ValueError(
            "Translated chunk changed LaTeX control sequences: "
            f"expected={source_sequences[:6]}, actual={translated_sequences[:6]}, "
            f"translated_output={_preview_text(translated_text)}"
        )


def _validate_delimiters_unchanged(
    *,
    source_text: str,
    translated_text: str,
) -> None:
    """構造に関わる区切り記号の個数が変化していないことを検査する.

    Args:
        source_text: 元チャンク本文.
        translated_text: 翻訳後本文.

    Returns:
        None: 常に ``None``.

    Raises:
        ValueError: 波括弧, 角括弧, ドル記号の個数が変化している場合.
    """
    normalized_source = _strip_soft_formatting_commands(source_text)
    normalized_translated = _strip_soft_formatting_commands(translated_text)
    for delimiter in ("{", "}", "[", "]", "$"):
        if normalized_source.count(delimiter) != normalized_translated.count(delimiter):
            raise ValueError(
                "Translated chunk changed LaTeX delimiter counts: "
                f"{delimiter}, translated_output={_preview_text(translated_text)}"
            )


def _validate_chunk_kind_constraints(
    *,
    chunk: TexChunk,
    translated_text: str,
) -> None:
    """チャンク種別ごとの自然言語制約を検査する.

    ``section_title``, ``paragraph_title``, ``caption``, ``list_item`` のような
    compact chunk は, 本文段落と違って 1 行, 非 Markdown, 過剰膨張なし,
    といった制約が強い. この関数はそうした種別依存の制約だけを担当する.

    Args:
        chunk: 元の翻訳対象チャンク.
        translated_text: 翻訳後本文.

    Returns:
        None: 常に ``None``.

    Raises:
        ValueError: 種別に対して不自然な改行, Markdown記法, 過剰な膨張がある場合.
    """
    if chunk.kind not in _COMPACT_CHUNK_KINDS:
        return

    stripped = translated_text.strip()
    if "\n" in stripped or "\r" in stripped:
        raise ValueError(
            "Compact chunk translation must stay on a single line: "
            f"{_preview_text(translated_text)}"
        )
    if _MARKDOWN_MARKER_RE.search(stripped):
        raise ValueError(
            "Compact chunk translation must not include Markdown markers: "
            f"{_preview_text(translated_text)}"
        )

    source_stripped = chunk.text.strip()
    if source_stripped and len(stripped) > max(
        len(source_stripped) * 4, len(source_stripped) + 40
    ):
        raise ValueError(
            "Compact chunk translation expanded too much: "
            f"{_preview_text(translated_text)}"
        )

    if (
        chunk.kind in {"section_title", "paragraph_title"}
        and not _SENTENCE_TERMINATOR_RE.search(source_stripped)
        and _SENTENCE_TERMINATOR_RE.search(stripped)
    ):
        raise ValueError(
            "Heading translation must not introduce sentence-ending punctuation: "
            f"{_preview_text(translated_text)}"
        )


def _preview_text(text: str, *, max_length: int = 120) -> str:
    """例外メッセージ向けに文字列を短い1行プレビューへ整形する.

    Args:
        text: 整形対象文字列.
        max_length: 出力する最大文字数.

    Returns:
        str: 1行へ潰して切り詰めたプレビュー文字列.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= max_length:
        return collapsed
    return f"{collapsed[: max_length - 3]}..."


def _is_citation_command(sequence: str) -> bool:
    """制御綴が ``\\cite`` 系かを判定する."""
    return sequence.startswith("\\cite")


def _strip_soft_formatting_commands(text: str) -> str:
    """装飾専用コマンドを落として本文だけを取り出す.

    ``\\textbf{...}`` や ``\\emph{...}`` は, 日本語化時に残っていても
    落ちていても意味上の差が小さい. そこで validator ではこれらを
    構造破壊と見なさず, 中身だけを比較対象へ残す.

    Args:
        text: 検査対象のTeX断片.

    Returns:
        str: 装飾コマンドの外殻を除いた文字列.
    """
    result: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] != "\\":
            result.append(text[cursor])
            cursor += 1
            continue

        command_match = re.match(r"\\([A-Za-z]+[*]?)", text[cursor:])
        if command_match is None:
            result.append(text[cursor])
            cursor += 1
            continue

        command_name = command_match.group(1)
        command_end = cursor + len(command_match.group(0))
        if (
            command_name not in _SOFT_FORMATTING_COMMANDS
            or command_end >= len(text)
            or text[command_end] != "{"
        ):
            result.append(command_match.group(0))
            cursor = command_end
            continue

        argument_end = _find_balanced_brace_end(text, command_end)
        if argument_end is None:
            result.append(command_match.group(0))
            cursor = command_end
            continue

        inner_text = text[command_end + 1 : argument_end]
        result.append(_strip_soft_formatting_commands(inner_text))
        cursor = argument_end + 1

    return "".join(result)


def _find_balanced_brace_end(text: str, open_index: int) -> int | None:
    """指定位置の ``{`` に対応する ``}`` を返す.

    Args:
        text: 探索対象文字列.
        open_index: 開き波括弧の位置.

    Returns:
        int | None: 対応する閉じ波括弧位置. 見つからない場合は ``None``.
    """
    depth = 0
    cursor = open_index
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            cursor += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cursor
        cursor += 1
    return None
