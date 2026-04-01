"""翻訳前に, 壊してほしくない inline LaTeX を退避, 復元するモジュール.

このモジュールは, 本文の自然な翻訳と LaTeX 構造の保全を両立するための
前処理層である. 数式, 参照, 引用, URL など, LLM に直接編集させると
壊れやすい断片を placeholder へ退避し, 翻訳後に元の TeX 断片へ戻す.
さらに, 通常翻訳が失敗したときに plain text だけを訳し直すための
segment fallback 用断片列もここで組み立てる.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from arxiv_honyaku.core.models import TexChunk

_PROTECTED_COMMAND_NAMES = frozenset(
    {
        "SI",
        "SIlist",
        "SIrange",
        "autoref",
        "Cref",
        "cite",
        "citealp",
        "citealt",
        "citeauthor",
        "citep",
        "citet",
        "citeyear",
        "citeyearpar",
        "cref",
        "eqref",
        "includegraphics",
        "label",
        "nocite",
        "num",
        "numlist",
        "numrange",
        "pageref",
        "qty",
        "qtyproduct",
        "qtyrange",
        "ref",
        "url",
    }
)
_UNPROTECTED_STANDALONE_COMMAND_NAMES = frozenset(
    {
        "begin",
        "end",
        "item",
    }
)
_PROTECTED_SINGLE_CHAR_COMMANDS = frozenset(
    {
        "\\",
        " ",
        "%",
        "&",
        "#",
        "_",
        "{",
        "}",
    }
)
_PROTECTED_SINGLE_CHAR_SYMBOL_ACCENT_COMMANDS = frozenset(
    {
        '"',
        "'",
        "`",
        "^",
        "~",
        "=",
        ".",
    }
)
_PROTECTED_SINGLE_CHAR_ALPHA_ACCENT_COMMANDS = frozenset(
    {
        "b",
        "c",
        "d",
        "H",
        "k",
        "r",
        "u",
        "v",
    }
)
_PLACEHOLDER_PREFIX = "XQK"
_MATH_PLACEHOLDER_PREFIX = "XQM"
ProtectedSpanKind = Literal["command", "math"]


@dataclass(slots=True, frozen=True)
class ProtectedSpan:
    """翻訳前に退避した断片.

    Attributes:
        placeholder: 翻訳時に埋め込む代替トークン.
        source_text: 復元時に戻す元のTeX断片.
        prompt_text: LLMへ渡す際に使う placeholder 化済み断片.
        kind: 保護断片の種別, 数式か通常コマンドかを表す.
    """

    placeholder: str
    source_text: str
    prompt_text: str
    kind: ProtectedSpanKind


@dataclass(slots=True, frozen=True)
class PreparedTranslationInput:
    """翻訳器へ渡す直前の, 復元情報付きチャンク情報.

    Attributes:
        source_text: 元のチャンク本文.
        prompt_text: placeholder 化後に, 実際に LLM へ渡す本文.
        protected_spans: 翻訳後復元と fallback 再構成に使う保護断片配列.
    """

    source_text: str
    prompt_text: str
    protected_spans: tuple[ProtectedSpan, ...]


@dataclass(slots=True, frozen=True)
class PreparedTranslationSegment:
    """前処理後入力を再分割した断片.

    ``PreparedTranslationInput`` を fallback 用にさらに分解した要素であり,
    ``translate=True`` の断片だけを LLM に渡し, ``False`` の断片は原文のまま
    出力へ差し戻す.

    Attributes:
        translate: ``True`` の場合のみ翻訳対象として扱う.
        text: 断片文字列.
    """

    translate: bool
    text: str


class ProtectedPlaceholderError(ValueError):
    """保護 placeholder の欠落, 重複, 不正な並び替えを検出した場合の例外."""


def prepare_translation_input(chunk: TexChunk) -> PreparedTranslationInput:
    """チャンク中の inline LaTeX を placeholder 化した翻訳入力へ変換する.

    数式や保護対象コマンドを左から順に抽出し, 元本文を ``prompt_text`` と
    ``protected_spans`` に分解する. command 系は可能な限り
    ``\\ref{XQK0000}`` や ``\\SI{XQK0001}`` のように LaTeX の外形を残し,
    math 系は ``$XQM0000$`` のように区切りだけを保持する. さらに
    ``\\llamathree`` のような stand-alone マクロも bare placeholder として
    保護する. ``\\textbf`` や ``\\emph`` のような装飾コマンドは,
    日本語化の自由度を優先して placeholder 化しない.

    Args:
        chunk: 翻訳対象チャンク.

    Returns:
        PreparedTranslationInput: LLM入力用本文と復元情報.
    """
    protected_ranges = _find_protected_ranges(chunk.text)
    if not protected_ranges:
        return PreparedTranslationInput(
            source_text=chunk.text,
            prompt_text=chunk.text,
            protected_spans=(),
        )

    prompt_parts: list[str] = []
    protected_spans: list[ProtectedSpan] = []
    cursor = 0
    for index, (start_index, end_index, kind) in enumerate(protected_ranges):
        prompt_parts.append(chunk.text[cursor:start_index])
        placeholder = _build_unique_placeholder(
            source_text=chunk.text,
            prefix=_PLACEHOLDER_PREFIX,
            index=index,
        )
        span_prompt_text = _build_span_prompt_text(
            source_text=chunk.text[start_index:end_index],
            kind=kind,
            placeholder=placeholder,
        )
        prompt_parts.append(span_prompt_text)
        protected_spans.append(
            ProtectedSpan(
                placeholder=placeholder,
                source_text=chunk.text[start_index:end_index],
                prompt_text=span_prompt_text,
                kind=kind,
            )
        )
        cursor = end_index
    prompt_parts.append(chunk.text[cursor:])
    return PreparedTranslationInput(
        source_text=chunk.text,
        prompt_text="".join(prompt_parts),
        protected_spans=tuple(protected_spans),
    )


def split_prepared_input_into_segments(
    prepared_input: PreparedTranslationInput,
    *,
    preserve_math_context: bool = True,
) -> list[PreparedTranslationSegment]:
    """前処理済み入力を, 翻訳対象断片と固定断片へ再分割する.

    数式 placeholder は翻訳対象断片の中へ残し, 引用や参照などの通常コマンドだけを
    固定断片として切り出す. これにより, fallback 時も数式前後の文章をまとめて翻訳しやすくする.
    ``preserve_math_context=False`` の場合は, 数式 placeholder も固定断片として扱い,
    すべての protected span を原文のまま保持する厳格 fallback 用の断片列を返す.

    Args:
        prepared_input: 事前退避済みの翻訳入力情報.
        preserve_math_context: 数式 placeholder を翻訳対象断片内へ残す場合は ``True``.

    Returns:
        list[PreparedTranslationSegment]: 順序保持された断片配列.
    """
    if not prepared_input.protected_spans:
        return [
            PreparedTranslationSegment(translate=True, text=prepared_input.prompt_text)
        ]

    segments: list[PreparedTranslationSegment] = []
    cursor = 0
    for span in prepared_input.protected_spans:
        prompt_index = prepared_input.prompt_text.find(span.prompt_text, cursor)
        if prompt_index < 0:
            raise ValueError(
                f"Missing placeholder in prepared prompt: {span.prompt_text}"
            )
        if preserve_math_context and span.kind == "math":
            continue
        if prompt_index > cursor:
            segments.append(
                PreparedTranslationSegment(
                    translate=True,
                    text=prepared_input.prompt_text[cursor:prompt_index],
                )
            )
        segments.append(
            PreparedTranslationSegment(
                translate=False,
                text=span.source_text,
            )
        )
        cursor = prompt_index + len(span.prompt_text)
    if cursor < len(prepared_input.prompt_text):
        segments.append(
            PreparedTranslationSegment(
                translate=True,
                text=prepared_input.prompt_text[cursor:],
            )
        )
    return segments


def restore_translation_output(
    prepared_input: PreparedTranslationInput,
    translated_text: str,
    *,
    restore_kinds: frozenset[ProtectedSpanKind] | None = None,
) -> str:
    """placeholder を元の inline LaTeX へ戻し, 構造破壊を検査する.

    復元前に, LLM が placeholder へ勝手に被せた数式区切りやコマンド外形を
    正規化してから, 各 protected span を元の ``source_text`` へ戻す.
    その過程で, placeholder の欠落や重複を検知する. なお, 日本語化では句や節の
    順序が自然に入れ替わることがあるため, placeholder の並び順自体は強制しない.

    Args:
        prepared_input: 事前退避済みの翻訳入力情報.
        translated_text: LLMが返した翻訳文字列.
        restore_kinds: 復元対象の保護断片種別集合, ``None`` の場合は全種別を復元する.

    Returns:
        str: inline LaTeX を復元した翻訳文字列.

    Raises:
        ValueError: プレースホルダが欠落, または重複した場合.
    """
    if not prepared_input.protected_spans:
        return translated_text

    restored_text = _normalize_placeholder_wrappers(
        prepared_input=prepared_input,
        translated_text=translated_text,
    )
    target_spans = tuple(
        span
        for span in prepared_input.protected_spans
        if restore_kinds is None or span.kind in restore_kinds
    )
    if not target_spans:
        return restored_text

    for span in target_spans:
        occurrences = restored_text.count(span.placeholder)
        if occurrences != 1:
            raise ProtectedPlaceholderError(
                "Translated chunk lost or duplicated a protected placeholder: "
                f"{span.placeholder} occurrences={occurrences} "
                f"raw_output={_preview_translation_text(translated_text)} "
                f"normalized_output={_preview_translation_text(restored_text)}"
            )

    for span in target_spans:
        restored_text = restored_text.replace(span.placeholder, span.source_text)
    return restored_text


def _normalize_placeholder_wrappers(
    *,
    prepared_input: PreparedTranslationInput,
    translated_text: str,
) -> str:
    """LLM が placeholder に付与した余計な LaTeX ラッパを取り除く.

    LLMが ``\\cite{PLACEHOLDER}``, ``\\cite PLACEHOLDER``, ``$PLACEHOLDER$`` のように
    placeholder をLaTeX断片として包み直すことがある. この関数は, 元の protected span を
    復元しやすい形へ正規化する.

    Args:
        prepared_input: 事前退避済みの翻訳入力情報.
        translated_text: LLMが返した翻訳文字列.

    Returns:
        str: placeholder 周辺を正規化した翻訳文字列.
    """
    normalized = translated_text
    for span in prepared_input.protected_spans:
        escaped_placeholder = re.escape(span.placeholder)
        normalized = normalized.replace(span.prompt_text, span.placeholder)
        normalized = re.sub(
            rf"\\[A-Za-z]+[*]?(?:\s*\[[^\[\]{{}}]*\]){{0,2}}\s*\{{\s*({escaped_placeholder})\s*\}}",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            rf"\\[A-Za-z]+[*]?\s+({escaped_placeholder})",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            rf"\\({escaped_placeholder})",
            r"\1",
            normalized,
        )
        if span.kind != "math":
            continue
        normalized = normalized.replace(f"${span.placeholder}$", span.placeholder)
        normalized = normalized.replace(rf"\({span.placeholder}\)", span.placeholder)
        normalized = normalized.replace(rf"\[{span.placeholder}\]", span.placeholder)
        normalized = re.sub(
            rf"\$\s*({escaped_placeholder})\s*\$",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            rf"\\\(\s*({escaped_placeholder})\s*\\\)",
            r"\1",
            normalized,
        )
        normalized = re.sub(
            rf"\\\[\s*({escaped_placeholder})\s*\\\]",
            r"\1",
            normalized,
        )
    return normalized


def _build_span_prompt_text(
    *,
    source_text: str,
    kind: ProtectedSpanKind,
    placeholder: str,
) -> str:
    """元断片から, LLMへ見せる placeholder 化済み断片を構築する.

    Args:
        source_text: 元の保護対象断片.
        kind: 保護断片の種別.
        placeholder: 埋め込む placeholder.

    Returns:
        str: LLM入力へ挿入する placeholder 化済み断片.
    """
    if kind == "math":
        return _build_math_prompt_text(source_text, placeholder)
    return _build_command_prompt_text(source_text, placeholder)


def _build_math_prompt_text(source_text: str, placeholder: str) -> str:
    """数式断片の区切りを残したまま placeholder 化した文字列を返す.

    Args:
        source_text: 元の数式断片.
        placeholder: 埋め込む placeholder.

    Returns:
        str: 数式の外側区切りだけを残した placeholder 文字列.
    """
    if source_text.startswith("$$") and source_text.endswith("$$"):
        return f"$${placeholder}$$"
    if source_text.startswith(r"\(") and source_text.endswith(r"\)"):
        return rf"\({placeholder}\)"
    if source_text.startswith(r"\[") and source_text.endswith(r"\]"):
        return rf"\[{placeholder}\]"
    if source_text.startswith("$") and source_text.endswith("$"):
        return f"${placeholder}$"
    return placeholder


def _build_command_prompt_text(source_text: str, placeholder: str) -> str:
    """コマンドの外形を残したまま placeholder 化した文字列を返す.

    参照, 引用, URL など, 引数自体を翻訳しないコマンド向けに
    ``\\ref{XQK0000}`` のような形を作る. こうすることで LLM は
    bare placeholder よりも元の LaTeX 文法を推測しやすくなる.

    Args:
        source_text: 元のコマンド断片.
        placeholder: 埋め込む placeholder.

    Returns:
        str: 可能ならコマンド外形を保持した placeholder 文字列, できない場合は bare placeholder.
    """
    if not source_text.startswith("\\"):
        return placeholder
    if len(source_text) >= 2 and source_text[1] in _PROTECTED_SINGLE_CHAR_COMMANDS:
        return placeholder

    command_name, cursor = _parse_command_name(source_text, 0)
    if command_name is None:
        return placeholder

    cursor = _consume_inline_whitespace(source_text, cursor)
    optional_ranges_consumed = 0
    while (
        cursor < len(source_text)
        and source_text[cursor] == "["
        and optional_ranges_consumed < 2
    ):
        optional_end = _consume_balanced(source_text, cursor, "[", "]")
        if optional_end is None:
            return placeholder
        cursor = _consume_inline_whitespace(source_text, optional_end)
        optional_ranges_consumed += 1

    if cursor >= len(source_text) or source_text[cursor] != "{":
        return placeholder
    required_end = _consume_balanced(source_text, cursor, "{", "}")
    if required_end is None:
        return placeholder
    return f"{source_text[: cursor + 1]}{placeholder}{source_text[required_end - 1 :]}"


def _preview_translation_text(text: str, *, max_length: int = 120) -> str:
    """ログや例外向けに翻訳文字列を短い1行プレビューへ整形する.

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


def _preview_placeholder_order(
    spans: tuple[ProtectedSpan, ...] | list[ProtectedSpan],
    *,
    max_items: int = 6,
) -> list[str]:
    """ログ向けに placeholder の並びと元断片を短く整形する.

    Args:
        spans: 表示対象の protected span 配列.
        max_items: 表示する最大件数.

    Returns:
        list[str]: ``placeholder=source`` 形式へ整えた短い配列.
    """
    previews = [
        f"{span.placeholder}={_preview_translation_text(span.source_text, max_length=24)}"
        for span in spans[:max_items]
    ]
    if len(spans) > max_items:
        previews.append("...")
    return previews


def _build_unique_placeholder(
    *,
    source_text: str,
    prefix: str,
    index: int,
) -> str:
    """元テキストと衝突しない短い placeholder を生成する.

    Args:
        source_text: 衝突判定対象の元テキスト.
        prefix: placeholder 接頭辞.
        index: 断片順序を表す整数.

    Returns:
        str: 元テキストに含まれない placeholder 文字列.
    """
    suffix = ""
    while True:
        placeholder = f"{prefix}{index:04X}{suffix}"
        if placeholder not in source_text:
            return placeholder
        suffix += "X"


def replace_math_ranges_with_placeholders(text: str) -> str:
    """数式断片を比較専用 placeholder へ置き換えた文字列を返す.

    これは翻訳入力生成用ではなく, validation 時に LaTeX 制御綴の比較を安定させる
    ための補助関数である.

    Args:
        text: 数式断片を含む文字列.

    Returns:
        str: 数式を placeholder へ置き換えた文字列.
    """
    parts: list[str] = []
    cursor = 0
    math_index = 0
    for start_index, end_index in _find_math_ranges(text):
        parts.append(text[cursor:start_index])
        parts.append(
            _build_unique_placeholder(
                source_text=text,
                prefix=_MATH_PLACEHOLDER_PREFIX,
                index=math_index,
            )
        )
        cursor = end_index
        math_index += 1
    parts.append(text[cursor:])
    return "".join(parts)


def _find_protected_ranges(text: str) -> list[tuple[int, int, ProtectedSpanKind]]:
    """保護対象の範囲を左から順に列挙する.

    math と command の両方を単一の走査で抽出し, 元文字列中の開始位置順に返す.
    後段ではこの範囲列を ``ProtectedSpan`` へ変換し, ``prompt_text`` の組み立てや
    復元順序検査に使う.

    Args:
        text: 解析対象チャンク本文.

    Returns:
        list[tuple[int, int, ProtectedSpanKind]]: ``(start, end, kind)`` の範囲配列.
    """
    ranges: list[tuple[int, int, ProtectedSpanKind]] = []
    cursor = 0
    while cursor < len(text):
        protected_end = _consume_math_range(text, cursor)
        if protected_end is not None:
            ranges.append((cursor, protected_end, "math"))
            cursor = protected_end
            continue

        protected_end = _consume_protected_command_range(text, cursor)
        if protected_end is None:
            cursor += 1
            continue
        ranges.append((cursor, protected_end, "command"))
        cursor = protected_end
    return ranges


def _find_math_ranges(text: str) -> list[tuple[int, int]]:
    """数式断片の範囲を左から順に列挙する.

    ``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]`` を対象に,
    外側区切り単位で math range を返す.

    Args:
        text: 解析対象チャンク本文.

    Returns:
        list[tuple[int, int]]: ``(start, end)`` の数式範囲配列.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        protected_end = _consume_math_range(text, cursor)
        if protected_end is None:
            cursor += 1
            continue
        ranges.append((cursor, protected_end))
        cursor = protected_end
    return ranges


def _consume_math_range(text: str, start_index: int) -> int | None:
    """数式断片なら終端位置の次を返す.

    Args:
        text: 解析対象チャンク本文.
        start_index: 先頭候補位置.

    Returns:
        int | None: 数式終端の次位置, 対象外なら ``None``.
    """
    if text.startswith("$$", start_index):
        return _find_unescaped_substring(text, "$$", start_index + 2)
    if text.startswith(r"\(", start_index):
        closing_index = text.find(r"\)", start_index + 2)
        if closing_index < 0:
            return None
        return closing_index + 2
    if text.startswith(r"\[", start_index):
        closing_index = text.find(r"\]", start_index + 2)
        if closing_index < 0:
            return None
        return closing_index + 2
    if text[start_index] != "$":
        return None
    return _find_unescaped_substring(text, "$", start_index + 1)


def _consume_protected_command_range(text: str, start_index: int) -> int | None:
    """保護対象コマンド断片なら終端位置の次を返す.

    Args:
        text: 解析対象チャンク本文.
        start_index: 先頭候補位置.

    Returns:
        int | None: コマンド終端の次位置, 対象外なら ``None``.
    """
    if start_index >= len(text) or text[start_index] != "\\":
        return None

    if (
        start_index + 1 < len(text)
        and text[start_index + 1] in _PROTECTED_SINGLE_CHAR_COMMANDS
    ):
        return start_index + 2
    if (
        start_index + 1 < len(text)
        and text[start_index + 1] in _PROTECTED_SINGLE_CHAR_SYMBOL_ACCENT_COMMANDS
    ):
        cursor = start_index + 2
        if cursor < len(text) and text[cursor] == "{":
            accent_end = _consume_balanced(text, cursor, "{", "}")
            if accent_end is None:
                return None
            return accent_end
        if cursor < len(text) and not text[cursor].isspace():
            return cursor + 1
        return cursor

    command_name, cursor = _parse_command_name(text, start_index)
    if command_name is None:
        return None
    if command_name in _PROTECTED_COMMAND_NAMES:
        cursor = _consume_inline_whitespace(text, cursor)
        optional_ranges_consumed = 0
        while (
            cursor < len(text) and text[cursor] == "[" and optional_ranges_consumed < 2
        ):
            optional_end = _consume_balanced(text, cursor, "[", "]")
            if optional_end is None:
                return None
            cursor = _consume_inline_whitespace(text, optional_end)
            optional_ranges_consumed += 1

        if cursor >= len(text) or text[cursor] != "{":
            return cursor
        required_end = _consume_balanced(text, cursor, "{", "}")
        if required_end is None:
            return None
        return required_end

    if _should_protect_standalone_command(
        text=text,
        command_name=command_name,
        cursor=cursor,
    ):
        return cursor

    if command_name not in _PROTECTED_SINGLE_CHAR_ALPHA_ACCENT_COMMANDS:
        return None

    if cursor < len(text) and text[cursor] == "{":
        accent_end = _consume_balanced(text, cursor, "{", "}")
        if accent_end is None:
            return None
        return accent_end
    if cursor < len(text) and not text[cursor].isspace():
        return cursor + 1
    return cursor


def _should_protect_standalone_command(
    *,
    text: str,
    command_name: str,
    cursor: int,
) -> bool:
    """引数を伴わない stand-alone マクロを保護対象にするか判定する.

    ``\\llamathree`` のような model 名マクロは, 翻訳時に落ちやすい一方で
    そのまま残せば十分なことが多い. ここでは, 露骨に構造制御用途のコマンドを
    除き, 引数無しの multi-letter command を保護対象として扱う.

    Args:
        text: 解析対象チャンク本文.
        command_name: 読み取ったコマンド名.
        cursor: コマンド名読み取り後の位置.

    Returns:
        bool: stand-alone command として保護する場合は ``True``.
    """
    if len(command_name) <= 1:
        return False
    if command_name in _UNPROTECTED_STANDALONE_COMMAND_NAMES:
        return False
    if cursor >= len(text):
        return True
    return text[cursor] not in {"{", "["}


def _parse_command_name(text: str, start_index: int) -> tuple[str | None, int]:
    """制御綴名と読み進め位置を返す.

    Args:
        text: 解析対象チャンク本文.
        start_index: 先頭の ``\\`` 位置.

    Returns:
        tuple[str | None, int]: コマンド名と, コマンド名読み取り後の位置.
    """
    cursor = start_index + 1
    while cursor < len(text) and text[cursor].isalpha():
        cursor += 1
    if cursor == start_index + 1:
        return None, start_index + 1
    if cursor < len(text) and text[cursor] == "*":
        cursor += 1
    return text[start_index + 1 : cursor], cursor


def _consume_inline_whitespace(text: str, start_index: int) -> int:
    """改行以外の空白を読み飛ばす.

    Args:
        text: 解析対象チャンク本文.
        start_index: 読み飛ばし開始位置.

    Returns:
        int: 空白読み飛ばし後の位置.
    """
    cursor = start_index
    while cursor < len(text) and text[cursor] in {" ", "\t"}:
        cursor += 1
    return cursor


def _consume_balanced(
    text: str,
    start_index: int,
    opening: str,
    closing: str,
) -> int | None:
    """対応する閉じ記号までを探索し, 終端位置の次を返す.

    Args:
        text: 解析対象文字列.
        start_index: 開き記号の位置.
        opening: 開き記号.
        closing: 閉じ記号.

    Returns:
        int | None: 閉じ記号直後の位置, 見つからない場合は ``None``.
    """
    if start_index >= len(text) or text[start_index] != opening:
        return None

    depth = 0
    cursor = start_index
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            cursor += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _find_unescaped_substring(
    text: str,
    target: str,
    start_index: int,
) -> int | None:
    """エスケープされていない部分文字列を探索し, 終端位置の次を返す.

    Args:
        text: 解析対象文字列.
        target: 探索対象文字列.
        start_index: 探索開始位置.

    Returns:
        int | None: 見つかった終端位置の次, 未検出なら ``None``.
    """
    cursor = start_index
    while cursor < len(text):
        match_index = text.find(target, cursor)
        if match_index < 0:
            return None
        if match_index == 0 or text[match_index - 1] != "\\":
            return match_index + len(target)
        cursor = match_index + len(target)
    return None
