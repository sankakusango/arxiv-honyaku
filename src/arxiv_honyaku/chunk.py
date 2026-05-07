"""⑥ 翻訳対象本文だけを chunk として切り出す.

入力: 置換済み (placeholder 入り) の本文文字列.
出力: Chunk のリスト. `text + join_after` を順に連結すると元の本文に戻る.

例:
    "\\section{Intro}\\nBody.\\n\\caption{A plot.}\\n"

    - "\\section{Intro}" は構造 chunk になり, 以降の section_path は "Intro".
    - "Body." は通常の翻訳候補 chunk.
    - "\\caption{" と "}" は構造 chunk, "A plot." は caption 用の翻訳候補 chunk.
"""
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from functools import partial
import re

_SECTION_COMMANDS = frozenset({
    "section", "subsection", "subsubsection", "chapter", "paragraph", "subparagraph",
})
_CAPTION_COMMANDS = frozenset({"caption", "captionof"})
_ABSTRACT_COMMANDS = frozenset({"abstract"})
_PLACEHOLDER_RE = re.compile(r"XQK\d{4,}")
_COMMAND_ONLY_RE = re.compile(
    r"\\(?:pagebreak|newpage|clearpage|cleardoublepage|linebreak|nolinebreak|"
    r"allowbreak|smallskip|medskip|bigskip|vspace|hspace|vskip|hskip|"
    r"centering|raggedright|raggedleft|noindent|indent)\*?"
    r"(?:\[[^\[\]{}]*\])?(?:\{[^{}]*\})?"
)


@dataclass
class Chunk:
    """チャンク 1 つ分.

    `text` は末尾改行を含まない. `join_after` に chunk 後続の改行を持たせる.

    例:
        元の 1 行が "Body.\\n" なら `text="Body."`, `join_after="\\n"`.
        復元時は全 chunk の `text + join_after` を順に結合する.

    `translatable=False` の chunk は翻訳に渡さない. 典型的には section 見出し,
    caption wrapper, placeholder だけの行が該当する.
    """

    index: int
    section_path: str
    text: str
    translatable: bool
    skip_reason: str | None = None
    join_after: str = ""


@dataclass
class _SpecialSpan:
    """section/caption command の範囲と, chunk_text への作用.

    `command_span` は command 全体の範囲. `translatable_span` がある場合は,
    command 内のその範囲だけを翻訳候補にする.

    例:
        "\\caption{A plot.}" なら `command_span` は command 全体,
        `translatable_span` は "A plot." の範囲.

        "\\section{Intro}" なら `translatable_span=None`,
        `next_section_path="Intro"`.
    """

    command_span: slice
    translatable_span: slice | None = None
    next_section_path: str | None = None
    translatable_section_suffix: str | None = None


@dataclass
class _JoinRecord:
    """chunk 本文と, 復元時に後ろへ戻す改行.

    `splitlines(keepends=True)` で得た 1 レコードを受け取り, 末尾改行だけを
    `join_after` に分ける.

    例:
        raw_line="abc\\n"  -> text="abc", join_after="\\n"
        raw_line="abc"     -> text="abc", join_after=""
        raw_line="\\n"     -> text="",    join_after="\\n"
    """

    raw_line: InitVar[str]
    text: str = field(init=False)
    join_after: str = field(init=False)

    def __post_init__(self, raw_line: str) -> None:
        self.text = raw_line
        self.join_after = ""
        for ending in ("\r\n", "\n", "\r"):
            if raw_line.endswith(ending):
                self.text = raw_line[:-len(ending)]
                self.join_after = ending
                break


@dataclass
class _ParsedCommand:
    """TeX command 名と command 名直後の位置.

    `after_name` は command 名と optional star を読んだ直後の index.

    例:
        "\\section*{Intro}" を先頭から読むと
        `name="section"`, `after_name` は "{" の位置.

        "\\%" のような 1 文字 command は `name="%"`.
    """

    name: str
    after_name: int

    @classmethod
    def read(cls, text: str, start: int) -> "_ParsedCommand | None":
        """`start` 位置の TeX command を読む.

        `start` が "\\" でない, または末尾の "\\" なら None を返す.
        command 引数自体は読まず, command 名の直後までだけを読む.
        """
        if start >= len(text) or text[start] != "\\" or start + 1 >= len(text):
            return None
        cursor = start + 1
        if not text[cursor].isalpha():
            return cls(name=text[cursor], after_name=cursor + 1)
        while cursor < len(text) and text[cursor].isalpha():
            cursor += 1
        name = text[start + 1:cursor]
        if cursor < len(text) and text[cursor] == "*":
            cursor += 1
        return cls(name=name, after_name=cursor)


def chunk_text(
    text: str,
    *,
    placeholders: Sequence[str] | None = None,
) -> list[Chunk]:
    """section 見出しを翻訳対象から外し, caption 内容と本文だけを chunk 化する.

    `placeholders` には `substitute()` が生成した placeholder を渡す. chunk が
    placeholder と command-only 記法だけなら `translatable=False` になる.

    例:
        入力: "\\section{Intro}\\nThis is XQK0001.\\n"
        出力:
          - "\\section{Intro}" は `skip_reason="structure"`
          - "This is XQK0001." は placeholder 以外の本文が残るので翻訳候補
    """
    chunks: list[Chunk] = []
    section_path = "preamble"
    append = partial(_append_chunks, chunks, placeholders=placeholders)
    cursor = 0

    while cursor < len(text):
        special = _find_next_special(text, cursor)
        if special is None:
            append(text[cursor:], section_path=section_path)
            break

        command_span = special.command_span
        append(text[cursor:command_span.start], section_path=section_path)

        translatable_span = special.translatable_span
        if translatable_span is None:
            append(
                text[command_span], section_path=section_path,
                skip_reason="structure",
            )
            section_path = special.next_section_path or section_path
            cursor = command_span.stop
            continue

        append(
            text[command_span.start:translatable_span.start],
            section_path=section_path, skip_reason="structure",
        )
        translatable_section_path = section_path
        if special.translatable_section_suffix:
            translatable_section_path = (
                special.translatable_section_suffix
                if section_path == "preamble"
                else f"{section_path}/{special.translatable_section_suffix}"
            )
        append(
            text[translatable_span], section_path=translatable_section_path,
            preserve_empty_records=True,
        )
        append(
            text[translatable_span.stop:command_span.stop],
            section_path=section_path, skip_reason="structure",
        )
        cursor = command_span.stop

    return _merge_continuing_chunks(chunks)


def _merge_continuing_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """途切れていない translatable chunk を 1 つにまとめる.

    直前 chunk が `.` で終わらず, `join_after` が単一の `"\\n"` で, 次 chunk も
    同 section_path で translatable のとき, 次 chunk を取り込む. TeX ソースの
    hard-wrap で文の途中で chunk が分割される問題を回避する.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and _continues_previous(merged[-1], chunk):
            prev = merged[-1]
            prev.text = prev.text + prev.join_after + chunk.text
            prev.join_after = chunk.join_after
            continue
        merged.append(chunk)
    for new_index, chunk in enumerate(merged):
        chunk.index = new_index
    return merged


def _continues_previous(prev: Chunk, current: Chunk) -> bool:
    return (
        prev.translatable
        and current.translatable
        and prev.section_path == current.section_path
        and prev.join_after == "\n"
        and not prev.text.endswith(".")
    )


def _find_next_special(text: str, start: int) -> _SpecialSpan | None:
    """次の section/caption command を返す.

    `start` 以降を左から走査し, chunk_text が特別扱いする command だけを返す.
    他の command は読み飛ばす.

    例:
        "\\label{x}\\caption{A}" では `\\label` を無視して caption の span を返す.
    """
    cursor = start
    while True:
        command_start = text.find("\\", cursor)
        if command_start < 0:
            return None
        cursor = command_start + 1
        parsed = _ParsedCommand.read(text, command_start)
        if parsed is None:
            continue
        if parsed.name in _SECTION_COMMANDS and (
            span := _consume_section_command(text, command_start, parsed.after_name)
        ) is not None:
            return span
        if parsed.name in _CAPTION_COMMANDS and (
            span := _consume_caption_command(
                text, command_start, parsed.after_name, name=parsed.name,
            )
        ) is not None:
            return span
        if parsed.name in _ABSTRACT_COMMANDS and (
            span := _consume_abstract_command(text, command_start, parsed.after_name)
        ) is not None:
            return span


def _append_chunks(
    chunks: list[Chunk],
    text: str,
    *,
    section_path: str,
    placeholders: Sequence[str] | None,
    skip_reason: str | None = None,
    preserve_empty_records: bool = False,
) -> None:
    """改行を `join_after` に分離しながら chunk を追加する.

    空行は通常は直前 chunk の `join_after` に吸収する. caption 本文では
    `preserve_empty_records=True` にして, caption 内の空行も独立 record として残す.
    """
    if not text:
        return
    for record in _split_join_records(text):
        if (
            not record.text and record.join_after
            and chunks and not preserve_empty_records
        ):
            chunks[-1].join_after += record.join_after
            continue
        chunks.append(_make_chunk(
            index=len(chunks),
            section_path=section_path,
            text=record.text,
            placeholders=placeholders,
            join_after=record.join_after,
            skip_reason=skip_reason,
        ))


def _split_join_records(text: str) -> list[_JoinRecord]:
    """末尾改行を chunk 本文から分離する.

    例:
        "a\\nb" -> [
            _JoinRecord(text="a", join_after="\\n"),
            _JoinRecord(text="b", join_after=""),
        ]
    """
    return [
        _JoinRecord(raw_line=line)
        for line in text.splitlines(keepends=True)
    ]


def _consume_section_command(
    text: str,
    start: int,
    cursor: int,
) -> _SpecialSpan | None:
    """section 系 command 全体を消費し, title を section_path に使う.

    optional argument は読み飛ばし, required argument の中身だけを
    `next_section_path` に入れる. 見出し自体は翻訳対象にしない.

    例:
        "\\section[Short]{Long Title}" -> next_section_path="Long Title"
    """
    cursor = _consume_optional_args(text, cursor)
    if cursor < 0:
        return None
    title_span = _consume_required_arg(text, cursor)
    if title_span is None:
        return None
    return _SpecialSpan(
        command_span=slice(start, title_span.stop + 1),
        next_section_path=text[title_span].strip() or None,
    )


def _consume_caption_command(
    text: str,
    start: int,
    cursor: int,
    *,
    name: str,
) -> _SpecialSpan | None:
    """caption command の required caption 引数だけ翻訳対象として返す.

    optional short caption は構造側に残し, 最後の required caption 本文だけを
    `translatable_span` にする.

    例:
        "\\caption[Short]{Long caption}" では "Long caption" だけ翻訳候補.
        "\\captionof{figure}{Long caption}" では "{figure}" は構造扱い.
    """
    cursor = _consume_optional_args(text, cursor)
    if cursor < 0:
        return None
    if name == "captionof":
        caption_type = _consume_required_arg(text, cursor)
        if caption_type is None:
            return None
        cursor = _consume_inline_space(text, caption_type.stop + 1)

    content_span = _consume_required_arg(text, cursor)
    if content_span is None:
        return None
    return _SpecialSpan(
        command_span=slice(start, content_span.stop + 1),
        translatable_span=content_span,
        translatable_section_suffix="caption",
    )


def _consume_abstract_command(
    text: str,
    start: int,
    cursor: int,
) -> _SpecialSpan | None:
    """abstract command の required 引数だけ翻訳対象として返す."""
    cursor = _consume_optional_args(text, cursor)
    if cursor < 0:
        return None
    content_span = _consume_required_arg(text, cursor)
    if content_span is None:
        return None
    return _SpecialSpan(
        command_span=slice(start, content_span.stop + 1),
        translatable_span=content_span,
        translatable_section_suffix="abstract",
    )


def _consume_optional_args(text: str, cursor: int) -> int:
    """command 直後の空白と optional arguments を消費する.

    例:
        cursor が "\\caption [short] {long}" の command 名直後を指す場合,
        戻り値は "{long}" の "{" の位置.

    bracket が閉じない場合は -1 を返す.
    """
    cursor = _consume_inline_space(text, cursor)
    while cursor < len(text) and text[cursor] == "[":
        cursor = _skip_balanced(text, cursor, open_ch="[", close_ch="]")
        if cursor < 0:
            return -1
        cursor = _consume_inline_space(text, cursor)
    return cursor


def _consume_required_arg(text: str, cursor: int) -> slice | None:
    """required argument の中身の範囲を返す.

    cursor は "{" の位置を指している必要がある. 戻り値は brace を含まない中身.

    例:
        "{a {b} c}" -> "a {b} c" の範囲.
        brace が閉じない場合は None.
    """
    if cursor >= len(text) or text[cursor] != "{":
        return None
    end = _skip_balanced(text, cursor, open_ch="{", close_ch="}")
    if end < 0:
        return None
    return slice(cursor + 1, end - 1)


def _consume_inline_space(text: str, cursor: int) -> int:
    """command と引数の間の空白を消費する.

    TeX では command と引数の間に改行が入ることがあるので, space/tab だけでなく
    CR/LF も消費する.
    """
    while cursor < len(text) and text[cursor] in " \t\r\n":
        cursor += 1
    return cursor


def _skip_balanced(text: str, start: int, *, open_ch: str, close_ch: str) -> int:
    """対応する閉じ記号の次の index を返す. 不整合時は -1.

    ネストした brace/bracket を数える. escaped character は構造として扱わない.

    例:
        "{a {b} c}" の "{" から読むと, 最後の "}" の次の index を返す.
        "{a \\} b}" では "\\}" は閉じ brace として数えない.
    """
    depth = 0
    cursor = start
    while cursor < len(text):
        ch = text[cursor]
        if ch == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return -1


def _make_chunk(
    *,
    index: int,
    section_path: str,
    text: str,
    placeholders: Sequence[str] | None,
    join_after: str,
    skip_reason: str | None,
) -> Chunk:
    """単一チャンクを生成し, translatable / skip_reason を判定する.

    `skip_reason` が指定されている場合は無条件で非翻訳 chunk.
    指定がない場合は, placeholder や command-only 記法を除いて本文が残るかで判定する.
    """
    reason = skip_reason
    if reason is None and not _has_translatable_text(text, placeholders):
        reason = "command_only"
    return Chunk(
        index=index, section_path=section_path, text=text,
        translatable=reason is None, skip_reason=reason, join_after=join_after,
    )


def _has_translatable_text(
    text: str,
    placeholders: Sequence[str] | None,
) -> bool:
    """placeholder と command-only 記法を除いた後に本文が残るかを返す.

    例:
        "XQK0001" -> False
        "\\pagebreak" -> False
        "This is XQK0001." -> True
    """
    if placeholders is None:
        stripped = _PLACEHOLDER_RE.sub("", text)
    else:
        stripped = text
        for placeholder in sorted(placeholders, key=len, reverse=True):
            stripped = stripped.replace(placeholder, "")
    stripped = _COMMAND_ONLY_RE.sub("", stripped).strip()
    return bool(stripped)
