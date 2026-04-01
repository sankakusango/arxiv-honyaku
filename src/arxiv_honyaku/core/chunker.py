"""TeX 本文を, 翻訳対象と非翻訳対象に分けながらチャンク化するモジュール.

このモジュールは, TeX 全文を壊れやすい構造と自然言語本文へ分解し,
翻訳器に渡してよい断片だけを ``TexChunk`` として切り出す責務を持つ.
preamble, 図表, 数式環境, bibliography のような保護対象に加え,
見出し, caption, ``\\iftoggle`` の分岐本文, 行頭コマンド付き本文も
ここで意味単位に整理する.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

from arxiv_honyaku.core.models import ChunkKind, TexChunk

_SECTION_PREFIXES = (
    "\\section",
    "\\subsection",
    "\\subsubsection",
    "\\chapter",
    "\\paragraph",
    "\\subparagraph",
)
_TRANSLATABLE_BRACED_COMMANDS: dict[str, tuple[ChunkKind, bool, bool, bool]] = {
    "section": ("section_title", False, True, True),
    "subsection": ("section_title", False, True, True),
    "subsubsection": ("section_title", False, True, True),
    "chapter": ("section_title", False, True, True),
    "para": ("paragraph_title", False, True, True),
    "paragraph": ("paragraph_title", False, True, True),
    "subparagraph": ("paragraph_title", False, True, True),
    "caption": ("caption", True, True, False),
    "footnote": ("body", False, True, True),
    "label": ("body", False, False, True),
    "textbf": ("body", False, True, True),
    "emph": ("body", False, True, True),
    "textit": ("body", False, True, True),
    "cref": ("body", False, False, True),
    "Cref": ("body", False, False, True),
    "ref": ("body", False, False, True),
    "eqref": ("body", False, False, True),
    "autoref": ("body", False, False, True),
    "pageref": ("body", False, False, True),
    "cite": ("body", True, False, True),
    "citealp": ("body", True, False, True),
    "citealt": ("body", True, False, True),
    "citeauthor": ("body", True, False, True),
    "citep": ("body", True, False, True),
    "citet": ("body", True, False, True),
    "citeyear": ("body", False, False, True),
    "citeyearpar": ("body", False, False, True),
}
_RAW_INLINE_COMMANDS = frozenset(
    {
        "label",
        "vspace",
        "hspace",
        "vskip",
        "hskip",
    }
)
_TABULAR_ENVIRONMENTS = {
    "tabular",
    "tabular*",
}
_PROTECTED_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "aligned*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
    "figure",
    "figure*",
    "table",
    "table*",
    "tabular",
    "tabular*",
    "tikzpicture",
    "verbatim",
    "verbatim*",
    "lstlisting",
    "minted",
    "thebibliography",
}
_BEGIN_ENVIRONMENT_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_ENVIRONMENT_RE = re.compile(r"\\end\{([^}]+)\}")


@dataclass(slots=True, frozen=True)
class _TeXPart:
    """TeX全文を構成する内部断片.

    Attributes:
        translate: ``True`` の場合のみLLM翻訳対象として扱う.
        text: 元文書の部分文字列.
        kind: 翻訳対象としての文脈種別.
    """

    translate: bool
    text: str
    kind: ChunkKind = "body"


def digest_text(text: str) -> str:
    """文字列からSHA-256ダイジェストを生成する.

    Args:
        text: ハッシュ化対象文字列.

    Returns:
        str: 16進表現のSHA-256ダイジェスト.
    """
    return sha256(text.encode("utf-8")).hexdigest()


def split_tex_into_chunks(text: str, *, max_chars: int) -> list[TexChunk]:
    """TeX 文字列を, 翻訳対象と保護対象を区別しつつチャンク分割する.

    この関数は, preamble, bibliography, 数式や図表などの壊れやすい環境,
    先頭が LaTeX コマンドの行を保護し, 自然言語本文ブロックだけを翻訳対象にする.
    ただし, ``\\caption{...}``, ``\\footnote{...}``, ``\\cref{...} provides ...`` のように
    コマンド骨格を保ったまま本文だけ翻訳したいケースは, 内部で raw と translate の
    部分へさらに分解する.
    返却チャンクを順に連結すると, 入力文字列へ完全に戻ることを保証する.

    Args:
        text: 分割対象のTeX全文.
        max_chars: 1チャンクあたりの最大文字数.

    Returns:
        list[TexChunk]: ``translate`` フラグ付きの順序保持チャンク配列.

    Raises:
        ValueError: ``max_chars`` が0以下の場合.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    parts = _split_tex_into_parts(text)
    chunks: list[TexChunk] = []
    current_line = 1
    for part in parts:
        if not part.text:
            continue
        if part.translate:
            chunks.extend(
                _split_translatable_part(
                    part.text,
                    max_chars=max_chars,
                    start_index=len(chunks),
                    kind=part.kind,
                    start_line=current_line,
                )
            )
        else:
            chunks.append(
                TexChunk(
                    index=len(chunks),
                    text=part.text,
                    digest=digest_text(part.text),
                    translate=False,
                    kind=part.kind,
                    start_line=current_line,
                )
            )
        current_line += part.text.count("\n")

    if not chunks:
        chunks.append(TexChunk(index=0, text="", digest=digest_text("")))
    return chunks


def _split_tex_into_parts(text: str) -> list[_TeXPart]:
    """TeX 全文を, 翻訳対象本文と保護対象本文へ分割する.

    ここでの出力は最終チャンクではなく, ``translate`` と ``kind`` を持つ
    粗い部分列である. その後 ``_split_translatable_part`` が最大文字数ベースで
    さらに細分化する.

    Args:
        text: 分割対象のTeX全文.

    Returns:
        list[_TeXPart]: ``translate`` と ``kind`` を含む順序保持配列.
    """
    lines = text.splitlines(keepends=True)
    parts: list[_TeXPart] = []
    buffer: list[str] = []
    current_translate: bool | None = None
    current_kind: ChunkKind | None = None
    protected_environment_stack: list[str] = []

    document_started = "\\begin{document}" not in text
    body_started = "\\maketitle" not in text
    bibliography_started = False

    def flush() -> None:
        nonlocal buffer, current_translate, current_kind
        if not buffer or current_translate is None or current_kind is None:
            buffer = []
            return
        parts.append(
            _TeXPart(
                translate=current_translate,
                text="".join(buffer),
                kind=current_kind,
            )
        )
        buffer = []
        current_kind = None

    def push_text(
        text_fragment: str,
        *,
        translate: bool,
        kind: ChunkKind = "body",
    ) -> None:
        nonlocal current_translate, current_kind
        if not text_fragment:
            return
        if current_translate is None:
            current_translate = translate
            current_kind = kind
        if current_translate != translate or current_kind != kind:
            flush()
            current_translate = translate
            current_kind = kind
        buffer.append(text_fragment)

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.lstrip()

        if bibliography_started:
            push_text(line, translate=False)
            line_index += 1
            continue

        if "\\begin{document}" in line:
            document_started = True
        if "\\maketitle" in line:
            body_started = True
        if "\\begin{thebibliography}" in line or stripped.startswith("\\bibliography"):
            bibliography_started = True
            push_text(line, translate=False)
            line_index += 1
            continue

        for match in _BEGIN_ENVIRONMENT_RE.finditer(line):
            env_name = match.group(1).strip()
            if env_name in _PROTECTED_ENVIRONMENTS:
                protected_environment_stack.append(env_name)

        consumed_lines = 1
        command_parts = None
        protected_command_block = None
        if document_started and body_started:
            if not any(
                environment_name in _TABULAR_ENVIRONMENTS
                for environment_name in protected_environment_stack
            ):
                supported_command = _split_supported_command_block(
                    lines,
                    start_index=line_index,
                )
                if supported_command is not None:
                    command_parts, consumed_lines = supported_command
                else:
                    protected_command_block = _consume_multiline_command_block(
                        lines,
                        start_index=line_index,
                    )
                    if protected_command_block is not None:
                        protected_text, consumed_lines = protected_command_block

        if command_parts is not None:
            for part in command_parts:
                push_text(
                    part.text,
                    translate=part.translate,
                    kind=part.kind,
                )
        elif protected_command_block is not None:
            push_text(protected_text, translate=False)
        elif (
            not document_started
            or not body_started
            or protected_environment_stack
            or _is_protected_line(stripped)
        ):
            push_text(line, translate=False)
        else:
            push_text(line, translate=True)

        for consumed_line in lines[line_index : line_index + consumed_lines]:
            for match in _END_ENVIRONMENT_RE.finditer(consumed_line):
                env_name = match.group(1).strip()
                if (
                    protected_environment_stack
                    and protected_environment_stack[-1] == env_name
                ):
                    protected_environment_stack.pop()

        line_index += consumed_lines

    flush()
    return parts


def _split_translatable_part(
    text: str,
    *,
    max_chars: int,
    start_index: int,
    kind: ChunkKind,
    start_line: int,
) -> list[TexChunk]:
    """翻訳対象本文を最大文字数ベースで細分化する.

    Args:
        text: 翻訳対象として抽出済みの本文.
        max_chars: 1チャンクあたりの最大文字数.
        start_index: 返却チャンク列の開始インデックス.
        kind: 翻訳対象本文の文脈種別.
        start_line: 入力本文が元文書内で始まる1始まり行番号.

    Returns:
        list[TexChunk]: 翻訳対象フラグ付きチャンク配列.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[TexChunk] = []
    current_lines: list[str] = []
    current_len = 0
    next_chunk_start_line = start_line
    current_chunk_start_line = start_line

    def append_chunk(chunk_text: str, *, chunk_start_line: int) -> None:
        chunks.append(
            TexChunk(
                index=start_index + len(chunks),
                text=chunk_text,
                digest=digest_text(chunk_text),
                translate=True,
                kind=kind,
                start_line=chunk_start_line,
            )
        )

    def flush() -> None:
        nonlocal \
            current_chunk_start_line, \
            current_len, \
            current_lines, \
            next_chunk_start_line
        if not current_lines:
            return
        chunk_text = "".join(current_lines)
        append_chunk(chunk_text, chunk_start_line=current_chunk_start_line)
        next_chunk_start_line = current_chunk_start_line + chunk_text.count("\n")
        current_lines = []
        current_len = 0
        current_chunk_start_line = next_chunk_start_line

    for line in lines:
        if not current_lines:
            current_chunk_start_line = next_chunk_start_line

        if len(line) > max_chars:
            flush()
            piece_start_line = next_chunk_start_line
            for start in range(0, len(line), max_chars):
                piece = line[start : start + max_chars]
                append_chunk(piece, chunk_start_line=piece_start_line)
                piece_start_line += piece.count("\n")
            next_chunk_start_line = piece_start_line
            current_chunk_start_line = next_chunk_start_line
            continue

        if current_len + len(line) > max_chars and current_len > 0:
            flush()

        if not current_lines:
            current_chunk_start_line = next_chunk_start_line
        current_lines.append(line)
        current_len += len(line)

        boundary = line.strip() == "" or line.lstrip().startswith(_SECTION_PREFIXES)
        if boundary and current_len >= int(max_chars * 0.65):
            flush()

    flush()
    return chunks


def _split_supported_command_block(
    lines: list[str],
    *,
    start_index: int,
) -> tuple[list[_TeXPart], int] | None:
    """翻訳対象にできる TeX コマンド塊を, raw と translate に分解する.

    見出し, caption, footnote, 行頭参照コマンド付き本文, 箇条書きなど,
    「コマンドそのものは保持しつつ, 引数や後続の自然言語だけ訳したい」断片を扱う.
    戻り値の ``int`` は, 対象コマンド塊を完成させるまでに消費した行数である.

    Args:
        lines: 元文書の行配列.
        start_index: 解析開始行の0始まりインデックス.

    Returns:
        tuple[list[_TeXPart], int] | None: 分解結果と消費行数, 対象外なら ``None``.
    """
    line = lines[start_index]
    item_parts = _split_item_line(line)
    if item_parts is not None:
        return item_parts, 1

    conditional_parts = _split_supported_conditional_block(
        lines,
        start_index=start_index,
    )
    if conditional_parts is not None:
        return conditional_parts

    for command_name, (
        content_kind,
        translate_optional_argument,
        translate_required_argument,
        translate_trailing_text,
    ) in _TRANSLATABLE_BRACED_COMMANDS.items():
        if not _starts_with_command_prefix(line, command_name):
            continue
        block_text = ""
        for consumed_lines, block_line in enumerate(lines[start_index:], start=1):
            block_text += block_line
            parts = _split_braced_command_text(
                block_text,
                command_name=command_name,
                content_kind=content_kind,
                translate_optional_argument=translate_optional_argument,
                translate_required_argument=translate_required_argument,
                translate_trailing_text=translate_trailing_text,
            )
            if parts is not None:
                return parts, consumed_lines
    return None


def _split_supported_conditional_block(
    lines: list[str],
    *,
    start_index: int,
) -> tuple[list[_TeXPart], int] | None:
    """翻訳対象にできる条件分岐コマンド塊を raw と translate に分解する.

    現状は ``\\iftoggle{...}{then}{else}`` を対象とし, 条件名は raw のまま保持し,
    then/else の中身だけを再帰的に通常本文として解析する. これにより,
    arXiv 向け分岐の中にある段落見出しや自然言語本文も翻訳対象へ戻せる.

    Args:
        lines: 元文書の行配列.
        start_index: 解析開始行の0始まりインデックス.

    Returns:
        tuple[list[_TeXPart], int] | None: 分解結果と消費行数, 対象外なら ``None``.
    """
    line = lines[start_index]
    if not _starts_with_command_prefix(line, "iftoggle"):
        return None

    block_text = ""
    for consumed_lines, block_line in enumerate(lines[start_index:], start=1):
        block_text += block_line
        parts = _split_iftoggle_text(block_text)
        if parts is not None:
            return parts, consumed_lines
    return None


def _consume_multiline_command_block(
    lines: list[str],
    *,
    start_index: int,
) -> tuple[str, int] | None:
    """複数行にまたがる一般 LaTeX コマンド塊を raw として保護する.

    allowlist に載っていない複数行コマンドは, 解釈を誤って壊すより
    原文保持を優先する. 典型例は独自マクロ定義やスタイル設定である.

    Args:
        lines: 元文書の行配列.
        start_index: 解析開始行の0始まりインデックス.

    Returns:
        tuple[str, int] | None: 保護対象の文字列と消費行数, 対象外なら ``None``.
    """
    line = lines[start_index]
    if not line.lstrip().startswith("\\"):
        return None

    brace_balance = _measure_unclosed_brace_balance(line)
    if brace_balance <= 0:
        return None

    block_lines = [line]
    for consumed_lines, block_line in enumerate(lines[start_index + 1 :], start=2):
        block_lines.append(block_line)
        brace_balance += _measure_unclosed_brace_balance(block_line)
        if brace_balance <= 0:
            return "".join(block_lines), consumed_lines
    return "".join(block_lines), len(block_lines)


def _split_iftoggle_text(text: str) -> list[_TeXPart] | None:
    """``\\iftoggle`` ブロックを raw と translate に分解する.

    1 個目の引数である toggle 名は raw で維持し, 2 個目, 3 個目のブランチ内容は
    それぞれ ``_split_tex_into_parts`` へ再帰的に渡す.

    Args:
        text: ``\\iftoggle{...}{...}{...}`` を含む文字列.

    Returns:
        list[_TeXPart] | None: 分解結果, まだ引数が閉じていなければ ``None``.
    """
    leading_whitespace = text[: len(text) - len(text.lstrip())]
    content = text[len(leading_whitespace) :]
    command_prefix = "\\iftoggle"
    if not content.startswith(command_prefix):
        return None

    cursor = len(command_prefix)
    parts: list[_TeXPart] = [
        _TeXPart(translate=False, text=leading_whitespace + command_prefix)
    ]
    cursor = _append_inline_whitespace(parts, content, cursor)

    first_argument_end = _consume_balanced(content, cursor, "{", "}")
    if first_argument_end is None:
        return None
    parts.append(_TeXPart(translate=False, text=content[cursor:first_argument_end]))
    cursor = first_argument_end
    cursor = _append_inline_whitespace(parts, content, cursor)

    for _ in range(2):
        branch_argument_end = _consume_balanced(content, cursor, "{", "}")
        if branch_argument_end is None:
            return None
        parts.append(_TeXPart(translate=False, text="{"))
        branch_text = content[cursor + 1 : branch_argument_end - 1]
        if branch_text:
            parts.extend(_split_tex_into_parts(branch_text))
        parts.append(_TeXPart(translate=False, text="}"))
        cursor = branch_argument_end
        cursor = _append_inline_whitespace(parts, content, cursor)

    if cursor < len(content):
        parts.append(_TeXPart(translate=False, text=content[cursor:]))
    return _merge_adjacent_parts(parts)


def _split_item_line(line: str) -> list[_TeXPart] | None:
    """``\\item`` 行を raw と translate に分解する.

    Args:
        line: 元の1行分のTeX文字列.

    Returns:
        list[_TeXPart] | None: ``\\item`` 行なら分解結果, 対象外なら ``None``.
    """
    leading_whitespace = line[: len(line) - len(line.lstrip())]
    content = line[len(leading_whitespace) :]
    if not content.startswith("\\item"):
        return None
    if len(content) > len("\\item") and content[len("\\item")].isalpha():
        return None

    cursor = len("\\item")
    parts: list[_TeXPart] = [
        _TeXPart(translate=False, text=leading_whitespace + content[:cursor])
    ]
    cursor = _append_inline_whitespace(parts, content, cursor)

    if cursor < len(content) and content[cursor] == "[":
        optional_end = _consume_balanced(content, cursor, "[", "]")
        if optional_end is None:
            return None
        _append_balanced_argument(
            parts,
            content,
            start_index=cursor,
            end_index=optional_end,
            opening="[",
            closing="]",
            translate_content=True,
            kind="list_item",
        )
        cursor = optional_end
        cursor = _append_inline_whitespace(parts, content, cursor)

    remainder = content[cursor:]
    if not remainder:
        return parts
    if remainder.lstrip().startswith("\\"):
        nested_parts = _split_prefixed_command_text(
            remainder,
            content_kind="list_item",
            translate_trailing_text=True,
        )
        if nested_parts is None:
            parts.append(_TeXPart(translate=False, text=remainder))
        else:
            parts.extend(nested_parts)
    else:
        parts.append(
            _TeXPart(
                translate=bool(remainder.strip()),
                text=remainder,
                kind="list_item",
            )
        )
    return _merge_adjacent_parts(parts)


def _split_braced_command_text(
    text: str,
    *,
    command_name: str,
    content_kind: ChunkKind,
    translate_optional_argument: bool,
    translate_required_argument: bool,
    translate_trailing_text: bool,
) -> list[_TeXPart] | None:
    """指定コマンド行を, コマンド構造と本文へ分解する.

    Args:
        text: 解析対象の文字列.
        command_name: 解析対象のコマンド名.
        content_kind: 主引数内部に付与する文脈種別.
        translate_optional_argument: 先頭の ``[...]`` 引数も翻訳対象にする場合は ``True``.
        translate_required_argument: 主引数 ``{...}`` の内部を翻訳対象にする場合は ``True``.
        translate_trailing_text: 主引数閉じ後ろの平文も翻訳対象にする場合は ``True``.

    Returns:
        list[_TeXPart] | None: 対応行なら分解結果, 対象外なら ``None``.
    """
    leading_whitespace = text[: len(text) - len(text.lstrip())]
    content = text[len(leading_whitespace) :]
    command_prefix = f"\\{command_name}"
    if not content.startswith(command_prefix):
        return None

    cursor = len(command_prefix)
    if cursor < len(content) and content[cursor] == "*":
        cursor += 1
    if cursor < len(content) and content[cursor].isalpha():
        return None

    parts: list[_TeXPart] = [
        _TeXPart(translate=False, text=leading_whitespace + content[:cursor])
    ]
    cursor = _append_inline_whitespace(parts, content, cursor)

    if cursor < len(content) and content[cursor] == "[":
        optional_end = _consume_balanced(content, cursor, "[", "]")
        if optional_end is None:
            return None
        _append_balanced_argument(
            parts,
            content,
            start_index=cursor,
            end_index=optional_end,
            opening="[",
            closing="]",
            translate_content=translate_optional_argument,
            kind=content_kind,
        )
        cursor = optional_end
        cursor = _append_inline_whitespace(parts, content, cursor)

    if cursor >= len(content) or content[cursor] != "{":
        return None

    argument_end = _consume_balanced(content, cursor, "{", "}")
    if argument_end is None:
        return None
    _append_balanced_argument(
        parts,
        content,
        start_index=cursor,
        end_index=argument_end,
        opening="{",
        closing="}",
        translate_content=translate_required_argument,
        kind=content_kind,
    )
    cursor = argument_end

    remainder = content[cursor:]
    if not remainder:
        return parts
    if translate_trailing_text and remainder.strip():
        nested_parts = _split_prefixed_command_text(
            remainder,
            content_kind=content_kind,
            translate_trailing_text=translate_trailing_text,
        )
        if nested_parts is None:
            if not remainder.lstrip().startswith("\\"):
                parts.append(
                    _TeXPart(translate=True, text=remainder, kind=content_kind)
                )
            else:
                parts.append(_TeXPart(translate=False, text=remainder))
        else:
            parts.extend(nested_parts)
    else:
        parts.append(_TeXPart(translate=False, text=remainder))
    return _merge_adjacent_parts(parts)


def _split_prefixed_command_text(
    text: str,
    *,
    content_kind: ChunkKind,
    translate_trailing_text: bool,
) -> list[_TeXPart] | None:
    """行頭の軽量コマンド列と後続本文を分解する.

    ``\\cref{...} provides ...`` や ``\\textbf{Term} is ...`` のように,
    行頭の短いコマンド列に続いて平文が来るケースを扱う. 先頭コマンド群だけを raw
    あるいは部分翻訳として切り出し, 末尾の自然言語を ``content_kind`` 付きで返す.

    Args:
        text: 解析対象文字列.
        content_kind: 翻訳対象に付与する文脈種別.
        translate_trailing_text: 後続平文を翻訳対象にする場合は ``True``.

    Returns:
        list[_TeXPart] | None: 分解に成功した場合は断片配列, 対象外なら ``None``.
    """
    if not text.lstrip().startswith("\\"):
        return None

    leading_whitespace = text[: len(text) - len(text.lstrip())]
    content = text[len(leading_whitespace) :]
    parts: list[_TeXPart] = []
    if leading_whitespace:
        parts.append(_TeXPart(translate=False, text=leading_whitespace))

    cursor = 0
    consumed_any_command = False
    while cursor < len(content) and content[cursor] == "\\":
        nested_parts = _split_single_command_prefix(
            content[cursor:],
            content_kind=content_kind,
        )
        if nested_parts is None:
            break
        parts.extend(nested_parts)
        consumed_any_command = True
        cursor += sum(len(part.text) for part in nested_parts)
        cursor = _append_inline_whitespace(parts, content, cursor)
        while cursor < len(content) and content[cursor] in {",", ";", ":", "."}:
            parts.append(_TeXPart(translate=False, text=content[cursor]))
            cursor += 1
        cursor = _append_inline_whitespace(parts, content, cursor)

    remainder = content[cursor:]
    if not consumed_any_command:
        return None
    if remainder:
        if (
            translate_trailing_text
            and not remainder.lstrip().startswith("\\")
            and remainder.strip()
        ):
            parts.append(_TeXPart(translate=True, text=remainder, kind=content_kind))
        else:
            parts.append(_TeXPart(translate=False, text=remainder))
    return _merge_adjacent_parts(parts)


def _split_single_command_prefix(
    text: str,
    *,
    content_kind: ChunkKind,
) -> list[_TeXPart] | None:
    """先頭 1 個の対応コマンドを分解する.

    ``_split_prefixed_command_text`` が複数コマンド列を左から食べ進めるための
    1 ステップ分のパーサであり, trailing text はここでは消費しない.

    Args:
        text: コマンドから始まる文字列.
        content_kind: 翻訳対象に付与する文脈種別.

    Returns:
        list[_TeXPart] | None: 対応コマンドなら分解結果, 対象外なら ``None``.
    """
    for command_name, (
        command_kind,
        translate_optional_argument,
        translate_required_argument,
        _translate_trailing_text,
    ) in _TRANSLATABLE_BRACED_COMMANDS.items():
        if not _starts_with_command_prefix(text, command_name):
            continue
        leading_whitespace = text[: len(text) - len(text.lstrip())]
        content = text[len(leading_whitespace) :]
        command_prefix = f"\\{command_name}"
        if not content.startswith(command_prefix):
            continue

        cursor = len(command_prefix)
        if cursor < len(content) and content[cursor] == "*":
            cursor += 1
        if cursor < len(content) and content[cursor].isalpha():
            continue

        parts: list[_TeXPart] = [
            _TeXPart(translate=False, text=leading_whitespace + content[:cursor])
        ]
        cursor = _append_inline_whitespace(parts, content, cursor)

        if cursor < len(content) and content[cursor] == "[":
            optional_end = _consume_balanced(content, cursor, "[", "]")
            if optional_end is None:
                return None
            _append_balanced_argument(
                parts,
                content,
                start_index=cursor,
                end_index=optional_end,
                opening="[",
                closing="]",
                translate_content=translate_optional_argument,
                kind=command_kind if command_kind != "body" else content_kind,
            )
            cursor = optional_end
            cursor = _append_inline_whitespace(parts, content, cursor)

        if cursor >= len(content) or content[cursor] != "{":
            return _merge_adjacent_parts(parts)

        argument_end = _consume_balanced(content, cursor, "{", "}")
        if argument_end is None:
            return None
        _append_balanced_argument(
            parts,
            content,
            start_index=cursor,
            end_index=argument_end,
            opening="{",
            closing="}",
            translate_content=translate_required_argument,
            kind=command_kind if command_kind != "body" else content_kind,
        )
        return _merge_adjacent_parts(parts)
    return None


def _append_inline_whitespace(
    parts: list[_TeXPart],
    content: str,
    start_index: int,
) -> int:
    """改行以外の空白を raw 部分として追加し, 次の位置を返す.

    Args:
        parts: 分解結果配列.
        content: 解析中の行内容.
        start_index: 現在位置.

    Returns:
        int: 空白を読み飛ばした後の位置.
    """
    cursor = start_index
    while cursor < len(content) and content[cursor] in {" ", "\t"}:
        cursor += 1
    if cursor > start_index:
        parts.append(_TeXPart(translate=False, text=content[start_index:cursor]))
    return cursor


def _measure_unclosed_brace_balance(text: str) -> int:
    """行コメントを除いた文字列内の未閉鎖波括弧バランスを返す.

    Args:
        text: 判定対象文字列.

    Returns:
        int: ``{`` を ``+1``, ``}`` を ``-1`` とした総和.
    """
    balance = 0
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "%":
            break
        if character == "{":
            balance += 1
        elif character == "}":
            balance -= 1
    return balance


def _starts_with_command_prefix(text: str, command_name: str) -> bool:
    """対象文字列が指定コマンドで始まるかを判定する.

    Args:
        text: 判定対象文字列.
        command_name: 先頭一致を確認するコマンド名.

    Returns:
        bool: 指定コマンドで始まる場合は ``True``.
    """
    leading_whitespace = text[: len(text) - len(text.lstrip())]
    content = text[len(leading_whitespace) :]
    command_prefix = f"\\{command_name}"
    return content.startswith(command_prefix)


def _append_balanced_argument(
    parts: list[_TeXPart],
    content: str,
    *,
    start_index: int,
    end_index: int,
    opening: str,
    closing: str,
    translate_content: bool,
    kind: ChunkKind,
) -> None:
    """均衡引数を raw と translate に分けて分解結果へ追加する.

    Args:
        parts: 分解結果配列.
        content: 解析中の行内容.
        start_index: 開始位置.
        end_index: 終了位置の次のインデックス.
        opening: 開き記号.
        closing: 閉じ記号.
        translate_content: 引数内部本文を翻訳対象にする場合は ``True``.
        kind: 翻訳対象断片に付与する文脈種別.

    Returns:
        None: 常に ``None``.
    """
    inner_text = content[start_index + 1 : end_index - 1]
    parts.append(_TeXPart(translate=False, text=opening))
    if inner_text:
        leading_whitespace_length = len(inner_text) - len(inner_text.lstrip())
        trailing_whitespace_length = len(inner_text) - len(inner_text.rstrip())
        leading_whitespace = inner_text[:leading_whitespace_length]
        trailing_whitespace = (
            inner_text[len(inner_text) - trailing_whitespace_length :]
            if trailing_whitespace_length > 0
            else ""
        )
        translatable_text = inner_text[
            leading_whitespace_length : len(inner_text) - trailing_whitespace_length
            if trailing_whitespace_length > 0
            else len(inner_text)
        ]
        if leading_whitespace:
            parts.append(_TeXPart(translate=False, text=leading_whitespace))
        if translatable_text:
            if translate_content:
                parts.extend(
                    _split_argument_text_into_parts(
                        translatable_text,
                        kind=kind,
                    )
                )
            else:
                parts.append(
                    _TeXPart(
                        translate=False,
                        text=translatable_text,
                        kind=kind,
                    )
                )
        if trailing_whitespace:
            parts.append(_TeXPart(translate=False, text=trailing_whitespace))
    parts.append(_TeXPart(translate=False, text=closing))


def _split_argument_text_into_parts(
    text: str,
    *,
    kind: ChunkKind,
) -> list[_TeXPart]:
    """引数内部の TeX を, 翻訳対象本文と raw 断片へ分解する.

    ``\\caption{...}`` や ``\\footnote{...}`` のように, 引数全体を 1 つの翻訳塊に
    すると ``\\label{...}``, ``%`` コメント, ``\\vspace{...}`` と自然言語が混在して
    しまう. ここでは引数内部を 1 行ずつ見直し, 行頭コマンドや行末コメントは raw
    のまま保持しつつ, 実際の自然言語だけを翻訳対象に切り出す.

    Args:
        text: 波括弧や角括弧の内側に入っているTeX文字列.
        kind: 翻訳対象断片に付与する文脈種別.

    Returns:
        list[_TeXPart]: 引数内部を再分解した断片配列.
    """
    parts: list[_TeXPart] = []
    for line in text.splitlines(keepends=True):
        line_without_comment, comment_suffix = _split_line_comment_suffix(line)
        parts.extend(_split_argument_line_into_parts(line_without_comment, kind=kind))
        if comment_suffix:
            parts.append(_TeXPart(translate=False, text=comment_suffix))
    return _split_parts_on_inline_raw_commands(_merge_adjacent_parts(parts))


def _split_argument_line_into_parts(
    text: str,
    *,
    kind: ChunkKind,
) -> list[_TeXPart]:
    """引数内部の 1 行を, 翻訳対象本文と raw 断片へ分解する.

    Args:
        text: 行末コメントを除いた1行分のTeX文字列.
        kind: 翻訳対象断片に付与する文脈種別.

    Returns:
        list[_TeXPart]: 1行分の再分解結果.
    """
    if not text:
        return []
    if not text.strip():
        return [_TeXPart(translate=False, text=text)]
    if text.lstrip().startswith("\\"):
        prefixed_parts = _split_prefixed_command_text(
            text,
            content_kind=kind,
            translate_trailing_text=True,
        )
        if prefixed_parts is not None:
            return _merge_adjacent_parts(prefixed_parts)
        supported_command = _split_supported_command_block([text], start_index=0)
        if supported_command is not None:
            command_parts, _consumed_lines = supported_command
            return _merge_adjacent_parts(command_parts)
        return [_TeXPart(translate=False, text=text)]
    return [_TeXPart(translate=True, text=text, kind=kind)]


def _split_line_comment_suffix(text: str) -> tuple[str, str]:
    """1 行を, 本文部分と行末コメント部分へ分ける.

    TeX の ``%`` は行末までをコメント化するため, 翻訳対象へ混ぜると
    ``.% \\vspace{...}`` のような危険な潰れ方を起こしやすい. 最初の未エスケープ
    ``%`` 以降を raw として切り出す.

    Args:
        text: 1行分のTeX文字列.

    Returns:
        tuple[str, str]: ``%`` より前の本文と, ``%`` を含むコメント部分.
    """
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "%":
            return text[:index], text[index:]
    return text, ""


def _split_parts_on_inline_raw_commands(parts: list[_TeXPart]) -> list[_TeXPart]:
    """翻訳対象断片から, 危険な inline LaTeX コマンドを raw へ切り出す.

    caption などの自然言語中には, 文末に ``\\label{...}`` や ``\\vspace{...}``
    が続くことがある. これらを翻訳文字列に残すと LLM が壊しやすいため,
    翻訳対象断片の中から機械的に raw へ退避する.

    Args:
        parts: 行単位で分解済みの断片配列.

    Returns:
        list[_TeXPart]: 危険な inline コマンドを raw に切り出した配列.
    """
    normalized_parts: list[_TeXPart] = []
    for part in parts:
        if not part.translate:
            normalized_parts.append(part)
            continue
        normalized_parts.extend(
            _split_text_on_inline_raw_commands(
                part.text,
                kind=part.kind,
            )
        )
    return _merge_adjacent_parts(normalized_parts)


def _split_text_on_inline_raw_commands(
    text: str,
    *,
    kind: ChunkKind,
) -> list[_TeXPart]:
    """翻訳対象文字列から, 危険な inline LaTeX コマンドを raw へ切り出す.

    Args:
        text: 翻訳対象の平文寄り文字列.
        kind: 翻訳対象断片に付与する文脈種別.

    Returns:
        list[_TeXPart]: inline raw コマンドを分離した配列.
    """
    parts: list[_TeXPart] = []
    cursor = 0
    last_text_start = 0
    while cursor < len(text):
        if text[cursor] != "\\":
            cursor += 1
            continue
        command_span_end = _consume_inline_raw_command(text, start_index=cursor)
        if command_span_end is None:
            cursor += 1
            continue
        if last_text_start < cursor:
            parts.append(
                _TeXPart(
                    translate=True,
                    text=text[last_text_start:cursor],
                    kind=kind,
                )
            )
        parts.append(
            _TeXPart(
                translate=False,
                text=text[cursor:command_span_end],
                kind=kind,
            )
        )
        cursor = command_span_end
        last_text_start = command_span_end

    if last_text_start < len(text):
        parts.append(
            _TeXPart(
                translate=True,
                text=text[last_text_start:],
                kind=kind,
            )
        )
    return parts or [_TeXPart(translate=True, text=text, kind=kind)]


def _consume_inline_raw_command(
    text: str,
    *,
    start_index: int,
) -> int | None:
    """危険な inline LaTeX コマンド 1 個ぶんの終端位置を返す.

    Args:
        text: 解析対象文字列.
        start_index: ``\\`` が現れた位置.

    Returns:
        int | None: コマンド末尾の次の位置, 対象外なら ``None``.
    """
    if start_index >= len(text) or text[start_index] != "\\":
        return None

    cursor = start_index + 1
    while cursor < len(text) and text[cursor].isalpha():
        cursor += 1
    if cursor == start_index + 1:
        return None

    command_name = text[start_index + 1 : cursor]
    if cursor < len(text) and text[cursor] == "*":
        command_name += "*"
        cursor += 1
    if command_name.rstrip("*") not in _RAW_INLINE_COMMANDS:
        return None

    while cursor < len(text) and text[cursor] in {" ", "\t"}:
        cursor += 1

    if cursor < len(text) and text[cursor] == "[":
        optional_end = _consume_balanced(text, cursor, "[", "]")
        if optional_end is None:
            return None
        cursor = optional_end
        while cursor < len(text) and text[cursor] in {" ", "\t"}:
            cursor += 1

    if cursor < len(text) and text[cursor] == "{":
        required_end = _consume_balanced(text, cursor, "{", "}")
        if required_end is None:
            return None
        cursor = required_end
    return cursor


def _consume_balanced(
    content: str,
    start_index: int,
    opening: str,
    closing: str,
) -> int | None:
    """対応する閉じ記号までを探索し, 終端位置の次を返す.

    Args:
        content: 解析対象文字列.
        start_index: 開き記号の位置.
        opening: 開き記号.
        closing: 閉じ記号.

    Returns:
        int | None: 閉じ記号直後の位置, 見つからない場合は ``None``.
    """
    if start_index >= len(content) or content[start_index] != opening:
        return None

    depth = 0
    cursor = start_index
    while cursor < len(content):
        character = content[cursor]
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


def _merge_adjacent_parts(
    parts: list[_TeXPart],
) -> list[_TeXPart]:
    """同じ翻訳フラグの連続部分をまとめる.

    Args:
        parts: 分解後の断片配列.

    Returns:
        list[_TeXPart]: 空要素を除去し, 隣接同種断片を結合した配列.
    """
    merged: list[_TeXPart] = []
    for part in parts:
        if not part.text:
            continue
        if (
            merged
            and merged[-1].translate == part.translate
            and merged[-1].kind == part.kind
        ):
            previous = merged[-1]
            merged[-1] = _TeXPart(
                translate=previous.translate,
                text=previous.text + part.text,
                kind=previous.kind,
            )
            continue
        merged.append(part)
    return merged


def _is_protected_line(stripped_line: str) -> bool:
    """1行単位で, 翻訳を避けるべきTeX行かどうかを判定する.

    Args:
        stripped_line: 行頭空白を除去した1行文字列.

    Returns:
        bool: 保護すべき行なら ``True``.
    """
    if not stripped_line:
        return True
    if stripped_line.startswith("%"):
        return True
    if stripped_line.startswith("\\"):
        return True
    return False
