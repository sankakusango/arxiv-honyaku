"""可逆操作③④: LaTeX 構造とコマンドを placeholder に退避し, 復元可能にする.

設計指針:
- placeholder は `XQK0001` のような短い token. TeX の特殊文字を含まず,
  自然な英文中で衝突しにくい (大文字+数字).
- 置換時に `original` を Substitution レコードに保存しておけば,
  復元時はその文字列を素朴に戻すだけで完全に元に戻る.
- `original` の中に別 placeholder が含まれる場合があるため, 復元は
  substitution リストを後ろから 1 回だけ処理する.
- 賢く分割しようとしすぎない. round-trip が壊れないことを最優先.
"""
from dataclasses import dataclass
import re

_PLACEHOLDER_PREFIX = "XQK"
_PLACEHOLDER_PREFIX_PLACEHOLDER = f"{_PLACEHOLDER_PREFIX}0000"
_PLACEHOLDER_PREFIX_KIND = "placeholder_prefix"
_PLACEHOLDER_RE = re.compile(rf"{_PLACEHOLDER_PREFIX}\d{{4,}}")

# ③ 構造単位置換の対象環境. caption を含みうる図表は外側だけを少し特別扱いする.
_MATH_ENVIRONMENTS = (
    # 数式
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "gather", "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
    "displaymath", "math",
)
_CAPTION_ENVIRONMENTS = (
    "figure", "figure*", "table", "table*",
    "wrapfigure", "wraptable",
)
_VERBATIM_ENVIRONMENTS = (
    "verbatim", "lstlisting", "minted",
)
_BIBLIOGRAPHY_ENVIRONMENTS = (
    "thebibliography",
)
_TABLE_BODY_ENVIRONMENTS = (
    "array",
    "longtable",
    "longtblr",
    "NiceArray",
    "NiceArray*",
    "NiceTabular",
    "NiceTabular*",
    "tabular",
    "tabular*",
    "tabularx",
    "tabulary",
    "tblr",
)
_BLOCK_ENVIRONMENTS = (
    *_MATH_ENVIRONMENTS,
    *_CAPTION_ENVIRONMENTS,
    *_VERBATIM_ENVIRONMENTS,
    *_BIBLIOGRAPHY_ENVIRONMENTS,
    *_TABLE_BODY_ENVIRONMENTS,
)

# ④ コマンド単位置換の除外対象.
# 基本は LaTeX コマンドを保護し, 自然言語を持つ構造/装飾コマンドだけを残す.
_UNPROTECTED_COMMANDS = frozenset({
    "section", "subsection", "subsubsection", "chapter",
    "paragraph", "subparagraph",
    "caption", "captionof",
    "abstract",
    "footnote",
    "textbf", "emph", "textit", "textsc", "textrm", "textsf", "texttt",
    "underline",
})

_INLINE_MATH_RE = re.compile(
    r"(?<!\\)\$\$.+?(?<!\\)\$\$"   # display $$...$$
    r"|(?<!\\)\$(?:\\.|[^$\\])*?(?<!\\)\$"  # inline $...$
    r"|\\\(.+?\\\)"                # \(...\)
    r"|\\\[.+?\\\]",               # \[...\]
    re.S,
)


@dataclass
class Substitution:
    """1つの placeholder 置換レコード."""

    placeholder: str
    kind: str
    original: str


def substitute(tex_text: str) -> tuple[str, list[Substitution]]:
    """tex_text に③④を適用し, 置換後本文と Substitution リストを返す.

    順序: 環境ブロック → インライン数式 → 連続コマンド の順. 後段は前段で生まれた
    placeholder を含む本文に対して動くので, 既存 placeholder を二重置換しない.
    """
    pool: list[Substitution] = []
    counter = [0]  # mutable container for nested closures
    used_placeholders = set(_PLACEHOLDER_RE.findall(tex_text))
    used_placeholders.add(_PLACEHOLDER_PREFIX_PLACEHOLDER)

    def make_placeholder() -> str:
        while True:
            counter[0] += 1
            placeholder = f"{_PLACEHOLDER_PREFIX}{counter[0]:04d}"
            if placeholder in used_placeholders:
                continue
            used_placeholders.add(placeholder)
            return placeholder

    out = tex_text
    if _PLACEHOLDER_PREFIX in out:
        out = out.replace(_PLACEHOLDER_PREFIX, _PLACEHOLDER_PREFIX_PLACEHOLDER)
        pool.append(Substitution(
            placeholder=_PLACEHOLDER_PREFIX_PLACEHOLDER,
            kind=_PLACEHOLDER_PREFIX_KIND,
            original=_PLACEHOLDER_PREFIX,
        ))

    # ③-a: 構造ブロック (begin/end ペア). ネストしない前提でゆるくマッチ.
    out = _substitute_environments(out, pool, make_placeholder)

    # ③-b: インライン数式.
    out = _substitute_inline_math(out, pool, make_placeholder)

    # ④: 連続する保護コマンド.
    out = _substitute_command_runs(out, pool, make_placeholder)

    return out, pool


def restore(text: str, substitutions: list[Substitution]) -> str:
    """`substitute` の結果を逆置換し, 元の本文に戻す.

    後から作った substitution の `original` に, 先に作った placeholder が入る
    ことがある. そのため後ろから 1 回だけ置換する. ループしないので,
    placeholder と original が互いに参照しても無限ループにはならない.
    """
    out = text
    for sub in reversed(substitutions):
        out = out.replace(sub.placeholder, sub.original)
    return out


# ----- internal helpers -----

def _substitute_environments(
    text: str,
    pool: list[Substitution],
    make_placeholder,
) -> str:
    """`\\begin{env}...\\end{env}` を placeholder へ. ネストしない単純マッチ."""
    out_parts: list[str] = []
    cursor = 0
    # 全環境を1つの regex で. 同じ env 名の begin/end をペアでマッチさせる.
    env_alt = "|".join(re.escape(e) for e in _BLOCK_ENVIRONMENTS)
    pattern = re.compile(
        rf"\\begin\{{(?P<env>{env_alt})\}}(?P<rest>.*?)\\end\{{(?P=env)\}}",
        re.S,
    )
    for match in pattern.finditer(text):
        out_parts.append(text[cursor:match.start()])
        original = match.group(0)
        env = match.group("env")
        kind = _classify_env_kind(env)
        if env in _CAPTION_ENVIRONMENTS:
            out_parts.append(_substitute_caption_environment(
                original, pool, make_placeholder,
            ))
        else:
            ph = make_placeholder()
            pool.append(Substitution(placeholder=ph, kind=kind, original=original))
            out_parts.append(ph)
        cursor = match.end()
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _substitute_caption_environment(
    text: str,
    pool: list[Substitution],
    make_placeholder,
) -> str:
    """図表環境のうち caption 以外を placeholder へ退避する.

    caption command 自体は本文に残すので, caption 引数は通常チャンクとして
    翻訳できる. caption がなければ環境全体を従来通り 1 placeholder にする.
    """
    caption_spans = _find_caption_spans(text)
    if not caption_spans:
        ph = make_placeholder()
        pool.append(Substitution(placeholder=ph, kind="figure_env", original=text))
        return ph

    out_parts: list[str] = []
    cursor = 0
    for start, end in caption_spans:
        _append_placeholder(
            out_parts, text[cursor:start], kind="figure_env",
            pool=pool, make_placeholder=make_placeholder,
        )
        out_parts.append(text[start:end])
        cursor = end
    _append_placeholder(
        out_parts, text[cursor:], kind="figure_env",
        pool=pool, make_placeholder=make_placeholder,
    )
    return "".join(out_parts)


def _append_placeholder(
    out_parts: list[str],
    original: str,
    *,
    kind: str,
    pool: list[Substitution],
    make_placeholder,
) -> None:
    """空でない `original` を placeholder として `out_parts` に追加する."""
    if not original:
        return
    ph = make_placeholder()
    pool.append(Substitution(placeholder=ph, kind=kind, original=original))
    out_parts.append(ph)


def _find_caption_spans(text: str) -> list[tuple[int, int]]:
    """`\\caption...{...}` の span を返す. ネストした brace は数える."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("\\", cursor)
        if start < 0:
            break
        parsed = _parse_command(text, start)
        if parsed is None:
            cursor = start + 1
            continue
        name, after_name = parsed
        if name in {"caption", "captionof"}:
            end = _consume_caption_command(
                text, after_name, required_args=2 if name == "captionof" else 1,
            )
            if end is not None:
                spans.append((start, end))
                cursor = end
                continue
        cursor = start + 1
    return spans


def _consume_caption_command(
    text: str,
    cursor: int,
    *,
    required_args: int,
) -> int | None:
    """caption command の optional/required arguments を消費する."""
    n = len(text)
    cursor = _consume_inline_space(text, cursor)
    while cursor < n and text[cursor] == "[":
        cursor = _skip_balanced(text, cursor, open_ch="[", close_ch="]")
        if cursor < 0:
            return None
        cursor = _consume_inline_space(text, cursor)

    for _ in range(required_args):
        if cursor >= n or text[cursor] != "{":
            return None
        cursor = _skip_balanced(text, cursor, open_ch="{", close_ch="}")
        if cursor < 0:
            return None
        cursor = _consume_inline_space(text, cursor)
    return cursor


def _substitute_inline_math(
    text: str,
    pool: list[Substitution],
    make_placeholder,
) -> str:
    """インライン数式を placeholder へ."""
    out_parts: list[str] = []
    cursor = 0
    for match in _INLINE_MATH_RE.finditer(text):
        out_parts.append(text[cursor:match.start()])
        ph = make_placeholder()
        pool.append(Substitution(
            placeholder=ph, kind="math_inline", original=match.group(0),
        ))
        out_parts.append(ph)
        cursor = match.end()
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _substitute_command_runs(
    text: str,
    pool: list[Substitution],
    make_placeholder,
) -> str:
    """連続する保護コマンド run を placeholder へ.

    `\\cite{a}~\\ref{b}\\label{c}` のように間に空白/`~`/`,` だけを挟む並びを 1 つに.
    引数 `{...}` `[...]` を含むコマンド全体を取り込む.
    """
    out_parts: list[str] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        # 次の置換対象コマンド先頭を探す.
        start = _find_next_substitutable_command(text, cursor)
        if start is None:
            out_parts.append(text[cursor:])
            break
        out_parts.append(text[cursor:start])
        # 連続 run を取り込む.
        run_end = _consume_command_run(text, start)
        original = text[start:run_end]
        ph = make_placeholder()
        pool.append(Substitution(
            placeholder=ph, kind="command_block", original=original,
        ))
        out_parts.append(ph)
        cursor = run_end
    return "".join(out_parts)


def _find_next_substitutable_command(text: str, start: int) -> int | None:
    """`text[start:]` 内で最初に現れる置換対象コマンド位置を返す."""
    cursor = start
    while cursor < len(text):
        command_start = text.find("\\", cursor)
        if command_start < 0:
            return None
        parsed = _parse_command(text, command_start)
        if parsed is not None and _is_substitutable_command(parsed[0]):
            return command_start
        cursor = command_start + 1
    return None


def _consume_command_run(text: str, start: int) -> int:
    """`start` から始まる連続コマンド run の終端 index を返す.

    1 コマンドの引数 `{...}` `[...]` を消費し, 区切り文字 (空白/`~`/`,`) のみを挟んで
    次の保護コマンドが続けば取り込む.
    """
    cursor = start
    n = len(text)
    while True:
        if cursor >= n or text[cursor] != "\\":
            break
        parsed = _parse_command(text, cursor)
        if parsed is None:
            break
        name, cursor = parsed
        if not _is_substitutable_command(name):
            break
        # 任意の `[...]` `{...}` 引数を消費.
        while cursor < n and text[cursor] in "[{":
            close = "]" if text[cursor] == "[" else "}"
            cursor = _skip_balanced(text, cursor, open_ch=text[cursor], close_ch=close)
            if cursor < 0:
                return n  # 不整合: 残り全部を取り込む
        # 区切り文字をスキップ.
        sep_start = cursor
        while cursor < n and text[cursor] in " \t~,":
            cursor += 1
        # 次が保護コマンドでなければ, 区切り文字は取り込まないので戻す.
        if cursor >= n or text[cursor] != "\\":
            return sep_start
        next_parsed = _parse_command(text, cursor)
        if next_parsed is None or not _is_substitutable_command(next_parsed[0]):
            return sep_start
    return cursor


def _parse_command(text: str, start: int) -> tuple[str, int] | None:
    """TeX command 名と, command 名直後の index を返す."""
    if start >= len(text) or text[start] != "\\" or start + 1 >= len(text):
        return None
    cursor = start + 1
    if not text[cursor].isalpha():
        return text[cursor], cursor + 1
    while cursor < len(text) and text[cursor].isalpha():
        cursor += 1
    name = text[start + 1:cursor]
    if cursor < len(text) and text[cursor] == "*":
        cursor += 1
    return name, cursor


def _is_substitutable_command(name: str) -> bool:
    """除外リストにない command は保護対象とみなす."""
    return name not in _UNPROTECTED_COMMANDS


def _consume_inline_space(text: str, cursor: int) -> int:
    """caption command の引数周辺の空白を消費する."""
    while cursor < len(text) and text[cursor] in " \t\r\n":
        cursor += 1
    return cursor


def _skip_balanced(text: str, start: int, *, open_ch: str, close_ch: str) -> int:
    """`text[start]` が `open_ch` のとき, 対応する `close_ch` の次の index を返す.

    `\\{` `\\}` はエスケープとして扱う. 不整合時は -1.
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _classify_env_kind(env: str) -> str:
    """環境名から kind 文字列を決める."""
    if env in _CAPTION_ENVIRONMENTS:
        return "figure_env"
    if env in _BIBLIOGRAPHY_ENVIRONMENTS:
        return "bibliography_env"
    if env in _VERBATIM_ENVIRONMENTS:
        return "verbatim_env"
    if env in _TABLE_BODY_ENVIRONMENTS:
        return "table_body_env"
    return "math_env"
