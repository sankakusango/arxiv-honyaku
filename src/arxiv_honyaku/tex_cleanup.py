"""翻訳前に TeX ソースを安全に正規化する.

verbatim / lstlisting / minted 環境内は空白や `%` 自体が本文なので保持する.
それ以外ではコメントを削除し, コンパイル結果に影響しにくい空白だけを畳む.
"""
import re

_VERBATIM_BEGIN_RE = re.compile(r"^\s*\\begin\{(verbatim|lstlisting|minted)(\*?)\}")
_VERBATIM_END_RE = re.compile(r"^\s*\\end\{(verbatim|lstlisting|minted)(\*?)\}")
_SI_NO_UNIT_RE = re.compile(r"(\\SI\{[^{}]*\})(?!\s*\{)")


def clean_tex(tex_text: str) -> str:
    """コメントと安全に捨てられる空白を除去して返す."""
    out_lines: list[str] = []
    in_verbatim = False
    previous_blank = False
    for line in tex_text.splitlines(keepends=True):
        if in_verbatim:
            out_lines.append(line)
            if _VERBATIM_END_RE.match(line):
                in_verbatim = False
            previous_blank = False
            continue
        if _VERBATIM_BEGIN_RE.match(line):
            in_verbatim = True
            out_lines.append(line)
            previous_blank = False
            continue

        line = _strip_comment(line)
        if line is None:
            continue
        line = _normalize_insignificant_whitespace(line)
        if _is_blank_line(line):
            if previous_blank or not line:
                continue
            previous_blank = True
        else:
            previous_blank = False
        out_lines.append(line)
    return "".join(out_lines)


def pad_unitless_si(tex_text: str) -> str:
    r"""単位引数を欠いた ``\SI{xxx}`` に空単位 ``{}`` を補う.

    siunitx の ``\SI`` は本来 ``\SI{値}{単位}`` の 2 引数だが, arXiv 投稿には
    単位を省略した ``\SI{8e-5}`` のような書き方が混入することがある.

    英語のままだと ``\SI{8e-5},`` のように直後が ASCII なので通る場合があるが,
    日本語化後は直後が和文になり, siunitx 内部で致命エラーになることがある.
    ここでは ``\SI{xxx}{}`` に正規化して空単位を渡す. 直後に既に ``{...}`` が
    続く正規の呼び出しは lookahead で除外する.
    """
    return _SI_NO_UNIT_RE.sub(r"\1{}", tex_text)


def _strip_comment(line: str) -> str | None:
    """行頭コメント行と行内コメントを除去する."""
    comment_start = _find_comment_start(line)
    if comment_start is None:
        return line
    body = line[:comment_start]
    if not body.strip():
        return None
    return body + _line_ending(line)


def _normalize_insignificant_whitespace(line: str) -> str:
    """TeX の結果に影響しにくい行末空白だけを削る."""
    ending = _line_ending(line)
    body = line[:-len(ending)] if ending else line
    return body.rstrip(" \t") + ending


def _is_blank_line(line: str) -> bool:
    """空白と改行だけの行かを返す."""
    return not line.strip()


def _find_comment_start(line: str) -> int | None:
    """最初の未エスケープ `%` の index を返す."""
    backslashes = 0
    for index, ch in enumerate(line):
        if ch == "%":
            if backslashes % 2 == 0:
                return index
            backslashes = 0
            continue
        if ch == "\\":
            backslashes += 1
            continue
        backslashes = 0
    return None


def _line_ending(line: str) -> str:
    """`line` の改行文字を返す."""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""
