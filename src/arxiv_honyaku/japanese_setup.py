"""日本語コンパイルのための CJK 注入と layout 補正.

メイン .tex に対しては preamble へ CJK パッケージを注入し本文を CJK 環境で包む.
全 .tex に対しては `japanese_layout_mode` に応じた wrapfigure 等の補正を行う.
"""
from typing import Literal
import re

JapaneseFontMode = Literal["compat", "paper-like"]
JapaneseLayoutMode = Literal["preserve", "adaptive", "safe"]

_CJK_PACKAGE_LINE = "\\usepackage{CJKutf8}\n"
_IPAEX_TYPE1_PACKAGE_LINE = "\\usepackage{ipaex-type1}\n"
_CJK_BEGIN_COMPAT = "\\begin{CJK}{UTF8}{min}\n"
_CJK_BEGIN_PAPER_LIKE = "\\begin{CJK}{UTF8}{ipxm}\n"
_CJK_END = "\\end{CJK}\n"

_CJK_BEGIN_RE = re.compile(r"\\begin\{CJK\}\{UTF8\}\{[^}]+\}\n?")
_WRAP_BEGIN_RE = re.compile(r"\\begin\{wrap(figure|table)\}(?:\[[^\]]*\])?\{[^{}]*\}\{[^{}]*\}")
_WRAP_END_RE = re.compile(r"\\end\{wrap(figure|table)\}")
_FLOAT_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>figure\*?|table\*?)\}(?P<opt>\[[^\]]*\])?(?P<body>.*?)\\end\{(?P=env)\}",
    re.S,
)
_NEGATIVE_VSPACE_RE = re.compile(r"\\vspace\*?\{\s*-[^{}]+\}")


def inject_cjk(tex_text: str, *, font_mode: JapaneseFontMode) -> str:
    """メイン .tex に CJK パッケージと本文ラッパを注入する. 既設なら重複追加しない."""
    if font_mode == "compat":
        required_packages = [_CJK_PACKAGE_LINE]
        cjk_begin = _CJK_BEGIN_COMPAT
    else:  # paper-like
        required_packages = [_CJK_PACKAGE_LINE, _IPAEX_TYPE1_PACKAGE_LINE]
        cjk_begin = _CJK_BEGIN_PAPER_LIKE

    out = tex_text
    if "\\begin{document}" in out:
        injection = "".join(p for p in required_packages if p not in out)
        if injection:
            out = out.replace(
                "\\begin{document}", f"{injection}\n\\begin{{document}}", 1,
            )

    if _CJK_BEGIN_RE.search(out):
        out = _CJK_BEGIN_RE.sub(cjk_begin, out, count=1)
    elif "\\begin{document}" in out:
        out = out.replace("\\begin{document}", f"\\begin{{document}}\n{cjk_begin}", 1)

    if _CJK_END.rstrip() not in out and "\\end{document}" in out:
        out = out.replace("\\end{document}", f"{_CJK_END}\\end{{document}}", 1)
    return out


def apply_layout(tex_text: str, *, mode: JapaneseLayoutMode) -> str:
    """`mode` に従って wrapfigure 等の補正を行う. preserve なら no-op."""
    if mode == "preserve":
        return tex_text
    out = tex_text
    if mode == "safe":
        out = _WRAP_BEGIN_RE.sub(lambda m: f"\\begin{{{m.group(1)}}}[!htbp]", out)
        out = _WRAP_END_RE.sub(lambda m: f"\\end{{{m.group(1)}}}", out)
        out = _neutralize_negative_vspace_in_floats(out)
    elif mode == "adaptive":
        # 簡易版: wrap を通常 float に置換するのみ. v04 の minipage 化は省略.
        out = _WRAP_BEGIN_RE.sub(lambda m: f"\\begin{{{m.group(1)}}}[!htbp]", out)
        out = _WRAP_END_RE.sub(lambda m: f"\\end{{{m.group(1)}}}", out)
    return out


def _neutralize_negative_vspace_in_floats(tex_text: str) -> str:
    """figure/table 内の負の \\vspace を 0pt に丸める."""
    def replace(match: re.Match[str]) -> str:
        env = match.group("env")
        opt = match.group("opt") or ""
        body = _NEGATIVE_VSPACE_RE.sub(r"\\vspace{0pt}", match.group("body"))
        return f"\\begin{{{env}}}{opt}{body}\\end{{{env}}}"
    return _FLOAT_BLOCK_RE.sub(replace, tex_text)
