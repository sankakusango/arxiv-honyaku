"""和訳済み TeX を日本語対応でコンパイルしやすく整形するモジュール.

このモジュールは, 翻訳後 TeX ツリーに対して日本語コンパイルに必要な注入を行う.
責務は大きく 2 つあり, 1 つはメイン TeX へ日本語フォント関連パッケージと
``CJK`` 環境を入れること, もう 1 つは日本語化で崩れやすい wrap 系レイアウトを
設定モードに応じて保守的に補正することである.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

_JAPANESE_CHARACTER_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)
_CJK_PACKAGE_LINE = "\\usepackage{CJKutf8}\n"
_IPAEX_TYPE1_PACKAGE_LINE = "\\usepackage{ipaex-type1}\n"
_CJK_BEGIN_COMPAT = "\\begin{CJK}{UTF8}{min}\n"
_CJK_BEGIN_PAPER_LIKE = "\\begin{CJK}{UTF8}{ipxm}\n"
_CJK_END = "\\end{CJK}\n"
_CJK_BEGIN_RE = re.compile(r"\\begin\{CJK\}\{UTF8\}\{[^}]+\}\n?")
_WRAP_BEGIN_RE = re.compile(
    r"\\begin\{wrap(figure|table)\}(?:\[[^\]]*\])?\{[^{}]*\}\{[^{}]*\}"
)
_WRAP_END_RE = re.compile(r"\\end\{wrap(figure|table)\}")
_WRAP_BLOCK_RE = re.compile(
    r"\\begin\{wrap(?P<kind>figure|table)\}(?:\[[^\]]*\])?\{(?P<side>[^{}]*)\}\{(?P<width>[^{}]*)\}(?P<body>.*?)\\end\{wrap(?P=kind)\}",
    re.S,
)
_FLOAT_BLOCK_RE = re.compile(
    r"\\begin\{(?P<envname>figure\*?|table\*?)\}(?P<opt>\[[^\]]*\])?"
    r"(?P<body>.*?)"
    r"\\end\{(?P=envname)\}",
    re.S,
)
_NEGATIVE_VSPACE_RE = re.compile(r"\\vspace\*?\{\s*-[^{}]+\}")
_RISKY_JAPANESE_BOUNDARY_COMMAND_RE = re.compile(
    r"(\\(?:SI|SIlist|SIrange|num|numlist|numrange|qty|qtyproduct|qtyrange)"
    r"(?:\[[^\]]*\])?(?:\{[^{}]*\})+)(?="
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f])"
)
_COMMENTED_LAYOUT_COMMAND_RE = re.compile(
    r"%[ \t]+(\\(?:vspace|hspace|vskip|hskip)\*?(?:\[[^\]]*\])?\{[^{}]*\})"
)


def translated_project_contains_japanese(source_dir: Path) -> bool:
    """翻訳済みTeXツリーに日本語文字が含まれるかを判定する.

    Args:
        source_dir: 判定対象の翻訳済みTeXルートディレクトリ.

    Returns:
        bool: いずれかの ``.tex`` ファイルに日本語文字が含まれる場合は ``True``.
    """
    for tex_path in sorted(source_dir.rglob("*.tex")):
        if _JAPANESE_CHARACTER_RE.search(
            tex_path.read_text(encoding="utf-8", errors="ignore")
        ):
            return True
    return False


def prepare_main_tex_for_japanese_pdf_compilation(
    tex_text: str,
    *,
    font_mode: JapaneseFontMode,
) -> str:
    """主文書に ``CJKutf8`` パッケージと本文ラッパを注入する.

    日本語文字を含む和訳済みTeXを ``pdflatex`` 系でコンパイルできるように,
    preamble へ必要な日本語フォント関連パッケージを追加し, 文書本文全体を
    ``CJK`` 環境で包む. ``font_mode="compat"`` は従来どおり ``min`` を使い,
    ``font_mode="paper-like"`` は IPAex 明朝 ``ipxm`` を使う.
    既に同等の設定が入っている場合は, 重複追加しない.

    Args:
        tex_text: 整形対象のメインTeX全文.
        font_mode: 日本語フォント注入方針.

    Returns:
        str: 日本語PDFコンパイル向けに整形済みのTeX全文.
    """
    prepared = tex_text
    if font_mode == "compat":
        required_package_lines = [_CJK_PACKAGE_LINE]
        desired_cjk_begin = _CJK_BEGIN_COMPAT
    elif font_mode == "paper-like":
        required_package_lines = [
            _CJK_PACKAGE_LINE,
            _IPAEX_TYPE1_PACKAGE_LINE,
        ]
        desired_cjk_begin = _CJK_BEGIN_PAPER_LIKE
    else:  # pragma: no cover - 呼び出し側の型保証を越えた防御.
        raise ValueError(f"Unsupported Japanese font mode: {font_mode}")

    if "\\begin{document}" in prepared:
        package_injection = "".join(
            package_line
            for package_line in required_package_lines
            if package_line not in prepared
        )
        if package_injection:
            prepared = prepared.replace(
                "\\begin{document}",
                f"{package_injection}\n\\begin{{document}}",
                1,
            )

    if _CJK_BEGIN_RE.search(prepared):
        prepared = _CJK_BEGIN_RE.sub(desired_cjk_begin, prepared, count=1)
    elif "\\begin{document}" in prepared:
        prepared = prepared.replace(
            "\\begin{document}",
            f"\\begin{{document}}\n{desired_cjk_begin}",
            1,
        )

    if _CJK_END.rstrip() not in prepared and "\\end{document}" in prepared:
        prepared = prepared.replace(
            "\\end{document}",
            f"{_CJK_END}\\end{{document}}",
            1,
        )

    return prepared


def prepare_translated_tex_tree_for_japanese_layout(
    source_dir: Path,
    *,
    mode: JapaneseLayoutMode,
) -> None:
    """日本語版で崩れやすい TeX レイアウトを木全体で保守的に整える.

    ``mode="preserve"`` では翻訳後TeXの構造をそのまま維持し, 何も変更しない.
    ``mode="adaptive"`` では, 回り込み float を左右寄せの通常 float へ変換し,
    幅と配置の雰囲気を残しつつ本文との衝突を避ける.
    ``mode="safe"`` では ``wrapfigure`` / ``wraptable`` を通常の
    ``figure`` / ``table`` に置き換える. 同時に ``figure`` / ``table`` 内の
    過度な負の ``\\vspace`` を ``0pt`` に丸める. 回り込み図は英語原稿では成立していても,
    日本語化後は段落幅や改行位置の変化で本文と干渉しやすいためである.

    Args:
        source_dir: 整形対象の翻訳済みTeXルートディレクトリ.
        mode: レイアウト補正方針.

    Returns:
        None: 常に ``None``.
    """
    if mode == "preserve":
        return

    for tex_path in sorted(source_dir.rglob("*.tex")):
        original_text = tex_path.read_text(encoding="utf-8")
        prepared_text = _normalize_risky_japanese_macro_boundaries(original_text)
        prepared_text = _normalize_commented_layout_commands(prepared_text)
        if mode == "adaptive":
            prepared_text = _adapt_wrap_float_environments(prepared_text)
        elif mode == "safe":
            prepared_text = _replace_wrap_float_environments(prepared_text)
            prepared_text = _neutralize_negative_vspace_in_floats(prepared_text)
        if prepared_text == original_text:
            continue
        tex_path.write_text(prepared_text, encoding="utf-8")


def _replace_wrap_float_environments(tex_text: str) -> str:
    """回り込み float 環境を通常 float 環境へ置き換える.

    Args:
        tex_text: 整形対象のTeX全文.

    Returns:
        str: 回り込み float を通常 float に置換したTeX全文.
    """
    replaced = _WRAP_BEGIN_RE.sub(
        lambda match: f"\\begin{{{match.group(1)}}}[!htbp]",
        tex_text,
    )
    return _WRAP_END_RE.sub(
        lambda match: f"\\end{{{match.group(1)}}}",
        replaced,
    )


def _adapt_wrap_float_environments(tex_text: str) -> str:
    """回り込み float を左右寄せの通常 float へ変換する.

    Args:
        tex_text: 整形対象のTeX全文.

    Returns:
        str: 元の左右配置や幅を残した通常 float へ置換したTeX全文.
    """
    return _WRAP_BLOCK_RE.sub(_convert_wrap_block_to_side_float, tex_text)


def _convert_wrap_block_to_side_float(match: re.Match[str]) -> str:
    """1つの wrap float ブロックを, 左右寄せ通常 float へ変換する.

    Args:
        match: ``wrapfigure`` / ``wraptable`` ブロックの正規表現マッチ.

    Returns:
        str: 左右寄せ ``minipage`` を内包した通常 float 文字列.
    """
    kind = match.group("kind")
    side = match.group("side").strip().upper()
    width = match.group("width").strip()
    body = match.group("body").strip("\n")
    alignment = "flushright" if side in {"R", "O"} else "flushleft"
    indented_body = _indent_block(body, prefix="    ")
    return (
        f"\\begin{{{kind}}}[!t]\n"
        f"\\begin{{{alignment}}}\n"
        f"\\begin{{minipage}}{{{width}}}\n"
        f"{indented_body}\n"
        f"\\end{{minipage}}\n"
        f"\\end{{{alignment}}}\n"
        f"\\end{{{kind}}}"
    )


def _indent_block(text: str, *, prefix: str) -> str:
    """複数行文字列の各行へ共通インデントを付ける.

    Args:
        text: インデント対象文字列.
        prefix: 各行へ付与する接頭辞.

    Returns:
        str: 各行へ接頭辞を付与した文字列.
    """
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def _normalize_risky_japanese_macro_boundaries(tex_text: str) -> str:
    """日本語直結で壊れやすいマクロ境界へ空ブロックを補う.

    ``siunitx`` 系コマンドは, 日本語が直後に来ると次トークンの読み取りが
    不安定になることがある. そこで ``\\SI{8e-07}まで`` のような箇所を
    ``\\SI{8e-07}{}まで`` へ機械的に正規化する.

    Args:
        tex_text: 整形対象のTeX全文.

    Returns:
        str: 危険な境界へ ``{}`` を補ったTeX全文.
    """
    return _RISKY_JAPANESE_BOUNDARY_COMMAND_RE.sub(r"\1{}", tex_text)


def _normalize_commented_layout_commands(tex_text: str) -> str:
    """行末コメントと inline レイアウト命令の危険な連結をほどく.

    caption では, ``.%`` の直後に ``\\vspace{...}`` を置いて改行したいケースが
    ある. ここが翻訳や整形で ``.% \\vspace{...}`` に潰れると, コメントが
    ``\\vspace`` と閉じ波括弧まで飲み込んで runaway を起こす. そのため
    コメント記号とレイアウト命令の間に改行を戻す.

    Args:
        tex_text: 整形対象のTeX全文.

    Returns:
        str: コメント連結を安全な改行へ戻したTeX全文.
    """
    return _COMMENTED_LAYOUT_COMMAND_RE.sub(r"%\n  \1", tex_text)


def _neutralize_negative_vspace_in_floats(tex_text: str) -> str:
    """``figure`` / ``table`` 内の負の ``\\vspace`` を保守的に無効化する.

    日本語化後は行高と改行位置が変わるため, 英語原稿で成立していた大きな負の
    ``\\vspace`` がキャプションや本文の重なりを起こしやすい. この関数は,
    float 環境内に限って負値指定を ``\\vspace{0pt}`` へ置き換える.

    Args:
        tex_text: 整形対象のTeX全文.

    Returns:
        str: 負の ``\\vspace`` を無効化したTeX全文.
    """

    def _replace_block(match: re.Match[str]) -> str:
        envname = match.group("envname")
        opt = match.group("opt") or ""
        body = match.group("body")
        normalized_body = _NEGATIVE_VSPACE_RE.sub(r"\\vspace{0pt}", body)
        return (
            f"\\begin{{{envname}}}{opt}"
            f"{normalized_body}"
            f"\\end{{{envname}}}"
        )

    return _FLOAT_BLOCK_RE.sub(_replace_block, tex_text)


JapaneseLayoutMode = Literal["preserve", "adaptive", "safe"]
JapaneseFontMode = Literal["compat", "paper-like"]
