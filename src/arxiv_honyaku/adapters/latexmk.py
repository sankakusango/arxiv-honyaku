"""latexmkを使ったPDFコンパイルアダプタモジュール."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess

from arxiv_honyaku.core.ports import PdfCompiler

LOGGER = logging.getLogger("arxiv_honyaku.adapters.latexmk")


class LatexmkCompiler(PdfCompiler):
    """TeXファイルをlatexmkでコンパイルする実装.

    Attributes:
        _binary: 呼び出すlatexmk実行ファイル名.
        _allow_shell_escape: ``True`` の場合, ``-shell-escape`` を付与して実行する.
        _texlive_versions: 試行順の TeX Live バージョン指定.
            ``None`` の場合は ``/opt/texlive`` から自動検出する.
    """

    def __init__(
        self,
        *,
        binary: str = "latexmk",
        allow_shell_escape: bool = False,
        texlive_versions: tuple[str, ...] | None = None,
    ) -> None:
        """利用するlatexmkバイナリ名を受け取る.

        Args:
            binary: latexmk実行ファイル名.
            allow_shell_escape: ``True`` の場合, ``-shell-escape`` を有効化する.
            texlive_versions: TeX Live バージョン試行順.
                例: ``("2025", "2023")`` や ``("2015", "2015")``.
        """
        self._binary = binary
        self._allow_shell_escape = allow_shell_escape
        self._texlive_versions = texlive_versions

    def compile(self, tex_file: Path, build_dir: Path) -> Path:
        """TeXをコンパイルし, 生成PDFパスを返す.

        Args:
            tex_file: コンパイル対象のメインTeXファイルパス.
            build_dir: ビルド成果物の出力ディレクトリ.

        Returns:
            Path: 生成されたPDFファイルパス.

        Raises:
            RuntimeError: latexmk実行が失敗した場合.
            FileNotFoundError: 成功終了してもPDFが見つからない場合.
        """
        compile_targets = _resolve_compile_targets(
            binary=self._binary,
            texlive_versions=self._texlive_versions,
        )
        if not compile_targets:
            raise RuntimeError("latexmk failed. No compile targets were resolved.")

        shell_escape_enabled = self._allow_shell_escape
        if shell_escape_enabled:
            LOGGER.info("latexmk is running with -shell-escape enabled for %s.", tex_file)

        last_completed: subprocess.CompletedProcess[bytes] | None = None
        attempted_targets: list[str] = []
        for index, (target_label, target_binary, target_bin_dir) in enumerate(
            compile_targets,
            start=1,
        ):
            attempted_targets.append(target_label)
            if target_binary is None:
                LOGGER.warning(
                    "Skipping TeX Live target %s because latexmk binary was not found.",
                    target_label,
                )
                continue

            cmd = _build_latexmk_command(
                binary=target_binary,
                tex_file=tex_file,
                build_dir=build_dir,
                shell_escape_enabled=shell_escape_enabled,
            )
            _reset_build_dir(build_dir)
            _mirror_source_directories(source_root=tex_file.parent, build_dir=build_dir)
            env = dict(os.environ)
            if target_bin_dir is not None:
                current_path = env.get("PATH", "")
                env["PATH"] = (
                    f"{target_bin_dir}:{current_path}" if current_path else target_bin_dir
                )
                LOGGER.info(
                    "Compiling with TeX Live target %s via PATH prefix: %s",
                    target_label,
                    target_bin_dir,
                )
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=tex_file.parent,
                    env=env,
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError:
                LOGGER.warning(
                    "Skipping TeX Live target %s because executable was not found: %s",
                    target_label,
                    target_binary,
                )
                continue
            last_completed = completed

            if completed.returncode == 0:
                pdf_path = build_dir / f"{tex_file.stem}.pdf"
                if not pdf_path.exists():
                    raise FileNotFoundError(f"Compiled PDF not found: {pdf_path}")
                return pdf_path

            if index < len(compile_targets):
                LOGGER.warning(
                    "latexmk failed with TeX Live target %s (%d/%d). Trying next target for %s.",
                    target_label,
                    index,
                    len(compile_targets),
                    tex_file,
                )

        raise RuntimeError(
            _build_latexmk_error_message(
                last_completed,
                attempted_targets=tuple(attempted_targets),
            )
        )


def _resolve_compile_targets(
    *,
    binary: str,
    texlive_versions: tuple[str, ...] | None,
) -> list[tuple[str, str | None, str | None]]:
    """コンパイル時に試行する ``latexmk`` バイナリ列を解決する.

    Args:
        binary: システムPATH上の ``latexmk`` コマンド名.
        texlive_versions: ユーザー指定のTeX Liveバージョン列.

    Returns:
        list[tuple[str, str | None, str | None]]:
            ``(表示ラベル, 実行バイナリ, PATH先頭に加えるbinディレクトリ)`` の配列.
            バイナリが見つからない場合は ``None`` が入る.
    """
    raw_targets = list(texlive_versions) if texlive_versions is not None else _discover_texlive_versions()
    if not raw_targets:
        raw_targets = ["system"]

    resolved_targets: list[tuple[str, str | None, str | None]] = []
    for raw_target in raw_targets:
        target = raw_target.strip()
        if not target:
            continue
        resolved_binary = _resolve_latexmk_binary_for_target(
            binary=binary,
            target=target,
        )
        resolved_bin_dir = str(Path(resolved_binary).parent) if resolved_binary else None
        if target.lower() in {"system", "path", "default"}:
            resolved_bin_dir = None
        resolved_targets.append(
            (
                target,
                resolved_binary,
                resolved_bin_dir,
            )
        )
    return resolved_targets


def _discover_texlive_versions(texlive_root: Path = Path("/opt/texlive")) -> list[str]:
    """``/opt/texlive`` 配下から利用可能な年バージョンを自動検出する.

    Args:
        texlive_root: TeX Live インストールルート.

    Returns:
        list[str]: 例 ``["2025", "2023"]`` のような降順バージョン列.
    """
    if not texlive_root.exists():
        return []
    versions = [
        child.name
        for child in texlive_root.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return sorted(versions, key=int, reverse=True)


def _resolve_latexmk_binary_for_target(
    *,
    binary: str,
    target: str,
    texlive_root: Path = Path("/opt/texlive"),
) -> str | None:
    """1つのターゲット指定から実行すべき ``latexmk`` バイナリを解決する.

    Args:
        binary: システムPATH上の ``latexmk`` コマンド名.
        target: ``system`` または TeX Live 年バージョン文字列.
        texlive_root: TeX Live インストールルート.

    Returns:
        str | None: 実行バイナリパス, 未解決時は ``None``.
    """
    lowered_target = target.lower()
    if lowered_target in {"system", "path", "default"}:
        return binary

    texlive_bin_root = texlive_root / target / "bin"
    if not texlive_bin_root.exists():
        return None
    candidates = sorted(
        path
        for path in texlive_bin_root.glob("*/latexmk")
        if path.is_file()
    )
    if not candidates:
        return None
    return str(candidates[0])


def _build_latexmk_command(
    *,
    binary: str,
    tex_file: Path,
    build_dir: Path,
    shell_escape_enabled: bool,
) -> list[str]:
    """``latexmk`` 実行コマンドを構築する.

    Args:
        binary: latexmk 実行バイナリ.
        tex_file: コンパイル対象メイン TeX.
        build_dir: ``-outdir`` 出力先.
        shell_escape_enabled: ``True`` の場合 ``-shell-escape`` を付与.

    Returns:
        list[str]: ``subprocess.run`` に渡すコマンド配列.
    """
    command = [
        binary,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={build_dir}",
        tex_file.name,
    ]
    if shell_escape_enabled:
        command.insert(1, "-shell-escape")
    return command


def _decode_process_output(output: bytes) -> str:
    """プロセス出力バイト列を, 置換付きUTF-8で安全に文字列化する.

    Args:
        output: ``subprocess.run`` が返した標準出力または標準エラーのバイト列.

    Returns:
        str: ログ表示向けに安全にデコード済みの文字列.
    """
    return output.decode("utf-8", errors="replace")


def _build_latexmk_error_message(
    completed: subprocess.CompletedProcess[bytes] | None,
    *,
    attempted_targets: tuple[str, ...] = (),
) -> str:
    """latexmk 失敗時に返す例外メッセージを構築する.

    Args:
        completed: 最後に実行した latexmk の結果.

    Returns:
        str: 末尾ログを含む例外メッセージ.
    """
    if completed is None:
        if attempted_targets:
            targets_text = ", ".join(attempted_targets)
            return (
                "latexmk failed. No executable target succeeded. "
                f"Tried targets: {targets_text}"
            )
        return "latexmk failed."
    stdout_text = _decode_process_output(completed.stdout)
    stderr_text = _decode_process_output(completed.stderr)
    message_lines = [
        "latexmk failed.",
        f"Tried targets: {', '.join(attempted_targets)}" if attempted_targets else "",
        stdout_text[-2000:],
        stderr_text[-2000:],
    ]
    return "\n".join(line for line in message_lines if line).strip()


def _reset_build_dir(build_dir: Path) -> None:
    """コンパイル試行ごとに build ディレクトリをクリーン再作成する.

    ``latexmk`` は前回失敗時の ``.fdb_latexmk`` を見ると, 次回試行でも
    ``Nothing to do`` で止まることがある. そのため外側 retry では毎回
    build ディレクトリを作り直し, 状態を完全に初期化してから再実行する.

    Args:
        build_dir: 再作成対象のビルド成果物ディレクトリ.

    Returns:
        None: 常に ``None``.
    """
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)


def _mirror_source_directories(*, source_root: Path, build_dir: Path) -> None:
    """ビルド出力先へ, ソースツリーのサブディレクトリ構造だけを複製する.

    ``latexmk -outdir=...`` は ``\\include{results/tables/foo}`` のような
    相対 include に対して, ``outdir/results/tables/foo.aux`` を開こうとする.
    そのため build 側に対応サブディレクトリが無いと, PDF 本体はまだ正常でも
    ``.aux`` 書き込みで停止する. この関数は translated ソースツリー配下の
    ディレクトリだけを build 側へ先回りで作り, そうした失敗を防ぐ.

    Args:
        source_root: latexmk の ``cwd`` として使う translated ツリールート.
        build_dir: ``-outdir`` に渡すビルド成果物ディレクトリ.

    Returns:
        None: 常に ``None``.
    """
    for source_dir in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        relative_dir = source_dir.relative_to(source_root)
        if not relative_dir.parts:
            continue
        (build_dir / relative_dir).mkdir(parents=True, exist_ok=True)
