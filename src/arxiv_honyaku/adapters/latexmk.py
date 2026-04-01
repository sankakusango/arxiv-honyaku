"""latexmkを使ったPDFコンパイルアダプタモジュール."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import time

from arxiv_honyaku.core.ports import PdfCompiler

LOGGER = logging.getLogger("arxiv_honyaku.adapters.latexmk")


class LatexmkCompiler(PdfCompiler):
    """TeXファイルをlatexmkでコンパイルする実装.

    Attributes:
        _binary: 呼び出すlatexmk実行ファイル名.
        _max_attempts: 外側で許可する最大試行回数.
        _retry_delay_seconds: 再試行前の待機秒数.
    """

    def __init__(
        self,
        *,
        binary: str = "latexmk",
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        """利用するlatexmkバイナリ名を受け取る.

        Args:
            binary: latexmk実行ファイル名.
            max_attempts: latexmk 呼び出し全体の最大試行回数.
            retry_delay_seconds: 失敗時に再試行まで待機する秒数.
        """
        self._binary = binary
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

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
        cmd = [
            self._binary,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={build_dir}",
            tex_file.name,
        ]
        last_completed: subprocess.CompletedProcess[bytes] | None = None
        for attempt in range(1, self._max_attempts + 1):
            _reset_build_dir(build_dir)
            _mirror_source_directories(source_root=tex_file.parent, build_dir=build_dir)
            completed = subprocess.run(
                cmd,
                cwd=tex_file.parent,
                capture_output=True,
                check=False,
            )
            last_completed = completed
            if completed.returncode == 0:
                pdf_path = build_dir / f"{tex_file.stem}.pdf"
                if not pdf_path.exists():
                    raise FileNotFoundError(f"Compiled PDF not found: {pdf_path}")
                return pdf_path

            if attempt < self._max_attempts:
                LOGGER.warning(
                    "latexmk failed. Retrying compile attempt %d/%d for %s.",
                    attempt,
                    self._max_attempts,
                    tex_file,
                )
                time.sleep(self._retry_delay_seconds)

        raise RuntimeError(_build_latexmk_error_message(last_completed))


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
) -> str:
    """latexmk 失敗時に返す例外メッセージを構築する.

    Args:
        completed: 最後に実行した latexmk の結果.

    Returns:
        str: 末尾ログを含む例外メッセージ.
    """
    if completed is None:
        return "latexmk failed."
    return "\n".join(
        [
            "latexmk failed.",
            _decode_process_output(completed.stdout)[-2000:],
            _decode_process_output(completed.stderr)[-2000:],
        ]
    ).strip()


def _reset_build_dir(build_dir: Path) -> None:
    """retry ごとに build ディレクトリをクリーン再作成する.

    ``latexmk`` は前回失敗時の ``.fdb_latexmk`` を見ると, retry でも
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
