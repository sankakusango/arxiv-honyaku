"""texソースをコンパイルする."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import logging
import os
import re

from tqdm import tqdm

logger = logging.getLogger(__name__)

LATEXMK_BINARY = "latexmk"
ALLOW_SHELL_ESCAPE = False
TEXLIVE_ROOT = Path("/opt/texlive")
ProgressCallback = Callable[[str, dict[str, object]], None]
_LATEXMK_LOG_NAMES = ("latexmk.stderr.log", "latexmk.stdout.log")
_LATEX_ERROR_PATTERNS = (
    re.compile(r"^! .+"),
    re.compile(r"^\./.+:\d+: .+"),
    re.compile(r"^.+:\d+: (?:LaTeX Error|Package .+ Error|Misplaced|Undefined control sequence|Improper|Missing \$ inserted|Extra alignment tab|Runaway argument|TeX capacity exceeded).+"),
    re.compile(r"^!!! Error: .+"),
    re.compile(r"^Fatal error occurred.+"),
    re.compile(r"^No pages of output\."),
)


@dataclass
class CompileTarget:
    """1回の latexmk 試行で使う TeX Live ターゲット.

    `label` (年バージョン) を渡すと `bin_dir` を `TEXLIVE_ROOT` 配下から自動解決する.
    解決に失敗した場合は `FileNotFoundError` を送出する.
    """

    label: str
    bin_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.label = str(self.label)
        candidates = sorted(
            (TEXLIVE_ROOT / self.label / "bin").glob(f"*/{LATEXMK_BINARY}")
        )
        if not candidates:
            raise FileNotFoundError(
                f"{LATEXMK_BINARY} not found under {TEXLIVE_ROOT / self.label}"
            )
        self.bin_dir = candidates[0].parent


def find_main_tex(source_dir: Path) -> Path:
    """\\documentclass を含む最初の .tex を返す."""
    for path in sorted(source_dir.rglob("*.tex")):
        if "\\documentclass" in path.read_text(encoding="utf-8", errors="ignore"):
            return path
    raise FileNotFoundError(f"No main .tex in {source_dir}")

def compile_tex(
    source_dir: Path,
    output_dir: Path,
    *,
    texlive_version: str,
) -> Path:
    """指定 TeX Live バージョンで1回 latexmk を回し, 生成 PDF パスを返す.

    `output_dir` は呼び出し側が用意した空ディレクトリを想定する. 中身は上書きされる.

    Args:
        source_dir: TeXソースツリーのルートディレクトリ. メインTeXは内部で自動検出する.
        output_dir: latexmk の `-outdir` として使うディレクトリ. 既存・空であること.
        texlive_version: 利用する TeX Live 年バージョン (例: `"2025"`).

    Returns:
        Path: 生成された PDF ファイルパス.

    Raises:
        RuntimeError: latexmk が非0終了した場合.
        FileNotFoundError: PDF または .tex が見つからない場合.
    """
    # latexmk は cwd を `tex_file.parent` に切り替えるので, `-outdir` が相対のままだと
    # その相対パスが新 cwd 基準で解決され, 出力先がディレクトリ階層分二重化する.
    output_dir = output_dir.resolve()
    tex_file = find_main_tex(source_dir)
    target = CompileTarget(label=texlive_version)
    _mirror_source_dirs(source_root=tex_file.parent, target=output_dir)
    cmd = _build_latexmk_command(
        binary=str(target.bin_dir / LATEXMK_BINARY),
        tex_file_name=tex_file.name,
        build_dir=output_dir,
    )
    # latexmk が呼び出す pdflatex などの補助ツールも同一 TeX Live から拾わせる.
    env = dict(os.environ)
    env["PATH"] = f"{target.bin_dir}:{env.get('PATH', '')}"

    completed = subprocess.run(
        cmd, cwd=tex_file.parent, env=env, capture_output=True, check=False,
    )
    # 成否によらずログを残しておけば, 後で `output_dir` を見れば原因を追える.
    (output_dir / "latexmk.stdout.log").write_bytes(completed.stdout)
    (output_dir / "latexmk.stderr.log").write_bytes(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"latexmk failed for TeX Live {texlive_version}. See {output_dir}"
        )

    pdf_path = output_dir / f"{tex_file.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Compiled PDF not found: {pdf_path}")
    return pdf_path


def compile_tex_trying_texlive_versions(
    source_dir: Path,
    build_root: Path,
    *,
    texlive_versions: list[str],
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
) -> Path:
    """`texlive_versions` を順に試し, 最初に成功した PDF パスを返す.

    試行ごとに `build_root/<タイムスタンプ>/` を作って `compile_tex` に渡すので,
    `compile_tex` 側のディレクトリ衝突は起きない.

    Args:
        source_dir: TeXソースツリーのルートディレクトリ.
        build_root: 各試行のビルドディレクトリを作る親ディレクトリ.
        texlive_versions: 試行順の TeX Live 年バージョン (例: `["2025", "2023"]`).

    Raises:
        RuntimeError: 全バージョンで失敗した場合.
    """
    source_dir = Path(source_dir)
    build_root = Path(build_root)
    versions = [str(version) for version in texlive_versions]
    failure_summaries: list[str] = []
    progress = tqdm(
        versions,
        desc=f"latexmk {source_dir.name}",
        unit="try",
        disable=not show_progress,
    )
    for index, version in enumerate(progress, start=1):
        progress.set_postfix_str(f"TeX Live {version}")
        attempt_dir = build_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        attempt_dir.mkdir(parents=True, exist_ok=False)
        if progress_callback is not None:
            progress_callback(
                "build_attempt_started",
                {
                    "current": index - 1,
                    "total": len(versions),
                    "version": version,
                    "attempt_dir": str(attempt_dir),
                    "source_dir": str(source_dir),
                },
            )
        try:
            pdf_path = compile_tex(source_dir, attempt_dir, texlive_version=version)
            if progress_callback is not None:
                progress_callback(
                    "build_attempt_succeeded",
                    {
                        "current": index,
                        "total": len(versions),
                        "version": version,
                        "attempt_dir": str(attempt_dir),
                        "source_dir": str(source_dir),
                        "pdf_path": str(pdf_path),
                    },
                )
            return pdf_path
        except (RuntimeError, FileNotFoundError) as error:
            summary = summarize_latexmk_failure(attempt_dir)
            if summary:
                failure_summaries.append(f"TeX Live {version}: {summary}")
            else:
                failure_summaries.append(f"TeX Live {version}: {error}")
            logger.warning(
                "latexmk failed with TeX Live %s: %s%s",
                version,
                error,
                f"\n{summary}" if summary else "",
            )
            if progress_callback is not None:
                progress_callback(
                    "build_attempt_failed",
                    {
                        "current": index,
                        "total": len(versions),
                        "version": version,
                        "attempt_dir": str(attempt_dir),
                        "source_dir": str(source_dir),
                        "error": str(error),
                        "summary": summary,
                    },
                )
    detail = "\n\n".join(failure_summaries)
    message = f"latexmk failed for all TeX Live targets. See logs under {build_root}/"
    if detail:
        message = f"{message}\n\nKey errors:\n{detail}"
    raise RuntimeError(message)


def summarize_latexmk_failure(attempt_dir: Path, *, max_lines: int = 14) -> str:
    """Extract concise error lines from latexmk stdout/stderr logs."""
    findings: list[str] = []
    seen: set[str] = set()

    for name in _LATEXMK_LOG_NAMES:
        path = Path(attempt_dir) / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or not _is_latex_error_line(stripped):
                continue
            _append_unique(findings, seen, f"{name}: {stripped}")
            for context in _following_latex_context(lines, index):
                _append_unique(findings, seen, f"{name}: {context}")
            if len(findings) >= max_lines:
                return "\n".join(findings[:max_lines])

    if findings:
        return "\n".join(findings[:max_lines])

    tails: list[str] = []
    for name in _LATEXMK_LOG_NAMES:
        path = Path(attempt_dir) / name
        if not path.exists():
            continue
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        for line in lines[-4:]:
            _append_unique(tails, seen, f"{name}: {line}")
    return "\n".join(tails[:max_lines])


def _is_latex_error_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in _LATEX_ERROR_PATTERNS)


def _following_latex_context(lines: list[str], index: int) -> list[str]:
    context: list[str] = []
    for line in lines[index + 1:index + 4]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("l.", "<argument>", "\\", "Transcript written")):
            context.append(stripped)
            continue
        if len(stripped) < 100 and stripped.startswith(("==>", "See ", "Refer to ")):
            context.append(stripped)
    return context


def _append_unique(items: list[str], seen: set[str], value: str) -> None:
    if value in seen:
        return
    seen.add(value)
    items.append(value)


def _build_latexmk_command(
    *,
    binary: str,
    tex_file_name: str,
    build_dir: Path,
) -> list[str]:
    """`latexmk` 実行コマンドを構築する.

    `cwd` をメイン TeX の親に設定する前提なので, ファイル名のみで参照する.
    """
    command = [
        binary,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={build_dir}",
        tex_file_name,
    ]
    if ALLOW_SHELL_ESCAPE:
        command.insert(1, "-shell-escape")
    return command


def _mirror_source_dirs(*, source_root: Path, target: Path) -> None:
    """`source_root` 配下のサブディレクトリ構造だけを `target` に複製する.

    `latexmk -outdir=...` は `\\include{results/tables/foo}` のような相対 include に対し
    `outdir/results/tables/foo.aux` を開こうとするため, 対応するサブディレクトリを
    先回りで作っておかないと `.aux` 書き込みで停止する.
    """
    for source_dir in source_root.rglob("*"):
        if not source_dir.is_dir():
            continue
        (target / source_dir.relative_to(source_root)).mkdir(parents=True, exist_ok=True)
