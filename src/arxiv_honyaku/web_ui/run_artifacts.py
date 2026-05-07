"""Filesystem discovery for completed arxiv-honyaku runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class PdfArtifact:
    """One translated PDF variant found under a run directory."""

    label: str
    font_mode: str | None
    layout_mode: str | None
    source_dir: Path
    pdf_path: Path


@dataclass(frozen=True)
class RunArtifacts:
    """Completed artifacts for one run directory."""

    run_dir: Path
    title: str
    pdfs: tuple[PdfArtifact, ...]


def discover_run_artifacts(runs_dir: Path) -> list[RunArtifacts]:
    """Return translated PDF artifacts already present in ``runs_dir``."""
    root = runs_dir.resolve()
    if not root.is_dir():
        return []
    return [
        artifacts
        for artifacts in (
            collect_run_artifacts(path)
            for path in sorted(root.iterdir())
            if path.is_dir()
        )
        if artifacts.pdfs
    ]


def collect_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Collect translated PDF artifacts for one run directory."""
    root = run_dir.resolve()
    translated_root = root / "translated"
    build_root = root / "translated_builds"
    pdfs: list[PdfArtifact] = []
    if translated_root.is_dir() and build_root.is_dir():
        for variant_build_root in sorted(path for path in build_root.iterdir() if path.is_dir()):
            label = variant_build_root.name
            pdf_path = latest_pdf(variant_build_root)
            source_dir = translated_root / label
            if pdf_path is None or not source_dir.is_dir():
                continue
            font_mode, layout_mode = split_variant_label(label)
            pdfs.append(PdfArtifact(
                label=label,
                font_mode=font_mode,
                layout_mode=layout_mode,
                source_dir=source_dir.resolve(),
                pdf_path=pdf_path.resolve(),
            ))
    return RunArtifacts(
        run_dir=root,
        title=extract_source_title(root),
        pdfs=tuple(pdfs),
    )


def split_variant_label(label: str) -> tuple[str | None, str | None]:
    """Split ``font--layout`` variant labels."""
    if "--" not in label:
        return None, None
    font_mode, layout_mode = label.split("--", 1)
    return font_mode or None, layout_mode or None


def latest_pdf(variant_build_root: Path) -> Path | None:
    """Return the newest PDF in a variant build directory."""
    pdfs = sorted(path for path in variant_build_root.glob("*/*.pdf") if path.is_file())
    return pdfs[-1] if pdfs else None


def extract_source_title(run_dir: Path) -> str:
    """Best-effort title extraction from local TeX sources."""
    source_dir = run_dir / "source"
    if not source_dir.is_dir():
        return ""
    for path in candidate_title_files(source_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = extract_latex_title(text)
        if title:
            return title
    return ""


def candidate_title_files(source_dir: Path) -> list[Path]:
    """Return likely top-level TeX files before falling back to all TeX files."""
    seen: set[Path] = set()
    candidates: list[Path] = []
    readme = source_dir / "00README.json"
    if readme.is_file():
        try:
            payload = json.loads(readme.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for item in payload.get("sources", []):
            if not isinstance(item, dict) or item.get("usage") != "toplevel":
                continue
            filename = item.get("filename")
            if not isinstance(filename, str):
                continue
            path = (source_dir / filename).resolve()
            if path.suffix == ".tex" and path.is_file() and path not in seen:
                seen.add(path)
                candidates.append(path)
    for path in sorted(source_dir.rglob("*.tex")):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def extract_latex_title(text: str) -> str:
    """Extract a plain-ish title from a LaTeX ``\\title{...}`` command."""
    text = strip_latex_comments(text)
    match = re.search(r"\\title(?:\s*\[[^\]]*\])?\s*\{", text)
    if match is None:
        return ""
    title = read_balanced_argument(text, match.end() - 1)
    return cleanup_latex_text(title)


def strip_latex_comments(text: str) -> str:
    """Remove simple LaTeX comments while preserving escaped percent signs."""
    lines = []
    for line in text.splitlines():
        escaped = False
        keep = []
        for char in line:
            if char == "%" and not escaped:
                break
            keep.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(keep))
    return "\n".join(lines)


def read_balanced_argument(text: str, open_brace_index: int) -> str:
    """Read a brace-balanced argument starting at ``open_brace_index``."""
    if open_brace_index >= len(text) or text[open_brace_index] != "{":
        return ""
    depth = 0
    escaped = False
    chars: list[str] = []
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if index == open_brace_index:
            depth = 1
            continue
        if char == "\\" and not escaped:
            escaped = True
            chars.append(char)
            continue
        if char == "{" and not escaped:
            depth += 1
        elif char == "}" and not escaped:
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
        escaped = False
    return "".join(chars)


def cleanup_latex_text(text: str) -> str:
    """Convert a small subset of LaTeX markup to compact display text."""
    value = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\[[^\]]*\])?", " ", value)
    value = value.replace("\\&", "&")
    value = value.replace("\\%", "%")
    value = value.replace("\\_", "_")
    value = value.replace("~", " ")
    value = value.replace("{", "").replace("}", "")
    return " ".join(value.split())
