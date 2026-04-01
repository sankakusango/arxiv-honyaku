"""TeXソース展開とメインファイル検出を担うアダプタモジュール."""

from __future__ import annotations

from pathlib import Path
import tarfile

from arxiv_honyaku.core.ports import ArchiveExtractor, MainTexLocator


class TarArchiveExtractor(ArchiveExtractor):
    """tar系アーカイブを安全に展開する実装."""

    def extract(self, archive_path: Path, destination: Path) -> None:
        """パストラバーサル検査付きでアーカイブを展開する.

        Args:
            archive_path: 展開対象アーカイブパス.
            destination: 展開先ディレクトリ.

        Raises:
            ValueError: 展開先外へ脱出する危険パスが含まれる場合.
        """
        destination.mkdir(parents=True, exist_ok=True)
        destination_resolved = destination.resolve()
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive.getmembers():
                target_path = (destination / member.name).resolve()
                if not str(target_path).startswith(str(destination_resolved)):
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            archive.extractall(destination)


class TeXMainFileLocator(MainTexLocator):
    """ヒューリスティックでメインTeXファイルを特定する実装."""

    def find(self, source_dir: Path) -> Path:
        """documentclass等を手掛かりに最有力ファイルを返す.

        Args:
            source_dir: 展開済みソースルートディレクトリ.

        Returns:
            Path: 推定されたメインTeXファイルパス.

        Raises:
            FileNotFoundError: ``.tex`` ファイルが1つも見つからない場合.
        """
        candidates = sorted(source_dir.rglob("*.tex"))
        if not candidates:
            raise FileNotFoundError("No .tex files found in source tree")

        scored: list[tuple[int, int, int, Path]] = []
        for path in candidates:
            content = path.read_text(encoding="utf-8", errors="ignore")
            score = 0
            if "\\documentclass" in content:
                score += 2
            if "\\begin{document}" in content:
                score += 1
            if score == 0:
                continue

            rel_depth = len(path.relative_to(source_dir).parts)
            size = path.stat().st_size
            scored.append((score, -rel_depth, size, path))

        if scored:
            scored.sort(reverse=True)
            return scored[0][3]
        return max(candidates, key=lambda path: path.stat().st_size)
