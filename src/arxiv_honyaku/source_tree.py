"""ソースツリーの構造を走査・保存する."""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class SourceEntry:
    """ソースツリー内の1エントリ."""

    path: Path  # `source_dir` からの相対パス
    is_dir: bool
    is_tex: bool


@dataclass
class SourceTree:
    """保存済み source_tree.json の内容."""

    source_root: Path
    entries: list[SourceEntry]


def list_source_entries(source_dir: Path) -> list[SourceEntry]:
    """`source_dir` 配下の全ファイル・ディレクトリを `SourceEntry` のリストで返す."""
    entries: list[SourceEntry] = []
    for path in sorted(source_dir.rglob("*")):
        is_dir = path.is_dir()
        entries.append(SourceEntry(
            path=path.relative_to(source_dir),
            is_dir=is_dir,
            is_tex=not is_dir and path.suffix == ".tex",
        ))
    return entries


def save_source_tree(source_dir: Path, output_path: Path) -> None:
    """`source_dir` を走査して結果を JSON に書き出す."""
    payload = {
        "source_root": _default_source_root(source_dir, output_path).as_posix(),
        "entries": [
            {
                "path": entry.path.as_posix(),
                "is_dir": entry.is_dir,
                "is_tex": entry.is_tex,
            }
            for entry in list_source_entries(source_dir)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_source_tree(source_tree_path: Path) -> SourceTree:
    """`save_source_tree` が書いた JSON を読み込む."""
    payload = json.loads(source_tree_path.read_text(encoding="utf-8"))
    return SourceTree(
        source_root=Path(payload["source_root"]),
        entries=[
            SourceEntry(
                path=Path(entry["path"]),
                is_dir=entry["is_dir"],
                is_tex=entry["is_tex"],
            )
            for entry in payload["entries"]
        ],
    )


def resolve_source_root(source_tree_path: Path) -> Path:
    """`source_tree.json` から実際の source root ディレクトリを返す."""
    tree = load_source_tree(source_tree_path)
    return source_tree_path.parent / tree.source_root


def _default_source_root(source_dir: Path, output_path: Path) -> Path:
    """`source_tree.json` から source root を辿るための相対パスを返す."""
    try:
        return source_dir.relative_to(output_path.parent)
    except ValueError:
        pass
    try:
        return source_dir.resolve().relative_to(output_path.parent.resolve())
    except ValueError:
        return Path(source_dir.name)
