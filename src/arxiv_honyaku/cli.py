"""arxiv_id を渡すと download → 翻訳 → PDF build まで一気に行う CLI."""
from pathlib import Path
import argparse

from .arxiv_source import download_and_unpack
from .config import load_config
from .source_tree import save_source_tree
from .translate import translate_source_tree_to_pdf


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="arxiv-honyaku",
        description="arxiv_id から翻訳済み PDF を生成する.",
    )
    parser.add_argument("arxiv_id", help="例: 1706.03762")
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の translations.jsonl を無視して翻訳をやり直す.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    run_dir = Path(config.run.runs_dir) / args.arxiv_id
    source_dir = run_dir / "source"
    source_tree_path = run_dir / "source_tree.json"
    prep_dir = run_dir / "prep"
    translations_jsonl = run_dir / "translations.jsonl"
    translated_dir = run_dir / "translated"
    build_root = run_dir / "translated_builds"

    if not source_dir.exists():
        print(f"Downloading arxiv source for {args.arxiv_id}")
        download_and_unpack(
            args.arxiv_id,
            download_dir=run_dir / "downloads",
            unpack_dir=source_dir,
        )

    save_source_tree(source_dir, source_tree_path)
    result = translate_source_tree_to_pdf(
        source_tree_path,
        prep_dir,
        translations_jsonl,
        translated_dir,
        build_root,
        config=config,
        force=args.force,
    )

    print(f"Saved prep JSON under {prep_dir}")
    print(f"Saved translations JSONL at {translations_jsonl}")
    print(
        "Translation chunks: "
        f"total={result.stats.total}, ok={result.stats.ok}, failed={result.stats.failed}"
    )
    for variant in result.variants:
        label = f"{variant.font_mode}--{variant.layout_mode}"
        print(f"[{label}] compiled PDF: {variant.pdf_path}")


if __name__ == "__main__":
    main()
