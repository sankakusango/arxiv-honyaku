"""prep JSON の translatable chunk を翻訳して JSONL に保存する."""
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import asyncio
import json

from tqdm import tqdm

from .build_latex import compile_tex_trying_texlive_versions
from .client import LLMClient
from .config import Config, load_config
from .japanese_setup import JapaneseFontMode, JapaneseLayoutMode
from .prepare_translation import (
    prepare_from_source_tree,
    reconstruct_translated_from_source_tree,
)
from .translation_logic import (
    TranslationChunk,
    TranslationClient,
    TranslationLogic,
    TranslationResult,
    build_translation_logic,
)

ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class TranslationRunStats:
    """翻訳実行の件数サマリ."""

    total: int
    ok: int
    failed: int


@dataclass(frozen=True)
class TranslationVariantResult:
    """1 つの (font_mode, layout_mode) 組み合わせでの再構成・ビルド結果."""

    font_mode: JapaneseFontMode
    layout_mode: JapaneseLayoutMode
    translated_paths: list[Path]
    pdf_path: Path


@dataclass(frozen=True)
class TranslationBuildResult:
    """prep, 翻訳, translated 再構成, build をまとめた結果."""

    prep_paths: list[Path]
    stats: TranslationRunStats
    variants: list[TranslationVariantResult]


def iter_translation_chunks(prep_dir: Path) -> Iterator[TranslationChunk]:
    """prep dir 内の JSON から翻訳対象 chunk を集める."""
    for json_path in sorted(prep_dir.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for chunk in payload["chunks"]:
            if not chunk["translatable"]:
                continue
            yield TranslationChunk(
                source_path=payload["source_path"],
                chunk_index=chunk["index"],
                section_path=chunk["section_path"],
                text=chunk["text"],
            )


def translate_prep_dir(
    prep_dir: Path,
    output_jsonl: Path,
    *,
    config: Config | None = None,
    logic: TranslationLogic | None = None,
    client: TranslationClient | None = None,
    max_concurrency: int | None = None,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    reuse_existing: bool = True,
) -> TranslationRunStats:
    """同期入口. `prep_dir` の翻訳対象 chunk を JSONL に書き出す."""
    return asyncio.run(translate_prep_dir_async(
        prep_dir,
        output_jsonl,
        config=config,
        logic=logic,
        client=client,
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
        show_progress=show_progress,
        reuse_existing=reuse_existing,
    ))


async def translate_prep_dir_async(
    prep_dir: Path,
    output_jsonl: Path,
    *,
    config: Config | None = None,
    logic: TranslationLogic | None = None,
    client: TranslationClient | None = None,
    max_concurrency: int | None = None,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    reuse_existing: bool = True,
) -> TranslationRunStats:
    """最大 `max_concurrency` 件まで同時に chunk 翻訳を走らせる."""
    resolved = config or load_config()
    concurrency = (
        resolved.llm.max_concurrency
        if max_concurrency is None
        else max_concurrency
    )
    if concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    llm_client = LLMClient.from_config(resolved) if client is None else client
    translation_logic = (
        build_translation_logic(resolved.llm.translation_logic)
        if logic is None
        else logic
    )
    semaphore = asyncio.Semaphore(concurrency)
    chunks = list(iter_translation_chunks(prep_dir))
    existing_ok = (
        _load_ok_translation_keys(output_jsonl)
        if reuse_existing
        else set()
    )
    needed_keys = {(chunk.source_path, chunk.chunk_index) for chunk in chunks}
    existing_ok &= needed_keys
    pending_chunks = [
        chunk for chunk in chunks
        if (chunk.source_path, chunk.chunk_index) not in existing_ok
    ]
    total = len(chunks)
    initial_ok = len(existing_ok)
    if progress_callback is not None:
        progress_callback(
            "translate_started",
            {"current": initial_ok, "total": total, "ok": initial_ok, "failed": 0},
        )
    tasks = [
        asyncio.create_task(_translate_one(
            chunk,
            client=llm_client,
            logic=translation_logic,
            semaphore=semaphore,
        ))
        for chunk in pending_chunks
    ]

    ok = initial_ok
    failed = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        total=total,
        desc="translating",
        unit="chunk",
        disable=not show_progress,
    )
    progress.update(initial_ok)
    mode = "a" if reuse_existing and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                handle.write(json.dumps(result.as_json(), ensure_ascii=False) + "\n")
                handle.flush()
                if result.status == "ok":
                    ok += 1
                else:
                    failed += 1
                progress.update(1)
                progress.set_postfix(ok=ok, failed=failed)
                if progress_callback is not None:
                    progress_callback(
                        "translate_chunk_finished",
                        {
                            "current": ok + failed,
                            "total": total,
                            "ok": ok,
                            "failed": failed,
                            "source_path": result.chunk.source_path,
                            "chunk_index": result.chunk.chunk_index,
                            "status": result.status,
                            "error": result.error,
                            "attempts": [
                                attempt.__dict__ for attempt in result.attempts
                            ],
                        },
                    )
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            progress.close()

    if progress_callback is not None:
        progress_callback(
            "translate_finished",
            {"current": ok + failed, "total": total, "ok": ok, "failed": failed},
        )
    return TranslationRunStats(total=ok + failed, ok=ok, failed=failed)


def translate_source_tree_to_pdf(
    source_tree_path: Path,
    prep_dir: Path,
    translations_jsonl: Path,
    translated_dir: Path,
    build_root: Path,
    *,
    config: Config | None = None,
    logic: TranslationLogic | None = None,
    client: TranslationClient | None = None,
    max_concurrency: int | None = None,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
) -> TranslationBuildResult:
    """source tree から prep, 翻訳, translated 再構成, PDF build まで実行する.

    prep と翻訳は mode 非依存なので 1 度だけ実行し, 全 (font_mode, layout_mode)
    の組み合わせについて再構成と PDF ビルドを行う. 各組み合わせの出力は
    `translated_dir/<font>--<layout>/` と `build_root/<font>--<layout>/` に置く.
    `force=False` で `translations_jsonl` が prep の全 translatable chunk を満たす
    なら翻訳ステップをスキップする.
    """
    resolved = config or load_config()
    prep_paths = prepare_from_source_tree(source_tree_path, prep_dir)
    if not force and _is_translation_complete(prep_dir, translations_jsonl):
        print(f"Skipping translation: {translations_jsonl} is already complete")
        stats = _stats_from_jsonl(translations_jsonl)
    else:
        stats = translate_prep_dir(
            prep_dir,
            translations_jsonl,
            config=resolved,
            logic=logic,
            client=client,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
            show_progress=show_progress,
            reuse_existing=not force,
        )
    variants: list[TranslationVariantResult] = []
    for font_mode in resolved.run.japanese_font_modes:
        for layout_mode in resolved.run.japanese_layout_modes:
            label = f"{font_mode}--{layout_mode}"
            variant_translated_dir = translated_dir / label
            variant_build_root = build_root / label
            translated_paths = reconstruct_translated_from_source_tree(
                source_tree_path,
                prep_dir,
                variant_translated_dir,
                translations_jsonl=translations_jsonl,
                font_mode=font_mode,
                layout_mode=layout_mode,
            )
            existing_pdf = None if force else _find_latest_pdf(variant_build_root)
            if existing_pdf is not None:
                print(f"Skipping build [{label}]: reusing {existing_pdf}")
                pdf_path = existing_pdf
            else:
                pdf_path = compile_tex_trying_texlive_versions(
                    source_dir=variant_translated_dir,
                    build_root=variant_build_root,
                    texlive_versions=resolved.run.texlive_versions,
                    progress_callback=progress_callback,
                    show_progress=show_progress,
                )
            variants.append(TranslationVariantResult(
                font_mode=font_mode,
                layout_mode=layout_mode,
                translated_paths=translated_paths,
                pdf_path=pdf_path,
            ))
    return TranslationBuildResult(
        prep_paths=prep_paths,
        stats=stats,
        variants=variants,
    )


async def _translate_one(
    chunk: TranslationChunk,
    *,
    client: TranslationClient,
    logic: TranslationLogic,
    semaphore: asyncio.Semaphore,
) -> TranslationResult:
    """1 chunk を翻訳する."""
    async with semaphore:
        return await logic.translate(chunk, client=client)


def _is_translation_complete(prep_dir: Path, jsonl_path: Path) -> bool:
    """prep の全 translatable chunk が JSONL に ok 翻訳として存在するかを返す."""
    if not jsonl_path.exists():
        return False
    needed = {
        (chunk.source_path, chunk.chunk_index)
        for chunk in iter_translation_chunks(prep_dir)
    }
    have: set[tuple[str, int]] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["status"] == "ok":
            have.add((record["source_path"], record["chunk_index"]))
    return needed.issubset(have)


def _load_ok_translation_keys(jsonl_path: Path) -> set[tuple[str, int]]:
    """Return source/chunk keys already translated successfully in JSONL."""
    if not jsonl_path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            keys.add((record["source_path"], record["chunk_index"]))
    return keys


def is_translation_complete(prep_dir: Path, jsonl_path: Path) -> bool:
    """prep の全 translatable chunk が JSONL に ok 翻訳として存在するかを返す."""
    return _is_translation_complete(prep_dir, jsonl_path)


def _find_latest_pdf(variant_build_root: Path) -> Path | None:
    """既存ビルド成果の中で最新タイムスタンプの PDF を返す. 無ければ None."""
    if not variant_build_root.exists():
        return None
    pdfs = sorted(variant_build_root.glob("*/*.pdf"))
    return pdfs[-1] if pdfs else None


def find_latest_pdf(variant_build_root: Path) -> Path | None:
    """既存ビルド成果の中で最新タイムスタンプの PDF を返す. 無ければ None."""
    return _find_latest_pdf(variant_build_root)


def _stats_from_jsonl(jsonl_path: Path) -> TranslationRunStats:
    ok = failed = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["status"] == "ok":
            ok += 1
        else:
            failed += 1
    return TranslationRunStats(total=ok + failed, ok=ok, failed=failed)


def stats_from_jsonl(jsonl_path: Path) -> TranslationRunStats:
    """translations JSONL から件数サマリを作る."""
    return _stats_from_jsonl(jsonl_path)
