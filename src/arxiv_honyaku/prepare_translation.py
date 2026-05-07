"""⑦ パイプライン全体を 1 つの `.tex` ファイルに対して走らせ, JSON へ保存する.

```
prepare_tex_file(tex_path, ...)
  ├─ ① clean_tex            (非可逆)
  ├─ ② pad_unitless_si      (非可逆)
  ├─ ③④ substitute         (可逆)
  ├─ ⑤⑥ chunk_text         (translatable 判定込み)
  └─ ⑦ JSON 出力
```

復元 (`restore_from_json`) は ②③ を逆に辿る. ① は非可逆なので戻せない.
日本語化 (CJK 注入と layout 補正) は翻訳結果に依存しない mode 切替で再生成
できるよう, 再構成 (`reconstruct_*_from_source_tree`) のタイミングで適用する.
"""
from dataclasses import asdict
from collections.abc import Mapping
from pathlib import Path
import json
import logging
import shutil

from .build_latex import find_main_tex
from .chunk import chunk_text
from .japanese_setup import (
    JapaneseFontMode,
    JapaneseLayoutMode,
    apply_layout,
    inject_cjk,
)
from .tex_cleanup import clean_tex, pad_unitless_si
from .source_tree import load_source_tree
from .substitute import Substitution, restore, substitute
from .translation_logic import validate_translation_text

logger = logging.getLogger(__name__)

JSON_VERSION = 1
_CHUNK_VISIBLE_SUBSTITUTION_KINDS = {"placeholder_prefix"}


def prep_json_path(tex_path: Path, *, source_dir: Path, prep_dir: Path) -> Path:
    """`.tex` の相対階層を保った JSON 出力先を返す."""
    rel = tex_path.relative_to(source_dir)
    return prep_dir / rel.with_suffix(".tex.json")


def prepare_tex_file(
    tex_path: Path,
    *,
    source_dir: Path,
    output_path: Path,
    is_main: bool,
) -> None:
    """1つの .tex に①〜⑥を適用し, JSON を `output_path` に書き出す."""
    raw = tex_path.read_text(encoding="utf-8")

    # ① 非可逆: コメント削除 + 意味を持ちにくい空白の整理.
    text = clean_tex(raw)
    # ② 非可逆: 単位引数なしの \SI{...} を siunitx の正規形へ寄せる.
    text = pad_unitless_si(text)
    irreversible_ops = ["clean_tex", "pad_unitless_si"]

    # ③④ 可逆: placeholder 置換.
    substituted_text, pool = substitute(text)

    # ⑤⑥ チャンク化 (translatable 判定はチャンク内で).
    chunks = chunk_text(
        substituted_text,
        placeholders=[
            s.placeholder for s in pool
            if s.kind not in _CHUNK_VISIBLE_SUBSTITUTION_KINDS
        ],
    )

    # ⑦ JSON 保存.
    payload = {
        "version": JSON_VERSION,
        "source_path": str(tex_path.relative_to(source_dir)),
        "is_main": is_main,
        "irreversible_ops": irreversible_ops,
        "substitutions": [asdict(s) for s in pool],
        "chunks": [
            {
                "index": c.index,
                "section_path": c.section_path,
                "translatable": c.translatable,
                "skip_reason": c.skip_reason,
                "text": c.text,
                "join_after": c.join_after,
            }
            for c in chunks
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def prepare_from_source_tree(
    source_tree_path: Path,
    prep_dir: Path,
) -> list[Path]:
    """`source_tree.json` と出力先 prep dir を指定して prep JSON を生成する."""
    tree = load_source_tree(source_tree_path)
    source_dir = source_tree_path.parent / tree.source_root
    main_tex = find_main_tex(source_dir)
    json_paths: list[Path] = []

    for entry in tree.entries:
        if not entry.is_tex:
            continue
        tex_path = source_dir / entry.path
        json_path = prep_json_path(tex_path, source_dir=source_dir, prep_dir=prep_dir)
        prepare_tex_file(
            tex_path,
            source_dir=source_dir,
            output_path=json_path,
            is_main=tex_path.resolve() == main_tex.resolve(),
        )
        json_paths.append(json_path)

    return json_paths


def reconstruct_from_source_tree(
    source_tree_path: Path,
    prep_dir: Path,
    output_dir: Path,
    *,
    font_mode: JapaneseFontMode,
    layout_mode: JapaneseLayoutMode,
) -> list[Path]:
    """`source_tree.json` と prep JSON から source tree を再構成する.

    TeX は prep JSON から復元し, 非 TeX ファイルは元 source tree からコピーする.
    復元した TeX には日本語化 (CJK 注入と layout 補正) を適用する.
    """
    return _reconstruct_source_tree(
        source_tree_path,
        prep_dir,
        output_dir,
        translations={},
        font_mode=font_mode,
        layout_mode=layout_mode,
    )


def reconstruct_translated_from_source_tree(
    source_tree_path: Path,
    prep_dir: Path,
    output_dir: Path,
    *,
    translations_jsonl: Path,
    font_mode: JapaneseFontMode,
    layout_mode: JapaneseLayoutMode,
) -> list[Path]:
    """source tree と翻訳 JSONL から translated source tree を再構成する.

    JSONL に成功翻訳がある chunk は翻訳文を使い, 失敗または欠落している chunk は
    原文 chunk のまま復元する. 非 TeX ファイルは元 source tree からコピーする.
    復元した TeX には日本語化 (CJK 注入と layout 補正) を適用する.
    """
    return _reconstruct_source_tree(
        source_tree_path,
        prep_dir,
        output_dir,
        translations=load_translation_jsonl(translations_jsonl),
        font_mode=font_mode,
        layout_mode=layout_mode,
    )


def _reconstruct_source_tree(
    source_tree_path: Path,
    prep_dir: Path,
    output_dir: Path,
    *,
    translations: Mapping[str, Mapping[int, str]],
    font_mode: JapaneseFontMode,
    layout_mode: JapaneseLayoutMode,
) -> list[Path]:
    tree = load_source_tree(source_tree_path)
    source_dir = source_tree_path.parent / tree.source_root
    output_paths: list[Path] = []

    for entry in tree.entries:
        output_path = output_dir / entry.path
        if entry.is_dir:
            output_path.mkdir(parents=True, exist_ok=True)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if entry.is_tex:
            payload = _load_prep_payload(prep_json_path(
                source_dir / entry.path,
                source_dir=source_dir,
                prep_dir=prep_dir,
            ))
            restored = _restore_from_payload(
                payload,
                translations.get(entry.path.as_posix(), {}),
            )
            output_path.write_text(
                apply_japanese_setup(
                    restored,
                    is_main=payload["is_main"],
                    font_mode=font_mode,
                    layout_mode=layout_mode,
                ),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source_dir / entry.path, output_path)
        output_paths.append(output_path)

    return output_paths


def apply_japanese_setup(
    tex_text: str,
    *,
    is_main: bool,
    font_mode: JapaneseFontMode,
    layout_mode: JapaneseLayoutMode,
) -> str:
    """復元 TeX に CJK 注入 (main のみ) と layout 補正を適用する."""
    text = tex_text
    if is_main:
        text = inject_cjk(text, font_mode=font_mode)
    return apply_layout(text, mode=layout_mode)


def load_translation_jsonl(jsonl_path: Path) -> dict[str, dict[int, str]]:
    """翻訳 JSONL から成功翻訳だけを `source_path -> chunk_index` で引ける形にする."""
    translations: dict[str, dict[int, str]] = {}
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["status"] != "ok":
            continue
        translated_text = record["translated_text"]
        if not isinstance(translated_text, str):
            raise ValueError(f"translated_text is missing at line {line_number}")

        source_path = record["source_path"]
        chunk_index = record["chunk_index"]
        file_translations = translations.setdefault(source_path, {})
        if chunk_index in file_translations:
            raise ValueError(
                f"duplicate translation: {source_path} chunk {chunk_index}"
            )
        file_translations[chunk_index] = translated_text
    return translations


def restore_from_json(
    json_path: Path,
    *,
    translations: Mapping[int, str] | None = None,
) -> str:
    """`prepare_tex_file` が書き出した JSON から ① 後の本文を復元する.

    手順: 全 chunk の text と join_after を順に連結 → substitutions を逆置換.
    """
    return _restore_from_payload(_load_prep_payload(json_path), translations)


def _load_prep_payload(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload["version"] != JSON_VERSION:
        raise ValueError(f"Unsupported JSON version: {payload['version']}")
    return payload


def _restore_from_payload(
    payload: dict,
    translations: Mapping[int, str] | None,
) -> str:
    substituted_text = "".join(
        _translated_chunk_text(c, translations) + c["join_after"]
        for c in payload["chunks"]
    )
    pool = [
        Substitution(placeholder=s["placeholder"], kind=s["kind"], original=s["original"])
        for s in payload["substitutions"]
    ]
    return restore(substituted_text, pool)


def _translated_chunk_text(
    chunk: dict,
    translations: Mapping[int, str] | None,
) -> str:
    """翻訳済み chunk があれば使い, 欠けていれば原文を返す."""
    source_text = chunk["text"]
    if not translations or not chunk["translatable"]:
        return source_text
    translated_text = translations.get(chunk["index"])
    if translated_text is None:
        return source_text
    validate_translation_text(source_text, translated_text)
    return translated_text


def apply_irreversible(raw_text: str) -> str:
    """prep の非可逆部分だけを適用したテキストを返す (再構成検証用の参照値生成)."""
    return pad_unitless_si(clean_tex(raw_text))
