"""翻訳ユースケースを統合実行する, オーケストレーションサービス."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
import re
import shutil

from arxiv_honyaku.core.arxiv_id import normalize_arxiv_ref, path_safe_arxiv_id
from arxiv_honyaku.core.chunker import digest_text, split_tex_into_chunks
from arxiv_honyaku.core.japanese_tex import (
    JapaneseFontMode,
    prepare_main_tex_for_japanese_pdf_compilation,
    prepare_translated_tex_tree_for_japanese_layout,
    translated_project_contains_japanese,
)
from arxiv_honyaku.core.latex_comments import strip_comment_only_lines
from arxiv_honyaku.core.models import (
    JapaneseLayoutMode,
    TexChunk,
    TranslateRequest,
    TranslateResult,
)
from arxiv_honyaku.core.progress import NullProgressReporter
from arxiv_honyaku.core.ports import (
    ArchiveExtractor,
    MainTexLocator,
    PdfCompiler,
    ProgressReporter,
    SourceDownloader,
)
from arxiv_honyaku.core.text_normalization import normalize_translated_text
from arxiv_honyaku.core.translation_runner import ChunkTranslationRunner

LOGGER = logging.getLogger("arxiv_honyaku.core.service")


@dataclass(slots=True, frozen=True)
class _TeXFilePlan:
    """1ファイル分の翻訳計画を保持する内部モデル."""

    tex_path: Path
    relative_path: str
    original_text: str
    chunks: list[TexChunk]


class ArxivHonyakuService:
    """arXiv翻訳パイプラインを実行するサービス.

    Attributes:
        _downloader: ソースアーカイブ取得ポート実装.
        _extractor: アーカイブ展開ポート実装.
        _tex_locator: メインTeX特定ポート実装.
        _translator: チャンク翻訳ランナー実装.
        _compiler: PDFコンパイルポート実装.
    """

    def __init__(
        self,
        *,
        downloader: SourceDownloader,
        extractor: ArchiveExtractor,
        tex_locator: MainTexLocator,
        translator: ChunkTranslationRunner,
        compiler: PdfCompiler,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        """依存ポート実装を注入し, サービスを初期化する.

        Args:
            downloader: ソースダウンロード実装.
            extractor: アーカイブ展開実装.
            tex_locator: メインTeX特定実装.
            translator: チャンク翻訳ランナー.
            compiler: PDFコンパイル実装.
            progress_reporter: 翻訳進捗通知ポート実装, ``None`` の場合は無通知.
        """
        self._downloader = downloader
        self._extractor = extractor
        self._tex_locator = tex_locator
        self._translator = translator
        self._compiler = compiler
        self._progress_reporter = progress_reporter or NullProgressReporter()

    async def translate(self, request: TranslateRequest) -> TranslateResult:
        """1本のarXiv論文を取得し, 翻訳し, PDFへコンパイルする.

        Args:
            request: 翻訳対象と実行パラメータを含むリクエスト.

        Returns:
            TranslateResult: 翻訳済みTeX, 生成PDF, 作業ディレクトリ情報.
        """
        arxiv_id = normalize_arxiv_ref(request.arxiv_ref)
        run_dir = request.workspace_root / path_safe_arxiv_id(arxiv_id)
        download_path = run_dir / "downloads" / "source.tar"
        source_dir = run_dir / "source"
        translated_dir = run_dir / "translated"
        build_dir = run_dir / "build"

        LOGGER.info(
            "Run settings: force=%s max_chunk_chars=%s translate_section_titles=%s "
            "japanese_layout_mode=%s japanese_font_mode=%s",
            request.force,
            request.max_chunk_chars,
            request.translate_section_titles,
            request.japanese_layout_mode,
            request.japanese_font_mode,
        )

        if request.force and run_dir.exists():
            shutil.rmtree(run_dir)

        if not request.force:
            completed_result = _load_completed_result(
                arxiv_id=arxiv_id,
                source_dir=source_dir,
                translated_dir=translated_dir,
                build_dir=build_dir,
                tex_locator=self._tex_locator,
                max_chunk_chars=request.max_chunk_chars,
                run_dir=run_dir,
                translate_section_titles=request.translate_section_titles,
            )
            if completed_result is not None:
                LOGGER.info(
                    "Skipping translation because compiled PDF already exists: %s",
                    completed_result.pdf_path,
                )
                return completed_result

        download_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        archive_path = self._downloader.download(arxiv_id, download_path)

        _recreate_dir(source_dir)
        self._extractor.extract(archive_path, source_dir)

        source_main_tex = self._tex_locator.find(source_dir)

        _recreate_dir(translated_dir)
        shutil.copytree(source_dir, translated_dir, dirs_exist_ok=True)

        file_plans = _build_file_plans(
            source_dir=source_dir,
            max_chunk_chars=request.max_chunk_chars,
            translate_section_titles=request.translate_section_titles,
        )
        total_chunks = sum(
            1
            for file_plan in file_plans
            for chunk in file_plan.chunks
            if chunk.translate
        )
        self._progress_reporter.on_translation_started(
            total_files=len(file_plans),
            total_chunks=total_chunks,
        )

        try:
            for file_plan in file_plans:
                self._progress_reporter.on_file_started(
                    relative_path=file_plan.relative_path,
                    total_chunks=sum(
                        1 for chunk in file_plan.chunks if chunk.translate
                    ),
                )
                translated_chunks = await self._translator.translate_all(
                    chunks=file_plan.chunks,
                    source_digest=digest_text(file_plan.original_text),
                    checkpoint_path=_checkpoint_path_for(
                        checkpoint_root=run_dir / "checkpoints",
                        source_root=source_dir,
                        tex_path=file_plan.tex_path,
                    ),
                    relative_path=file_plan.relative_path,
                )
                translated_chunks = _normalize_translated_chunks(
                    chunks=file_plan.chunks,
                    translated_chunks=translated_chunks,
                )
                translated_tex_path = translated_dir / file_plan.tex_path.relative_to(
                    source_dir
                )
                translated_tex_path.parent.mkdir(parents=True, exist_ok=True)
                translated_tex_path.write_text(
                    "".join(translated_chunks),
                    encoding="utf-8",
                )

            translated_main_tex = translated_dir / source_main_tex.relative_to(
                source_dir
            )
            _prepare_translated_main_tex_for_compilation(
                translated_dir=translated_dir,
                translated_main_tex=translated_main_tex,
                japanese_layout_mode=request.japanese_layout_mode,
                japanese_font_mode=request.japanese_font_mode,
            )
            _recreate_dir(build_dir)
            pdf_path = self._compiler.compile(translated_main_tex, build_dir)
        except Exception:
            self._progress_reporter.on_translation_finished(successful=False)
            raise

        self._progress_reporter.on_translation_finished(successful=True)
        return TranslateResult(
            arxiv_id=arxiv_id,
            run_dir=run_dir,
            translated_main_tex=translated_main_tex,
            pdf_path=pdf_path,
            total_chunks=total_chunks,
        )


def _recreate_dir(path: Path) -> None:
    """ディレクトリを再作成する.

    Args:
        path: 再作成対象ディレクトリパス.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_latex_text(path: Path) -> str:
    """TeXファイルを複数エンコーディングで読み込む.

    Args:
        path: 読み込むTeXファイルパス.

    Returns:
        str: デコード済みTeX本文文字列.
    """
    data = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _iter_translation_targets(source_dir: Path) -> list[Path]:
    """翻訳対象にするTeXファイル一覧を返す.

    Args:
        source_dir: 展開済みソースルートディレクトリ.

    Returns:
        list[Path]: ソースルート配下の ``.tex`` ファイルを相対順で並べた配列.
    """
    return sorted(source_dir.rglob("*.tex"))


def _build_file_plans(
    *,
    source_dir: Path,
    max_chunk_chars: int,
    translate_section_titles: bool,
) -> list[_TeXFilePlan]:
    """翻訳対象TeXファイル群の処理計画を構築する.

    Args:
        source_dir: 展開済みソースルートディレクトリ.
        max_chunk_chars: 1チャンクあたりの最大文字数.
        translate_section_titles: ``False`` の場合, ``\\section`` 系タイトルを翻訳対象から外す.

    Returns:
        list[_TeXFilePlan]: ファイルごとの処理計画配列.
    """
    file_plans: list[_TeXFilePlan] = []
    for tex_path in _iter_translation_targets(source_dir):
        original_text = strip_comment_only_lines(_read_latex_text(tex_path))
        file_plans.append(
            _TeXFilePlan(
                tex_path=tex_path,
                relative_path=tex_path.relative_to(source_dir).as_posix(),
                original_text=original_text,
                chunks=_apply_chunk_translation_options(
                    split_tex_into_chunks(
                        original_text,
                        max_chars=max_chunk_chars,
                    ),
                    translate_section_titles=translate_section_titles,
                ),
            )
        )
    return file_plans


def _load_completed_result(
    *,
    arxiv_id: str,
    source_dir: Path,
    translated_dir: Path,
    build_dir: Path,
    tex_locator: MainTexLocator,
    max_chunk_chars: int,
    run_dir: Path,
    translate_section_titles: bool,
) -> TranslateResult | None:
    """既存の成功済み成果物から, 返却用結果オブジェクトを復元する.

    Args:
        arxiv_id: 正規化済みarXiv ID.
        source_dir: 前回実行時のソースツリーディレクトリ.
        translated_dir: 前回実行時の翻訳済みツリーディレクトリ.
        build_dir: 前回実行時のビルド成果物ディレクトリ.
        tex_locator: メインTeX特定ポート実装.
        max_chunk_chars: チャンク分割時の最大文字数.
        run_dir: 実行単位の作業ディレクトリ.
        translate_section_titles: ``False`` の場合, ``\\section`` 系タイトルを翻訳対象から外す.

    Returns:
        TranslateResult | None: 成功済み成果物を復元できた場合は結果, できない場合は ``None``.
    """
    if not translated_dir.exists() or not build_dir.exists():
        return None

    try:
        translated_main_tex = tex_locator.find(translated_dir)
    except FileNotFoundError:
        return None

    pdf_path = build_dir / f"{translated_main_tex.stem}.pdf"
    if not pdf_path.exists():
        pdf_candidates = sorted(build_dir.glob("*.pdf"))
        if len(pdf_candidates) != 1:
            return None
        pdf_path = pdf_candidates[0]

    total_chunks = _count_total_chunks(
        source_dir=source_dir,
        max_chunk_chars=max_chunk_chars,
        translate_section_titles=translate_section_titles,
    )
    return TranslateResult(
        arxiv_id=arxiv_id,
        run_dir=run_dir,
        translated_main_tex=translated_main_tex,
        pdf_path=pdf_path,
        total_chunks=total_chunks,
    )


def _count_total_chunks(
    *,
    source_dir: Path,
    max_chunk_chars: int,
    translate_section_titles: bool,
) -> int:
    """ソースツリーから翻訳対象チャンク総数を数える.

    Args:
        source_dir: 元ソースツリーディレクトリ.
        max_chunk_chars: チャンク分割時の最大文字数.
        translate_section_titles: ``False`` の場合, ``\\section`` 系タイトルを翻訳対象から外す.

    Returns:
        int: 翻訳対象チャンクの総数. ソースが無い場合は ``0``.
    """
    if not source_dir.exists():
        return 0

    return sum(
        1
        for file_plan in _build_file_plans(
            source_dir=source_dir,
            max_chunk_chars=max_chunk_chars,
            translate_section_titles=translate_section_titles,
        )
        for chunk in file_plan.chunks
        if chunk.translate
    )


def _apply_chunk_translation_options(
    chunks: list[TexChunk],
    *,
    translate_section_titles: bool,
) -> list[TexChunk]:
    """チャンク列へ翻訳対象制御オプションを反映する.

    Args:
        chunks: chunker が生成した元チャンク配列.
        translate_section_titles: ``False`` の場合, ``section_title`` チャンクを原文維持へ切り替える.

    Returns:
        list[TexChunk]: 翻訳対象設定を反映した新しいチャンク配列.
    """
    if translate_section_titles:
        return chunks

    return [
        replace(chunk, translate=False)
        if chunk.translate and chunk.kind == "section_title"
        else chunk
        for chunk in chunks
    ]


def _checkpoint_path_for(
    *,
    checkpoint_root: Path,
    source_root: Path,
    tex_path: Path,
) -> Path:
    """TeXファイルごとのチェックポイントパスを返す.

    Args:
        checkpoint_root: チェックポイント保存ルート.
        source_root: 翻訳元ソースルート.
        tex_path: 対応するTeXファイルパス.

    Returns:
        Path: TeXファイルに一意に対応するJSONチェックポイントパス.
    """
    relative_path = tex_path.relative_to(source_root)
    return checkpoint_root / relative_path.with_suffix(".json")


def _prepare_translated_main_tex_for_compilation(
    *,
    translated_dir: Path,
    translated_main_tex: Path,
    japanese_layout_mode: JapaneseLayoutMode,
    japanese_font_mode: JapaneseFontMode,
) -> None:
    """翻訳後主文書へ, 必要に応じて日本語コンパイル対応を注入する.

    Args:
        translated_dir: 翻訳済みTeXツリールート.
        translated_main_tex: 翻訳済みメインTeXファイルパス.
        japanese_layout_mode: 日本語レイアウト補正方針.
        japanese_font_mode: 日本語フォント注入方針.

    Returns:
        None: 常に ``None``.
    """
    LOGGER.info(
        "Preparing translated TeX for Japanese compilation: main=%s layout_mode=%s font_mode=%s",
        translated_main_tex,
        japanese_layout_mode,
        japanese_font_mode,
    )
    if not translated_project_contains_japanese(translated_dir):
        LOGGER.info(
            "Skipping Japanese TeX preparation because no Japanese characters were found under: %s",
            translated_dir,
        )
        return

    before_text = translated_main_tex.read_text(encoding="utf-8")
    before_cjk_family = _extract_cjk_family(before_text)
    before_has_ipaex = "\\usepackage{ipaex-type1}" in before_text

    prepare_translated_tex_tree_for_japanese_layout(
        translated_dir,
        mode=japanese_layout_mode,
    )
    after_layout_text = translated_main_tex.read_text(encoding="utf-8")
    if after_layout_text != before_text:
        LOGGER.info(
            "Japanese layout normalization changed main TeX before font injection."
        )
    prepared_text = prepare_main_tex_for_japanese_pdf_compilation(
        after_layout_text,
        font_mode=japanese_font_mode,
    )
    after_cjk_family = _extract_cjk_family(prepared_text)
    after_has_ipaex = "\\usepackage{ipaex-type1}" in prepared_text
    LOGGER.info(
        "Japanese TeX preparation summary: cjk_family=%s->%s ipaex-type1=%s->%s",
        before_cjk_family,
        after_cjk_family,
        before_has_ipaex,
        after_has_ipaex,
    )
    if japanese_font_mode == "paper-like" and after_cjk_family != "ipxm":
        LOGGER.warning(
            "paper-like mode requested but CJK family is not ipxm after preparation: %s",
            after_cjk_family,
        )
    translated_main_tex.write_text(prepared_text, encoding="utf-8")


_CJK_FAMILY_RE = re.compile(r"\\begin\{CJK\}\{UTF8\}\{([^}]+)\}")


def _extract_cjk_family(tex_text: str) -> str:
    """TeX本文中の CJK family 指定（例: min/ipxm）を抽出する."""
    match = _CJK_FAMILY_RE.search(tex_text)
    if match is None:
        return "(none)"
    return match.group(1)


def _normalize_translated_chunks(
    *,
    chunks: list[TexChunk],
    translated_chunks: list[str],
) -> list[str]:
    """翻訳済みチャンク列へ機械的な日本語正規化を適用する.

    Args:
        chunks: 元のチャンク配列.
        translated_chunks: チャンク順の翻訳済み文字列配列.

    Returns:
        list[str]: 翻訳対象チャンクだけを正規化した文字列配列.
    """
    return [
        text if not chunk.translate else normalize_translated_text(text)
        for chunk, text in zip(chunks, translated_chunks, strict=True)
    ]
