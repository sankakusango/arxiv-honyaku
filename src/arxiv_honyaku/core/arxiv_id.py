"""arXiv識別子を正規化し, パス安全な形式へ変換するユーティリティ."""

from __future__ import annotations

from urllib.parse import urlparse
import re

_ID_PATTERN = re.compile(
    r"^(?:"
    r"\d{4}\.\d{4,5}(?:v\d+)?"
    r"|"
    r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
    r")$",
    re.IGNORECASE,
)


def normalize_arxiv_ref(value: str) -> str:
    """arXiv参照文字列を正規化済みIDへ変換する.

    Args:
        value: arXiv ID, または arXiv URL.

    Returns:
        str: 検証済みかつ正規化済みのarXiv ID.

    Raises:
        ValueError: 入力が空, URLが不正, ID形式が不正な場合.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("arXiv reference is empty")

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if not parts:
            raise ValueError(f"Invalid arXiv URL: {value}")
        token = parts[-1]
        if token == "pdf" and len(parts) >= 2:
            token = parts[-2]
        if token.endswith(".pdf"):
            token = token[:-4]
        raw = token

    if not _ID_PATTERN.fullmatch(raw):
        raise ValueError(f"Invalid arXiv identifier: {value}")
    return raw


def path_safe_arxiv_id(arxiv_id: str) -> str:
    """arXiv IDをファイルパスで安全な文字列へ変換する.

    Args:
        arxiv_id: 正規化済みarXiv ID.

    Returns:
        str: ``/`` を ``_`` に置換したパス安全文字列.
    """
    return arxiv_id.replace("/", "_")
