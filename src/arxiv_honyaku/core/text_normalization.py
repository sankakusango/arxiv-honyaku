"""翻訳済み日本語テキストの機械的な正規化を行うモジュール."""

from __future__ import annotations

_PUNCTUATION_TRANSLATION_TABLE = str.maketrans(
    {
        "、": ",",
        "。": ".",
        "，": ",",
        "．": ".",
        "､": ",",
        "｡": ".",
    }
)


def normalize_translated_text(text: str) -> str:
    """翻訳済みテキストの句読点とスペースを正規化する.

    Args:
        text: 正規化対象の翻訳済みテキスト.

    Returns:
        str: 句読点をカンマ, ピリオドへ統一し, 全角スペースを半角スペースへ置換した文字列.
    """
    normalized = text.replace("\u3000", " ")
    return normalized.translate(_PUNCTUATION_TRANSLATION_TABLE)
