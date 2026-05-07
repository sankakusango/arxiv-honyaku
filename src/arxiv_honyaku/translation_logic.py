"""1 chunk をどう翻訳し, 失敗時にどう再試行するかを扱う."""
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
import re

from openai.types.chat import ChatCompletionMessageParam

_PLACEHOLDER_RE = re.compile(r"XQK\d{4,}")
_CONTROL_SEQUENCE_RE = re.compile(r"\\(?:[A-Za-z]+[*]?|.)")
_OPEN_TAG = "<translated>"
_CLOSE_TAG = "</translated>"
_NO_SPACE_AFTER_PLACEHOLDER = frozenset(".,;:!?)]}")
_LATEX_SPECIAL_CHAR_ESCAPES = {
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\textasciicircum{}",
}
_IGNORED_CONTROL_SEQUENCES = frozenset({
    r"\#",
    r"\$",
    r"\%",
    r"\&",
    r"\_",
    r"\textasciicircum",
})

ALLOW_PLACEHOLDER_REORDER = True


@dataclass(frozen=True)
class TranslationChunk:
    """prep JSON 内の翻訳対象 chunk."""

    source_path: str
    chunk_index: int
    section_path: str
    text: str


@dataclass(frozen=True)
class TranslationAttempt:
    """1 chunk に対する 1 回の翻訳試行."""

    attempt: int
    ok: bool
    error: str | None = None


@dataclass
class TranslationResult:
    """JSONL に書き出す 1 chunk 分の翻訳結果."""

    chunk: TranslationChunk
    translated_text: str | None
    status: str
    logic: str
    attempts: list[TranslationAttempt] = field(default_factory=list)
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        """JSONL に書ける dict にする."""
        return {
            "source_path": self.chunk.source_path,
            "chunk_index": self.chunk.chunk_index,
            "section_path": self.chunk.section_path,
            "source_text": self.chunk.text,
            "translated_text": self.translated_text,
            "status": self.status,
            "logic": self.logic,
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "error": self.error,
        }


class TranslationClient(Protocol):
    """翻訳ロジックから見た LLM client の最小 interface."""

    async def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
    ) -> str:
        """messages を渡して応答本文を返す."""


class TranslationLogic(Protocol):
    """LLM ごとの翻訳処理差し替え口."""

    name: str

    async def translate(
        self,
        chunk: TranslationChunk,
        *,
        client: TranslationClient,
    ) -> TranslationResult:
        """1 chunk を翻訳する."""


class ChatTranslationLogic:
    """普通の chat model 向け翻訳ロジック.

    1 回目の翻訳結果が空, placeholder 欠落などで不正なら, その応答とエラー内容を
    次の user message に入れてもう 1 回だけ翻訳する.
    """

    name = "general_chat"

    async def translate(
        self,
        chunk: TranslationChunk,
        *,
        client: TranslationClient,
    ) -> TranslationResult:
        attempts: list[TranslationAttempt] = []
        messages = _initial_messages(chunk)

        for attempt_number in (1, 2):
            response = None
            try:
                response = await client.complete(messages)
            except Exception as error:
                message = str(error)
            else:
                try:
                    translated = extract_translated_text(response)
                    translated = normalize_translation_text(chunk.text, translated)
                    validate_translation_text(chunk.text, translated)
                    _validate_control_sequences(chunk.text, translated)
                    _validate_prompt_leakage(translated)
                except ValueError as error:
                    message = str(error)
                else:
                    attempts.append(TranslationAttempt(
                        attempt=attempt_number,
                        ok=True,
                    ))
                    return TranslationResult(
                        chunk=chunk,
                        translated_text=translated,
                        status="ok",
                        logic=self.name,
                        attempts=attempts,
                    )

            attempts.append(TranslationAttempt(
                attempt=attempt_number,
                ok=False,
                error=message,
            ))
            if attempt_number == 2:
                return TranslationResult(
                    chunk=chunk,
                    translated_text=None,
                    status="failed",
                    logic=self.name,
                    attempts=attempts,
                    error=message,
                )
            messages = _retry_messages(
                chunk,
                previous_response=response,
                error=message,
            )

        raise AssertionError("unreachable")


_LOGIC_FACTORIES = {
    ChatTranslationLogic.name: ChatTranslationLogic,
}


def build_translation_logic(name: str) -> TranslationLogic:
    """config の logic 名から翻訳ロジックを作る."""
    try:
        return _LOGIC_FACTORIES[name]()
    except KeyError as error:
        choices = ", ".join(sorted(_LOGIC_FACTORIES))
        raise ValueError(f"Unknown translation logic: {name}. Choose from {choices}") from error


def _initial_messages(chunk: TranslationChunk) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                "あなたは学術LaTeX文書の翻訳者である. "
                "英語本文を自然な日本語の学術論文調へ翻訳し, 常体・である調で書く. "
                "専門用語, 人名, データセット名, モデル名は定着した日本語訳が"
                "明らかな場合だけ訳し, それ以外は原語のまま残す. "
                "LaTeXコマンド, 環境, 引数, 引用キー, ラベル, 参照, 数式, "
                "XQK0001 のような placeholder は一切変更しない. "
                "<source> の中身だけを過不足なく翻訳し, それ以外の文脈情報を"
                "出力に混ぜない. Markdown, 説明, 前置き, 完全文書, preamble は"
                "出力しない."
            ),
        },
        {
            "role": "user",
            "content": (
                "以下のLaTeX断片を英語から日本語へ翻訳してください.\n"
                "ルール:\n"
                "1. 翻訳対象は <source>...</source> の中身だけである.\n"
                "2. <source> tag 自体や, tag 外の文字列を出力に含めない.\n"
                "3. 文体は学術論文らしい常体・である調にする.\n"
                "4. 専門用語, 人名, データセット名, モデル名を無理に翻訳しない.\n"
                "5. LaTeXコマンド, 環境, 引数, 引用キー, ラベル, 参照, 数式は変更しない.\n"
                "6. XQK0001 のような placeholder は綴り, 個数, 順序を保つ.\n"
                "7. 著者の we は文脈に応じて「本論文では」「本稿では」などへ自然に訳す.\n"
                "8. 翻訳結果は必ず <translated>...</translated> の中に入れる.\n\n"
                "<source>\n"
                f"{chunk.text}"
                "\n</source>"
            ),
        },
    ]


def _retry_messages(
    chunk: TranslationChunk,
    *,
    previous_response: str | None,
    error: str,
) -> list[ChatCompletionMessageParam]:
    messages = _initial_messages(chunk)
    if previous_response is not None:
        messages.append({"role": "assistant", "content": previous_response})
    messages.append({
        "role": "user",
        "content": (
            "前回の翻訳結果は不正でした.\n"
            f"Error: {error}\n\n"
            "同じ <source>...</source> の中身だけをもう一度翻訳してください. "
            "エラーを修正し, tag 外の文脈情報を混ぜず, "
            "XQK placeholder と LaTeX 構造を厳密に保持し, "
            "<translated>...</translated> の中に翻訳結果だけを入れてください."
        ),
    })
    return messages


def extract_translated_text(response: str) -> str:
    """LLM応答から `<translated>...</translated>` の中身だけを取り出す.

    tag の前後に「以下が翻訳結果です」のような任意文字列が付くことは許す.
    一方で tag の欠落, open/close の個数不一致, 複数ペアは構造不明なのでエラーにする.
    """
    open_count = response.count(_OPEN_TAG)
    close_count = response.count(_CLOSE_TAG)
    if open_count == 0 and close_count == 0:
        raise ValueError("translated tags are missing")
    if open_count != close_count:
        raise ValueError(
            f"translated tag count mismatch: open={open_count}, close={close_count}"
        )
    if open_count != 1:
        raise ValueError(f"expected exactly one translated tag pair: {open_count}")

    start = response.find(_OPEN_TAG) + len(_OPEN_TAG)
    end = response.find(_CLOSE_TAG, start)
    translated = response[start:end].strip()
    if not translated:
        raise ValueError("translated text is empty")
    return translated


def normalize_translation_text(source: str, translated: str) -> str:
    """翻訳後本文へ機械的に安全な補正をかける."""
    text = _normalize_japanese_punctuation(translated.strip())
    text = _unwrap_placeholder_commands(source, text)
    text = _escape_bare_latex_special_chars(source, text)
    text = _preserve_placeholder_spacing(source, text)
    return _preserve_outer_whitespace(source, text)


def validate_translation_text(source: str, translated: str) -> None:
    """翻訳前後で XQK placeholder の種類, 個数, 順序が同じことを検査する."""
    source_placeholders = _PLACEHOLDER_RE.findall(source)
    translated_placeholders = _PLACEHOLDER_RE.findall(translated)
    if source_placeholders == translated_placeholders:
        return

    source_counts = Counter(source_placeholders)
    translated_counts = Counter(translated_placeholders)
    if source_counts == translated_counts:
        if ALLOW_PLACEHOLDER_REORDER:
            return
        raise ValueError("placeholder order changed")

    missing = [
        placeholder
        for placeholder, count in source_counts.items()
        if translated_counts[placeholder] < count
    ]
    extra = [
        placeholder
        for placeholder, count in translated_counts.items()
        if source_counts[placeholder] < count
    ]
    errors: list[str] = []
    if missing:
        errors.append(f"missing placeholders: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"extra placeholders: {', '.join(sorted(extra))}")
    raise ValueError("; ".join(errors))


def _validate_control_sequences(source: str, translated: str) -> None:
    source_counts = _control_sequence_counts(source)
    translated_counts = _control_sequence_counts(translated)
    if source_counts != translated_counts:
        raise ValueError(
            "control sequences changed: "
            f"expected={dict(source_counts)}, actual={dict(translated_counts)}"
        )


def _control_sequence_counts(text: str) -> Counter[str]:
    protected = _PLACEHOLDER_RE.sub("", text)
    return Counter(
        sequence
        for sequence in _CONTROL_SEQUENCE_RE.findall(protected)
        if sequence not in _IGNORED_CONTROL_SEQUENCES
    )


def _validate_prompt_leakage(translated: str) -> None:
    stripped = translated.lstrip()
    if stripped.startswith(("Section:", "節:", "セクション:")):
        raise ValueError("prompt metadata leaked into translation")


def _unwrap_placeholder_commands(source: str, translated: str) -> str:
    source_commands = _control_sequence_counts(source)
    parts: list[str] = []
    cursor = 0

    while cursor < len(translated):
        start = translated.find("\\", cursor)
        if start < 0:
            parts.append(translated[cursor:])
            break

        parsed = _parse_placeholder_command(translated, start)
        if parsed is None:
            parts.append(translated[cursor:start + 1])
            cursor = start + 1
            continue

        command, end, placeholder = parsed
        parts.append(translated[cursor:start])
        if source_commands[command]:
            parts.append(translated[start:end])
        else:
            parts.append(placeholder)
        cursor = end

    return "".join(parts)


def _parse_placeholder_command(text: str, start: int) -> tuple[str, int, str] | None:
    match = re.match(r"\\[A-Za-z]+\*?", text[start:])
    if match is None:
        return None

    command = match.group()
    cursor = _skip_inline_space(text, start + len(command))
    while cursor < len(text) and text[cursor] == "[":
        cursor = _skip_balanced(text, cursor, open_ch="[", close_ch="]")
        if cursor is None:
            return None
        cursor = _skip_inline_space(text, cursor)

    if cursor >= len(text) or text[cursor] != "{":
        return None
    arg_start = cursor + 1
    end = _skip_balanced(text, cursor, open_ch="{", close_ch="}")
    if end is None:
        return None

    arg = text[arg_start:end - 1]
    placeholders = _PLACEHOLDER_RE.findall(arg)
    if len(placeholders) != 1 or "\\" in arg or "{" in arg or "}" in arg:
        return None

    after = _skip_inline_space(text, end)
    if after < len(text) and text[after] in "[{":
        return None
    return command, end, placeholders[0]


def _skip_inline_space(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    return cursor


def _skip_balanced(
    text: str,
    start: int,
    *,
    open_ch: str,
    close_ch: str,
) -> int | None:
    depth = 0
    cursor = start
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == open_ch:
            depth += 1
        elif text[cursor] == close_ch:
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _normalize_japanese_punctuation(text: str) -> str:
    text = text.replace("\u3000", " ")
    replacements = {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    text = re.sub(r"[、，､]\s*", ", ", text)
    text = re.sub(r"[。．｡]\s*", ". ", text)
    text = re.sub(r"[：]\s*", ": ", text)
    text = re.sub(r"[；]\s*", "; ", text)
    text = re.sub(r"[？]\s*", "? ", text)
    text = re.sub(r"[！]\s*", "! ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def _escape_bare_latex_special_chars(source: str, text: str) -> str:
    r"""平文に混じった LaTeX 特殊文字を escape する.

    既存の control sequence はそのまま残し, command 引数内の平文は処理する.
    例: ``\footnote{AT&T}`` は ``\footnote{AT\&T}`` になる.
    """
    escapes = dict(_LATEX_SPECIAL_CHAR_ESCAPES)
    if _has_bare_latex_special_char(source, "&"):
        # In tabular chunks, bare ampersands are column separators, not text.
        escapes.pop("&", None)

    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            match = _CONTROL_SEQUENCE_RE.match(text, cursor)
            if match is not None:
                parts.append(match.group())
                cursor = match.end()
                continue

        replacement = escapes.get(text[cursor])
        if replacement is None:
            parts.append(text[cursor])
        else:
            parts.append(replacement)
        cursor += 1
    return "".join(parts)


def _has_bare_latex_special_char(text: str, char: str) -> bool:
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            match = _CONTROL_SEQUENCE_RE.match(text, cursor)
            if match is not None:
                cursor = match.end()
                continue
        if text[cursor] == char:
            return True
        cursor += 1
    return False


def _preserve_placeholder_spacing(source: str, translated: str) -> str:
    placeholders_needing_space = {
        match.group()
        for match in _PLACEHOLDER_RE.finditer(source)
        if match.end() < len(source) and source[match.end()].isspace()
    }
    if not placeholders_needing_space:
        return translated

    parts: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(translated):
        parts.append(translated[cursor:match.end()])
        cursor = match.end()
        if match.group() not in placeholders_needing_space:
            continue
        if cursor >= len(translated):
            continue
        next_char = translated[cursor]
        if next_char.isspace() or next_char in _NO_SPACE_AFTER_PLACEHOLDER:
            continue
        parts.append(" ")
    parts.append(translated[cursor:])
    return "".join(parts)


def _preserve_outer_whitespace(source: str, translated: str) -> str:
    if not source.strip():
        return translated.strip()
    leading = len(source) - len(source.lstrip())
    trailing = len(source) - len(source.rstrip())
    trailing_start = len(source) - trailing if trailing else len(source)
    return f"{source[:leading]}{translated.strip()}{source[trailing_start:]}"
