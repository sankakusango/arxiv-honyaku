"""OpenAI互換 Chat Completions API の薄い async client."""
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .config import Config, load_config


class LLMClient:
    """OpenAI互換 API へ推論リクエストを投げる最小単位."""

    def __init__(
        self,
        *,
        client_options: Mapping[str, Any],
        generation_defaults: Mapping[str, Any],
    ) -> None:
        self._client = AsyncOpenAI(**dict(client_options))
        self._generation_defaults = dict(generation_defaults)

    @classmethod
    def from_config(cls, config: Config | None = None) -> "LLMClient":
        """config.toml の `[llm.client]` / `[llm.generation]` から生成する."""
        load_dotenv()
        resolved = config or load_config()
        return cls(
            client_options=resolved.llm.client,
            generation_defaults=resolved.llm.generation,
        )

    async def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
    ) -> str:
        """chat completion を実行し, 応答本文だけを返す."""
        completion = await self._client.chat.completions.create(
            messages=list(messages),
            stream=False,
            **self._generation_defaults,
        )
        return _extract_openai_content(completion)


def _extract_openai_content(completion: ChatCompletion) -> str:
    """OpenAI 応答から message content 文字列を取り出す."""
    if not completion.choices:
        raise ValueError(
            "OpenAI response has no choices: "
            f"{_summarize_openai_completion(completion)}"
        )

    choice = completion.choices[0]
    content = choice.message.content
    if isinstance(content, str) and content.strip():
        return content.strip()

    raise ValueError(
        "OpenAI response content is empty: "
        f"{_summarize_openai_completion(completion)}"
    )


def _summarize_openai_completion(completion: ChatCompletion) -> str:
    """Return a compact, non-secret summary of a chat completion response."""
    parts = [
        f"id={_safe_value(getattr(completion, 'id', None))}",
        f"model={_safe_value(getattr(completion, 'model', None))}",
        f"created={_safe_value(getattr(completion, 'created', None))}",
        f"choices={len(getattr(completion, 'choices', []) or [])}",
    ]
    usage = _summarize_usage(getattr(completion, "usage", None))
    if usage:
        parts.append(f"usage={usage}")

    choices = getattr(completion, "choices", []) or []
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        parts.extend([
            f"choice0.finish_reason={_safe_value(getattr(choice, 'finish_reason', None))}",
            f"choice0.index={_safe_value(getattr(choice, 'index', None))}",
            f"message.role={_safe_value(getattr(message, 'role', None))}",
            f"message.content={_summarize_content(content)}",
        ])
        refusal = getattr(message, "refusal", None)
        if refusal:
            parts.append(f"message.refusal={_safe_value(refusal)}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append(f"message.tool_calls={len(tool_calls)}")
        function_call = getattr(message, "function_call", None)
        if function_call:
            parts.append("message.function_call=present")
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict) and extra:
            parts.append(f"message.extra_keys={','.join(sorted(extra)[:8])}")
    return "; ".join(parts)


def _summarize_usage(usage: Any) -> str:
    if usage is None:
        return ""
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, Mapping):
        raw = usage
    else:
        raw = {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if hasattr(usage, key)
        }
    selected = {
        key: raw.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if raw.get(key) is not None
    }
    return ",".join(f"{key}={value}" for key, value in selected.items())


def _summarize_content(content: Any) -> str:
    if content is None:
        return "None"
    if isinstance(content, str):
        preview = content.strip().replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return f"str(len={len(content)}, preview={preview!r})"
    if isinstance(content, list):
        item_types = ",".join(type(item).__name__ for item in content[:5])
        return f"list(len={len(content)}, item_types={item_types})"
    return type(content).__name__


def _safe_value(value: Any) -> str:
    if value is None:
        return "None"
    text = str(value).replace("\n", "\\n")
    if len(text) > 120:
        text = text[:117] + "..."
    return text
