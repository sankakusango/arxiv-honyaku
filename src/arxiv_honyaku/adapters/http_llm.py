"""OpenAI互換APIで翻訳を実行する, LLMアダプタモジュール."""

from __future__ import annotations

from arxiv_honyaku.core.ports import LLMTranslator
from arxiv_honyaku.prompts import PromptProfile

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as import_error:
    AsyncOpenAI = None
    _OPENAI_IMPORT_ERROR = import_error
else:
    _OPENAI_IMPORT_ERROR = None


class OpenAICompatibleTranslator(LLMTranslator):
    """OpenAI互換のChat Completions APIを使う翻訳実装.

    Attributes:
        _provider: プロバイダ識別子.
        _model: 利用モデル名.
        _prompt_profile: 翻訳時に使うプロンプトプロファイル.
        _client: AsyncOpenAIクライアント.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        prompt_profile: PromptProfile,
        base_url: str | None,
        api_key: str | None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """モデル接続情報とプロンプトプロファイルを設定する.

        Args:
            provider: プロバイダ識別子.
            model: 利用モデル名.
            prompt_profile: system/user promptを提供するプロファイル.
            base_url: OpenAI互換APIベースURL.
            api_key: APIキー.
            timeout_seconds: API呼び出しタイムアウト秒数.

        Raises:
            RuntimeError: ``openai`` パッケージが未導入の場合.
        """
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai package is required for LLM translation. "
                "Install dependencies and retry."
            ) from _OPENAI_IMPORT_ERROR
        self._provider = provider
        self._model = model
        self._prompt_profile = prompt_profile
        client_options: dict[str, object] = {"timeout": timeout_seconds}
        if base_url:
            client_options["base_url"] = base_url
        if api_key:
            client_options["api_key"] = api_key
        self._client = AsyncOpenAI(**client_options)

    async def translate(self, text: str, *, temperature: float) -> str:
        """TeXチャンクを翻訳し, 翻訳本文のみを返す.

        Args:
            text: 翻訳対象のTeXチャンク本文.
            temperature: 当該試行で使う推論 temperature.

        Returns:
            str: 翻訳済みTeX本文.
        """
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=self._prompt_profile.build_messages(
                provider=self._provider,
                model=self._model,
                chunk_text=text,
            ),
        )
        content = _extract_openai_content(completion)
        return self._prompt_profile.extract_translated_text(content)


def _extract_openai_content(completion: object) -> str:
    """OpenAIレスポンスからテキスト本文を抽出する.

    Args:
        completion: ``chat.completions.create`` の戻り値オブジェクト.

    Returns:
        str: 応答本文文字列.

    Raises:
        ValueError: choices, message, content が期待形式でない場合.
    """
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI response has no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("OpenAI response has no message")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            return stripped
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") != "text":
                    continue
                raw_text = item.get("text")
                if isinstance(raw_text, str):
                    text_parts.append(raw_text)
                continue

            item_type = getattr(item, "type", None)
            if item_type != "text":
                continue
            raw_text = getattr(item, "text", None)
            if isinstance(raw_text, str):
                text_parts.append(raw_text)
        joined = "".join(text_parts).strip()
        if joined:
            return joined
    raise ValueError("OpenAI response content is empty")
