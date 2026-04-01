"""FastAPIベースのWebインターフェースモジュール."""

from __future__ import annotations

from dataclasses import asdict

from arxiv_honyaku.api import translate_arxiv_async
from arxiv_honyaku.config import AppSettings


def create_app():
    """Web UIと翻訳APIエンドポイントを持つFastAPIアプリを生成する.

    Returns:
        FastAPI: 構築済みアプリインスタンス.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class TranslateInput(BaseModel):
        arxiv_ref: str

    app = FastAPI(title="arxiv-honyaku")
    settings = AppSettings.load()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """トップページHTMLを返す.

        Returns:
            str: 埋め込みHTML文字列.
        """
        return """
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>arxiv-honyaku</title>
    <style>
      body { font-family: "Noto Sans JP", sans-serif; margin: 2rem auto; max-width: 700px; line-height: 1.6; }
      input, button { font-size: 1rem; padding: 0.7rem; }
      input { width: 100%; margin-bottom: 0.8rem; }
      button { cursor: pointer; }
      pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }
    </style>
  </head>
  <body>
    <h1>arxiv-honyaku</h1>
    <p>arXiv ID または URL を入力して翻訳を実行します.</p>
    <input id="arxivRef" placeholder="2401.01234 or https://arxiv.org/abs/2401.01234" />
    <button id="runButton">翻訳開始</button>
    <pre id="result"></pre>
    <script>
      const runButton = document.getElementById("runButton");
      const result = document.getElementById("result");
      runButton.addEventListener("click", async () => {
        result.textContent = "実行中...";
        const arxiv_ref = document.getElementById("arxivRef").value;
        try {
          const response = await fetch("/translate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({arxiv_ref}),
          });
          const data = await response.json();
          result.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
          result.textContent = String(error);
        }
      });
    </script>
  </body>
</html>
"""

    @app.post("/translate")
    async def translate(payload: TranslateInput):
        """翻訳APIリクエストを実行する.

        Args:
            payload: arXiv参照文字列を含む入力モデル.

        Returns:
            dict[str, object]: 翻訳結果を辞書化したレスポンス.

        Raises:
            HTTPException: 翻訳処理で例外が発生した場合.
        """
        try:
            outcome = await translate_arxiv_async(
                payload.arxiv_ref,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return asdict(outcome)

    return app
