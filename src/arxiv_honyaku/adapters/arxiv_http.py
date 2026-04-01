"""arXivソースをHTTPで取得するアダプタモジュール."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
import shutil

from arxiv_honyaku.core.ports import SourceDownloader


class HttpArxivDownloader(SourceDownloader):
    """arXivのe-printエンドポイントからソースを取得するダウンローダ.

    Attributes:
        _timeout_seconds: HTTPタイムアウト秒数.
    """

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        """HTTPタイムアウト秒数を指定して初期化する.

        Args:
            timeout_seconds: 1リクエストあたりのタイムアウト秒数.
        """
        self._timeout_seconds = timeout_seconds

    def download(self, arxiv_id: str, destination: Path) -> Path:
        """指定arXiv IDのソースアーカイブを保存してパスを返す.

        Args:
            arxiv_id: 正規化済みarXiv ID.
            destination: 保存先アーカイブパス.

        Returns:
            Path: 保存済みアーカイブパス.
        """
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": "arxiv-honyaku/0.1"})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            with destination.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        return destination
