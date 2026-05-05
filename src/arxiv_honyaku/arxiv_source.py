"""arxivからtexソースをダウンロードする."""
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import shutil
import tarfile

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/"
USER_AGENT = "arxiv-honyaku/0.1"

def is_safe(arxiv_id: str) -> bool:
    "arxiv_idがsafeか判定する."
    return not (".." in arxiv_id or arxiv_id.startswith("/") or ":" in arxiv_id)

def download(arxiv_id: str, destination: Path) -> None:
    """指定arXiv IDのソースアーカイブを保存する.

    Args:
        arxiv_id: 正規化済みarXiv ID.
        destination: 保存先アーカイブパス.

    Returns:
        Path: 保存済みアーカイブパス.
    """
    if not is_safe(arxiv_id):
        raise ValueError(f"unsafe arxiv_id: {arxiv_id!r}")

    url = urljoin(ARXIV_EPRINT_URL, arxiv_id)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response:
        with destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)

def unpack(archive_path: Path, extract_dir: Path) -> None:
    """tarアーカイブをextract_dirに展開する."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar_archive:
        tar_archive.extractall(extract_dir, filter="data")
