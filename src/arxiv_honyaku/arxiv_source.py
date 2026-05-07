"""arxivからソースや公式PDFをダウンロードする."""
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import shutil
import tarfile

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/"
ARXIV_PDF_URL = "https://arxiv.org/pdf/"
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


def download_pdf(arxiv_id: str, destination: Path) -> Path:
    """指定arXiv IDの公式PDFを保存する."""
    if not is_safe(arxiv_id):
        raise ValueError(f"unsafe arxiv_id: {arxiv_id!r}")

    url = urljoin(ARXIV_PDF_URL, f"{arxiv_id}.pdf")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response:
        with temporary.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    if not _looks_like_pdf(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded file is not a PDF: {arxiv_id}")
    temporary.replace(destination)
    return destination


def _looks_like_pdf(path: Path) -> bool:
    """PDF header を軽く確認する."""
    with path.open("rb") as fh:
        return fh.read(5) == b"%PDF-"


def unpack(archive_path: Path, extract_dir: Path) -> None:
    """tarアーカイブをextract_dirに展開する."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar_archive:
        tar_archive.extractall(extract_dir, filter="data")

def download_and_unpack(arxiv_id: str, download_dir: Path, unpack_dir: Path) -> None:
    """ダウンロードと解凍をする."""
    tar_path = download_dir / "source.tar"
    download(arxiv_id, tar_path)
    unpack(tar_path, unpack_dir)
