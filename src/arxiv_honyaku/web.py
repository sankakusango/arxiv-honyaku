"""Small built-in web UI for arxiv-honyaku.

The server intentionally uses only the Python standard library.  It is aimed at
private use by a handful of people, with opaque per-user links instead of a
login flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree
import argparse
import json
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import threading
import traceback
import uuid

from .arxiv_source import download_and_unpack
from .build_latex import compile_tex_trying_texlive_versions
from .config import Config, load_config
from .japanese_setup import JapaneseLayoutMode
from .prepare_translation import (
    prepare_from_source_tree,
    reconstruct_translated_from_source_tree,
)
from .source_tree import save_source_tree
from .translate import (
    find_latest_pdf,
    is_translation_complete,
    iter_translation_chunks,
    stats_from_jsonl,
    translate_prep_dir,
)
from .web_assets import render_admin_html, render_app_html


LayoutMode = Literal["preserve", "adaptive", "safe"]
ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "arxiv-honyaku/0.1"
ARXIV_NEW_STYLE_RE = re.compile(r"(?P<base>\d{4}\.\d{4,5})(?P<version>v\d+)?", re.I)
ARXIV_LEGACY_RE = re.compile(
    r"(?P<base>[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?",
    re.I,
)
SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ParsedArxivId:
    """arXiv id split into base id and optional version."""

    base_id: str
    version: str | None

    @property
    def effective_id(self) -> str:
        return f"{self.base_id}{self.version or ''}"

    @property
    def version_label(self) -> str:
        if self.version is None:
            raise ValueError("arXiv version has not been resolved")
        return self.version


@dataclass(frozen=True)
class ArxivMetadata:
    """Small arXiv API metadata payload used by the UI."""

    parsed: ParsedArxivId
    title: str


class JobCancelled(RuntimeError):
    """Raised when a user requests cancellation."""


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the web UI."""
    parser = argparse.ArgumentParser(
        prog="arxiv-honyaku-web",
        description="arxiv-honyaku の内輪向け Web UI を起動する.",
    )
    parser.add_argument("--host", default=os.environ.get("ARXIV_HONYAKU_WEB_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ARXIV_HONYAKU_WEB_PORT", "8000")),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Web UI の SQLite DB パス. 省略時は runs/web.sqlite3.",
    )
    parser.add_argument(
        "--create-user",
        metavar="NAME",
        help="サーバを起動せずユーザーリンクだけ作成する.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="発行リンクのベース URL. 省略時は http://host:port.",
    )
    args = parser.parse_args(argv)

    app = WebApplication(
        host=args.host,
        port=args.port,
        db_path=args.db,
        base_url=args.base_url,
    )

    if args.create_user:
        user = app.store.create_user(args.create_user)
        print(app.user_url(user["token"]))
        return

    print(f"User admin: {app.admin_url()}")
    print(f"Serving on http://{args.host}:{args.port}")
    app.serve_forever()


def extract_arxiv_id(raw_value: str) -> ParsedArxivId:
    """Extract an arXiv id from a bare id or arxiv.org URL."""
    value = unquote(raw_value.strip())
    if not value:
        raise ValueError("arXiv ID or URL is empty")

    parsed = urlparse(value)
    haystacks = [value]
    if parsed.netloc:
        path = parsed.path.strip("/")
        if path.startswith(("abs/", "pdf/", "e-print/")):
            path = path.split("/", 1)[1]
        if path.endswith(".pdf"):
            path = path[:-4]
        haystacks.insert(0, path)

    for haystack in haystacks:
        for pattern in (ARXIV_NEW_STYLE_RE, ARXIV_LEGACY_RE):
            match = pattern.search(haystack)
            if match is not None:
                version = match.group("version")
                return ParsedArxivId(
                    base_id=match.group("base"),
                    version=version.lower() if version else None,
                )
    raise ValueError(f"arXiv ID could not be extracted from: {raw_value}")


def resolve_arxiv_version(parsed: ParsedArxivId) -> ParsedArxivId:
    """Ensure an arXiv id has an explicit version."""
    return resolve_arxiv_metadata(parsed).parsed


def resolve_arxiv_metadata(parsed: ParsedArxivId) -> ArxivMetadata:
    """Ensure an arXiv id has an explicit version and title if available."""
    if parsed.version is None:
        return fetch_arxiv_metadata(parsed.base_id)
    try:
        metadata = fetch_arxiv_metadata(parsed.effective_id)
    except Exception:
        return ArxivMetadata(parsed=parsed, title="")
    if metadata.parsed.base_id != parsed.base_id or metadata.parsed.version != parsed.version:
        return ArxivMetadata(parsed=parsed, title=metadata.title)
    return metadata


def fetch_latest_arxiv_version(base_id: str) -> ParsedArxivId:
    """Look up the latest explicit arXiv version for a base id."""
    return fetch_arxiv_metadata(base_id).parsed


def fetch_arxiv_metadata(arxiv_id: str) -> ArxivMetadata:
    """Look up an explicit arXiv version and title."""
    query = urlencode({"id_list": arxiv_id, "max_results": "1"})
    request = Request(
        f"{ARXIV_API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=60) as response:
        payload = response.read()

    root = ElementTree.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        raise ValueError(f"arXiv id not found: {base_id}")
    entry_id = entry.findtext("atom:id", namespaces=namespace)
    if not entry_id:
        raise ValueError(f"arXiv response did not include an entry id for: {arxiv_id}")
    resolved = extract_arxiv_id(entry_id)
    if resolved.version is None:
        fallback = extract_arxiv_id(arxiv_id)
        if fallback.version is None:
            raise ValueError(f"arXiv response did not include an explicit version for: {arxiv_id}")
        resolved = fallback
    title = entry.findtext("atom:title", default="", namespaces=namespace)
    return ArxivMetadata(parsed=resolved, title=normalize_space(title))


def normalize_space(value: str) -> str:
    """Collapse arXiv title whitespace."""
    return " ".join(value.split())


def safe_run_name(arxiv_id: str) -> str:
    """Return a filesystem-safe directory name for a run."""
    return SAFE_SEGMENT_RE.sub("_", arxiv_id).strip("._") or "paper"


def pdf_download_filename(candidate: dict[str, Any]) -> str:
    """Return the user-facing PDF filename for a candidate."""
    stem = safe_run_name(f"{candidate['paper_id']}{candidate['version_label']}")
    return f"{stem}.pdf"


def utc_now() -> str:
    """Current UTC timestamp for storage."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(payload: Any) -> bytes:
    """Encode JSON response bytes."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    """Read a JSON request body."""
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def path_within(root: Path, relative_path: str) -> Path:
    """Resolve a relative path and ensure it stays under root."""
    if relative_path.startswith("/") or "\x00" in relative_path:
        raise ValueError("unsafe path")
    root_resolved = root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("unsafe path") from error
    return target


def read_log_file(path: Path) -> str:
    """Read a log file with replacement for broken bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


class WebStore:
    """SQLite store for the private web UI."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.init_db()

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        with self.lock, self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    token TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_versions (
                    paper_id TEXT NOT NULL,
                    version_label TEXT NOT NULL,
                    effective_arxiv_id TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, version_label)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    user_token TEXT NOT NULL,
                    paper_id TEXT,
                    version_label TEXT,
                    effective_arxiv_id TEXT,
                    workspace_id TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    overall_current REAL NOT NULL,
                    overall_total REAL NOT NULL,
                    phase_current REAL NOT NULL,
                    phase_total REAL NOT NULL,
                    message TEXT NOT NULL,
                    selected_layout_modes TEXT NOT NULL,
                    force INTEGER NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pdf_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    version_label TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    font_mode TEXT,
                    layout_mode TEXT,
                    source_dir TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    is_primary INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_paper_meta (
                    user_token TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    starred INTEGER NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_token, paper_id)
                );
                CREATE TABLE IF NOT EXISTS board_posts (
                    post_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    user_token TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tex_workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    version_label TEXT NOT NULL,
                    user_token TEXT NOT NULL,
                    base_candidate_id TEXT NOT NULL,
                    source_dir TEXT NOT NULL,
                    build_root TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_token, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_candidates_paper_version ON pdf_candidates(paper_id, version_label);
                CREATE INDEX IF NOT EXISTS idx_posts_paper_created ON board_posts(paper_id, created_at);
                """
            )
            self.ensure_column("papers", "title", "TEXT NOT NULL DEFAULT ''")
            self.ensure_column("jobs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0")

    def ensure_column(self, table: str, column: str, definition: str) -> None:
        """Add a column for older SQLite DBs."""
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def get_admin_token(self) -> str:
        """Return the admin token, creating it if needed."""
        with self.lock, self.connection:
            row = self.connection.execute(
                "SELECT value FROM settings WHERE key = 'admin_token'"
            ).fetchone()
            if row is not None:
                return str(row["value"])
            token = secrets.token_urlsafe(24)
            self.connection.execute(
                "INSERT INTO settings(key, value) VALUES('admin_token', ?)",
                (token,),
            )
            return token

    def create_user(self, display_name: str) -> dict[str, Any]:
        """Create a user link token."""
        name = display_name.strip()
        if not name:
            raise ValueError("display_name is required")
        now = utc_now()
        token = secrets.token_urlsafe(18)
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO users(token, display_name, created_at) VALUES(?, ?, ?)",
                (token, name, now),
            )
        return {"token": token, "display_name": name, "created_at": now}

    def list_users(self) -> list[dict[str, Any]]:
        """List users."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT token, display_name, created_at FROM users ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_user(self, token: str) -> dict[str, Any] | None:
        """Return a user by token."""
        with self.lock:
            row = self.connection.execute(
                "SELECT token, display_name, created_at FROM users WHERE token = ?",
                (token,),
            ).fetchone()
        return dict(row) if row is not None else None

    def upsert_paper_version(
        self,
        *,
        paper_id: str,
        version_label: str,
        effective_arxiv_id: str,
        run_dir: Path,
        title: str = "",
    ) -> None:
        """Ensure paper and version rows exist."""
        now = utc_now()
        clean_title = normalize_space(title)
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO papers(paper_id, title, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = CASE
                        WHEN excluded.title != '' THEN excluded.title
                        ELSE papers.title
                    END,
                    updated_at = excluded.updated_at
                """,
                (paper_id, clean_title, now, now),
            )
            self.connection.execute(
                """
                INSERT INTO paper_versions(
                    paper_id, version_label, effective_arxiv_id, run_dir, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, version_label) DO UPDATE SET
                    effective_arxiv_id = excluded.effective_arxiv_id,
                    run_dir = excluded.run_dir,
                    updated_at = excluded.updated_at
                """,
                (paper_id, version_label, effective_arxiv_id, str(run_dir), now, now),
            )

    def touch_paper(self, paper_id: str) -> None:
        """Update a paper timestamp."""
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE papers SET updated_at = ? WHERE paper_id = ?",
                (utc_now(), paper_id),
            )

    def list_papers(self, user_token: str) -> list[dict[str, Any]]:
        """List shared papers with user-specific metadata."""
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT
                    p.paper_id,
                    p.title,
                    p.created_at,
                    p.updated_at,
                    COALESCE(m.starred, 0) AS starred,
                    COALESCE(m.note, '') AS note,
                    (
                        SELECT COUNT(*) FROM pdf_candidates c
                        WHERE c.paper_id = p.paper_id
                    ) AS candidate_count,
                    (
                    SELECT j.status FROM jobs j
                        WHERE j.paper_id = p.paper_id
                        ORDER BY j.created_at DESC
                        LIMIT 1
                    ) AS latest_status
                FROM papers p
                LEFT JOIN user_paper_meta m
                    ON m.paper_id = p.paper_id AND m.user_token = ?
                ORDER BY COALESCE(m.starred, 0) DESC, p.updated_at DESC
                """,
                (user_token,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_paper_detail(self, user_token: str, paper_id: str) -> dict[str, Any] | None:
        """Return detail for one paper."""
        with self.lock:
            paper = self.connection.execute(
                "SELECT paper_id, title, created_at, updated_at FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if paper is None:
                return None
            meta = self.connection.execute(
                """
                SELECT starred, note FROM user_paper_meta
                WHERE user_token = ? AND paper_id = ?
                """,
                (user_token, paper_id),
            ).fetchone()
            versions = self.connection.execute(
                """
                SELECT paper_id, version_label, effective_arxiv_id, run_dir, created_at, updated_at
                FROM paper_versions
                WHERE paper_id = ?
                ORDER BY
                    CASE
                        WHEN version_label GLOB 'v[0-9]*' THEN CAST(SUBSTR(version_label, 2) AS INTEGER)
                        ELSE 0
                    END DESC,
                    version_label DESC
                """,
                (paper_id,),
            ).fetchall()
            candidates = self.connection.execute(
                """
                SELECT candidate_id, paper_id, version_label, job_id, label, font_mode,
                       layout_mode, source_dir, pdf_path, is_primary, created_at
                FROM pdf_candidates
                WHERE paper_id = ?
                ORDER BY is_primary DESC, (pdf_path = '') ASC, created_at ASC
                """,
                (paper_id,),
            ).fetchall()
            posts = self.connection.execute(
                """
                SELECT bp.post_id, bp.paper_id, bp.user_token, bp.body, bp.created_at,
                       u.display_name
                FROM board_posts bp
                JOIN users u ON u.token = bp.user_token
                WHERE bp.paper_id = ? AND bp.deleted_at IS NULL
                ORDER BY bp.created_at ASC
                """,
                (paper_id,),
            ).fetchall()
            jobs = self.connection.execute(
                """
                SELECT job_id, job_type, status, phase, message, created_at, updated_at,
                       version_label, overall_current, overall_total, phase_current,
                       phase_total, cancel_requested
                FROM jobs
                WHERE paper_id = ?
                ORDER BY created_at DESC
                LIMIT 8
                """,
                (paper_id,),
            ).fetchall()

        meta_payload = dict(meta) if meta is not None else {"starred": 0, "note": ""}
        version_payload = [dict(row) for row in versions]
        default_version = version_payload[0]["version_label"] if version_payload else None
        return {
            "paper": dict(paper),
            "meta": meta_payload,
            "versions": version_payload,
            "default_version": default_version,
            "candidates": [dict(row) for row in candidates],
            "posts": [
                {
                    **dict(row),
                    "can_delete": row["user_token"] == user_token,
                }
                for row in posts
            ],
            "jobs": [dict(row) for row in jobs],
        }

    def get_paper_versions(
        self,
        paper_id: str,
        version_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return version rows for a paper."""
        if version_label:
            sql = """
                SELECT paper_id, version_label, effective_arxiv_id, run_dir, created_at, updated_at
                FROM paper_versions
                WHERE paper_id = ? AND version_label = ?
            """
            args = (paper_id, version_label)
        else:
            sql = """
                SELECT paper_id, version_label, effective_arxiv_id, run_dir, created_at, updated_at
                FROM paper_versions
                WHERE paper_id = ?
            """
            args = (paper_id,)
        with self.lock:
            rows = self.connection.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def list_workspaces_for_paper(
        self,
        paper_id: str,
        version_label: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return TeX workspaces for a paper."""
        if version_label:
            sql = """
                SELECT workspace_id, paper_id, version_label, user_token, base_candidate_id,
                       source_dir, build_root, created_at
                FROM tex_workspaces
                WHERE paper_id = ? AND version_label = ?
            """
            args = (paper_id, version_label)
        else:
            sql = """
                SELECT workspace_id, paper_id, version_label, user_token, base_candidate_id,
                       source_dir, build_root, created_at
                FROM tex_workspaces
                WHERE paper_id = ?
            """
            args = (paper_id,)
        with self.lock:
            rows = self.connection.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def insert_job(
        self,
        *,
        job_id: str,
        job_type: str,
        user_token: str,
        paper_id: str | None,
        version_label: str | None,
        effective_arxiv_id: str | None,
        workspace_id: str | None = None,
        selected_layout_modes: list[str] | None = None,
        force: bool = False,
        message: str = "queued",
    ) -> None:
        """Insert a job row."""
        now = utc_now()
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO jobs(
                    job_id, job_type, user_token, paper_id, version_label,
                    effective_arxiv_id, workspace_id, status, phase,
                    overall_current, overall_total, phase_current, phase_total,
                    message, selected_layout_modes, force, cancel_requested,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', 0, 1, 0, 1, ?, ?, ?, 0, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    user_token,
                    paper_id,
                    version_label,
                    effective_arxiv_id,
                    workspace_id,
                    message,
                    json.dumps(selected_layout_modes or [], ensure_ascii=False),
                    1 if force else 0,
                    now,
                    now,
                ),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update selected job fields."""
        if not fields:
            return
        fields["updated_at"] = utc_now()
        allowed = {
            "status",
            "phase",
            "overall_current",
            "overall_total",
            "phase_current",
            "phase_total",
            "message",
            "cancel_requested",
            "updated_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values())
        values.append(job_id)
        with self.lock, self.connection:
            self.connection.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                values,
            )

    def append_log(self, job_id: str, level: str, message: str) -> None:
        """Append a job log line."""
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO job_logs(job_id, level, message, created_at) VALUES(?, ?, ?, ?)",
                (job_id, level.upper(), message, utc_now()),
            )

    def get_job_payload(self, user_token: str, job_id: str) -> dict[str, Any] | None:
        """Return job and logs for a user."""
        with self.lock:
            job = self.connection.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND user_token = ?",
                (job_id, user_token),
            ).fetchone()
            if job is None:
                return None
            logs = self.connection.execute(
                """
                SELECT id, level, message, created_at FROM job_logs
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
            candidates = self.connection.execute(
                """
                SELECT candidate_id, paper_id, version_label, job_id, label, font_mode,
                       layout_mode, source_dir, pdf_path, is_primary, created_at
                FROM pdf_candidates
                WHERE job_id = ?
                ORDER BY created_at ASC
                """,
                (job_id,),
            ).fetchall()
        return {
            "job": dict(job),
            "logs": [dict(row) for row in logs],
            "candidates": [dict(row) for row in candidates],
        }

    def list_active_jobs(self, user_token: str) -> list[dict[str, Any]]:
        """List active jobs for a user."""
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT job_id, job_type, paper_id, version_label, status, phase,
                       message, created_at, updated_at, overall_current, overall_total,
                       phase_current, phase_total, cancel_requested
                FROM jobs
                WHERE user_token = ? AND status NOT IN ('done', 'failed', 'cancelled')
                ORDER BY
                    CASE status WHEN 'running' THEN 0 WHEN 'canceling' THEN 1 ELSE 2 END,
                    created_at ASC
                """,
                (user_token,),
            ).fetchall()
        return [dict(row) for row in rows]

    def request_cancel_job(self, user_token: str, job_id: str) -> dict[str, Any] | None:
        """Mark a job for cancellation."""
        with self.lock, self.connection:
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE user_token = ? AND job_id = ?",
                (user_token, job_id),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status in {"done", "failed", "cancelled"}:
                return dict(row)
            new_status = "cancelled" if status == "queued" else "canceling"
            now = utc_now()
            self.connection.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1, status = ?, message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (new_status, "cancel requested", now, job_id),
            )
        self.append_log(job_id, "info", "cancel requested")
        payload = self.get_job_payload(user_token, job_id)
        return payload["job"] if payload is not None else None

    def is_cancel_requested(self, job_id: str) -> bool:
        """Return whether the user requested cancellation."""
        with self.lock:
            row = self.connection.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def add_candidate(
        self,
        *,
        paper_id: str,
        version_label: str,
        job_id: str,
        label: str,
        font_mode: str | None,
        layout_mode: str | None,
        source_dir: Path,
        pdf_path: Path | None,
        is_primary: bool,
    ) -> dict[str, Any]:
        """Add a PDF candidate. `pdf_path=None` はビルド失敗 variant を表す."""
        candidate_id = uuid.uuid4().hex
        now = utc_now()
        pdf_path_str = "" if pdf_path is None else str(pdf_path)
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO pdf_candidates(
                    candidate_id, paper_id, version_label, job_id, label,
                    font_mode, layout_mode, source_dir, pdf_path, is_primary, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    paper_id,
                    version_label,
                    job_id,
                    label,
                    font_mode,
                    layout_mode,
                    str(source_dir),
                    pdf_path_str,
                    1 if is_primary else 0,
                    now,
                ),
            )
        self.touch_paper(paper_id)
        return {
            "candidate_id": candidate_id,
            "paper_id": paper_id,
            "version_label": version_label,
            "job_id": job_id,
            "label": label,
            "font_mode": font_mode,
            "layout_mode": layout_mode,
            "source_dir": str(source_dir),
            "pdf_path": pdf_path_str,
            "is_primary": 1 if is_primary else 0,
            "created_at": now,
        }

    def has_candidates(self, paper_id: str, version_label: str) -> bool:
        """Return whether candidates already exist."""
        with self.lock:
            row = self.connection.execute(
                """
                SELECT 1 FROM pdf_candidates
                WHERE paper_id = ? AND version_label = ?
                LIMIT 1
                """,
                (paper_id, version_label),
            ).fetchone()
        return row is not None

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Return a candidate."""
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM pdf_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_star(self, user_token: str, paper_id: str, starred: bool) -> bool:
        """Set user star metadata."""
        now = utc_now()
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO user_paper_meta(user_token, paper_id, starred, note, updated_at)
                VALUES(?, ?, ?, '', ?)
                ON CONFLICT(user_token, paper_id) DO UPDATE SET
                    starred = excluded.starred,
                    updated_at = excluded.updated_at
                """,
                (user_token, paper_id, 1 if starred else 0, now),
            )
        return starred

    def set_note(self, user_token: str, paper_id: str, note: str) -> None:
        """Set user note metadata."""
        now = utc_now()
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO user_paper_meta(user_token, paper_id, starred, note, updated_at)
                VALUES(?, ?, 0, ?, ?)
                ON CONFLICT(user_token, paper_id) DO UPDATE SET
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (user_token, paper_id, note, now),
            )
        self.touch_paper(paper_id)

    def add_post(self, user_token: str, paper_id: str, body: str) -> dict[str, Any]:
        """Add a board post."""
        text = body.strip()
        if not text:
            raise ValueError("post body is required")
        now = utc_now()
        post_id = uuid.uuid4().hex
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO board_posts(post_id, paper_id, user_token, body, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (post_id, paper_id, user_token, text, now),
            )
        self.touch_paper(paper_id)
        return {"post_id": post_id, "paper_id": paper_id, "body": text, "created_at": now}

    def delete_post(self, user_token: str, post_id: str) -> bool:
        """Soft-delete a post owned by the user."""
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE board_posts
                SET deleted_at = ?
                WHERE post_id = ? AND user_token = ? AND deleted_at IS NULL
                """,
                (utc_now(), post_id, user_token),
            )
        return cursor.rowcount > 0

    def create_workspace(
        self,
        *,
        user_token: str,
        candidate: dict[str, Any],
        source_dir: Path,
        build_root: Path,
    ) -> dict[str, Any]:
        """Create a TeX editing workspace row."""
        workspace_id = uuid.uuid4().hex
        now = utc_now()
        with self.lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO tex_workspaces(
                    workspace_id, paper_id, version_label, user_token, base_candidate_id,
                    source_dir, build_root, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    candidate["paper_id"],
                    candidate["version_label"],
                    user_token,
                    candidate["candidate_id"],
                    str(source_dir),
                    str(build_root),
                    now,
                ),
            )
        return {
            "workspace_id": workspace_id,
            "paper_id": candidate["paper_id"],
            "version_label": candidate["version_label"],
            "user_token": user_token,
            "base_candidate_id": candidate["candidate_id"],
            "source_dir": str(source_dir),
            "build_root": str(build_root),
            "created_at": now,
        }

    def get_workspace(self, user_token: str, workspace_id: str) -> dict[str, Any] | None:
        """Return a workspace owned by a user."""
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM tex_workspaces WHERE workspace_id = ? AND user_token = ?",
                (workspace_id, user_token),
            ).fetchone()
        return dict(row) if row is not None else None


class WebApplication:
    """Application state and background jobs."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        db_path: Path | None,
        base_url: str | None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = base_url or f"http://{host}:{port}"
        self.config = load_config()
        self.runs_dir = Path(self.config.run.runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.store = WebStore(db_path or (self.runs_dir / "web.sqlite3"))
        self.admin_token = self.store.get_admin_token()
        self.migrate_legacy_latest_versions()
        self.migrate_missing_paper_titles()
        concurrency = int(os.environ.get("ARXIV_HONYAKU_WEB_CONCURRENCY", "1"))
        self.semaphore = threading.Semaphore(max(1, concurrency))
        self.httpd: ThreadingHTTPServer | None = None

    @property
    def layout_modes(self) -> list[str]:
        """Configured selectable layout modes."""
        modes = list(dict.fromkeys(self.config.run.japanese_layout_modes))
        return modes or ["safe"]

    def user_url(self, token: str) -> str:
        """Absolute user URL."""
        return f"{self.base_url}/u/{token}"

    def admin_url(self) -> str:
        """Absolute admin URL."""
        return f"{self.base_url}/admin/{self.admin_token}"

    def migrate_legacy_latest_versions(self) -> None:
        """Best-effort migration of old 'latest' records to explicit vN labels."""
        with self.store.lock:
            rows = self.store.connection.execute(
                """
                SELECT paper_id, effective_arxiv_id
                FROM paper_versions
                WHERE version_label = 'latest'
                """
            ).fetchall()
        for row in rows:
            paper_id = str(row["paper_id"])
            try:
                resolved = fetch_latest_arxiv_version(paper_id)
            except Exception as error:
                print(f"Could not resolve legacy latest version for {paper_id}: {error}")
                continue
            new_label = resolved.version_label
            new_effective_id = resolved.effective_id
            now = utc_now()
            with self.store.lock, self.store.connection:
                existing = self.store.connection.execute(
                    """
                    SELECT 1 FROM paper_versions
                    WHERE paper_id = ? AND version_label = ?
                    """,
                    (paper_id, new_label),
                ).fetchone()
                if existing is None:
                    self.store.connection.execute(
                        """
                        UPDATE paper_versions
                        SET version_label = ?, effective_arxiv_id = ?, updated_at = ?
                        WHERE paper_id = ? AND version_label = 'latest'
                        """,
                        (new_label, new_effective_id, now, paper_id),
                    )
                else:
                    self.store.connection.execute(
                        """
                        DELETE FROM paper_versions
                        WHERE paper_id = ? AND version_label = 'latest'
                        """,
                        (paper_id,),
                    )
                self.store.connection.execute(
                    """
                    UPDATE jobs
                    SET version_label = ?,
                        effective_arxiv_id = CASE
                            WHEN effective_arxiv_id IS NULL THEN NULL
                            ELSE ?
                        END,
                        updated_at = ?
                    WHERE paper_id = ? AND version_label = 'latest'
                    """,
                    (new_label, new_effective_id, now, paper_id),
                )
                self.store.connection.execute(
                    """
                    UPDATE pdf_candidates
                    SET version_label = ?
                    WHERE paper_id = ? AND version_label = 'latest'
                    """,
                    (new_label, paper_id),
                )
                self.store.connection.execute(
                    """
                    UPDATE tex_workspaces
                    SET version_label = ?
                    WHERE paper_id = ? AND version_label = 'latest'
                    """,
                    (new_label, paper_id),
                )

    def migrate_missing_paper_titles(self) -> None:
        """Best-effort fill for paper titles added after the initial UI."""
        with self.store.lock:
            rows = self.store.connection.execute(
                "SELECT paper_id FROM papers WHERE title = '' OR title IS NULL"
            ).fetchall()
        for row in rows:
            paper_id = str(row["paper_id"])
            try:
                metadata = fetch_arxiv_metadata(paper_id)
            except Exception as error:
                print(f"Could not fetch title for {paper_id}: {error}")
                continue
            if not metadata.title:
                continue
            with self.store.lock, self.store.connection:
                self.store.connection.execute(
                    """
                    UPDATE papers
                    SET title = ?, updated_at = ?
                    WHERE paper_id = ? AND (title = '' OR title IS NULL)
                    """,
                    (metadata.title, utc_now(), paper_id),
                )

    def serve_forever(self) -> None:
        """Start serving HTTP."""
        handler = self.make_handler()
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.httpd.serve_forever()

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        """Create a request handler bound to this application."""
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "arxiv-honyaku-web/0.1"

            def do_GET(self) -> None:
                app.handle(self, "GET")

            def do_POST(self) -> None:
                app.handle(self, "POST")

            def do_PUT(self) -> None:
                app.handle(self, "PUT")

            def do_DELETE(self) -> None:
                app.handle(self, "DELETE")

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    def handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """Dispatch a request."""
        try:
            parsed = urlparse(handler.path)
            raw_parts = [part for part in parsed.path.split("/") if part]
            parts = [unquote(part) for part in raw_parts]
            query = parse_qs(parsed.query)

            if method == "GET" and parts == []:
                self.redirect(handler, self.admin_url())
                return
            if method == "GET" and len(parts) == 2 and parts[0] == "u":
                self.handle_user_page(handler, parts[1])
                return
            if method == "GET" and len(parts) == 2 and parts[0] == "admin":
                self.handle_admin_page(handler, parts[1])
                return
            if parts and parts[0] == "api":
                self.handle_api(handler, method, parts[1:], query)
                return
            if method == "GET" and len(parts) in {2, 3} and parts[0] == "pdf":
                self.handle_pdf(handler, parts[1])
                return
            self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")
        except ValueError as error:
            self.send_error(handler, HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self.send_error(handler, HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def handle_user_page(self, handler: BaseHTTPRequestHandler, token: str) -> None:
        """Render the app page for a user token."""
        user = self.store.get_user(token)
        if user is None:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown user link")
            return
        html = render_app_html({
            "token": token,
            "user": user,
            "layout_modes": self.layout_modes,
        })
        self.send_bytes(handler, html.encode("utf-8"), content_type="text/html; charset=utf-8")

    def handle_admin_page(self, handler: BaseHTTPRequestHandler, token: str) -> None:
        """Render admin page."""
        if token != self.admin_token:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown admin link")
            return
        html = render_admin_html({
            "admin_token": token,
            "users": self.users_with_urls(),
        })
        self.send_bytes(handler, html.encode("utf-8"), content_type="text/html; charset=utf-8")

    def handle_api(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        parts: list[str],
        query: dict[str, list[str]],
    ) -> None:
        """Dispatch JSON API routes."""
        if len(parts) >= 2 and parts[0] == "admin":
            self.handle_admin_api(handler, method, parts[1:])
            return
        if len(parts) < 2 or parts[0] != "u":
            self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")
            return
        token = parts[1]
        user = self.store.get_user(token)
        if user is None:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown user")
            return
        route = parts[2:]
        if method == "GET" and route == ["state"]:
            self.send_json(handler, {
                "user": user,
                "papers": self.store.list_papers(token),
                "jobs": self.store.list_active_jobs(token),
            })
            return
        if method == "POST" and route == ["translate"]:
            payload = read_json_body(handler)
            result = self.create_translation_job(
                user_token=token,
                raw_input=str(payload.get("input", "")),
                layouts=payload.get("layouts"),
                force=bool(payload.get("force", False)),
            )
            self.send_json(handler, result, status=HTTPStatus.CREATED)
            return
        if method == "GET" and len(route) == 2 and route[0] == "jobs":
            job = self.store.get_job_payload(token, route[1])
            if job is None:
                self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown job")
            else:
                self.send_json(handler, job)
            return
        if method == "POST" and len(route) == 3 and route[0] == "jobs" and route[2] == "cancel":
            job = self.store.request_cancel_job(token, route[1])
            if job is None:
                self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown job")
            else:
                self.send_json(handler, {"job": job})
            return
        if len(route) >= 2 and route[0] == "papers":
            self.handle_paper_api(handler, method, token, route[1], route[2:])
            return
        if len(route) == 2 and route[0] == "posts" and method == "DELETE":
            deleted = self.store.delete_post(token, route[1])
            self.send_json(handler, {"deleted": deleted})
            return
        if len(route) >= 2 and route[0] == "candidates":
            self.handle_candidate_api(handler, method, token, route[1], route[2:])
            return
        if len(route) >= 2 and route[0] == "workspaces":
            self.handle_workspace_api(handler, method, token, route[1], route[2:], query)
            return
        self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")

    def handle_admin_api(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        parts: list[str],
    ) -> None:
        """Admin JSON API."""
        if not parts or parts[0] != self.admin_token:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown admin")
            return
        route = parts[1:]
        if method == "POST" and route == ["users"]:
            payload = read_json_body(handler)
            self.store.create_user(str(payload.get("display_name", "")))
            self.send_json(handler, {"users": self.users_with_urls()}, status=HTTPStatus.CREATED)
            return
        self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")

    def handle_paper_api(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        token: str,
        paper_id: str,
        route: list[str],
    ) -> None:
        """Paper-specific JSON API."""
        if method == "GET" and route == []:
            detail = self.store.get_paper_detail(token, paper_id)
            if detail is None:
                self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown paper")
            else:
                self.send_json(handler, detail)
            return
        if method == "GET" and route == ["build-logs"]:
            query = parse_qs(urlparse(handler.path).query)
            version_label = query.get("version", [None])[0]
            self.send_json(handler, self.build_logs_payload(paper_id, version_label))
            return
        if method == "POST" and route == ["star"]:
            payload = read_json_body(handler)
            starred = self.store.set_star(token, paper_id, bool(payload.get("starred", False)))
            self.send_json(handler, {"starred": 1 if starred else 0})
            return
        if method == "POST" and route == ["note"]:
            payload = read_json_body(handler)
            self.store.set_note(token, paper_id, str(payload.get("note", "")))
            self.send_json(handler, {"ok": True})
            return
        if method == "POST" and route == ["posts"]:
            payload = read_json_body(handler)
            post = self.store.add_post(token, paper_id, str(payload.get("body", "")))
            self.send_json(handler, post, status=HTTPStatus.CREATED)
            return
        self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")

    def handle_candidate_api(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        token: str,
        candidate_id: str,
        route: list[str],
    ) -> None:
        """Candidate-specific JSON API."""
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown candidate")
            return
        if method == "GET" and route == ["tex-files"]:
            files = self.list_tex_files(Path(candidate["source_dir"]))
            self.send_json(handler, {"files": files})
            return
        if method == "POST" and route == ["workspace"]:
            workspace = self.create_workspace(token, candidate)
            self.send_json(handler, {"workspace": workspace}, status=HTTPStatus.CREATED)
            return
        self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")

    def handle_workspace_api(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        token: str,
        workspace_id: str,
        route: list[str],
        query: dict[str, list[str]],
    ) -> None:
        """TeX workspace JSON API."""
        workspace = self.store.get_workspace(token, workspace_id)
        if workspace is None:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown workspace")
            return
        source_dir = Path(workspace["source_dir"])
        if method == "GET" and route == ["files"]:
            relative = query.get("path", [""])[0]
            path = path_within(source_dir, relative)
            if not path.is_file():
                self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown file")
                return
            self.send_json(handler, {"path": relative, "text": path.read_text(encoding="utf-8", errors="replace")})
            return
        if method == "PUT" and route == ["files"]:
            relative = query.get("path", [""])[0]
            path = path_within(source_dir, relative)
            if path.suffix != ".tex" or not path.is_file():
                self.send_error(handler, HTTPStatus.BAD_REQUEST, "only .tex files can be edited")
                return
            payload = read_json_body(handler)
            path.write_text(str(payload.get("text", "")), encoding="utf-8")
            self.send_json(handler, {"ok": True})
            return
        if method == "POST" and route == ["build"]:
            result = self.create_rebuild_job(token, workspace)
            self.send_json(handler, result, status=HTTPStatus.CREATED)
            return
        self.send_error(handler, HTTPStatus.NOT_FOUND, "not found")

    def handle_pdf(self, handler: BaseHTTPRequestHandler, candidate_id: str) -> None:
        """Serve a stored PDF candidate."""
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            self.send_error(handler, HTTPStatus.NOT_FOUND, "unknown PDF")
            return
        pdf_path = Path(candidate["pdf_path"])
        if not pdf_path.is_file():
            self.send_error(handler, HTTPStatus.NOT_FOUND, "PDF file is missing")
            return
        self.send_file(
            handler,
            pdf_path,
            content_type="application/pdf",
            download_filename=pdf_download_filename(candidate),
        )

    def users_with_urls(self) -> list[dict[str, Any]]:
        """List users with absolute URLs."""
        return [
            {**user, "url": self.user_url(user["token"])}
            for user in self.store.list_users()
        ]

    def build_logs_payload(
        self,
        paper_id: str,
        version_label: str | None,
    ) -> dict[str, Any]:
        """Return persisted latexmk log files for a paper/version."""
        versions = self.store.get_paper_versions(paper_id, version_label)
        attempts: list[dict[str, Any]] = []
        for version in versions:
            run_dir = Path(version["run_dir"])
            attempts.extend(self.collect_translated_build_logs(version, run_dir))
            attempts.extend(self.collect_manual_build_logs(version))
        attempts.sort(key=lambda item: item["created_at"], reverse=True)
        return {
            "paper_id": paper_id,
            "version_label": version_label,
            "attempts": attempts,
        }

    def collect_translated_build_logs(
        self,
        version: dict[str, Any],
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        """Collect build logs from translated_builds."""
        build_root = run_dir / "translated_builds"
        if not build_root.exists():
            return []
        attempts: list[dict[str, Any]] = []
        for attempt_dir in sorted({path.parent for path in build_root.rglob("latexmk.*.log")}):
            variant = attempt_dir.parent.name
            attempts.append(self.build_attempt_payload(
                source="translation",
                paper_id=version["paper_id"],
                version_label=version["version_label"],
                label=variant,
                attempt_dir=attempt_dir,
            ))
        return attempts

    def collect_manual_build_logs(self, version: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect build logs from manual TeX workspaces."""
        attempts: list[dict[str, Any]] = []
        for workspace in self.store.list_workspaces_for_paper(
            version["paper_id"],
            version["version_label"],
        ):
            build_root = Path(workspace["build_root"])
            if not build_root.exists():
                continue
            for attempt_dir in sorted({path.parent for path in build_root.rglob("latexmk.*.log")}):
                attempts.append(self.build_attempt_payload(
                    source="manual",
                    paper_id=version["paper_id"],
                    version_label=version["version_label"],
                    label=f"manual {workspace['workspace_id'][:8]}",
                    attempt_dir=attempt_dir,
                ))
        return attempts

    def build_attempt_payload(
        self,
        *,
        source: str,
        paper_id: str,
        version_label: str,
        label: str,
        attempt_dir: Path,
    ) -> dict[str, Any]:
        """Return one latexmk attempt payload."""
        files = []
        for name in ("latexmk.stderr.log", "latexmk.stdout.log"):
            path = attempt_dir / name
            if not path.exists():
                continue
            stat = path.stat()
            files.append({
                "name": name,
                "path": str(path),
                "size": stat.st_size,
                "text": read_log_file(path),
            })
        return {
            "source": source,
            "paper_id": paper_id,
            "version_label": version_label,
            "label": label,
            "attempt": attempt_dir.name,
            "attempt_dir": str(attempt_dir),
            "created_at": datetime.fromtimestamp(
                attempt_dir.stat().st_mtime,
                timezone.utc,
            ).isoformat(timespec="seconds"),
            "status": "success" if list(attempt_dir.glob("*.pdf")) else "failed",
            "files": files,
        }

    def create_translation_job(
        self,
        *,
        user_token: str,
        raw_input: str,
        layouts: Any,
        force: bool,
    ) -> dict[str, Any]:
        """Create and start a translation job."""
        metadata = resolve_arxiv_metadata(extract_arxiv_id(raw_input))
        parsed = metadata.parsed
        selected_layouts = self.normalize_layouts(layouts)
        effective_id = parsed.effective_id
        version_label = parsed.version_label
        run_dir = self.runs_dir / safe_run_name(effective_id)
        self.store.upsert_paper_version(
            paper_id=parsed.base_id,
            version_label=version_label,
            effective_arxiv_id=effective_id,
            run_dir=run_dir,
            title=metadata.title,
        )
        job_id = uuid.uuid4().hex
        self.store.insert_job(
            job_id=job_id,
            job_type="translate",
            user_token=user_token,
            paper_id=parsed.base_id,
            version_label=version_label,
            effective_arxiv_id=effective_id,
            selected_layout_modes=selected_layouts,
            force=force,
            message="queued",
        )
        thread = threading.Thread(
            target=self.run_translation_job,
            args=(job_id, parsed.base_id, version_label, effective_id, run_dir, selected_layouts, force),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "paper_id": parsed.base_id, "version_label": version_label}

    def normalize_layouts(self, value: Any) -> list[LayoutMode]:
        """Validate selected layout modes."""
        if not isinstance(value, list):
            selected = self.layout_modes
        else:
            selected = [str(item) for item in value]
        allowed = set(self.layout_modes)
        normalized = [item for item in selected if item in allowed]
        if not normalized:
            raise ValueError("at least one layout mode must be selected")
        return [cast(LayoutMode, item) for item in dict.fromkeys(normalized)]

    def config_for_layouts(self, layouts: list[LayoutMode]) -> Config:
        """Return config with layout modes overridden for one job."""
        run = self.config.run.model_copy(
            update={"japanese_layout_modes": cast(list[JapaneseLayoutMode], layouts)}
        )
        return self.config.model_copy(update={"run": run})

    def run_translation_job(
        self,
        job_id: str,
        paper_id: str,
        version_label: str,
        effective_id: str,
        run_dir: Path,
        layouts: list[LayoutMode],
        force: bool,
    ) -> None:
        """Run download, translation, and builds in a background thread."""
        variant_total = len(self.config.run.japanese_font_modes) * len(layouts)
        overall_total = 4 + max(1, variant_total)
        self.log(job_id, "info", "waiting for translation slot")
        self.store.update_job(
            job_id,
            status="queued",
            phase="queued",
            overall_current=0,
            overall_total=overall_total,
            phase_current=0,
            phase_total=1,
            message="waiting",
        )
        with self.semaphore:
            try:
                self.raise_if_cancelled(job_id)
                self.store.update_job(job_id, status="running", message="started")
                config = self.config_for_layouts(layouts)
                source_dir = run_dir / "source"
                source_tree_path = run_dir / "source_tree.json"
                prep_dir = run_dir / "prep"
                translations_jsonl = run_dir / "translations.jsonl"
                translated_dir = run_dir / "translated"
                build_root = run_dir / "translated_builds"
                run_dir.mkdir(parents=True, exist_ok=True)

                current = 0
                self.raise_if_cancelled(job_id)
                self.set_phase(job_id, "download", current, overall_total, 0, 1, "source")
                if not source_dir.exists():
                    self.log(job_id, "info", f"downloading source: {effective_id}")
                    download_and_unpack(
                        effective_id,
                        download_dir=run_dir / "downloads",
                        unpack_dir=source_dir,
                    )
                else:
                    self.log(job_id, "info", f"reusing source: {source_dir}")
                current += 1
                self.raise_if_cancelled(job_id)
                self.set_phase(job_id, "source_tree", current, overall_total, 0, 1, "source tree")
                save_source_tree(source_dir, source_tree_path)
                current += 1

                self.raise_if_cancelled(job_id)
                self.set_phase(job_id, "prepare", current, overall_total, 0, 1, "preparing chunks")
                prep_paths = prepare_from_source_tree(source_tree_path, prep_dir)
                self.log(job_id, "info", f"prepared files: {len(prep_paths)}")
                current += 1

                chunks = list(iter_translation_chunks(prep_dir))
                self.raise_if_cancelled(job_id)
                self.set_phase(job_id, "translate", current, overall_total, 0, len(chunks), "translating")
                if not force and is_translation_complete(prep_dir, translations_jsonl):
                    stats = stats_from_jsonl(translations_jsonl)
                    self.log(job_id, "info", f"translation cache complete: ok={stats.ok}, failed={stats.failed}")
                    self.set_phase(
                        job_id,
                        "translate",
                        current,
                        overall_total,
                        stats.total,
                        max(1, stats.total),
                        "translation cached",
                    )
                else:
                    def translate_progress(event: str, payload: dict[str, object]) -> None:
                        self.raise_if_cancelled(job_id)
                        phase_current = float(payload.get("current", 0) or 0)
                        phase_total = float(payload.get("total", 1) or 1)
                        self.set_phase(
                            job_id,
                            "translate",
                            current,
                            overall_total,
                            phase_current,
                            phase_total,
                            f"ok={payload.get('ok', 0)} failed={payload.get('failed', 0)}",
                        )
                        if event == "translate_chunk_finished" and payload.get("status") != "ok":
                            self.log(
                                job_id,
                                "error",
                                (
                                    f"chunk failed: {payload.get('source_path')}#"
                                    f"{payload.get('chunk_index')}: {payload.get('error')}"
                                ),
                            )
                            for attempt in cast(list[dict[str, Any]], payload.get("attempts", [])):
                                if not attempt.get("ok"):
                                    self.log(
                                        job_id,
                                        "error",
                                        f"attempt {attempt.get('attempt')}: {attempt.get('error')}",
                                    )

                    stats = translate_prep_dir(
                        prep_dir,
                        translations_jsonl,
                        config=config,
                        progress_callback=translate_progress,
                        show_progress=False,
                        reuse_existing=not force,
                    )
                    self.log(job_id, "info", f"translation done: total={stats.total}, ok={stats.ok}, failed={stats.failed}")
                current += 1

                built_count = 0
                build_errors = 0
                make_primary = not self.store.has_candidates(paper_id, version_label)
                for font_mode in config.run.japanese_font_modes:
                    for layout_mode in layouts:
                        self.raise_if_cancelled(job_id)
                        label = f"{font_mode}--{layout_mode}"
                        self.set_phase(job_id, "reconstruct", current, overall_total, 0, 1, label)
                        variant_translated_dir = translated_dir / label
                        variant_build_root = build_root / label
                        self.log(job_id, "info", f"reconstructing {label}")
                        reconstruct_translated_from_source_tree(
                            source_tree_path,
                            prep_dir,
                            variant_translated_dir,
                            translations_jsonl=translations_jsonl,
                            font_mode=font_mode,
                            layout_mode=layout_mode,
                        )
                        self.set_phase(
                            job_id,
                            "build",
                            current,
                            overall_total,
                            0,
                            len(config.run.texlive_versions),
                            label,
                        )
                        pdf_path: Path | None = None
                        try:
                            existing_pdf = None if force else find_latest_pdf(variant_build_root)
                            if existing_pdf is not None:
                                pdf_path = existing_pdf
                                self.log(job_id, "info", f"reusing PDF [{label}]: {pdf_path}")
                            else:
                                pdf_path = compile_tex_trying_texlive_versions(
                                    source_dir=variant_translated_dir,
                                    build_root=variant_build_root,
                                    texlive_versions=config.run.texlive_versions,
                                    progress_callback=self.build_progress_callback(job_id, label, current, overall_total),
                                    show_progress=False,
                                )
                        except Exception as error:
                            build_errors += 1
                            self.log(job_id, "error", f"build failed [{label}]: {error}")
                        candidate = self.store.add_candidate(
                            paper_id=paper_id,
                            version_label=version_label,
                            job_id=job_id,
                            label=label,
                            font_mode=str(font_mode),
                            layout_mode=str(layout_mode),
                            source_dir=variant_translated_dir,
                            pdf_path=pdf_path,
                            is_primary=pdf_path is not None and make_primary and built_count == 0,
                        )
                        if pdf_path is not None:
                            built_count += 1
                            self.log(job_id, "info", f"PDF candidate [{label}]: {candidate['candidate_id']}")
                        else:
                            self.log(job_id, "info", f"failed candidate registered [{label}]: {candidate['candidate_id']}")
                        current += 1

                if built_count == 0:
                    raise RuntimeError("no PDF candidate was built")
                message = f"done: PDFs={built_count}"
                if build_errors:
                    message += f", build_errors={build_errors}"
                self.store.update_job(
                    job_id,
                    status="done",
                    phase="done",
                    overall_current=overall_total,
                    overall_total=overall_total,
                    phase_current=1,
                    phase_total=1,
                    message=message,
                )
                self.log(job_id, "info", message)
                self.store.touch_paper(paper_id)
            except JobCancelled:
                self.store.update_job(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    message="cancelled",
                    cancel_requested=1,
                )
                self.log(job_id, "info", "cancelled")
            except Exception as error:
                self.log(job_id, "error", str(error))
                self.log(job_id, "error", traceback.format_exc())
                self.store.update_job(
                    job_id,
                    status="failed",
                    phase="failed",
                    message=str(error),
                )

    def build_progress_callback(
        self,
        job_id: str,
        label: str,
        overall_current: int,
        overall_total: int,
    ):
        """Create a progress callback for a build variant."""
        def callback(event: str, payload: dict[str, object]) -> None:
            self.raise_if_cancelled(job_id)
            phase_current = float(payload.get("current", 0) or 0)
            phase_total = float(payload.get("total", 1) or 1)
            version = payload.get("version", "")
            self.set_phase(
                job_id,
                "build",
                overall_current,
                overall_total,
                phase_current,
                phase_total,
                f"{label} / TeX Live {version}",
            )
            if event == "build_attempt_started":
                self.log(job_id, "info", f"latexmk start [{label}] TeX Live {version}")
            elif event == "build_attempt_succeeded":
                self.log(job_id, "info", f"latexmk success [{label}] TeX Live {version}")
            elif event == "build_attempt_failed":
                self.log(job_id, "error", f"latexmk failed [{label}] TeX Live {version}: {payload.get('error')}")
                summary = str(payload.get("summary", "") or "").strip()
                if summary:
                    self.log(job_id, "error", f"latexmk key errors [{label}] TeX Live {version}\n{summary}")
                self.append_latex_logs(job_id, Path(str(payload.get("attempt_dir", ""))))

        return callback

    def create_workspace(self, user_token: str, candidate: dict[str, Any]) -> dict[str, Any]:
        """Create a copy of a candidate source tree for direct TeX editing."""
        source = Path(candidate["source_dir"])
        if not source.is_dir():
            raise ValueError("candidate source directory is missing")
        workspace_root = (
            self.runs_dir
            / safe_run_name(str(candidate["paper_id"]))
            / "manual_edits"
            / user_token[:10]
            / uuid.uuid4().hex
        )
        workspace_source = workspace_root / "source"
        workspace_build = workspace_root / "builds"
        shutil.copytree(source, workspace_source)
        workspace_build.mkdir(parents=True, exist_ok=True)
        workspace = self.store.create_workspace(
            user_token=user_token,
            candidate=candidate,
            source_dir=workspace_source,
            build_root=workspace_build,
        )
        workspace["files"] = self.list_tex_files(workspace_source)
        return workspace

    def create_rebuild_job(self, user_token: str, workspace: dict[str, Any]) -> dict[str, Any]:
        """Create and start a background rebuild job from an edited workspace."""
        job_id = uuid.uuid4().hex
        self.store.insert_job(
            job_id=job_id,
            job_type="rebuild",
            user_token=user_token,
            paper_id=workspace["paper_id"],
            version_label=workspace["version_label"],
            effective_arxiv_id=None,
            workspace_id=workspace["workspace_id"],
            selected_layout_modes=[],
            force=False,
            message="queued",
        )
        thread = threading.Thread(
            target=self.run_rebuild_job,
            args=(job_id, workspace),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "paper_id": workspace["paper_id"], "version_label": workspace["version_label"]}

    def run_rebuild_job(self, job_id: str, workspace: dict[str, Any]) -> None:
        """Run latexmk for an edited TeX workspace."""
        self.log(job_id, "info", "waiting for build slot")
        self.store.update_job(
            job_id,
            status="queued",
            phase="queued",
            overall_current=0,
            overall_total=1,
            phase_current=0,
            phase_total=1,
            message="waiting",
        )
        with self.semaphore:
            try:
                self.raise_if_cancelled(job_id)
                source_dir = Path(workspace["source_dir"])
                build_root = Path(workspace["build_root"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
                self.store.update_job(job_id, status="running", message="building edited TeX")
                pdf_path = compile_tex_trying_texlive_versions(
                    source_dir=source_dir,
                    build_root=build_root,
                    texlive_versions=self.config.run.texlive_versions,
                    progress_callback=self.build_progress_callback(job_id, "manual", 0, 1),
                    show_progress=False,
                )
                self.raise_if_cancelled(job_id)
                label = "manual " + datetime.now().strftime("%Y-%m-%d %H:%M")
                candidate = self.store.add_candidate(
                    paper_id=workspace["paper_id"],
                    version_label=workspace["version_label"],
                    job_id=job_id,
                    label=label,
                    font_mode=None,
                    layout_mode="manual",
                    source_dir=source_dir,
                    pdf_path=pdf_path,
                    is_primary=False,
                )
                self.log(job_id, "info", f"PDF candidate [manual]: {candidate['candidate_id']}")
                self.store.update_job(
                    job_id,
                    status="done",
                    phase="done",
                    overall_current=1,
                    overall_total=1,
                    phase_current=1,
                    phase_total=1,
                    message="done",
                )
            except JobCancelled:
                self.store.update_job(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    message="cancelled",
                    cancel_requested=1,
                )
                self.log(job_id, "info", "cancelled")
            except Exception as error:
                self.log(job_id, "error", str(error))
                self.log(job_id, "error", traceback.format_exc())
                self.store.update_job(job_id, status="failed", phase="failed", message=str(error))

    def set_phase(
        self,
        job_id: str,
        phase: str,
        overall_current: float,
        overall_total: float,
        phase_current: float,
        phase_total: float,
        message: str,
    ) -> None:
        """Update job phase progress."""
        status = "canceling" if self.store.is_cancel_requested(job_id) else "running"
        self.store.update_job(
            job_id,
            status=status,
            phase=phase,
            overall_current=overall_current,
            overall_total=max(1, overall_total),
            phase_current=phase_current,
            phase_total=max(1, phase_total),
            message=message,
        )

    def raise_if_cancelled(self, job_id: str) -> None:
        """Raise when a job has been cancelled by the user."""
        if self.store.is_cancel_requested(job_id):
            raise JobCancelled("cancelled")

    def log(self, job_id: str, level: str, message: str) -> None:
        """Append a job log line."""
        self.store.append_log(job_id, level, message)

    def append_latex_logs(self, job_id: str, attempt_dir: Path) -> None:
        """Append latexmk stdout/stderr logs to the job log."""
        for name in ("latexmk.stderr.log", "latexmk.stdout.log"):
            path = attempt_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                self.log(job_id, "error", f"{name}\n{text}")

    def list_tex_files(self, source_dir: Path) -> list[str]:
        """List editable .tex files."""
        if not source_dir.is_dir():
            return []
        return [
            str(path.relative_to(source_dir))
            for path in sorted(source_dir.rglob("*.tex"))
            if path.is_file()
        ]

    def redirect(self, handler: BaseHTTPRequestHandler, location: str) -> None:
        """Send an HTTP redirect."""
        handler.send_response(HTTPStatus.FOUND)
        handler.send_header("Location", location)
        handler.end_headers()

    def send_json(
        self,
        handler: BaseHTTPRequestHandler,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Send a JSON response."""
        self.send_bytes(
            handler,
            json_dumps(payload),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def send_error(
        self,
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        message: str,
    ) -> None:
        """Send an error response."""
        self.send_json(handler, {"error": message}, status=status)

    def send_file(
        self,
        handler: BaseHTTPRequestHandler,
        path: Path,
        *,
        content_type: str | None = None,
        download_filename: str | None = None,
    ) -> None:
        """Send a local file."""
        resolved_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        headers = {}
        if download_filename is not None:
            encoded = quote(download_filename)
            headers["Content-Disposition"] = (
                f"inline; filename=\"{download_filename}\"; filename*=UTF-8''{encoded}"
            )
        self.send_bytes(handler, data, content_type=resolved_type, headers=headers)

    def send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        data: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send a bytes response."""
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(data)


if __name__ == "__main__":
    main()
