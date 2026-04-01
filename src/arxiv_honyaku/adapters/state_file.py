"""JSONファイルベースでチェックポイントを保存するアダプタモジュール."""

from __future__ import annotations

from pathlib import Path
import json

from arxiv_honyaku.core.models import ChunkProgress, TranslationCheckpoint
from arxiv_honyaku.core.ports import CheckpointRepository


class JsonCheckpointRepository(CheckpointRepository):
    """チェックポイントをJSONで保存, 復元するリポジトリ実装."""

    def load(self, checkpoint_path: Path) -> TranslationCheckpoint | None:
        """JSONファイルからチェックポイントを読み込む.

        Args:
            checkpoint_path: 読み込むチェックポイントファイルパス.

        Returns:
            TranslationCheckpoint | None: 読み込み結果, ファイル未存在時は ``None``.
        """
        if not checkpoint_path.exists():
            return None
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        chunks: dict[int, ChunkProgress] = {}
        raw_chunks = data.get("chunks", {})
        for index_text, raw in raw_chunks.items():
            index = int(index_text)
            chunks[index] = ChunkProgress(
                digest=raw["digest"],
                status=raw.get("status", "pending"),
                attempts=int(raw.get("attempts", 0)),
                translated_text=raw.get("translated_text"),
                last_error=raw.get("last_error"),
            )
        return TranslationCheckpoint(
            source_digest=data["source_digest"],
            chunks=chunks,
        )

    def save(self, checkpoint_path: Path, checkpoint: TranslationCheckpoint) -> None:
        """チェックポイントをJSONとして保存する.

        Args:
            checkpoint_path: 保存先チェックポイントファイルパス.
            checkpoint: 保存するチェックポイントオブジェクト.
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            "source_digest": checkpoint.source_digest,
            "chunks": {
                str(index): {
                    "digest": progress.digest,
                    "status": progress.status,
                    "attempts": progress.attempts,
                    "translated_text": progress.translated_text,
                    "last_error": progress.last_error,
                }
                for index, progress in checkpoint.chunks.items()
            },
        }
        checkpoint_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
