"""Local V1 artifact store."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.common.ids import new_artifact_id
from motion_agent.state.schemas import ArtifactHandle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactStore:
    """Filesystem-backed artifact store for local Phase 1 development."""

    def __init__(self, root: str | Path = "data/artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        artifact_type: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactHandle:
        metadata = metadata or {}
        fingerprint = stable_fingerprint(
            {"artifact_type": artifact_type, "payload": payload, "metadata": metadata},
            namespace="artifact",
        )
        artifact_id = artifact_id or new_artifact_id(artifact_type, fingerprint)
        artifact_dir = self.root / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        payload_path = artifact_dir / "payload.pkl"
        metadata_path = artifact_dir / "metadata.json"

        if not payload_path.exists():
            with payload_path.open("wb") as f:
                pickle.dump(payload, f)

        handle = ArtifactHandle(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            uri=str(payload_path),
            fingerprint=fingerprint,
            metadata=metadata,
            created_at=utc_now(),
        )
        if not metadata_path.exists():
            metadata_path.write_text(handle.model_dump_json(indent=2), encoding="utf-8")
        return handle

    def get(self, artifact_id: str) -> Any:
        with (self.root / artifact_id / "payload.pkl").open("rb") as f:
            return pickle.load(f)

    def exists(self, artifact_id: str) -> bool:
        return (self.root / artifact_id / "payload.pkl").exists()

    def metadata(self, artifact_id: str) -> ArtifactHandle:
        data = json.loads((self.root / artifact_id / "metadata.json").read_text(encoding="utf-8"))
        return ArtifactHandle.model_validate(data)

