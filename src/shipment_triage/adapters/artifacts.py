"""Restrictive atomic filesystem writes for self-contained run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

from shipment_triage.domain.runs import StoredArtifactResult, StoredArtifactStatus


class ArtifactError(RuntimeError):
    """An artifact path or existing file violates the local safety contract."""


def _relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ArtifactError("artifact path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError("artifact path must stay below its configured root")
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_private(destination: Path, payload: bytes) -> None:
    """Atomically replace one local file with owner-only permissions."""

    if destination.exists() and destination.is_symlink():
        raise ArtifactError("artifact destination cannot be a symbolic link")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


class RunArtifactWriter:
    """Write one run directory and safely inspect artifacts from prior runs."""

    def __init__(self, output_root: str | Path, run_id: str) -> None:
        self.output_root = Path(output_root)
        if self.output_root.is_symlink():
            raise ArtifactError("output root cannot be a symbolic link")
        root_existed = self.output_root.exists()
        self.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not root_existed:
            self.output_root.chmod(0o700)
        self.run_id = run_id
        self.run_root = self._resolve_output(run_id)
        if self.run_root.is_symlink():
            raise ArtifactError("run directory cannot be a symbolic link")
        self.run_root.mkdir(mode=0o700, exist_ok=False)

    def _resolve_output(self, relative: str) -> Path:
        path = _relative_path(relative)
        candidate = self.output_root.joinpath(*path.parts)
        current = self.output_root
        for part in path.parts[:-1]:
            current /= part
            if current.exists() and current.is_symlink():
                raise ArtifactError("artifact parent cannot be a symbolic link")
        return candidate

    def _resolve_run(self, relative: str) -> Path:
        path = _relative_path(relative)
        return self._resolve_output(f"{self.run_id}/{path.as_posix()}")

    def write_bytes(self, relative: str, payload: bytes) -> tuple[str, str]:
        destination = self._resolve_run(relative)
        atomic_write_private(destination, payload)
        return relative, _sha256(payload)

    def write_json(self, relative: str, value: BaseModel | dict[str, Any]) -> tuple[str, str]:
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        payload = (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
        return self.write_bytes(relative, payload)

    def write_jsonl(self, relative: str, values: tuple[BaseModel, ...]) -> tuple[str, str]:
        lines = [
            json.dumps(
                value.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            for value in values
        ]
        payload = (("\n".join(lines) + "\n") if lines else "").encode()
        return self.write_bytes(relative, payload)

    def output_relative(self, run_relative: str) -> str:
        return f"{self.run_id}/{_relative_path(run_relative).as_posix()}"

    def inspect_or_restore(
        self,
        output_relative: str,
        *,
        expected_hash: str,
        payload: bytes,
    ) -> StoredArtifactResult:
        payload_hash = _sha256(payload)
        if payload_hash != expected_hash:
            return StoredArtifactResult(
                status=StoredArtifactStatus.CONFLICT,
                artifact_hash=payload_hash,
            )
        destination = self._resolve_output(output_relative)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ArtifactError("stored artifact path is not a regular file")
            existing_hash = _sha256(destination.read_bytes())
            status = (
                StoredArtifactStatus.MATCHED
                if existing_hash == expected_hash
                else StoredArtifactStatus.CONFLICT
            )
            return StoredArtifactResult(status=status, artifact_hash=existing_hash)
        atomic_write_private(destination, payload)
        return StoredArtifactResult(
            status=StoredArtifactStatus.RESTORED,
            artifact_hash=payload_hash,
        )


__all__ = [
    "ArtifactError",
    "RunArtifactWriter",
    "atomic_write_private",
]
