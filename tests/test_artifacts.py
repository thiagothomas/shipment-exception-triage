import hashlib
import stat
from pathlib import Path

import pytest

from shipment_triage.adapters.artifacts import ArtifactError, RunArtifactWriter
from shipment_triage.domain.runs import StoredArtifactStatus

RUN_ID = "20260716T120000Z-abcdef12-123abc"


def test_writer_uses_restrictive_modes_and_safe_relative_paths(tmp_path: Path) -> None:
    writer = RunArtifactWriter(tmp_path / "runs", RUN_ID)

    relative, artifact_hash = writer.write_bytes("evidence/abc.json", b"{}\n")
    path = tmp_path / "runs" / RUN_ID / relative

    assert artifact_hash == hashlib.sha256(b"{}\n").hexdigest()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    with pytest.raises(ArtifactError):
        writer.write_bytes("../escape.txt", b"unsafe")


def test_restore_never_overwrites_conflicting_existing_bytes(tmp_path: Path) -> None:
    writer = RunArtifactWriter(tmp_path / "runs", RUN_ID)
    output_relative = f"{RUN_ID}/edi/abc.edi"
    payload = b"expected"
    expected_hash = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "runs" / output_relative
    path.parent.mkdir(mode=0o700)
    path.write_bytes(b"different")

    result = writer.inspect_or_restore(
        output_relative,
        expected_hash=expected_hash,
        payload=payload,
    )

    assert result.status is StoredArtifactStatus.CONFLICT
    assert path.read_bytes() == b"different"


def test_writer_rejects_symlink_destination(tmp_path: Path) -> None:
    writer = RunArtifactWriter(tmp_path / "runs", RUN_ID)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    destination = tmp_path / "runs" / RUN_ID / "summary.json"
    destination.symlink_to(outside)

    with pytest.raises(ArtifactError, match="symbolic link"):
        writer.write_bytes("summary.json", b"replacement")
    assert outside.read_text() == "outside"
