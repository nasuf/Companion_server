"""Legacy screenshot migration cannot lose/overwrite other users' media."""

import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import persist_feedback_media as media

KEY = "11111111111111111111111111111111_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
PEER = "22222222222222222222222222222222_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"


def archive(items):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content, kind in items:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            if kind == tarfile.REGTYPE:
                entry.size = len(content)
                tar.addfile(entry, io.BytesIO(content))
            else:
                entry.linkname = "/outside"
                tar.addfile(entry)
    buffer.seek(0)
    return buffer


def inspect(monkeypatch, destination, *, running=False, mounts=()):
    info = {
        "Config": {"WorkingDir": "/app", "Env": ["DATABASE_PASSWORD=PRIVATE_SENTINEL"]},
        "Mounts": list(mounts),
        "State": {"Running": running},
    }
    run = Mock(
        return_value=SimpleNamespace(returncode=0, stdout=json.dumps([info]).encode())
    )
    monkeypatch.setattr(subprocess, "run", run)
    return run


def test_migration_preserves_both_users_and_is_repeatable(tmp_path):
    items = [
        (KEY, b"original", tarfile.REGTYPE),
        (PEER, b"peer original", tarfile.REGTYPE),
    ]
    first = media.merge_archive(archive(items), tmp_path)
    assert first == {
        "status": "copied",
        "copied_files": 2,
        "matched_files": 0,
        "copied_bytes": 21,
    }
    again = media.merge_archive(archive(items), tmp_path)
    assert again["copied_files"] == 0 and again["matched_files"] == 2
    assert (tmp_path / KEY).read_bytes() == b"original"
    assert (tmp_path / PEER).read_bytes() == b"peer original"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([KEY, PEER])


def test_conflict_preserves_existing_file_and_removes_private_temporary(tmp_path):
    (tmp_path / KEY).write_bytes(b"must remain")
    with pytest.raises(RuntimeError, match="conflict"):
        media.merge_archive(archive([(KEY, b"different", tarfile.REGTYPE)]), tmp_path)
    assert (tmp_path / KEY).read_bytes() == b"must remain"
    assert list(tmp_path.iterdir()) == [tmp_path / KEY]


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../" + KEY, tarfile.REGTYPE),
        ("/" + KEY, tarfile.REGTYPE),
        (KEY, tarfile.SYMTYPE),
        (KEY, tarfile.LNKTYPE),
        ("other/" + KEY, tarfile.REGTYPE),
        ("unowned.jpg", tarfile.REGTYPE),
    ],
)
def test_unexpected_members_are_not_extracted(tmp_path, name, kind):
    with pytest.raises(RuntimeError):
        media.merge_archive(archive([(name, b"private", kind)]), tmp_path)
    assert not list(tmp_path.iterdir())


def test_live_container_cannot_be_copied_or_recreated_without_stop(
    tmp_path, monkeypatch
):
    destination = tmp_path / "screenshots"
    inspect(monkeypatch, destination, running=True)
    spawn = Mock()
    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert media.preserve(["sudo", "docker"], "server", destination, check=True)[0] == 1
    with pytest.raises(RuntimeError, match="Stop server"):
        media.preserve(["sudo", "docker"], "server", destination)
    assert not destination.exists()
    spawn.assert_not_called()


def test_current_persistent_volume_is_not_copied_again(tmp_path, monkeypatch):
    destination = tmp_path / "screenshots"
    inspect(
        monkeypatch,
        destination,
        running=True,
        mounts=[
            {
                "Type": "bind",
                "Destination": "/app/var/feedback_media",
                "Source": str(destination),
            }
        ],
    )
    assert media.preserve(["docker"], "server", destination, check=True) == (
        0,
        {"status": "already_persistent"},
    )


def test_custom_volume_is_not_duplicated_into_a_second_host_location(
    tmp_path, monkeypatch
):
    inspect(
        monkeypatch,
        tmp_path,
        mounts=[
            {
                "Type": "volume",
                "Destination": "/app/var/feedback_media",
                "Source": "/other-media",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="explicit migration"):
        media.preserve(["docker"], "server", tmp_path, check=True)


def test_docker_error_does_not_masquerade_as_an_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=1, stderr=b"permission denied PRIVATE_SENTINEL"
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        media.preserve(["docker"], "server", tmp_path)


def test_cold_start_without_container_has_no_legacy_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(returncode=1, stderr=b"No such object: server")
        ),
    )
    destination = tmp_path / "new"
    assert media.preserve(["docker"], "server", destination) == (
        0,
        {"status": "no_container"},
    )
    assert destination.is_dir()


def test_deployment_checks_and_copies_before_recreating_server():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/deploy.yml"
    ).read_text()
    start = workflow.index("source scripts/deploy_feedback_media.sh")
    assert start < workflow.index("server prisma migrate deploy", start)
    assert start < workflow.index(
        "up -d --no-deps --no-build --pull never server", start
    )


@pytest.mark.parametrize(
    "check,copy,stop,allowed,stopped",
    [
        (0, 0, 0, True, False),
        (1, 0, 0, True, True),
        (2, 0, 0, False, False),
        (1, 2, 0, False, True),
        (1, 0, 9, False, True),
    ],
)
def test_preservation_shell_blocks_replacement_on_failure(
    tmp_path, check, copy, stop, allowed, stopped
):
    import os

    block = (
        Path(__file__).resolve().parents[1] / "scripts/deploy_feedback_media.sh"
    ).read_text()
    log = tmp_path / "calls"
    stub = f"""set -euo pipefail
DOCKER=docker
docker() {{ printf 'STOP\\n' >> "$CALL_LOG"; return {stop}; }}
python3() {{
 case "$*" in
  *--check*) printf 'CHECK\\n' >> "$CALL_LOG"; return {check};;
  *) printf 'COPY\\n' >> "$CALL_LOG"; return {copy};;
 esac
}}
"""
    result = subprocess.run(
        ["bash"],
        input=stub + block + "\nprintf REPLACEMENT_ALLOWED\\n\n",
        env={"PATH": os.environ["PATH"], "CALL_LOG": str(log)},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert (result.returncode == 0) is allowed
    assert ("REPLACEMENT_ALLOWED" in result.stdout) is allowed
    calls = log.read_text().splitlines()
    assert ("STOP" in calls) is stopped
    if stopped and stop == 0:
        assert calls.index("STOP") < calls.index("COPY")
    if check == 2 or stop:
        assert "COPY" not in calls
