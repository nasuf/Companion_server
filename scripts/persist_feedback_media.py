"""Preserve legacy container screenshots before adding their persistent mount.

Run after stopping the old server and before recreating it. Never overwrites a
different file, follows a link, or stores private staging files outside the
user-prefixed destination. A mounted custom legacy location requires an explicit
operator migration, so this tool cannot leave an unnoticed second host copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import threading


KEY = re.compile(r"^[0-9a-fA-F-]{32,36}_[0-9a-f]{32}\.jpg$")
MAX_FILE_BYTES = 16 * 1024 * 1024


def inspect_source(docker: list[str], container: str, destination: Path):
    result = subprocess.run(
        [*docker, "inspect", container], capture_output=True, timeout=15
    )
    if result.returncode:
        if b"No such object" in result.stderr or b"No such container" in result.stderr:
            return "no_container", None
        raise RuntimeError("Cannot inspect feedback media source")
    info = json.loads(result.stdout)[0]
    env = dict(
        item.split("=", 1) for item in info["Config"].get("Env", []) if "=" in item
    )
    source = PurePosixPath(env.get("FEEDBACK_MEDIA_DIR") or "var/feedback_media")
    if not source.is_absolute():
        source = PurePosixPath(info["Config"].get("WorkingDir") or "/") / source
    if ".." in source.parts:
        raise RuntimeError("Noncanonical feedback media location requires migration")
    for mount in info.get("Mounts", []):
        root = PurePosixPath(mount["Destination"])
        if source == root or root in source.parents:
            original = Path(mount["Source"]).joinpath(*source.relative_to(root).parts)
            if (
                mount.get("Type") == "bind"
                and original.resolve() == destination.resolve()
            ):
                return "already_persistent", source
            raise RuntimeError(
                "Custom feedback media mount requires explicit migration"
            )
    return ("running" if info["State"]["Running"] else "stopped"), source


def merge_archive(stream, destination: Path) -> dict:
    if destination.is_symlink():
        raise RuntimeError("Feedback destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    copied = matched = size = 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError("Unsafe feedback archive member")
            if entry.isdir() and name.name in {".", "", "feedback_media"}:
                continue
            if (
                not entry.isfile()
                or len(name.parts) > 2
                or (len(name.parts) == 2 and name.parts[0] != "feedback_media")
                or not KEY.fullmatch(name.name)
                or not 0 < entry.size <= MAX_FILE_BYTES
            ):
                raise RuntimeError("Unexpected feedback archive entry")
            target = destination / name.name
            owner = name.name.split("_", 1)[0]
            # A killed migration can only leave a file the user eraser finds.
            handle, temporary = tempfile.mkstemp(
                prefix=owner + "_preserve_", suffix=".tmp", dir=destination
            )
            staging = Path(temporary)
            digest = hashlib.sha256()
            try:
                with (
                    os.fdopen(handle, "wb") as writer,
                    archive.extractfile(entry) as reader,
                ):
                    remaining = entry.size
                    while remaining:
                        block = reader.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise RuntimeError("Incomplete feedback archive")
                        writer.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    writer.flush()
                    os.fsync(writer.fileno())
                staging.chmod(0o644)
                try:
                    os.link(staging, target)
                    copied += 1
                    size += entry.size
                except FileExistsError:
                    if (
                        not target.is_file()
                        or target.is_symlink()
                        or target.stat().st_size != entry.size
                        or hashlib.sha256(target.read_bytes()).digest()
                        != digest.digest()
                    ):
                        raise RuntimeError("Feedback file conflict; original retained")
                    matched += 1
            finally:
                staging.unlink(missing_ok=True)
    for directory in (destination, destination.parent):
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "status": "copied",
        "copied_files": copied,
        "matched_files": matched,
        "copied_bytes": size,
    }


def preserve(
    docker: list[str], container: str, destination: Path, *, check=False, timeout=600
):
    state, source = inspect_source(docker, container, destination)
    if state in {"already_persistent", "no_container"}:
        if not check:
            destination.mkdir(parents=True, exist_ok=True)
        return 0, {"status": state}
    if check:
        return 1, {"status": "requires_stopped_copy"}
    if state == "running":
        raise RuntimeError("Stop server before preserving feedback screenshots")
    process = subprocess.Popen(
        [*docker, "cp", f"{container}:{source}/.", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    watchdog = threading.Timer(timeout, process.kill)
    watchdog.start()
    try:
        # Peek before parsing: an absent optional directory is valid, any other
        # Docker/read/disk error must halt publication and preserve the old container.
        first = process.stdout.peek(1)
        if not first:
            error = process.stderr.read()
            code = process.wait(timeout=15)
            if code and b"Could not find the file" in error:
                destination.mkdir(parents=True, exist_ok=True)
                return 0, {"status": "no_legacy_screenshots"}
            raise RuntimeError("Feedback archive unavailable")
        report = merge_archive(process.stdout, destination)
        if process.wait(timeout=15):
            raise RuntimeError("Feedback archive copy incomplete")
        return 0, report
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        process.stdout.close()
        process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--container", default="companion-server")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        code, report = preserve(
            shlex.split(args.docker), args.container, args.destination, check=args.check
        )
        print(json.dumps(report))
        return code
    except Exception as error:
        reason = str(error) if type(error) is RuntimeError else type(error).__name__
        print(json.dumps({"status": "failed", "reason": reason}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
