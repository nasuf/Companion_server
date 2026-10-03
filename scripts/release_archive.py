#!/usr/bin/env python3
"""Private, verified rollback artifacts. No database access or container restart.

This standalone stdlib script also ships with Web so neither deployment depends
on the other repository being deployed first. Never print archived configuration.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid


class ArchiveError(RuntimeError):
    pass


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def docker(command, *args):
    # inspect can contain credentials; suppress stderr even on failures.
    result = subprocess.run([*shlex.split(command), *args], capture_output=True,
                            check=False, timeout=1200)
    if result.returncode:
        raise ArchiveError("Docker operation failed: " + args[0])
    return result.stdout


def private_root(root):
    if root.is_symlink():
        raise ArchiveError("Archive root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.geteuid():
        raise ArchiveError("Archive root has a different owner")
    root.chmod(0o700)


@contextlib.contextmanager
def locked(root):
    private_root(root)
    with (root / ".lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def enough_space(root, estimate):
    if shutil.disk_usage(root).free < estimate:
        raise ArchiveError("Insufficient free space; existing production is untouched")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def snapshot_file(source, target):
    if not source.is_file() or source.is_symlink():
        raise ArchiveError("Required regular configuration file is missing")
    shutil.copyfile(source, target)
    target.chmod(0o600)
    return file_digest(target)


def safe_member(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArchiveError("Unsafe archive path")
    if not member.isfile() and not member.isdir():
        raise ArchiveError("Archive links/devices are unsupported")


def validate_oci_identity(archive, expected_image, config_digest):
    """Containerd's image ID may be an OCI index, rather than a config digest."""
    if "index.json" not in archive.getnames():
        raise ArchiveError("Saved image config does not match running image")
    index = json.load(archive.extractfile("index.json"))
    descriptor = next((d for d in index["manifests"] if d["digest"] == expected_image), None)
    if descriptor is None:
        raise ArchiveError("Running OCI identity is absent from the saved index")
    verified = set()
    configs = set()

    def blob(descriptor, metadata=False):
        value = descriptor["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ArchiveError("Unsupported OCI blob identity")
        name = "blobs/sha256/" + value.split(":", 1)[1]
        member = archive.getmember(name)
        if member.size != descriptor["size"]:
            raise ArchiveError("OCI blob size mismatch")
        with archive.extractfile(member) as stream:
            if "sha256:" + digest(stream) != value:
                raise ArchiveError("OCI blob checksum mismatch")
        return json.load(archive.extractfile(member)) if metadata else None

    def walk(descriptor, depth=0):
        if depth > 16:
            raise ArchiveError("OCI index nesting exceeds the limit")
        value = descriptor["digest"]
        if value in verified:
            return
        document = blob(descriptor, metadata=True)
        if "manifests" in document:
            for child in document["manifests"]:
                walk(child, depth+1)
        else:
            config = document["config"]
            blob(config)
            configs.add(config["digest"])
            for layer in document["layers"]:
                blob(layer)
        verified.add(value)

    walk(descriptor)
    if config_digest not in configs:
        raise ArchiveError("Saved compatibility config is outside the running OCI image")


def validate_image(path, expected_image):
    """Validate docker-save config identity and every layer, without extraction."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            safe_member(member)
        if len({m.name for m in members}) != len(members):
            raise ArchiveError("Duplicate image archive members")
        manifest = json.load(archive.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise ArchiveError("Expected exactly one saved image")
        item = manifest[0]
        if item.get("RepoTags"):
            raise ArchiveError("Expected image saved by immutable ID without mutable tags")
        config_bytes = archive.extractfile(item["Config"]).read()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        if config_digest != expected_image:
            validate_oci_identity(archive, expected_image, config_digest)
        config = json.loads(config_bytes)
        layers = item["Layers"]
        diff_ids = config["rootfs"]["diff_ids"]
        if len(layers) != len(diff_ids):
            raise ArchiveError("Saved image layer count mismatch")
        for name, expected in zip(layers, diff_ids):
            with archive.extractfile(name) as stream:
                compressed = stream.read(2) == b"\x1f\x8b"
                stream.seek(0)
                # Classic Docker saves raw layer tar; containerd saves gzip blobs.
                if compressed:
                    with gzip.GzipFile(fileobj=stream) as layer:
                        actual = digest(layer)
                else:
                    actual = digest(stream)
                if "sha256:" + actual != expected:
                    raise ArchiveError("Saved image layer checksum mismatch")


def tree_index(root):
    if not root.is_dir() or root.is_symlink():
        raise ArchiveError("Static site directory is missing or is a symlink")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ArchiveError("Static site links/devices are unsupported")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "sha256": file_digest(path), "size": path.stat().st_size}
    if "index.html" not in result:
        raise ArchiveError("Static site index.html is missing")
    return result


def validate_static(path, expected):
    actual = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            safe_member(member)
            if not member.isfile():
                raise ArchiveError("Static archive must contain regular files only")
            if member.isfile():
                if member.name in actual:
                    raise ArchiveError("Duplicate static file")
                with archive.extractfile(member) as stream:
                    actual[member.name] = {"sha256": digest(stream), "size": member.size}
    if actual != expected:
        raise ArchiveError("Static archive content mismatch")


def verify(path):
    if path.is_symlink() or not path.is_dir():
        raise ArchiveError("Archive directory is missing or is a symlink")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["kind"] not in {"server", "web"}:
        raise ArchiveError("Unsupported archive manifest")
    required = ({"image.tar.gz", "container.json", "image.json", ".env", "docker-compose.deploy.yml"}
                if manifest["kind"] == "server" else {"dist.tar.gz", "apple-app-site-association"})
    if set(manifest["files"]) != required:
        raise ArchiveError("Archive configuration inventory is incomplete")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or (path / name).is_symlink():
            raise ArchiveError("Invalid artifact filename")
        if file_digest(path / name) != expected:
            raise ArchiveError("Artifact checksum mismatch: " + name)
    if manifest["kind"] == "server":
        validate_image(path / "image.tar.gz", manifest["image_id"])
    else:
        validate_static(path / "dist.tar.gz", manifest["static_files"])
    return manifest


def finish_archive(root, temporary, manifest):
    manifest["files"] = {p.name: file_digest(p) for p in temporary.iterdir()}
    write_json(temporary / "manifest.json", manifest)
    verify(temporary)
    # Persist artifact bytes before publishing the completion directory/pointer.
    for path in temporary.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    final = root / (manifest["created_at"].replace(":", "").replace("+", "_")
                    + "-" + uuid.uuid4().hex[:8])
    os.replace(temporary, final)
    pointer = root / (".latest-" + uuid.uuid4().hex)
    pointer.write_text(final.name + "\n")
    pointer.chmod(0o600)
    with pointer.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pointer, root / "latest")
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return final


def base_manifest(kind, incoming):
    return {"version": 1, "kind": kind, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "incoming_commit": incoming, "database_included": False}


def archive_server(root, source, command, container, incoming, allow_missing=False):
    with locked(root):
        if allow_missing and not (source / ".env").exists() and shutil.which(shlex.split(command)[0]) is None:
            return None
        names = docker(command, "container", "ls", "-a", "--format", "{{.Names}}")
        if container not in names.decode().splitlines():
            if allow_missing and not (source / ".env").exists():
                return None
            raise ArchiveError("Existing production container is missing")
        inspect = json.loads(docker(command, "inspect", container))[0]
        state = inspect["State"]
        if not state["Running"] or state.get("Health", {}).get("Status") != "healthy":
            raise ArchiveError("Existing production container is not healthy")
        image_id = inspect["Image"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ArchiveError("Invalid running image identity")
        image = json.loads(docker(command, "image", "inspect", image_id))[0]
        enough_space(root, image["Size"] * 2 + 2 * 1024**3)
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
        try:
            manifest = base_manifest("server", incoming)
            manifest.update(image_id=image_id, format="docker-save", container=container,
                            scope="image, host env/compose and actual container config; no data volumes")
            write_json(temporary / "container.json", inspect)
            write_json(temporary / "image.json", image)
            for name in (".env", "docker-compose.deploy.yml"):
                snapshot_file(source / name, temporary / name)
            # Save to disk with a bounded timeout, then compress without buffering
            # the image in RAM. Capacity preflight accounts for both files.
            raw = temporary / "image.tar"
            with raw.open("wb") as output:
                result = subprocess.run([*shlex.split(command), "image", "save", image_id],
                                        stdout=output, stderr=subprocess.DEVNULL, timeout=1200)
                if result.returncode:
                    raise ArchiveError("Docker image save failed")
            with gzip.open(temporary / "image.tar.gz", "wb", compresslevel=1) as output:
                with raw.open("rb") as source_image:
                    shutil.copyfileobj(source_image, output, length=1024 * 1024)
            raw.unlink()
            (temporary / "image.tar.gz").chmod(0o600)
            current = json.loads(docker(command, "inspect", container))[0]
            if any(current[k] != inspect[k] for k in ("Image", "Config", "HostConfig")):
                raise ArchiveError("Production container changed while archiving")
            if not current["State"]["Running"] or current["State"].get("Health", {}).get("Status") != "healthy":
                raise ArchiveError("Production health changed while archiving")
            for name in (".env", "docker-compose.deploy.yml"):
                if file_digest(source / name) != file_digest(temporary / name):
                    raise ArchiveError("Production configuration changed while archiving")
            return finish_archive(root, temporary, manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def archive_web(root, source, association, incoming, allow_missing=False):
    with locked(root):
        if not source.exists() and allow_missing:
            return None
        index = tree_index(source)
        enough_space(root, sum(f["size"] for f in index.values()) * 2 + 128 * 1024**2)
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
        try:
            manifest = base_manifest("web", incoming)
            manifest.update(format="static-tar", static_files=index,
                            scope="static dist and Universal Links file; no database or Nginx config")
            snapshot_file(association, temporary / "apple-app-site-association")
            with tarfile.open(temporary / "dist.tar.gz", "w:gz", compresslevel=1) as archive:
                for name in index:
                    archive.add(source / name, arcname=name, recursive=False)
            (temporary / "dist.tar.gz").chmod(0o600)
            if tree_index(source) != index or file_digest(association) != file_digest(
                    temporary / "apple-app-site-association"):
                raise ArchiveError("Production static files changed while archiving")
            return finish_archive(root, temporary, manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def load_image(path, command):
    manifest = verify(path)
    if manifest["kind"] != "server":
        raise ArchiveError("Not a server image archive")
    # Saved by immutable ID: loading does not retag/replace the current deployment.
    docker(command, "image", "load", "--input", str(path / "image.tar.gz"))
    actual = json.loads(docker(command, "image", "inspect", manifest["image_id"]))[0]
    if actual["Id"] != manifest["image_id"]:
        raise ArchiveError("Loaded image identity mismatch")
    return manifest


def restore_web(path, destination, association):
    manifest = verify(path)
    if manifest["kind"] != "web" or destination.is_symlink() or association.is_symlink():
        raise ArchiveError("Invalid static restore destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
    previous = destination.with_name(destination.name + ".before-restore-" + uuid.uuid4().hex[:8])
    previous_association = association.with_name(association.name + ".before-restore-" + uuid.uuid4().hex[:8])
    staged_association = None
    try:
        with tarfile.open(path / "dist.tar.gz", "r:gz") as archive:
            for member in archive:
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o644)
        staging.chmod(0o755)
        for directory in staging.rglob("*"):
            if directory.is_dir():
                directory.chmod(0o755)
        if tree_index(staging) != manifest["static_files"]:
            raise ArchiveError("Restored static content mismatch")
        association.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=association.parent, delete=False) as output:
            output.write((path / "apple-app-site-association").read_bytes())
            staged_association = Path(output.name)
        staged_association.chmod(0o644)
        if destination.exists():
            os.replace(destination, previous)
        try:
            if association.exists():
                os.replace(association, previous_association)
            os.replace(staging, destination)
            os.replace(staged_association, association)
        except OSError:
            if not staging.exists() and destination.exists():
                shutil.rmtree(destination)
            if previous.exists():
                os.replace(previous, destination)
            if previous_association.exists():
                os.replace(previous_association, association)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if staged_association is not None and staged_association.exists():
            staged_association.unlink()
    return manifest


def prune(root, keep_days=14, minimum=3):
    if keep_days < 14 or minimum < 3:
        raise ArchiveError("Retention must keep at least 14 days and 3 archives")
    with locked(root):
        archives = []
        for path in root.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                manifest = verify(path)  # Fail closed if any old artifact is corrupt.
                archives.append((dt.datetime.fromisoformat(manifest["created_at"]), path))
        archives.sort(reverse=True)
        latest = (root / "latest").read_text().strip() if (root / "latest").exists() else None
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)
        for created, path in archives[minimum:]:
            if created < cutoff and path.name != latest:
                shutil.rmtree(path)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["server", "web", "verify", "load-image", "restore-web", "prune"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--association", type=Path)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--container", default="companion-server")
    parser.add_argument("--incoming-commit", default="unknown")
    parser.add_argument("--allow-first-deploy", action="store_true")
    args = parser.parse_args()
    required = {"server": ("root", "source"), "web": ("root", "source", "association"),
                "verify": ("archive",), "load-image": ("archive",),
                "restore-web": ("archive", "source", "association"), "prune": ("root",)}
    for name in required[args.action]:
        if getattr(args, name) is None:
            parser.error("--" + name + " is required for " + args.action)
    try:
        if args.action == "server":
            path = archive_server(args.root, args.source, args.docker, args.container,
                                  args.incoming_commit, args.allow_first_deploy)
        elif args.action == "web":
            path = archive_web(args.root, args.source, args.association,
                               args.incoming_commit, args.allow_first_deploy)
        elif args.action == "prune":
            prune(args.root)
            path = args.root
        else:
            if args.action == "verify":
                verify(args.archive)
            elif args.action == "load-image":
                load_image(args.archive, args.docker)
            else:
                restore_web(args.archive, args.source, args.association)
            path = args.archive
        print("Release archive verified: " + str(path) if path else "First deployment: no previous artifact")
        return 0
    except (ArchiveError, OSError, ValueError, KeyError, TypeError, tarfile.TarError,
            subprocess.SubprocessError) as error:
        # Error messages from parsers/filesystem may expose private config; only
        # our controlled messages are safe for GitHub Actions logs.
        print(str(error) if isinstance(error, ArchiveError) else "Release archive operation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
