"""Rollback artifact tests use synthetic files and a fake Docker boundary."""
import datetime as dt
import hashlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_archive", ROOT / "scripts/release_archive.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def saved_image():
    layer = b"synthetic layer; no customer data"
    config = json.dumps({"rootfs": {"diff_ids": ["sha256:" + hashlib.sha256(layer).hexdigest()]}}).encode()
    image_id = "sha256:" + hashlib.sha256(config).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in [("config.json", config), ("layer.tar", layer), ("manifest.json", json.dumps([
                {"Config": "config.json", "Layers": ["layer.tar"], "RepoTags": None}]).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return image_id, buffer.getvalue()


def saved_oci_image():
    image_id, classic = saved_image()
    with tarfile.open(fileobj=io.BytesIO(classic)) as old:
        config = old.extractfile("config.json").read()
        layer = gzip.compress(old.extractfile("layer.tar").read())
    blobs = {}

    def descriptor(data, kind):
        value = "sha256:" + hashlib.sha256(data).hexdigest()
        blobs["blobs/sha256/" + value.split(":", 1)[1]] = data
        return {"digest": value, "size": len(data), "mediaType": kind}

    config_ref = descriptor(config, "application/vnd.oci.image.config.v1+json")
    layer_ref = descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")
    manifest = descriptor(json.dumps({"schemaVersion": 2, "config": config_ref, "layers": [layer_ref]}).encode(),
                          "application/vnd.oci.image.manifest.v1+json")
    index = descriptor(json.dumps({"schemaVersion": 2, "manifests": [manifest]}).encode(),
                       "application/vnd.oci.image.index.v1+json")
    blobs["index.json"] = json.dumps({"manifests": [index]}).encode()
    blobs["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    blobs["manifest.json"] = json.dumps([{"Config": "blobs/sha256/" + image_id.split(":", 1)[1],
        "Layers": ["blobs/sha256/"+layer_ref["digest"].split(":", 1)[1]], "RepoTags": None}]).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in blobs.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return index["digest"], buffer.getvalue()


class ReleaseArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "archives"
        self.site = self.base / "dist"
        (self.site / "assets").mkdir(parents=True)
        (self.site / "index.html").write_text("synthetic v1")
        (self.site / "assets/app.js").write_text("v1")
        self.association = self.base / "association"
        self.association.write_text('{"synthetic":true}')
        self.server = self.base / "server"
        self.server.mkdir()
        (self.server / ".env").write_text("JWT_SECRET=synthetic-do-not-log\n")
        (self.server / "docker-compose.deploy.yml").write_text("services: {}\n")
        self.image_id, self.tar = saved_image()
        self.inspect = {"Image": self.image_id, "Config": {"Env": ["KEY=synthetic-do-not-log"]},
                        "HostConfig": {}, "State": {"Running": True, "Health": {"Status": "healthy"}}}
        self.image = {"Id": self.image_id, "Size": 10}
        self.calls = []

    def docker(self, command, *args):
        self.calls.append(args)
        if args[0] == "container":
            return b"companion-server\n"
        if args[0] == "inspect":
            return json.dumps([self.inspect]).encode()
        if args[:2] == ("image", "inspect"):
            return json.dumps([self.image]).encode()
        if args[:2] == ("image", "load"):
            return b"loaded"
        raise AssertionError(args)

    def save(self, args, **kwargs):
        self.calls.append(tuple(args))
        self.assertEqual(args[1:3], ["image", "save"])
        kwargs["stdout"].write(self.tar)
        return subprocess.CompletedProcess(args, 0)

    def create_server(self):
        with patch.object(release, "docker", self.docker), patch.object(release.subprocess, "run", self.save):
            return release.archive_server(self.root, self.server, "docker", "companion-server", "incoming")

    def create_web(self):
        return release.archive_web(self.root, self.site, self.association, "incoming")

    def test_server_saves_exact_running_identity_config_and_private_permissions(self):
        archive = self.create_server()
        manifest = release.verify(archive)
        self.assertEqual(manifest["image_id"], self.image_id)
        self.assertEqual(manifest["format"], "docker-save")
        self.assertFalse(manifest["database_included"])
        self.assertEqual((self.root / "latest").read_text().strip(), archive.name)
        self.assertEqual(archive.stat().st_mode & 0o777, 0o700)
        for path in archive.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(any("stop" in call or "restart" in call for call in self.calls))

    def test_containerd_oci_index_and_compressed_layers_are_verified(self):
        self.image_id, self.tar = saved_oci_image()
        self.inspect["Image"] = self.image_id
        self.image["Id"] = self.image_id
        archive = self.create_server()
        self.assertEqual(release.verify(archive)["image_id"], self.image_id)

    def test_oci_identity_outside_saved_index_rejected(self):
        self.image_id, self.tar = saved_oci_image()
        self.inspect["Image"] = "sha256:"+"0"*64
        with self.assertRaises(release.ArchiveError):
            self.create_server()
        self.assertFalse((self.root / "latest").exists())

    def test_corrupted_oci_blob_rejected(self):
        self.image_id, self.tar = saved_oci_image()
        self.inspect["Image"] = self.image_id
        self.tar = self.tar.replace(b'"schemaVersion": 2', b'"schemaVersion": 3')
        with self.assertRaises(release.ArchiveError):
            self.create_server()

    def test_save_failure_keeps_previous_pointer_and_configuration(self):
        previous = self.create_server()
        original = (self.server / ".env").read_bytes()
        with patch.object(release, "docker", self.docker), patch.object(release.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 23)), self.assertRaises(release.ArchiveError):
            release.archive_server(self.root, self.server, "docker", "companion-server", "bad")
        self.assertEqual((self.root / "latest").read_text().strip(), previous.name)
        self.assertEqual((self.server / ".env").read_bytes(), original)
        self.assertFalse(list(self.root.glob(".partial-*")))

    def test_disk_full_fails_before_save_or_production_changes(self):
        with patch.object(release, "docker", self.docker), patch.object(release.shutil, "disk_usage",
                return_value=type("Disk", (), {"free": 0})()), patch.object(release.subprocess, "run") as save:
            with self.assertRaises(release.ArchiveError):
                release.archive_server(self.root, self.server, "docker", "companion-server", "bad")
            save.assert_not_called()
        self.assertFalse((self.root / "latest").exists())

    def test_unhealthy_container_is_not_archived(self):
        self.inspect["State"]["Health"]["Status"] = "unhealthy"
        with self.assertRaises(release.ArchiveError):
            self.create_server()

    def test_daemon_failure_cannot_skip_as_first_deployment(self):
        (self.server / ".env").unlink()
        with patch.object(release.shutil, "which", return_value="/bin/docker"), patch.object(release,
                "docker", side_effect=release.ArchiveError("Docker operation failed")):
            with self.assertRaises(release.ArchiveError):
                release.archive_server(self.root, self.server, "docker", "companion-server", "new", True)

    def test_empty_first_deployment_can_skip(self):
        (self.server / ".env").unlink()
        with patch.object(release.shutil, "which", return_value=None):
            self.assertIsNone(release.archive_server(self.root, self.server, "docker", "companion-server", "new", True))
        self.assertIsNone(release.archive_web(self.root, self.base / "missing", self.association, "new", True))

    def test_container_change_blocks_archive_publication(self):
        calls = 0

        def changed(command, *args):
            nonlocal calls
            if args[0] == "inspect":
                calls += 1
                if calls == 2:
                    self.inspect["Config"] = {"Env": ["NEW=synthetic"]}
            return self.docker(command, *args)

        with patch.object(release, "docker", changed), patch.object(release.subprocess, "run", self.save):
            with self.assertRaises(release.ArchiveError):
                release.archive_server(self.root, self.server, "docker", "companion-server", "new")
        self.assertFalse((self.root / "latest").exists())

    def test_corrupt_file_and_incomplete_inventory_rejected_before_load(self):
        archive = self.create_server()
        (archive / ".env").write_text("tampered")
        with patch.object(release, "docker") as load, self.assertRaises(release.ArchiveError):
            release.load_image(archive, "docker")
        load.assert_not_called()
        manifest = json.loads((archive / "manifest.json").read_text())
        manifest["files"].pop(".env")
        release.write_json(archive / "manifest.json", manifest)
        with self.assertRaises(release.ArchiveError):
            release.verify(archive)

    def test_image_identity_and_layers_rejected_even_with_updated_outer_hash(self):
        archive = self.create_server()
        manifest = json.loads((archive / "manifest.json").read_text())
        manifest["image_id"] = "sha256:" + "0" * 64
        release.write_json(archive / "manifest.json", manifest)
        with self.assertRaises(release.ArchiveError):
            release.verify(archive)
        manifest["image_id"] = self.image_id
        release.write_json(archive / "manifest.json", manifest)
        with tarfile.open(archive / "image.tar.gz", "w:gz") as image:
            self.tar = self.tar.replace(b"synthetic layer", b"corrupted layer")
            with tarfile.open(fileobj=io.BytesIO(self.tar)) as source:
                for member in source:
                    image.addfile(member, source.extractfile(member))
        manifest["files"]["image.tar.gz"] = release.file_digest(archive / "image.tar.gz")
        release.write_json(archive / "manifest.json", manifest)
        with self.assertRaises(release.ArchiveError):
            release.verify(archive)

    def test_load_verified_image_and_reject_loaded_identity_mismatch(self):
        archive = self.create_server()
        with patch.object(release, "docker", self.docker):
            self.assertEqual(release.load_image(archive, "docker")["image_id"], self.image_id)
            self.image["Id"] = "sha256:" + "0" * 64
            with self.assertRaises(release.ArchiveError):
                release.load_image(archive, "docker")

    def test_web_restore_verifies_content_and_keeps_previous(self):
        archive = self.create_web()
        (self.site / "index.html").write_text("synthetic v2")
        self.association.write_text("v2")
        release.restore_web(archive, self.site, self.association)
        self.assertEqual((self.site / "index.html").read_text(), "synthetic v1")
        self.assertEqual(self.association.read_text(), '{"synthetic":true}')
        self.assertEqual((self.site / "assets").stat().st_mode & 0o777, 0o755)
        previous = next(self.base.glob("dist.before-restore-*"))
        self.assertEqual((previous / "index.html").read_text(), "synthetic v2")

    def test_web_restore_install_failure_rolls_back_both_paths(self):
        archive = self.create_web()
        (self.site / "index.html").write_text("synthetic v2")
        self.association.write_text("v2")
        replace = os.replace

        def fail(source, target):
            if Path(target) == self.association and Path(source).name.startswith("tmp"):
                raise OSError("synthetic installation failure")
            return replace(source, target)

        with patch.object(release.os, "replace", fail), self.assertRaises(OSError):
            release.restore_web(archive, self.site, self.association)
        self.assertEqual((self.site / "index.html").read_text(), "synthetic v2")
        self.assertEqual(self.association.read_text(), "v2")

    def test_web_rejects_symlinks_and_tampered_archive(self):
        archive = self.create_web()
        (archive / "dist.tar.gz").write_bytes(b"truncated")
        with self.assertRaises(release.ArchiveError):
            release.restore_web(archive, self.site, self.association)
        self.assertEqual((self.site / "index.html").read_text(), "synthetic v1")
        (self.site / "link").symlink_to(self.association)
        with self.assertRaises(release.ArchiveError):
            self.create_web()

    def test_unsafe_tar_path_is_rejected(self):
        path = self.base / "unsafe.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("../index.html")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        with self.assertRaises(release.ArchiveError):
            release.validate_static(path, {})

    def test_retention_keeps_window_minimum_and_latest(self):
        paths = [self.create_web() for _ in range(6)]
        now = dt.datetime.now(dt.timezone.utc)
        for index, path in enumerate(paths):
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["created_at"] = (now - dt.timedelta(days=40-index)).isoformat()
            release.write_json(path / "manifest.json", manifest)
        (self.root / "latest").write_text(paths[0].name)
        release.prune(self.root)
        self.assertEqual({p.name for p in self.root.iterdir() if p.is_dir()},
                         {p.name for p in [paths[0], *paths[-3:]]})
        recent = self.create_web()
        release.prune(self.root)
        self.assertTrue(recent.exists())
        with self.assertRaises(release.ArchiveError):
            release.prune(self.root, keep_days=1)

    def test_corrupt_archive_blocks_all_retention_deletions(self):
        paths = [self.create_web() for _ in range(4)]
        (paths[0] / "dist.tar.gz").write_bytes(b"bad")
        with self.assertRaises(release.ArchiveError):
            release.prune(self.root)
        self.assertTrue(all(p.exists() for p in paths))

    def test_cli_failure_never_logs_private_configuration(self):
        archive = self.create_server()
        (archive / "container.json").write_text("synthetic-do-not-log")
        result = subprocess.run([os.sys.executable, str(ROOT / "scripts/release_archive.py"),
            "verify", "--archive", str(archive)], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("synthetic-do-not-log", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
