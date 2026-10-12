"""Real synthetic archive rotation, including safety and failure boundaries."""
import json
import importlib.util
from pathlib import Path
import os
import subprocess
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location("archive_capacity_fixtures", Path(__file__).with_name("test_release_archive.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

ROOT = fixtures.ROOT
release = fixtures.release


class CapacityTests(unittest.TestCase):
    setUp = fixtures.ReleaseArchiveTests.setUp
    create_web = fixtures.ReleaseArchiveTests.create_web
    create_server = fixtures.ReleaseArchiveTests.create_server
    docker = fixtures.ReleaseArchiveTests.docker
    save = fixtures.ReleaseArchiveTests.save

    def rotate(self, **kwargs):
        return release.prune(self.root, max_archives=kwargs.pop("max_archives", 5),
                             max_bytes=kwargs.pop("max_bytes", 4 * 1024**3), **kwargs)

    def size(self, path):
        return release.archive_inventory(path)[1]

    def test_recent_dense_releases_are_bounded_by_count(self):
        paths = [self.create_web() for _ in range(12)]
        result = self.rotate()
        self.assertEqual(result["removed_count"], 7)
        self.assertEqual({p for p in paths if p.exists()}, set(paths[-5:]))
        self.assertEqual((self.root / "latest").read_text().strip(), paths[-1].name)
        self.assertEqual(json.loads((self.root / ".retention-status.json").read_text())["status"], "ok")
        self.assertEqual(self.rotate()["removed_count"], 0)

    def test_noop_capacity_check_does_not_scan_image_payloads(self):
        self.create_web()
        with patch.object(release, "verify") as verify:
            result = self.rotate()
        verify.assert_not_called()
        self.assertEqual(result["verified_count"], 0)
        self.assertEqual(result["verification"], "not_required_no_deletions")

    def test_cold_location_metadata_is_preserved_but_links_are_rejected(self):
        for _ in range(6):
            self.create_web()
        index = self.root / "cold-archive-locations.json"
        index.write_text('{"cold":"private historical location"}')
        self.rotate()
        self.assertEqual(index.read_text(), '{"cold":"private historical location"}')
        index.unlink()
        index.symlink_to(self.association)
        with self.assertRaises(release.ArchiveError):
            self.rotate()

    def test_byte_limit_overrides_count_and_keeps_three(self):
        paths = [self.create_web() for _ in range(8)]
        limit = sum(self.size(p) for p in paths[-3:])
        result = self.rotate(max_bytes=limit)
        self.assertEqual(result["retained_count"], 3)
        self.assertLessEqual(result["retained_bytes"], limit)
        self.assertEqual({p for p in paths if p.exists()}, set(paths[-3:]))

    def test_oversized_protected_set_fails_without_deletion(self):
        paths = [self.create_web() for _ in range(8)]
        with self.assertRaisesRegex(release.ArchiveError, "Protected"):
            self.rotate(max_bytes=1)
        self.assertTrue(all(p.exists() for p in paths))
        status = json.loads((self.root / ".retention-status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertIsNone(status["retained_bytes"])

    def test_old_latest_is_protected_alongside_newest_three(self):
        paths = [self.create_web() for _ in range(8)]
        (self.root / "latest").write_text(paths[0].name)
        result = self.rotate(max_archives=4)
        self.assertEqual(result["retained_count"], 4)
        self.assertEqual({p for p in paths if p.exists()}, {paths[0], *paths[-3:]})
        with self.assertRaisesRegex(release.ArchiveError, "Protected"):
            self.rotate(max_archives=3)

    def test_empty_and_first_deploy_sets_do_not_invent_three_backups(self):
        self.root.mkdir()
        self.assertEqual(self.rotate()["retained_count"], 0)
        one = self.create_web()
        result = self.rotate()
        self.assertEqual(result["retained_count"], 1)
        self.assertTrue(one.exists())

    def test_missing_or_traversal_pointer_fails_without_deletion(self):
        paths = [self.create_web() for _ in range(6)]
        pointer = self.root / "latest"
        for value in ("../outside", "missing", ""):
            pointer.write_text(value)
            with self.assertRaises(release.ArchiveError):
                self.rotate()
            self.assertTrue(all(p.exists() for p in paths))
        pointer.unlink()
        with self.assertRaises(release.ArchiveError):
            self.rotate()

    def test_symlink_pointer_and_archive_never_touch_outside(self):
        paths = [self.create_web() for _ in range(6)]
        outside = self.base / "outside"
        outside.write_text("protected")
        pointer = self.root / "latest"
        pointer.unlink()
        pointer.symlink_to(outside)
        with self.assertRaises(release.ArchiveError):
            self.rotate()
        pointer.unlink()
        pointer.write_text(paths[-1].name)
        (paths[0] / "dist.tar.gz").unlink()
        (paths[0] / "dist.tar.gz").symlink_to(outside)
        with self.assertRaises(release.ArchiveError):
            self.rotate()
        self.assertTrue(all(p.exists() for p in paths))
        self.assertEqual(outside.read_text(), "protected")

    def test_bad_inventory_or_timestamp_aborts_before_any_deletion(self):
        paths = [self.create_web() for _ in range(6)]
        extra = paths[0] / "unexpected"
        extra.mkdir()
        with self.assertRaises(release.ArchiveError):
            self.rotate()
        extra.rmdir()
        manifest = json.loads((paths[0] / "manifest.json").read_text())
        manifest["created_at"] = "2026-10-12T00:00:00"
        release.write_json(paths[0] / "manifest.json", manifest)
        with self.assertRaises(release.ArchiveError):
            self.rotate()
        self.assertTrue(all(p.exists() for p in paths))

    def test_corrupt_retained_image_blocks_every_deletion(self):
        paths = [self.create_server() for _ in range(6)]
        (paths[-1] / "image.tar.gz").write_bytes(b"corrupt")
        with self.assertRaises(release.ArchiveError):
            self.rotate()
        self.assertTrue(all(p.exists() for p in paths))

    def test_discarded_payload_does_not_require_expensive_layer_validation(self):
        paths = [self.create_server() for _ in range(6)]
        (paths[0] / "image.tar.gz").write_bytes(b"corrupt obsolete payload")
        original = release.verify
        checked = []

        def verify(path):
            checked.append(path)
            return original(path)

        with patch.object(release, "verify", verify):
            result = self.rotate()
        self.assertEqual(set(checked), set(paths[-5:]))
        self.assertEqual(result["removed_count"], 1)
        self.assertFalse(paths[0].exists())

    def test_dry_run_verifies_but_does_not_delete_or_replace_status(self):
        paths = [self.create_web() for _ in range(7)]
        status = self.root / ".retention-status.json"
        status.write_text("previous")
        result = self.rotate(dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["planned_removed_count"], 2)
        self.assertEqual(result["removed_count"], 0)
        self.assertTrue(all(p.exists() for p in paths))
        self.assertEqual(status.read_text(), "previous")

    def test_partial_archives_require_inspection_not_blind_cleanup(self):
        paths = [self.create_web() for _ in range(6)]
        (self.root / ".partial-interrupted").mkdir()
        with self.assertRaisesRegex(release.ArchiveError, "Incomplete"):
            self.rotate()
        self.assertTrue(all(p.exists() for p in paths))

    def test_status_symlink_is_atomically_replaced_without_following(self):
        self.create_web()
        outside = self.base / "outside"
        outside.write_text("protected")
        status = self.root / ".retention-status.json"
        status.symlink_to(outside)
        self.rotate()
        self.assertFalse(status.is_symlink())
        self.assertEqual(outside.read_text(), "protected")
        self.assertEqual(status.stat().st_mode & 0o777, 0o600)

    def test_both_limits_and_minimum_are_mandatory(self):
        self.create_web()
        for options in ({"max_archives": 5}, {"max_bytes": 500},
                        {"max_archives": 2, "max_bytes": 500},
                        {"max_archives": 5, "max_bytes": -1}):
            with self.assertRaises(release.ArchiveError):
                release.prune(self.root, **options)

    def test_cli_reports_counts_without_private_config(self):
        for _ in range(6):
            self.create_server()
        result = subprocess.run([os.sys.executable, str(ROOT / "scripts/release_archive.py"),
            "prune", "--root", str(self.root), "--max-archives", "5", "--max-bytes", str(4*1024**3),
            "--dry-run"], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[0])["planned_removed_count"], 1)
        self.assertNotIn("synthetic-do-not-log", result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "posix", "flock requires POSIX")
    def test_process_lock_prevents_rotation_during_archive_publication(self):
        paths = [self.create_web() for _ in range(6)]
        process = None
        try:
            with release.locked(self.root):
                process = subprocess.Popen([os.sys.executable, str(ROOT / "scripts/release_archive.py"),
                    "prune", "--root", str(self.root), "--max-archives", "5", "--max-bytes", str(4*1024**3)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.communicate(timeout=0.2)
                self.assertTrue(all(p.exists() for p in paths))
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout.splitlines()[0])["removed_count"], 1)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
