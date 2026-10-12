"""Exercise the generated host job without installing anything on this host."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest

_spec = importlib.util.spec_from_file_location("archive_maintenance_fixtures", Path(__file__).with_name("test_release_archive.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
ROOT = fixtures.ROOT


class HostMaintenanceTests(unittest.TestCase):
    setUp = fixtures.ReleaseArchiveTests.setUp
    create_web = fixtures.ReleaseArchiveTests.create_web

    def test_generated_host_job_enforces_both_roots_and_reports_failures(self):
        parent = self.base / "archives"
        parent.mkdir()
        paths = {}
        for kind, count in (("server", 7), ("web", 12)):
            self.root = parent / kind
            paths[kind] = [self.create_web() for _ in range(count)]
        installer = (ROOT / "scripts/install_release_archive_maintenance.sh").read_text()
        wrapper = installer.split("<<'ROTATE'\n", 1)[1].split("\nROTATE\n", 1)[0]
        wrapper = wrapper.replace("/app/companion-release-archives", str(parent)).replace(
            "/usr/local/lib/companion/release_archive.py", str(ROOT / "scripts/release_archive.py"))
        commands = self.base / "bin"
        commands.mkdir()
        log = self.base / "logger.txt"
        for name, script in {"ionice": '#!/bin/sh\nshift 2\nexec "$@"\n',
                            "logger": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_LOG"\n'}.items():
            file = commands / name
            file.write_text(script)
            file.chmod(0o755)
        env = dict(os.environ, PATH=str(commands)+os.pathsep+os.environ["PATH"], TEST_LOG=str(log))
        result = subprocess.run(["sh"], input=wrapper, env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(p.exists() for p in paths["server"]), 5)
        self.assertEqual(sum(p.exists() for p in paths["web"]), 10)
        status = json.loads((parent / "web/.retention-status.json").read_text())
        self.assertEqual(status["max_bytes"], 268435456)
        self.root = parent / "server"
        self.create_web()  # Force another rotation; no-op checks need no full validation.
        (paths["server"][-1] / "dist.tar.gz").write_bytes(b"corrupt")
        result = subprocess.run(["sh"], input=wrapper, env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertIn("rotation FAILED: server", log.read_text())
        self.assertIn("rotation ok: web", log.read_text())
        self.assertEqual(sum(p.exists() for p in paths["server"]), 5)

    def test_install_rejects_privileged_or_injected_owners_before_sudo(self):
        installer = ROOT / "scripts/install_release_archive_maintenance.sh"
        for owner in ("root", "bad;whoami", "with space"):
            result = subprocess.run(["bash", str(installer), owner], text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unprivileged owner", result.stderr)
        syntax = subprocess.run(["bash", "-n", str(installer)], capture_output=True)
        self.assertEqual(syntax.returncode, 0)

    def test_deployment_and_cron_share_explicit_limits(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
        maintenance = (ROOT / "scripts/deploy_host_maintenance.sh").read_text()
        installer = (ROOT / "scripts/install_release_archive_maintenance.sh").read_text()
        self.assertIn("--max-archives 5 --max-bytes 4294967296", workflow)
        self.assertIn("--max-archives 5 --max-bytes 4294967296", maintenance)
        self.assertIn("server) count=5; bytes=4294967296", installer)
        self.assertIn("web) count=10; bytes=268435456", installer)
        self.assertIn("17 * * * * %s", installer)
        self.assertIn("install_release_archive_maintenance.sh", maintenance)


if __name__ == "__main__":
    unittest.main()
