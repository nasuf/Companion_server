"""数据库备份的查看 / 触发 / 删除.

删除接口拿用户传来的字符串去碰文件系统, 是这套功能里唯一能造成真实破坏的地方,
所以路径穿越单独重点守。防护用**白名单**而不是过滤 "../" —— 黑名单式过滤总有
漏网写法 (URL 编码、绝对路径、符号链接), 白名单没有。

另一类要守的是"不可用时的表现": 备份目录没挂载或镜像缺 pg_dump 时, 接口必须
明确报缺什么, 而不是返回空列表让人以为一份备份都没做过, 更不能抛 500。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.ops import db_backup


@pytest.fixture()
def backup_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(db_backup, "BACKUP_DIR", tmp_path)
    return tmp_path


def _make(dir_: Path, name: str, size: int = 1024) -> Path:
    path = dir_ / name
    path.write_bytes(b"x" * size)
    return path


class TestPathTraversal:
    @pytest.mark.parametrize("name", [
        "../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "/etc/passwd",
        "companion-20260728-1508.dump/../../../etc/passwd",
        "companion-.dump",
        "companion-20260728-1508.dump.bak",
        "arbitrary.dump",
        "",
        ".",
    ])
    def test_rejects_anything_not_matching_the_backup_name(self, backup_dir, name):
        with pytest.raises(ValueError):
            db_backup.delete_backup(name)

    def test_symlink_pointing_outside_is_refused(self, backup_dir, tmp_path):
        """名字合法不代表指向合法 —— 目录里放一个指向别处的符号链接就能绕过纯
        文件名校验, 所以解析后还要确认仍落在备份目录里."""
        outside = tmp_path.parent / "outside.dump"
        outside.write_bytes(b"secret")
        link = backup_dir / "companion-20260728-1508.dump"
        link.symlink_to(outside)
        _make(backup_dir, "companion-20260727-0300.dump")
        with pytest.raises(ValueError):
            db_backup.delete_backup(link.name)
        assert outside.exists(), "符号链接指向的目标不能被删掉"


class TestDelete:
    def test_deletes_a_real_backup(self, backup_dir):
        _make(backup_dir, "companion-20260728-1508.dump")
        _make(backup_dir, "companion-20260727-0300.dump")
        db_backup.delete_backup("companion-20260728-1508.dump")
        assert {b.name for b in db_backup.list_backups()} == {
            "companion-20260727-0300.dump"
        }

    def test_refuses_to_delete_the_only_remaining_backup(self, backup_dir):
        """误点一下就没有任何退路, 代价不对称 —— 要清空得走文件系统."""
        _make(backup_dir, "companion-20260728-1508.dump")
        with pytest.raises(ValueError, match="最后一份"):
            db_backup.delete_backup("companion-20260728-1508.dump")
        assert db_backup.list_backups()

    def test_missing_file_reports_not_found(self, backup_dir):
        _make(backup_dir, "companion-20260727-0300.dump")
        with pytest.raises(FileNotFoundError):
            db_backup.delete_backup("companion-20260728-1508.dump")


class TestListing:
    def test_newest_first(self, backup_dir):
        for name in ("companion-20260726-0300.dump",
                     "companion-20260728-0300.dump",
                     "companion-20260727-0300.dump"):
            _make(backup_dir, name)
        assert [b.name[:22] for b in db_backup.list_backups()] == [
            "companion-20260728-030", "companion-20260727-030", "companion-20260726-030",
        ]

    def test_ignores_files_that_are_not_backups(self, backup_dir):
        _make(backup_dir, "companion-20260728-0300.dump")
        _make(backup_dir, "companion-20260728-0300.dump.partial")
        _make(backup_dir, "notes.txt")
        assert len(db_backup.list_backups()) == 1

    def test_partial_files_never_appear(self, backup_dir):
        """.partial 是写到一半的文件, 出现在列表里会让人以为有一份可用备份."""
        _make(backup_dir, "companion-20260728-0300.dump.partial")
        assert db_backup.list_backups() == []


class TestAvailability:
    def test_reports_missing_directory_instead_of_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(db_backup, "BACKUP_DIR", tmp_path / "nope")
        state = db_backup.availability()
        assert state["available"] is False
        assert any("未挂载" in r for r in state["reasons"])

    def test_unreadable_directory_does_not_raise(self, tmp_path, monkeypatch):
        """目录存在但没权限时 is_dir() 抛 PermissionError —— 报告可用性的函数
        自己崩掉, 界面会拿到 500 而不是原因说明."""
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        blocked.chmod(0o000)
        monkeypatch.setattr(db_backup, "BACKUP_DIR", blocked / "inner")
        try:
            state = db_backup.availability()
            assert state["available"] is False
            assert state["reasons"]
            assert db_backup.list_backups() == []
        finally:
            blocked.chmod(0o755)

    def test_reports_missing_pg_dump(self, backup_dir, monkeypatch):
        monkeypatch.setattr(db_backup.shutil, "which", lambda _: None)
        state = db_backup.availability()
        assert state["available"] is False
        assert any("pg_dump" in r for r in state["reasons"])


@pytest.mark.asyncio
async def test_create_refuses_when_environment_is_incomplete(tmp_path, monkeypatch):
    """环境不满足时必须明确失败, 不能产出一个空文件让人以为备份成功了."""
    monkeypatch.setattr(db_backup, "BACKUP_DIR", tmp_path / "missing")
    with pytest.raises(RuntimeError):
        await db_backup.create_backup()
