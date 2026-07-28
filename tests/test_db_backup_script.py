"""数据库备份脚本的性质守卫.

2026-07-28: 一条过宽的 DELETE 清掉 1691 行变更日志, 排查恢复手段时才发现整台机器
没有任何备份 —— 无定时 dump, archive_mode=off, 无 WAL 归档. 当天丢的是可再生的
日志; 下一次若是 memories_* 就没有退路.

脚本本身没法在 CI 里跑 (要真数据库和 docker), 所以这里守的是几条"错了会让备份
在需要它的那天才失效"的性质. 备份的失效方式很特别: 平时完全没有症状。
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "companion-db-backup.sh"
DEPLOY = ROOT / ".github" / "workflows" / "deploy.yml"


@pytest.fixture(scope="module")
def script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_uses_custom_format_for_selective_restore(script):
    """-Fc 才能只恢复单张表。事故通常是一张表被误改, 全库回滚会把其它表这段时间
    的正常写入一起抹掉 —— 那是第二次事故。"""
    assert "-Fc" in script


def test_verifies_the_dump_can_be_read(script):
    """没验证过能读的备份不算备份。截断或损坏要在当天暴露, 不能等到真出事。"""
    assert "pg_restore --list" in script
    assert "TABLE DATA" in script


def test_writes_to_a_partial_file_first(script):
    """中途被打断时不能留下一个看起来完好的残缺备份 —— 那比没有更危险。"""
    assert ".partial" in script
    assert "mv " in script


def test_failure_is_loud(script):
    """静默失败的备份比没有备份更糟: 你以为有。"""
    assert "user.err" in script or "-p user.err" in script
    assert "exit 1" in script


def test_rotation_is_bounded_but_not_too_short(script):
    """留太少等于没有 —— 今天这种事故隔了几小时才发现。"""
    assert "KEEP_DAYS=14" in script
    assert "-mtime" in script


def test_rotation_runs_after_the_new_backup_is_confirmed(script):
    """先删旧的再发现今天没备成, 就是自己把退路清空了。"""
    verify_at = script.index("pg_restore --list")
    delete_at = script.index("-mtime")
    assert verify_at < delete_at, "轮转必须排在校验之后"


def test_deploy_reinstalls_it(script):
    """只装在机器上不算数 —— 重建实例就没了。"""
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert "/usr/local/bin/companion-db-backup.sh" in deploy
    assert "/etc/cron.d/companion-db-backup" in deploy


def test_embedded_copy_matches_the_source(script):
    """deploy.yml 里内嵌了脚本副本, 两处会漂移。改了源文件忘了同步, 服务器上跑的
    还是旧版, 而本地看到的是新版 —— 这种不一致只有在恢复失败那天才会发现。"""
    deploy = DEPLOY.read_text(encoding="utf-8")
    body = deploy.split("<<'BACKUP'\n", 1)[1].split("\n            BACKUP\n", 1)[0]
    embedded = "\n".join(
        line[12:] if line.startswith(" " * 12) else line
        for line in body.splitlines()
    )
    assert embedded.strip() == script.strip(), (
        "scripts/companion-db-backup.sh 与 deploy.yml 内嵌的副本不一致 —— "
        "改了一处要同步另一处，否则服务器上跑的是旧版本。"
    )
