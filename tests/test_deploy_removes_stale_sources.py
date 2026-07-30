"""部署必须让"删除"也能传播到生产, 且删除范围不能溢出到源码之外.

从 git 删掉的文件如果继续留在 VPS 的源码目录里, 会被烘进之后每一个镜像 —— 生产上跑
着的代码就成了"当前代码 + 历史上所有删过的代码"。2026-07-29 实测残留 9 个文件。最危险
的一类是重命名: 一个模块从 `memory/reflection/signals.py` 改名到
`memory/behaviour_signals.py` 之后, 两个路径在生产上同时存在, 任何还引用旧路径的地方
拿到的是改名前的版本 —— 而且不会报任何错。

2026-07-30 换 scp 为 rsync 之后, 保证这件事的机制从"复制前先 rm -rf"变成了传输本身的
`--delete`。**属性没变, 守的东西也就没变**, 只是钉的位置换了。同时多出一个原先不存在的
风险: `--delete` 的杀伤范围由 source 列表决定, 一旦有人把 source 写成 `.` 或给目标目录
带上尾斜杠, 它就会连 VPS 上的 .env / secrets/ 一起删 —— 那些东西不在传输范围里, 删掉
补不回来。所以下面既守"删得够", 也守"别删过头"。
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

_DEPLOY_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"

# 必须完整同步的代码目录。`--delete` 只在实际传过去的目录内部生效, 所以漏掉任何一个,
# 那个目录里的删除就永远不会传播到生产 —— 正是本文件开头那个事故的形态。
_CODE_DIRS = ["app", "jobs", "prisma", "scripts"]

_TARGET_DIR = "/app/companion-server"

# 带参数的 rsync 选项, 解析位置参数时要连它们的值一起跳过。
_OPTS_WITH_VALUE = {"-e", "--rsh", "--exclude", "--include", "--filter", "--rsync-path"}


def _steps() -> list[dict]:
    jobs = yaml.safe_load(_DEPLOY_YML.read_text())["jobs"]
    for job in jobs.values():
        steps = job.get("steps") or []
        if any("rsync" in str(s.get("run", "")) for s in steps):
            return steps
    raise AssertionError("部署里找不到 rsync 步骤 —— 源码是怎么上生产的?")


def _rsync_argv() -> list[str]:
    """把 run 块里那条反斜杠续行的 rsync 命令还原成 argv。"""
    run = next(str(s.get("run", "")) for s in _steps() if "rsync" in str(s.get("run", "")))
    joined = run.replace("\\\n", " ")
    line = next(
        stripped
        for stripped in (ln.strip() for ln in joined.splitlines())
        if stripped.startswith("rsync ")
    )
    return shlex.split(line)


def _flags() -> set[str]:
    """展开短选项组合: -azc 视为同时给了 -a / -z / -c。"""
    flags: set[str] = set()
    skip_next = False
    for arg in _rsync_argv()[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in _OPTS_WITH_VALUE:
            skip_next = True
            continue
        if arg.startswith("--"):
            flags.add(arg)
        elif arg.startswith("-"):
            flags.update(f"-{ch}" for ch in arg[1:])
    return flags


def _positionals() -> list[str]:
    positionals: list[str] = []
    skip_next = False
    for arg in _rsync_argv()[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in _OPTS_WITH_VALUE:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        positionals.append(arg)
    return positionals


def _sources() -> list[str]:
    return _positionals()[:-1]


def _destination() -> str:
    return _positionals()[-1]


def test_transfer_propagates_deletions():
    """没有 --delete, 从 git 删掉的文件就会永远留在生产上。"""
    assert "--delete" in _flags(), (
        "rsync 少了 --delete —— 删除不会传播到 VPS, 旧文件会被烘进之后每一个镜像"
    )


def test_scp_action_is_not_reintroduced():
    """scp 只覆盖不删除, 退回去等于同时丢掉删除语义和增量传输。"""
    offenders = [s.get("name") for s in _steps() if "scp-action" in str(s.get("uses", ""))]
    assert not offenders, f"这些步骤退回了 scp-action: {offenders}"


@pytest.mark.parametrize("directory", _CODE_DIRS)
def test_every_code_directory_is_synced(directory):
    """--delete 只在传过去的目录内生效, 漏一个目录那里的删除就永远不生效。"""
    assert directory in _sources(), (
        f"{directory} 不在 rsync 范围里 —— 其中被删除的文件会残留在生产上"
    )


def test_delete_cannot_reach_outside_the_synced_sources():
    """.env / secrets/ 就在目标目录下, 且不由部署产生 —— 删掉补不回来。

    rsync 的 --delete 只清理"由 source 推导出来的目录", 所以只要每个 source 都是显式
    的相对路径, 目标根目录下的其它东西就是安全的。危险写法是把整个工作区当 source
    (`.` / `./` / `*`), 那样目标根目录本身成为被同步的目录, .env 和 secrets/ 会因为
    "发送端没有"而被删除。
    """
    for source in _sources():
        assert source not in {".", "./", "*", "./*"}, (
            f"source 写成了 {source!r} —— --delete 会连目标目录下的 .env / secrets/ 一起删"
        )
        assert not source.startswith("/"), f"source {source!r} 是绝对路径, 应为仓库内相对路径"
        assert not source.endswith("/"), (
            f"source {source!r} 带尾斜杠 —— 语义变成同步目录内容而非目录本身, "
            "会把删除范围抬到目标根目录"
        )


def test_sync_target_is_the_project_directory():
    destination = _destination()
    assert destination.endswith(f":{_TARGET_DIR}/"), (
        f"rsync 目标是 {destination!r}, 期望 {_TARGET_DIR}/"
    )


def test_incremental_transfer_does_not_rely_on_mtime():
    """增量的前提是按内容比对, 不是按时间戳。

    git 不存 mtime, 每次 checkout 出来的文件时间戳都是全新的。rsync 默认用
    (size, mtime) 快速判定, 在这个前提下会认为每个文件都变了 —— 16MB 的头像会重新
    参与传输协商, 换 rsync 想省的那十分钟就白省了。删掉 -c 不会有任何报错, 只会静默
    变慢, 所以钉在这里。
    """
    flags = _flags()
    assert "-c" in flags or "--checksum" in flags, (
        "rsync 少了 --checksum —— git checkout 的 mtime 每次都变, "
        "默认的 size+mtime 判定会让增量失效且不报错"
    )
