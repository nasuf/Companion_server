"""部署必须让"删除"也能传播到生产.

`scp-action` 只覆盖不删除。所以从 git 删掉的文件会永远留在 VPS 的源码目录里, 并被
烘进之后每一个镜像 —— 生产上跑着的代码始终是"当前代码 + 历史上所有删过的代码"。

2026-07-29 实测残留 9 个文件。最危险的一类是重命名: 一个模块从
`memory/reflection/signals.py` 改名到 `memory/behaviour_signals.py` 之后, 两个路径
在生产上同时存在, 任何还引用旧路径的地方拿到的是改名前的版本 —— 而且不会报任何错。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_DEPLOY_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


def _jobs() -> dict:
    return yaml.safe_load(_DEPLOY_YML.read_text())["jobs"]


def _deploy_steps() -> list[dict]:
    for job in _jobs().values():
        steps = job.get("steps") or []
        if any("scp-action" in str(s.get("uses", "")) for s in steps):
            return steps
    raise AssertionError("找不到含 scp 的部署 job")


def _copied_paths() -> list[str]:
    for step in _deploy_steps():
        if "scp-action" in str(step.get("uses", "")):
            return [p.strip() for p in step["with"]["source"].split(",")]
    raise AssertionError("找不到 scp 步骤")


def test_stale_sources_are_cleared_before_copying():
    """清理必须在复制之前 —— 反过来会把刚传上去的代码删掉。"""
    steps = _deploy_steps()
    clear_index = next(
        (i for i, s in enumerate(steps) if "rm -rf" in str(s.get("with", {}).get("script", ""))),
        None,
    )
    copy_index = next(
        i for i, s in enumerate(steps) if "scp-action" in str(s.get("uses", ""))
    )
    assert clear_index is not None, (
        "部署里没有清理步骤 —— 从 git 删掉的文件会永远留在生产上"
    )
    assert clear_index < copy_index, "清理步骤排在了复制之后"


@pytest.mark.parametrize("directory", ["app", "jobs", "prisma", "scripts"])
def test_every_copied_code_directory_is_also_cleared(directory):
    """清理范围与复制范围必须一致。漏掉一个目录, 那个目录里的删除就永远不生效。"""
    steps = _deploy_steps()
    script = next(
        str(s.get("with", {}).get("script", "")) for s in steps
        if "rm -rf" in str(s.get("with", {}).get("script", ""))
    )
    assert directory in _copied_paths(), f"{directory} 不在 scp 范围里, 这条测试该更新"
    assert re.search(rf"rm -rf[^\n]*\b{directory}\b", script), (
        f"{directory} 会被复制但不会被清理 —— 其中删除的文件会残留"
    )


def test_clearing_does_not_touch_anything_outside_the_copied_set():
    """.env / secrets / 备份都不在 scp 范围里, 清掉它们不会被下一步补回来。"""
    steps = _deploy_steps()
    script = next(
        str(s.get("with", {}).get("script", "")) for s in steps
        if "rm -rf" in str(s.get("with", {}).get("script", ""))
    )
    removed = re.findall(r"rm -rf\s+(.+)", script)
    targets = [t for line in removed for t in line.split()]
    copied = set(_copied_paths())
    for target in targets:
        assert target in copied, (
            f"清理步骤删了 {target}, 但它不在复制范围里 —— 删掉就补不回来了"
        )


def test_clearing_is_scoped_to_the_project_directory():
    """rm -rf 必须在 cd 之后执行, 且 cd 失败时不能继续往下删。"""
    steps = _deploy_steps()
    script = next(
        str(s.get("with", {}).get("script", "")) for s in steps
        if "rm -rf" in str(s.get("with", {}).get("script", ""))
    )
    lines = [line.strip() for line in script.strip().splitlines() if line.strip()]
    cd_line = next((i for i, line in enumerate(lines) if line.startswith("cd ")), None)
    rm_line = next(i for i, line in enumerate(lines) if line.startswith("rm -rf"))
    assert cd_line is not None and cd_line < rm_line, "rm 之前没有 cd"
    assert "||" in lines[cd_line], "cd 失败时没有中止 —— 会在错误的目录里执行 rm"
    assert not any(line.startswith("rm -rf /") for line in lines), "出现了绝对路径 rm"
