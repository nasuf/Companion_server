"""deploy.yml 写进 .env 的默认值必须与代码默认值一致.

一个配置项一旦同时存在于代码默认和部署默认, 两处就可能分歧, 而分歧是静默的:
行为取决于 GitHub 变量有没有设过, 从代码里读不出来。

这不是假想的失败模式。嵌入模型就是这样出过事: 代码换了模型, 部署的环境变量还指
着旧的, 检索质量悄悄退化, 最后是靠启动时加告警才抓到。

这里只盯**会改变数据的**开关。普通配置项分歧了顶多行为不同, 而记忆整合会归档
原始记忆 —— 它的开关状态必须是从代码里一眼能确认的。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import Settings

_DEPLOY_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"

# (env 变量名, Settings 字段名)
_COUPLED_FLAGS = [
    ("MEMORY_CONSOLIDATION_ENABLED", "memory_consolidation_enabled"),
    ("MEMORY_CONSOLIDATION_WORKSPACES", "memory_consolidation_workspaces"),
    # 并发上限两侧必须一致: 代码默认是「没设 GH 变量时」的实际值, deploy 里的
    # fallback 是「运维看到的默认」。两者漂开的话, 调优时按看到的数推理会算错。
    ("LLM_MAX_CONCURRENCY", "llm_max_concurrency"),
    ("LLM_BACKGROUND_MAX_CONCURRENCY", "llm_background_max_concurrency"),
]


def _deploy_default(env_name: str) -> str | None:
    """从 `NAME=${{ vars.NAME || 'x' }}` 里取出那个 'x'; 没有 fallback 则返回空串。"""
    text = _DEPLOY_YML.read_text()
    match = re.search(rf"^\s*{env_name}=(.+)$", text, flags=re.M)
    if match is None:
        return None
    expression = match.group(1)
    fallback = re.search(r"\|\|\s*'([^']*)'", expression)
    return fallback.group(1) if fallback else ""


@pytest.mark.parametrize("env_name,field", _COUPLED_FLAGS)
def test_deploy_default_matches_code_default(env_name, field):
    deployed = _deploy_default(env_name)
    assert deployed is not None, (
        f"{env_name} 不在 deploy.yml 里 —— 那么它只能靠改代码 + 重新部署来调整。"
        "对会归档原始数据的任务来说, 应急开关必须能不改代码就翻。"
    )

    code_default = Settings.model_fields[field].default
    expected = (
        str(code_default).lower() if isinstance(code_default, bool) else str(code_default)
    )
    assert deployed == expected, (
        f"{env_name} 在 deploy.yml 里默认 {deployed!r}, 代码里是 {expected!r}。"
        "两处分歧时行为取决于 GitHub 变量有没有设过, 从代码读不出来。"
    )


@pytest.mark.parametrize("env_name", ["MEMORY_CONSOLIDATION_WORKSPACES"])
def test_canary_allowlist_is_actually_settable(env_name):
    """白名单存在的意义是"出事时先缩小范围"。要是只能靠改代码 + 重新部署才能设,
    那它就不是应急手段。"""
    assert f"vars.{env_name}" in _DEPLOY_YML.read_text()
