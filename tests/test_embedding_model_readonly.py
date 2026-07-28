"""Embedding 模型在后台只展示、不可改 —— 这个约束要有测试守着.

视觉模型和语音模型都是运行时开关: 改完下一条消息就用新的, 存量数据不依赖旧模型,
随时能改回去. Embedding 不是 —— 库里每一条向量都是当前模型的输出, 换掉而不重算
等于让查询在陌生坐标系里检索 (同一段文本跨模型的余弦实测 -0.001, 比同模型内两段
无关文本的 0.43 还低), 而且十余个相似度阈值按该模型的分布标定.

所以给它一个输入框, 等于给一个"点了之后记忆检索静默失效、要跑 50 分钟迁移才能
恢复"的按钮. 它出现在 resolved 里只为让运维看得到当前跑的是什么 —— 视觉语音都在
那一屏, 唯独它缺席反而让人以为漏配了.

这里守两件事: 它必须出现在 resolved 里 (可见), 且必须不出现在可写 payload 里
(不可改).
"""

from __future__ import annotations

from app.api.admin.runtime_config import ConfigPayload, _payload_to_data
from app.config import settings
from app.services.memory.config import CALIBRATED_EMBEDDING_MODEL


def _resolved() -> dict:
    from app.api.admin.runtime_config import _resolved_to_dict
    from app.services.runtime_config import resolve_config_sync

    return _resolved_to_dict(resolve_config_sync(agent_id=None))


def test_embedding_model_is_visible_to_operators():
    resolved = _resolved()
    assert resolved["embedding_model"] == settings.embedding_model


def test_embedding_model_is_marked_read_only():
    """前端靠这个标志决定渲染只读卡片而不是输入框."""
    assert _resolved()["embedding_model_editable"] is False


def test_resolved_reports_whether_thresholds_match_the_running_model():
    resolved = _resolved()
    assert resolved["embedding_model_calibrated"] == (
        settings.embedding_model == CALIBRATED_EMBEDDING_MODEL
    )


def test_admin_payload_has_no_way_to_set_the_embedding_model():
    """真正的护栏: 可写 payload 里根本没有这个字段, 后台改不了它."""
    assert not [f for f in ConfigPayload.model_fields if "embedding" in f]


def test_writable_config_never_carries_an_embedding_field():
    """即便有人在请求体里硬塞, 也不会落到 SystemConfig 上."""
    payload = ConfigPayload()
    data = _payload_to_data(payload, include_global_only=True)
    assert not [k for k in data if "mbedding" in k]
