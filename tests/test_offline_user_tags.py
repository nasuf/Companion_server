import pytest

from app.services.offline import repository as offline_repo
from app.services.offline.user_tags import derive_user_tags


def test_derive_user_tags_converts_memories_to_profile_labels():
    rows = [
        {
            "summary": "用户喜欢歌曲《The Gypsy Song》，想与AI一起听",
            "content": "用户喜欢歌曲《The Gypsy Song》，想与AI一起听",
            "main_category": "偏好",
            "sub_category": "审美爱好",
        },
        {
            "summary": "用户自称个人开发者",
            "content": "用户自称个人开发者",
            "main_category": "身份",
            "sub_category": "其他",
        },
        {
            "summary": "用户选择 Renpy 引擎配合 Codex 制作视觉小说",
            "content": "用户选择 Renpy 引擎配合 Codex 制作视觉小说",
            "main_category": "偏好",
            "sub_category": "其他",
        },
        {
            "summary": "职业与经济",
            "content": "职业与经济",
            "main_category": "身份",
            "sub_category": "职业与经济",
        },
        {
            "summary": "用户目前跟伴侣处于冷战状态，对方不理他",
            "content": "用户目前跟伴侣处于冷战状态，对方不理他",
            "main_category": "生活",
            "sub_category": "人际",
        },
    ]

    assert derive_user_tags(rows, limit=9) == [
        "独立游戏创作",
        "音乐爱好者",
        "个人开发者",
        "AI工具玩家",
    ]


def test_derive_user_tags_uses_colorful_placeholder_when_not_enough_real_tags():
    rows = [
        {
            "summary": "用户喜欢吃菠萝",
            "content": "用户喜欢吃菠萝",
            "main_category": "偏好",
            "sub_category": "饮食喜好",
        }
    ]

    assert derive_user_tags(rows, limit=5) == ["菠萝爱好者"]


@pytest.mark.asyncio
async def test_list_user_tags_reads_user_memories_only(monkeypatch):
    captured_sql = ""

    async def fake_query_raw(*args, **kwargs):
        nonlocal captured_sql
        captured_sql = args[0]
        return [
            {
                "summary": "用户喜欢周末去看小型艺术展",
                "content": "用户喜欢周末去看小型艺术展",
                "main_category": "偏好",
                "sub_category": "审美爱好",
            }
        ]

    monkeypatch.setattr(offline_repo.db, "query_raw", fake_query_raw)

    assert await offline_repo.list_user_tags("user-1", "workspace-1", limit=5) == [
        "艺术展爱好者",
    ]
    assert "FROM memories_user" in captured_sql
    assert "memories_ai" not in captured_sql
