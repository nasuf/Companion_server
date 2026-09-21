"""识图会话端到端（in-memory repo 驱动 recognize_on_photo 全编排）。

验证 spec §4.8/§4.9 在真实引擎下的连锁行为：首命中必出→满 3 关闭、同指纹去重、
已触发条件不复用、概率未过不产出且条件保持未触发、未命中走暗示分支。
"""

from unittest.mock import AsyncMock

from app.services.offline import recognition


class _FakeRepo:
    def __init__(self):
        # 物品带 category（PM #3/#4：拍摄「物品」大类），识图主体按名/类目匹配。
        self.conditions = [
            {"id": "c1", "short_name": "杯子", "category": "餐具"},
            {"id": "c2", "short_name": "天空", "category": "天空"},
            {"id": "c3", "short_name": "小路", "category": "道路"},
        ]
        self.triggered: set[str] = set()
        self.fragments: list[dict] = []
        self.fingerprints: set[str] = set()

    async def list_untriggered_conditions(self, rid):
        return [c for c in self.conditions if c["id"] not in self.triggered]

    async def list_all_conditions(self, rid):
        return list(self.conditions)

    async def count_fragments(self, rid):
        return len(self.fragments)

    async def fragment_fingerprint_exists(self, rid, fp):
        return fp in self.fingerprints

    async def get_prewritten_fragment(self, condition_id, tier):
        return "预写回忆"

    async def mark_condition_triggered(self, cid):
        self.triggered.add(cid)

    async def mark_media_fragment_cover(self, mid, fp):
        pass

    async def create_fragment(self, **kw):
        frag = {
            "id": f"f{len(self.fragments)}",
            "tier": kw["tier"],
            "text": kw["text"],
            "lead_in": kw.get("lead_in"),
        }
        self.fragments.append(frag)
        self.fingerprints.add(kw["content_fingerprint"])
        return frag


def _install(monkeypatch, *, produce_random=0.0, subjects_queue):
    fake = _FakeRepo()
    monkeypatch.setattr(recognition, "repo", fake)
    # 口语化交付确定化；写回 AI 记忆与 WS 推送置空避免触碰 DB/网络。
    monkeypatch.setattr(recognition, "_verbalize", AsyncMock(return_value="思绪正文"))
    monkeypatch.setattr(recognition, "remember_offline_fragment", lambda **kw: None)
    monkeypatch.setattr(recognition, "_handle_miss", AsyncMock(return_value=None))
    monkeypatch.setattr(
        recognition.chat_emit, "emit_thought_fragment", AsyncMock(return_value="m1")
    )
    # 概率门槛确定化：random() 恒为 produce_random。
    monkeypatch.setattr(recognition.random, "random", lambda: produce_random)
    queue = list(subjects_queue)

    async def _detect(_desc):
        return queue.pop(0) if queue else []

    monkeypatch.setattr(recognition, "_detect_subjects", _detect)
    return fake


def _subj(type_name):
    return [{"type": type_name, "confidence": 0.9}]


def _activity():
    return {
        "id": "a1", "user_id": "u1", "workspace_id": "w1",
        "agent_id": "ag1", "conversation_id": "c1",
        "status": "accepted", "reached": True, "title": "植物园",
    }


async def _fire(desc="描述", mid="m"):
    return await recognition.recognize_on_photo(
        activity=_activity(), ctx={"conversation_id": "c1", "agent_id": "ag1"},
        photo_description=desc, media_id=mid, source_message_id="msg",
    )


async def test_full_session_ladder_and_cap(monkeypatch):
    fake = _install(
        monkeypatch,
        produce_random=0.0,  # 概率恒过（<p），凸显阶梯的产出/关闭
        subjects_queue=[
            _subj("杯子"), _subj("天空"), _subj("小路"), _subj("杯子"),
        ],
    )
    assert await _fire(mid="m1") is not None  # 0 条→首命中 100% 必出
    assert await _fire(mid="m2") is not None  # 1 条→random 0<0.4 出
    assert await _fire(mid="m3") is not None  # 2 条→random 0<0.25 出
    assert await _fire(mid="m4") is None      # 3 条→硬上限关闭
    assert len(fake.fragments) == 3
    assert fake.triggered == {"c1", "c2", "c3"}


async def test_probability_not_passed_keeps_condition_untriggered(monkeypatch):
    fake = _install(
        monkeypatch,
        produce_random=0.9,  # 恒不过（>0.4/0.25）
        subjects_queue=[
            _subj("杯子"),  # 0 条 100% 出
            _subj("天空"),  # 1 条 0.9>0.4 不出
        ],
    )
    assert await _fire(mid="m1") is not None
    assert await _fire(mid="m2") is None
    assert len(fake.fragments) == 1
    # 概率未过：天空物品保持未触发，后续仍可命中
    assert "c2" not in fake.triggered


async def test_duplicate_fingerprint_not_reproduced(monkeypatch):
    fake = _install(
        monkeypatch,
        produce_random=0.0,
        subjects_queue=[_subj("杯子")],
    )
    # 预置「杯子」将生成的指纹 → 命中后走去重分支，不产出。
    fp = recognition.content_fingerprint(["杯子", "杯子"])
    fake.fingerprints.add(fp)
    assert await _fire(mid="m1") is None
    assert len(fake.fragments) == 0
    assert "c1" not in fake.triggered  # 未产出 → 条件保持未触发


async def test_miss_invokes_hint_branch(monkeypatch):
    fake = _install(
        monkeypatch,
        produce_random=0.0,
        subjects_queue=[_subj("汽车")],  # 无任何物品可匹配
    )
    assert await _fire(mid="m1") is None
    assert fake.fragments == []
    recognition._handle_miss.assert_awaited_once()  # 未命中 → 走暗示分支


async def test_no_conditions_no_fragment(monkeypatch):
    fake = _install(monkeypatch, subjects_queue=[])
    fake.conditions = []
    assert await _fire() is None
    assert fake.fragments == []
