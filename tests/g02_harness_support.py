"""G02 paired orchestration fixture; no real database or business actions.

Keep the production reply generator/prompt renderer. Only domain IO and model
responses are replaced, so differences in lazy prompts and tier routing remain
observable. Also usable by the opt-in live-model qualification runner.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.chat import orchestrator as chat, reply_generate
from tests.graph_harness_support import configure_chat


def synthetic_agent():
    return SimpleNamespace(
        id="a-1", name="小岚", age=24, gender="female", occupation="陶艺师",
        city="杭州", values={}, personality={},
    )


def configure_pair(patches):
    io = configure_chat(patches)
    from app.services.chat import intent_handlers, multi_intent
    # Handlers have their own imported DB/time aliases. Keep real handler and
    # model logic, but resolve current activity and repeat history synthetically.
    patches.setattr(intent_handlers, "db", io.db)
    patches.setattr(intent_handlers, "resolve_implicit_time", AsyncMock(
        return_value=(datetime(2026, 10, 4, 4, tzinfo=timezone.utc), "在家整理工作台"),
    ))

    def fire(coro):
        # Legacy text short-circuits schedule persistence. Eagerly complete only
        # the synthetic save mock, leaving all actual business background work
        # closed. Otherwise its awaited count would measure the fixture itself.
        if getattr(coro, "cr_frame", None) and coro.cr_frame.f_locals.get("self") is chat._save_replies:
            try:
                coro.send(None)
            except StopIteration:
                pass
        else:
            io.background.append(getattr(getattr(coro, "cr_code", None), "co_name", "mock"))
            coro.close()

    patches.setattr(chat, "_fire_background", fire)
    patches.setattr(multi_intent, "_fire_background", fire)
    from app.services import music
    from app.services.offline import module_settings
    from app.services.prompting import store

    patches.setattr(music, "get_active_co_listening", AsyncMock(return_value=None))
    patches.setattr(module_settings, "is_activity_enabled", AsyncMock(return_value=False))
    for name, value in (
        ("sample_expression_habits", []), ("get_relation_meta", {}),
        ("load_ai_mood", None), ("detect_l1_contradiction", None),
        ("get_cached_schedule", []),
    ):
        patches.setattr(chat, name, AsyncMock(return_value=value))
    patches.setattr(chat, "save_pending_contradiction", AsyncMock())
    patches.setattr(chat, "compute_reengagement_gap_seconds", lambda *_, **__: 0.0)
    patches.setattr(chat, "pick_reply_count_target", lambda *_: 2)
    patches.setattr(chat, "build_time_context", lambda: "2026-10-04 12:00，星期日")
    patches.setattr(chat, "actual_delay_seconds", lambda _: None)

    # A fresh in-memory store uses actual code defaults without touching Redis.
    class PromptCache:
        def __init__(self):
            self.values = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value, **_):
            self.values[key] = value

    cache = PromptCache()
    patches.setattr(store, "get_redis", AsyncMock(return_value=cache))
    patches.setattr(store, "db", SimpleNamespace(prompttemplate=SimpleNamespace(
        find_unique=AsyncMock(return_value=None),
    )))
    patches.setattr(store, "_enabled_local_cache", {})
    patches.setattr(chat, "_generate_reply", reply_generate.generate_reply)
    io.main_calls = []
    io.tier_calls = []

    async def main(messages, **kwargs):
        io.main_calls.append((messages, kwargs))
        return "先歇一会儿吧||今天挺累的呀[EMO:中性/20]", False

    patches.setattr(reply_generate, "_run_main_llm", main)
    for kind in ("weak", "medium", "strong", "l3"):
        async def tier(_kind=kind, **kwargs):
            io.tier_calls.append((_kind, kwargs))
            return "先歇一会儿吧||今天挺累的呀"
        patches.setattr(chat, f"_memory_{kind}_reply", tier)
    io.agent = synthetic_agent()
    io.now = datetime(2026, 10, 4, 4, tzinfo=timezone.utc)
    return io
