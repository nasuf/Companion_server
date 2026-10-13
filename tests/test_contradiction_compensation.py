"""No best-effort unarchive: the old fact survives every rejected replacement."""

import pytest
from app.services.memory.manual_write import MemoryWriteError
from app.services.memory.storage import repo
from tests.test_memory_evidence_postgres import origins
from tests.test_contradiction_write_postgres import confirming


async def test_embedding_failure_never_archives_old(confirming):
    f = confirming
    f.embedding.side_effect = RuntimeError("embedding unavailable")
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await f.apply()
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_duplicate_rejection_preserves_old_without_compensation(confirming):
    f = confirming
    await repo.create(userId=f.uid, workspaceId=f.wid, content="用户住在乙城", level=2)
    with pytest.raises(MemoryWriteError, match="duplicate_fact"):
        await f.apply()
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_other_singleton_blocks_replacement_without_archiving_old(confirming):
    f = confirming
    await repo.create(
        userId=f.uid,
        workspaceId=f.wid,
        content="用户现居丙城",
        level=1,
        mainCategory="身份",
        subCategory="现居地",
    )
    with pytest.raises(MemoryWriteError, match="singleton_conflict"):
        await f.apply()
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_reminder_cannot_skip_schedule_workflow(confirming):
    f = confirming
    with pytest.raises(MemoryWriteError, match="taxonomy_not_allowed"):
        await f.apply(
            analysis={
                **f.analysis,
                "new_memory_main_category": "生活",
                "new_memory_sub_category": "提醒",
            }
        )
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived
