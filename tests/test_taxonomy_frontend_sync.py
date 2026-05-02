"""CI guard: frontend `Companion_web/src/utils.ts` taxonomy constants must
include every sub-category that backend `app/services/memory/taxonomy.py`
defines for L1 (admin UI must be able to render every backend category).

Catches drift like Round 4: backend added "提醒"/"重要日期"/"其他特殊事件"
to `_LIFE_BASE`, frontend `LIFE_BASE` was forgotten → admin couldn't see
those memories.

Direction is `backend ⊆ frontend`. Frontend may carry extras (legacy aliases,
forward-compat) — that's fine; backend additions being silently dropped
from the UI is what we want to catch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UTILS_TS = _REPO_ROOT / "Companion_web" / "src" / "utils.ts"


def _parse_ts_array(source: str, name: str) -> list[str]:
    """Extract `const <name> = [ "a", "b", ... ];` array literal from TS source.

    Limitations: only supports flat string-literal arrays (no spreads, no
    interpolation). The constants we care about are flat lists.
    """
    pattern = rf"const\s+{re.escape(name)}\s*=\s*\[(.*?)\]\s*;"
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        raise AssertionError(
            f"Could not locate `const {name} = [...]` in {_UTILS_TS}. "
            "Frontend taxonomy refactored? Update test parser."
        )
    body = m.group(1)
    # Strip line + block comments before extracting strings, so canonical
    # names that appear in commentary (e.g. "提醒") don't pollute the result.
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    # Capture both "..." and '...' quoted strings.
    return [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', body)]


@pytest.fixture(scope="module")
def frontend_source() -> str:
    if not _UTILS_TS.exists():
        pytest.skip(f"Frontend not present at {_UTILS_TS}")
    return _UTILS_TS.read_text(encoding="utf-8")


def _assert_backend_subset(
    *,
    name: str,
    backend: tuple[str, ...] | list[str],
    frontend: list[str],
) -> None:
    missing = [s for s in backend if s not in frontend]
    assert not missing, (
        f"Frontend `{name}` in utils.ts is missing backend sub-categories: "
        f"{missing}. Add them to keep the admin UI in sync with "
        f"app/services/memory/taxonomy.py."
    )


def test_frontend_life_base_matches_backend(frontend_source: str) -> None:
    from app.services.memory.taxonomy import _LIFE_BASE
    _assert_backend_subset(
        name="LIFE_BASE",
        backend=_LIFE_BASE,
        frontend=_parse_ts_array(frontend_source, "LIFE_BASE"),
    )


def test_frontend_identity_full_matches_backend(frontend_source: str) -> None:
    from app.services.memory.taxonomy import _IDENTITY_FULL
    _assert_backend_subset(
        name="IDENTITY_FULL",
        backend=_IDENTITY_FULL,
        frontend=_parse_ts_array(frontend_source, "IDENTITY_FULL"),
    )


def test_frontend_preference_l1_matches_backend(frontend_source: str) -> None:
    from app.services.memory.taxonomy import _PREFERENCE_L1
    _assert_backend_subset(
        name="PREFERENCE_L1",
        backend=_PREFERENCE_L1,
        frontend=_parse_ts_array(frontend_source, "PREFERENCE_L1"),
    )


def test_frontend_emotion_full_matches_backend(frontend_source: str) -> None:
    from app.services.memory.taxonomy import _EMOTION_FULL
    _assert_backend_subset(
        name="EMOTION_FULL",
        backend=_EMOTION_FULL,
        frontend=_parse_ts_array(frontend_source, "EMOTION_FULL"),
    )


def test_frontend_thought_l1_matches_backend(frontend_source: str) -> None:
    from app.services.memory.taxonomy import _THOUGHT_L1
    _assert_backend_subset(
        name="THOUGHT_L1",
        backend=_THOUGHT_L1,
        frontend=_parse_ts_array(frontend_source, "THOUGHT_L1"),
    )
