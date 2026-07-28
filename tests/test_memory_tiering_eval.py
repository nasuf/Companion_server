"""分层评测的非 LLM 部分守卫.

评测本身要真实判定数据才能跑, 但分组判据和显著性检验是纯函数 —— 这些出错会让
结论悄悄反向, 所以单独钉住. 尤其是分组: 把缺 provenance 的行默认归进"建号人设",
就会凭空放大那一组的样本并稀释它的有用率, 而输出看起来完全正常.
"""

from __future__ import annotations

from evals.memory_tiering.run_eval import GROUPS, _permutation_p, _useful_rate


def _row(level: int, provenance: str | None, verdict: str) -> dict:
    row = {"level": level, "verdict": verdict}
    if provenance is not None:
        row["provenance"] = provenance
    return row


def _group(name: str):
    return next(select for label, select in GROUPS if label == name)


def test_init_persona_group_only_takes_creation_time_rows():
    select = _group("L1 · 建号人设")
    assert select(_row(1, "init", "有用"))
    assert not select(_row(1, "daily_summary", "有用"))
    assert not select(_row(2, "init", "有用"))


def test_rows_without_provenance_never_count_as_persona():
    """缺元数据的行必须落在人设组之外 —— 猜一个来源会把结论算歪."""
    assert not _group("L1 · 建号人设")(_row(1, None, "有用"))


def test_groups_do_not_overlap():
    """一条记忆只能进一组, 否则总数对不上, 各组占比也失真."""
    for row in (_row(1, "init", "有用"), _row(1, "daily_summary", "有用"),
                _row(2, "init", "无用"), _row(3, "init", "无用")):
        hits = [label for label, select in GROUPS if select(row)]
        assert len(hits) == 1, f"{row} 命中了 {hits}"


def test_useful_rate_counts_only_the_top_verdict():
    rows = [{"verdict": v} for v in ("有用", "沾边", "无用", "有用")]
    assert _useful_rate(rows, "verdict") == 0.5
    assert _useful_rate([], "verdict") == 0.0


def test_permutation_test_detects_a_real_gap():
    poor = [{"verdict": "无用"} for _ in range(90)] + [{"verdict": "有用"}] * 10
    good = [{"verdict": "有用"} for _ in range(60)] + [{"verdict": "无用"}] * 40
    assert _permutation_p(poor, good, "verdict") < 0.01


def test_permutation_test_stays_quiet_when_groups_match():
    a = [{"verdict": "有用"}] * 20 + [{"verdict": "无用"}] * 20
    b = [{"verdict": "有用"}] * 20 + [{"verdict": "无用"}] * 20
    assert _permutation_p(a, b, "verdict") > 0.05


def test_permutation_test_is_one_sided():
    """只问"b 是否更好". 双侧会把"人设反而更有用"也报成显著, 那是相反的结论."""
    better = [{"verdict": "有用"}] * 30 + [{"verdict": "无用"}] * 10
    worse = [{"verdict": "有用"}] * 5 + [{"verdict": "无用"}] * 35
    assert _permutation_p(better, worse, "verdict") > 0.5
