"""Conservative read constraints from committed, source-bound fact changes.

These only remove incompatible evidence; they never manufacture a validity
interval. In particular, a day-precision move describes a possible transition
window, not a midnight boundary.
"""

from app.services.time_domain.timestamps import exclusive_end, parse_time


def historical_change_filter(memory_id, period, changes):
    relevant = [
        row
        for row in changes
        if memory_id in {row["old_memory_id"], row["new_memory_id"]}
        and row["kind"] in {"transition", "correction"}
    ]
    if not relevant:
        return True, "no_change_constraint", ()
    identities = tuple(sorted(row["id"] for row in relevant))
    for row in relevant:
        outgoing = row["old_memory_id"] == memory_id
        if outgoing and row["kind"] == "correction":
            # A retracted mistake is not a formerly true fact. Losing its source
            # must not undo the committed retraction and resurrect the mistake.
            return False, "retrospective_correction", identities
        if not row.get("source_available"):
            return False, "change_source_unavailable", identities
        if row["kind"] != "transition" or period is None:
            continue
        semantics = row.get("semantics") or {}
        boundary = semantics.get("boundary") or {}
        start, end = parse_time(boundary.get("start")), parse_time(
            boundary.get("end_exclusive")
        )
        requested_end = exclusive_end(period.end, period.end_boundary)
        if start and end and start < end:
            if outgoing and requested_end > start:
                return False, "old_fact_after_or_inside_transition", identities
            if not outgoing and period.start < end:
                return False, "new_fact_before_or_inside_transition", identities
        elif outgoing:
            observed = parse_time(semantics.get("statement_at"))
            if observed and period.start >= observed:
                return False, "old_fact_after_confirmed_change_observation", identities
        else:
            # 边界未知的新事实: 旧说法被说出时它显然还没成立, 所问时段整个在那之前
            # 就不能拿新事实回答 (TIME07)。
            old_observed = parse_time(row.get("old_statement_at"))
            if old_observed and requested_end <= old_observed:
                return False, "new_fact_before_old_statement", identities
    return True, "compatible_with_change", identities
