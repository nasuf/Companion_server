"""Reuse existing evaluators without running their service-connected CLIs."""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch


def grade_durative(case, reply):
    """Hard lexical direction rule; semantic state correctness stays unmeasured."""
    from evals.durative_state.judge import build_prompt

    return {
        "violations": [s for s in case.must_not_contain if s in reply],
        "semantic_status": "not_run",
        "judge_prompt": build_prompt(
            state_line=case.state_line,
            duration_hint=case.duration_hint,
            kind=case.kind,
            message=case.message,
            reply=reply,
        ),
    }


def grade_identity(case, reply):
    from evals.persona_drift.standard import MIN_IDENTITY_ANCHORED, MAX_PERSONA_LEAK

    return {
        "anchored": all(a in reply for a in case["answer_anchors"]),
        "leak": any(s in reply for s in ("作为AI", "作为人工智能", "作为语言模型")),
        "identity_threshold": MIN_IDENTITY_ANCHORED,
        "leak_threshold": MAX_PERSONA_LEAK,
        "voice_status": "not_run",
    }


async def run(output: Path, manifest: dict):
    from app.services.llm.models import get_embedding_model
    from evals.memory_recall import run_eval as recall
    from evals.temporal_recall import run_eval as temporal
    from evals.memory_lifecycle.policy import AmvlPolicy
    from evals.durative_state.cases import CASES as STATES
    from evals.external.longmemeval import recall_at_k, all_evidence_at_k
    from .replay import embeddings

    texts = sorted(
        {s.text for s in recall.SEED_BANK}
        | {c.enhanced_query or c.query for c in recall.CASES}
        | {s.text for s in temporal.SEED_BANK}
        | {c.query for c in temporal.CASES}
    )
    vectors, digest = await embeddings(
        [{"query": texts[0], "memories": [{"text": t} for t in texts]}],
        get_embedding_model(),
        output,
    )
    if digest != manifest["embedding"]["digest"]:
        raise ValueError("Adapter embedding differs from qualified production model")

    async def embed(items):
        return [vectors[t] for t in items]

    async def embed_all(items):
        return {t: vectors[t] for t in items}

    # Recall has no built-in clock freezer; reuse the temporal evaluator's
    # compatible datetime proxy rather than silently using today's wall clock.
    with temporal._freeze_time_for_ranking():
        r = await recall.evaluate_cases(embed)
    with patch.object(temporal, "_embed_all", embed_all):
        t = await temporal.run()
    policy = AmvlPolicy()
    simulations = []
    for protected in (False, True):
        for days in (30, 90, 180, 365):
            state = policy.step(
                policy.initial(0.7, 2, 0, protected=protected), days, False, False
            )
            simulations.append({"protected": protected, "days": days, **asdict(state)})
    result = {
        "embedding_digest": digest,
        "memory_recall": r,
        "temporal_recall": {
            "cases": len(t),
            "strict_hits": sum(x.hit for x in t),
            "recalled": sum(x.recalled for x in t),
            "results": [asdict(x) for x in t],
        },
        "memory_lifecycle": {
            "policy": "current cumulative production apply_usage",
            "simulations": simulations,
        },
        "durative_state": {
            "cases": len(STATES),
            "adapter": "grade_durative",
            "status": "registered_not_run",
        },
        "persona_drift": {"adapter": "grade_identity", "voice_status": "not_run"},
        "longmemeval": {
            "status": "not_run",
            "reason": "No external dataset selected; canonical turn grader reused",
            "canonical_grader_check": recall_at_k(
                ["q:s0:t0:c0", "q:s0:t0:c1"], {"q:s0:t0:c1"}, 1
            )
            == 1
            and all_evidence_at_k(["q:s0:t0:c0"], {"q:s0:t0:c1"}, 1),
        },
        "limitations": [
            "Legacy evaluator clocks/datasets are independent strata, not pooled into the 240-case baseline.",
            "Registered adapters are not live model quality evidence.",
        ],
    }
    (output / "adapters-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result
