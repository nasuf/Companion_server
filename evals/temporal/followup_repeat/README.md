# TIME06 follow-up repeat cases

Frozen before the candidate was scored (sha256 recorded in the release evidence).
`development.json` was used for tuning; `holdout.json` and `holdout2.json` are
held out. Each case gives dated history (minutes before a fixed `NOW`), the
user's current message and one generated optional question; `expected` is
whether that question may be sent. Baseline is the TIME04 lexical repeat check
over the 3-hour session; the candidate is `followup_decision.decide_followup`.

    python -m evals.temporal.followup_repeat.run_eval evals/temporal/followup_repeat/holdout.json
