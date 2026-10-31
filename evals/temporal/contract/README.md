# TIME02B deterministic contract baseline

`cases.jsonl` freezes 48 authored expression families at five calendar boundaries
(240 cases). Expected answers use calendar truth, not the parser under test.
The manifest pins the file hash. Altering a case requires an explicit review and
preservation of the previous report. Pytest runs every case in the release gate.

Run with an isolated checkout without `.env`:

```
python -m evals.temporal_contract.run_eval --output /absolute/new/report.json
```

To compare a previous checkout, invoke this runner with `--source-root` pointing
at that checkout. All failed cases remain in the report; absent operation fields
are failures, not zero-valued successes. Parsing and diagnostic timing is measured
in milliseconds, on the same machine and frozen cases.

These are **deterministic boundary regressions**, not 240 independent multi-turn
model scenarios and not the future 120 development / 120 held-out qualification
set. Timing covers the pure query contract, not whole-chat latency or retrieval.
TIME03–TIME11 capabilities (fact validity, durable topic state, deduplicated counts,
duration computation, recovery and longitudinal quality) are not certified here.

Operation classification is a bounded lexical hint. `not_executed` means the
query asks for evidence that has **not** been computed by this contract. Neither
an explicit date range nor a matching candidate count proves a lifetime total.
No raw user history, credentials, production DB connection or model call is used.
