# Time domain ownership

CONV01.03 consolidates existing temporal behavior; it does not certify event
validity storage, durable topic state, complete historical counts or checkpoint
recovery. Those remain TIME03–TIME11 scopes.

- `timestamps`: UTC timestamp normalization and half-open boundaries.
- `clock`: current execution clock and the single process-local NTP calibration;
  existing holiday/status adapters remain here without new startup IO.
- `parser`, `binding`, `operations`, `query`: deterministic language/date parsing,
  source spans, bounded query intent and evidence eligibility. No model calls.
- `context`, `limits`: invocation-owned TurnClock, verified source interval,
  bounded active dialogue and prompt/diagnostic projections.
- `timeline`: bounded memory evidence projection; mention dates are not events.
- `followup`: current session optional-question policy; not durable topic state.
- `recall`, `guard`: existing model-backed answer/check adapters. Pure parsers
  do not import these modules; importing the package does not initialize them.
- `expression`: existing natural-language time rendering.
- `reminders`: pure target resolution shared by direct and pending reminders.
  Storage, scheduling, leases and delivery stay in their owning domains.

The graph builds one trusted TurnClock and passes it into both reminder paths.
Source time interprets the expression; fresh execution time checks expiry. An
unknown graph clock never falls back to processing time. Legacy standalone
callers retain their explicitly diagnosed synchronous receipt/processing contract.
Both endpoints of an aggregate interval must resolve the same target(s).

The original twelve module paths are module-object aliases, not copies or star
exports. This preserves mutable NTP state and historical patch/import consumers.
Runtime imports use canonical paths; a regression test rejects new imports back
to aliases. Any eventual alias deletion needs the CONV01.05 inventory/approval.

No prompt versions, database schema, persisted Trace keys or client protocols are
renamed. New direct-reminder diagnostics are additive. Reminder memory/trigger
atomicity remains outside this structural consolidation; partial creation must be
reported explicitly, never as complete success or an invitation to retry all.
