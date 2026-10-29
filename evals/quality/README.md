# Administrative quality evaluation

Daily sampling and prompt/route regression run through one SQL-leased executor.
The renderer runs in a fresh process with no provider keys and an audit hook
denying network and subprocess access. Only the parent can call a provider,
after reserving the cost ceiling. Nightly GitHub Actions uses dedicated EVAL
keys and the same server-side budget via repository-bound OIDC. These keys are
not supplied to the production deployment or a renderer.

The quick bank selects 30 scenes. Full runs retain all existing scenario-bank
adapters, including eight two-turn search scenarios. Critical scenes repeat
three times. Real replies are appended between turns; future turns never reach
the judge. Long-companion replay exercises reactive turns, not its proactive
scheduler. Human-confirmed cases replay exact captured model inputs only; they
do not reconstruct historical business state. Missing exact inputs are unknown.

## Search and billing

Search replay uses the real Ark Responses route and rendered search prompt.
Evaluation requests explicitly cap one tool round, one keyword, ten results,
public-web source, and output tokens. Product search settings are untouched.
This bounded profile is included in reports and corpus versioning; it does not
claim to test production's multi-keyword/multi-round search limits.

For `doubao-seed-character-260628`, reserve two model inferences at the full
128k context and the upper tier (input 1.2/output 6 CNY per million tokens),
plus 0.004 CNY/search, with no assumed cache or free quota. The model uses the
implicit cache; no explicit cache is created. Usage must prove exactly one
public-web search. Inputs above the approved 32k band remain unqualified and
retain their reservation; the upper tier is a budget safety bound only.
Normal settlement uses the immutable production price snapshot plus tool cost.

Official references checked 2026-10-29:

- https://docs.volcengine.com/docs/ark/model-list?lang=en
- https://docs.volcengine.com/docs/ark/model-pricing?lang=en
- https://docs.volcengine.com/docs/ark/web-search?lang=en

No partial run is a pass. The first complete, judged run records a baseline;
it is not a claim that all replies are correct. Subsequent matching banks report
critical failure deltas and the configured noncritical noise band. Budgets,
interruptions, missing evidence and unconfigured routes remain visible. Gates
warn and notify; they never roll back or block product publication automatically.
