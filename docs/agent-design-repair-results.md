# Agent reliability repair verification — 2026-09-06

The companion agent's reliability changes were checked using the same
[50-case catalog](../benchmarks/agent_design/CASES.md). Thirty-seven cases have
executable adapters: 29 use DeepSeek V4 Flash with thinking disabled, and eight
use fixed responses to exercise the real runtime at reproducible timing and
token boundaries. Thirteen cases remain unsupported, including all ten A2A
protocol cases. Skills are outside this batch.

## Full runs and interpretation

Each supported case ran once per snapshot. The fixtures, oracle and budgets
were retained after the baseline's documented offline scoring correction.

| Snapshot | Mechanism passes | Failures | Unassessed | Unsupported | Real API calls | Real tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, after offline oracle review | 26 | 8 | 3 | 13 | 95 | 75,055 |
| First repair | 30 | 4 | 3 | 13 | 89 | 69,282 |
| Second repair | 32 | 2 | 3 | 13 | 90 | 71,464 |
| Third repair, latest full run | 31 | 3 | 3 | 13 | 92 | 72,263 |

These are single-run observations, not stable success rates or an architecture
ablation. Multiple mechanisms changed between snapshots, and model behavior
varies. A mechanism pass can include correctly refusing or terminating an
impossible task; it does not always mean business success. Fixed-response token
counts are excluded from real API usage.

The third repair still failed TOOL-07 (pagination incomplete), STATE-06
(compensation claimed without the required tool effect), and CONTEXT-04
(cross-turn task incomplete and 62 tokens over its 6,000-token allowance).
The three unassessed cases lack either the intended fault trigger or complete
mechanism coverage. Unsupported and unassessed cases are never counted as
passes.

## Final budget patch: separate targeted run

After the latest full run, DeepSeek input estimation and error usage accounting
changed again. Four preselected deterministic cases, BUDGET-04/05/06/08, passed.
The one preselected live case, CONTEXT-04, still failed to finish its task.

- CONTEXT-04 used five API calls and 5,378 / 6,000 tokens. The runtime ledger,
  model spans and API records agreed on all 5,378 tokens; there was no overshoot,
  unknown usage or pending reservation.
- The model read the wrong configuration twice. With 622 tokens remaining,
  the next input and minimum output reservation did not fit, so the agent
  stopped before sending another request. The read-only constraint held, but
  the business task did not finish.
- All five targeted traces closed normally: 82 spans, no structural diagnostics.
  A separate offline regression covers generation-length errors with known
  provider usage. This live rerun did not reproduce that error branch.

The targeted results are not combined with the earlier full run to produce a
new success rate. No full run of the final source snapshot is claimed.

## Input-count calibration and accounting

The companion implementation now uses pinned official DeepSeek V4 encoding and
tokenizer assets, with an explicit margin of `max(16, ceil(raw_tokens * 0.01))`.
Recounting the third run's 92 saved inputs found 79 underestimates with the old
JSON-character estimate. Official raw counts matched 68 requests, underestimated
23 by at most eight tokens, and overestimated one by 14. Adding the margin
removed underestimation within this sample; the largest overestimate was 30.

This is retrospective calibration, not held-out validation or a strict future
billing guarantee. It covers non-thinking requests from one backend fingerprint,
with actual input sizes of 340–2,695 tokens. The companion project preserves
asset revisions, hashes, the MIT notice and explicit fallback metadata. See the
[official encoder documentation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/encoding/README.md).

The evaluation adapter also preserves valid usage when a provider returns a
generation error. Failed model spans retain that usage; unknown usage stays
unknown instead of silently becoming zero. Per-request output caps are passed
through wrappers without changing shared provider configuration.

## Reproduction and publication scope

The final offline suites passed 691 tests in this evaluation repository and
700 in the companion harness. The evaluation suite uses no paid model calls.
Live commands and trace coverage are documented in
[the execution guide](agent-design-cases-and-tracing.md).

Run manifests, per-call API records, local HTML viewers and full traces remain
in ignored local result directories. This repository publishes the code,
fixtures, source/license declarations and this aggregate summary. Generate
new results in a fresh output directory; do not overwrite historical runs or
present an older snapshot's scores as a new run of current code.
