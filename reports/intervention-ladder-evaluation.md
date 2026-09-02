# Intervention Ladder Evaluation: what fixes silent step-skipping, and what only appears to

**System under test:** the real `ReActAgent` in `agent-harness-from-scratch`,
LLM `deepseek-chat`, over that repo's 100-tool synthetic kit.
**Toolkit used:** `agent_eval.stats` (`bootstrap_ci`, `paired_bootstrap_test`).
**Benchmark data & statistics:** `benchmarks/intervention_ladder/`.
**Sweep driver:** lives in the sibling repo
(`examples/chain_task_gen.py`, `examples/intervention_ladder_test.py`) — that
is where the tools, the agent and the conditions are defined. This repo owns
the measurement.

---

## 1. Objective

The sibling repo's tool-scaling experiments established that tool *count*
(6→100) and tool *description length* (6.5x inflation) cause no measurable
degradation, and that the real variable is **chain length**: past five steps
the agent silently skips steps it judges redundant while still producing the
right answer. A hierarchical main-agent + specialist-subagent architecture was
built as the mitigation, reported as 57% → 67%.

Two things about that were never checked, and both turned out to matter more
than the mitigation itself:

1. Nobody tried simply **telling the model not to skip steps** — despite the
   shipped system prompt ending with *"when you have enough information,
   respond with a final answer and do not call any more tools."*
2. Nobody checked whether the hierarchical result was **attributable to the
   architecture at all**.

This evaluation answers both, and in doing so retires the 57% → 67% number.

## 2. Why the original benchmark could not answer this

The original evidence was 21 hand-written 5-step tasks, one run each. Two
independent problems, both found empirically here:

**Run-to-run variance swamps the effect.** Re-running the *unchanged* baseline
on those 21 tasks scored **71.4%** against the 57.1% recorded three weeks
earlier — same prompt, same model, same tasks; three tasks flipped to passing,
none regressed. A 14-point swing with nothing changed. The claimed 57% → 67%
improvement is inside that band.

**No headroom.** At a 71.4% baseline only 6 of 21 tasks are broken, each worth
4.76 points, so the maximum observable improvement is six tasks.

Everything below therefore uses a new, larger, harder task set. This is also
why the 5-step pilot dump is retained but marked *do not cite*.

## 3. Measurement design

### 3.1 Tasks: generated, trigger-enriched, self-labelling

8-step chains are produced by a random walk over a tool **type graph**
(`number → number`, `string → number`, `date → bool`, …), so each step may only
consume what the previous one produced. Ground truth is obtained by
**executing the chain** — every tool in the kit is a pure function, so the
expected sequence is whatever the walk took and the expected answer is whatever
the last call actually returned. Labelling cost is zero and labels cannot drift
from tool behaviour.

Chains are enriched toward five "redundancy pressures" reverse-engineered from
the original failure cases — a no-op round, an inferable boolean, a small
mental-arithmetic statistic, the same tool twice, an identity round trip. All
60 tasks carry at least one (T1 78%, T5 73%, T4 62%, T3 32%, T2 18%).

Two calibration decisions materially changed the corpus and are worth
recording, because both were wrong on the first pass:

- Restricting no-op rounds to *integer* values starved T1 to 35%, because any
  unit conversion earlier in a chain produces a float. Generalising to "round
  to the value's own current precision" — rounding 9.999975 to six places is
  exactly as much a no-op as rounding 70 to zero — restored T1 to 78%, close to
  its 14/21 dominance in the original hand-written set. A corpus whose trigger
  mix does not match the phenomenon measures a different phenomenon.
- Uniform sampling of trigger moves let two-step moves (round trips, doubled
  tools) eat the step budget; T5 reached 82% before weights were added.

### 3.2 Metrics, and why more than one is needed

| Metric | What it catches |
|---|---|
| Exact tool-sequence match | strict, all-or-nothing correctness |
| **Step recall** (multiset overlap ÷ required) | *skipping* specifically; one dropped step in an 8-step chain scores 0.875, not 0 — far lower variance |
| Final-answer accuracy | whether the intervention broke the actual product |
| Failure kind: `skip` / `extra` / `other` | which direction the error goes |
| Extra calls | over-correction, directly |

Separating `skip` from `extra` is the design decision that carries §5.2: an
intervention that says "never skip a step" can trade one failure for the other
and look perfectly neutral on exact match.

### 3.3 Statistical design

k=3 trials per task per condition. **The unit of analysis is a task, not a
run** — the per-task pass rate across trials is a continuous score in
{0, ⅓, ⅔, 1}, and conditions are compared as matched pairs over the same 60
tasks. Treating the 180 runs of a condition as independent samples would
triple-count every task and understate the intervals.

Paired bootstrap, 10,000 resamples of the per-task deltas, plus percentile
bootstrap CIs — both from `agent_eval.stats`, which exists so that results in
this project arrive as intervals rather than point estimates.

## 4. Localising the failure before intervening

For the hierarchical conditions it was necessary to know *which layer* loses
the step: does the coordinator fail to delegate a sub-task, or does a
correctly-delegated specialist not execute it? The runner records each run's
full delegation log (group, sub-task text, tools actually called), so a missing
required tool can be attributed to a layer.

On a 15-task diagnostic: **4 of 4 missing tools were dropped inside a
correctly-delegated specialist; the coordinator under-delegated zero times.**

That has two consequences. First, the specialist prompt — not the coordinator
prompt — is the only place a prompt intervention can act, and worked examples
naming concrete tools cannot go in the coordinator prompt at all, since it
holds only five `delegate_*` tools. Second, and more importantly, it sent us to
read the specialist prompt, where the confound in §5.3 was sitting in plain
text.

## 5. Findings

Full tables: `benchmarks/intervention_ladder/RESULTS.md`.

### 5.1 Only demonstrations move the headline metric

`fewshot` is the sole condition with a significant exact-match gain:
**0.567 vs 0.389, +17.8pp, p=0.0066**. It wins on four of five trigger classes.

### 5.2 A rule stops the skipping and creates the opposite failure

`instruction` did exactly what it was told — step recall 0.838 → **0.972**,
`skip` failures 55 → **6**, both p<0.0001 — and bought **nothing** on exact
match (−0.6pp, p=0.86), because the skips were replaced one-for-one by
unrequested calls (185 → 393 extra calls; `extra` failures 26 → 81). Final
answers collapsed from 78.9% to **49.4%**.

An abstract rule is an undirected command. The model complies by calling more
tools everywhere, including where it should not. The demonstrations in §5.1
have statistically indistinguishable step recall (0.974) but half the
over-correction, because a worked example communicates the upper bound as well
as the lower one.

### 5.3 The architecture's benefit was its prompt

The shipped specialist prompt already contained an anti-skipping instruction:

> "call every tool call the task actually requires, even one whose result looks
> like it wouldn't change (e.g. rounding a number that is already at the target
> precision); do not skip a step just because you are confident you already
> know the answer."

So the hierarchical arm was never a pure architectural intervention, and no
previous measurement separated the two. `hierarchical_bare` deletes that one
sentence and changes nothing else:

| | exact match | step recall | skip failures |
|---|---|---|---|
| flat baseline | 0.389 | 0.838 | 55 |
| **hierarchical_bare** | 0.333 (p=0.35) | **0.867 (p=0.227)** | **52** |
| hierarchical | 0.439 | 0.949 (p<0.0001) | 20 |

Stripped of the sentence, routing does **not** significantly reduce skipping,
and its exact match falls *below* doing nothing at all. The entire measured
benefit of shrinking the coordinator's tool surface from 100 tools to five came
from one sentence in a subagent prompt. The architecture on its own buys seven
delegations per run and no measurable accuracy.

### 5.4 Routing and demonstrations do not stack

Putting the worked example where §4 says steps are dropped —
`hierarchical_fewshot` — scores **0.411 (+0.022, p=0.78)**, indistinguishable
from both plain `hierarchical` and the baseline, with step recall a hair
*lower* (0.943 vs 0.949).

The failure composition explains it: under routing, skipping is already handled
(20–22 `skip` failures), and what remains is `other` (36–39) and `extra` (45) —
coordination errors, including runs that delegate 20 times to complete an
8-step chain. A specialist-level prompt cannot fix a coordination failure. The
same demonstration worth +17.8pp flat is worth nothing here because routing
moved the bottleneck rather than removing it.

### 5.5 The headline metric alone would have misled

`fewshot` wins exact match by 17.8 points while being 13.3 points *worse* than
doing nothing on final-answer accuracy (p=0.024); `instruction` is 29.4 points
worse (p<0.0001). Only the hierarchical arms leave answers intact.

At five steps this dimension had no resolving power whatsoever — answer
accuracy was 100% in every condition, which is precisely why the original
experiment's conclusions did not survive contact with longer chains.

## 6. Threats to validity

- **One model, one sweep.** All numbers are `deepseek-chat` at k=3.
- **Synthetic tasks.** Generated chains are occasionally absurd. That is the
  redundancy pressure under test, not natural traffic.
- **Few-shot examples are pattern-adjacent.** They avoid the task set's tools
  but do demonstrate the T1 and T5 patterns, the two largest trigger classes,
  so part of the `fewshot` gain on those classes is in-distribution.
- **`instruction` varies two things** — removing the early-stop clause and
  adding the completeness rule. The separating ablation is cheap and unrun.
- **`hierarchical_bare` removes one sentence, not the concept.** It shows the
  shipped prompt carries the benefit; it does not prove that no specialist
  prompt could make routing pay off.
- **The coordinator prompt was never tuned.** It was held fixed across all
  three hierarchical arms, so the coordination failures §5.4 blames were never
  themselves targeted.
- **The 5-step pilot is k=1 on 21 tasks** and is retained only as the
  motivation for this benchmark. Its own conclusions reversed here.

## 7. What this changes downstream

1. **57% → 67% should not be cited.** It is inside the baseline's run-to-run
   band, and §5.3 shows the mechanism credited for it was a prompt sentence.
2. **Any future "beat the hierarchical baseline" claim must name which
   specialist prompt it means.**
3. **The problem is now well posed.** Every intervention that states a rule
   stops the skipping; none stops it without inducing unrequested calls or
   degrading answers. What is missing is something that judges whether a
   *specific* step is required at a *specific* point — which a static rule
   cannot express.
