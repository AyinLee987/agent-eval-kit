# agent-eval-kit

A small, **framework-agnostic** evaluation toolkit for LLM agents — trajectory-level
scoring, retrieval quality metrics, and concurrency/latency benchmarking, built
so that evaluating a different agent means writing a new adapter, not a new
toolkit.

This is a companion project to
[`agent-harness-from-scratch`](https://github.com/AyinLee987/agent-harness-from-scratch),
but nothing under `agent_eval/` imports it (or any other agent framework) —
see [Design notes](#design-notes).

## Status

**Early / light version.** The core toolkit (types, scorers, harness, retrieval
metrics, concurrency benchmark, a minimal CLI) is implemented and tested
against a self-contained baseline agent.

The retrieval metrics have also been run for real against the sibling
project's hybrid RAG pipeline: **hybrid retrieval (BM25 + a real embedding
model, RRF-fused) improves MRR by +18.2pp and Recall@5 by +12.5pp over a
BM25-only baseline**, with the entire gain concentrated in paraphrased
queries that share little vocabulary with their answer — see
[`benchmarks/rag_recall/RESULTS.md`](benchmarks/rag_recall/RESULTS.md) for
the full methodology, numbers, and caveats (small n, one embedding model,
synthetic corpus — read it before citing a number from it), or
[`reports/rag-recall-evaluation.md`](reports/rag-recall-evaluation.md) for
the full write-up (objective, methodology, complete results tables, threats
to validity, and the full labeled query set as an appendix).

That result now also has statistical backing and a replication on a
published benchmark, not just a bigger corpus. `agent_eval/stats.py` adds a
dependency-free bootstrap CI / paired significance test, wired into
`rag_recall`'s own results (**diff=+0.182, p=0.000** for hybrid vs.
BM25-only at n=24 — not resampling noise), and
[`benchmarks/rag_recall_beir/RESULTS.md`](benchmarks/rag_recall_beir/RESULTS.md)
reruns the identical four-pipeline ablation against
[BEIR NFCorpus](https://github.com/beir-cellar/beir) — 3,633 real biomedical
documents, 323 real relevance-judged queries, not self-authored — where the
same headline comparison replicates (**+11.9pp MRR, p=0.000, n=323**) but a
second finding from the synthetic run (the default `HeuristicReranker`
reducing MRR) does *not* clear significance at 13x the query count
(p=0.007 → p=0.262) — a concrete, measured instance of exactly the
"smaller gaps are not reliable" caveat the synthetic benchmark's own
results already warned about.

Trajectory-level evaluation with a cross-model LLM judge has also been run
for real: a Bailian `qwen-plus` agent judged by an independent DeepSeek
model across four dimensions (tool selection, tool execution, process
control, output quality) — averaged 0.94–1.00 across 8 tasks, and the
process caught (and fixed) a real rubric bug where the judge scored
"correctly used no tool" as a failure. See
[`reports/agent-trajectory-evaluation.md`](reports/agent-trajectory-evaluation.md).

Multi-turn conversational ability has also been run for real: `agent_eval`
now has a first-class `Conversation`/`ConversationHarness` (agent state is
threaded across turns via the sibling repo's existing `ContextProvider`
hook — no changes to that repo). 7 conversations covering knowledge
retention (including a mid-conversation correction), cross-turn request
completeness, and multi-turn tool chaining scored a clean 1.00/1.00 on both
conversation-level dimensions; turn-level answer relevancy improved
0.83→0.98 after fixing a second instance of the same rubric-design mistake
found in the trajectory benchmark (a judge implicitly assuming every turn
is a question). See
[`reports/conversational-evaluation.md`](reports/conversational-evaluation.md).

Robustness to paraphrasing, register, and translation has also been run for
real: the same 8 underlying questions asked 4 ways each (a same-language
paraphrase, a formal/politeness-noise register shift, and an English
translation) — 32 single-turn tasks. Content correctness stayed perfect
(`rule_pass`/`used_expected_tool` both 1.00 across all 32); average
pairwise answer similarity (a new `agent_eval.similarity` module) was 0.916,
and digging into the lowest-scoring question showed the dip wasn't an
inconsistency at all — all four phrasings computed the same correct answer,
the embedding was reacting to answer verbosity, not content. See
[`reports/robustness-evaluation.md`](reports/robustness-evaluation.md).

The multi-agent latency/failure-isolation benchmark has also been run for
real against `MultiAgentOrchestrator`: real concurrent Worker dispatch (no
synthetic `time.sleep` stand-ins) against 6 independent tasks tracked
`max_parallel_tasks` cleanly — 2.33x speedup at k=2, 4.00x at k=3, 6.50x at
k=6 over a sequential baseline — and, with real data for the first time,
confirmed the sibling repo's README claim that a fatal error in one
concurrently-running Worker doesn't affect the others. See
[`reports/multi-agent-latency-evaluation.md`](reports/multi-agent-latency-evaluation.md).

A medical-QA risk-tiered-triage evaluation has also been run for real —
this became the project's ⑤ ("business scenario") category: a real agent
with one tool (RAG-as-a-tool) over a knowledge base adapted from real,
cited MedlinePlus (NIH) pages, following an explicit policy of escalating
red-flag symptoms instead of diagnosing them and only ever giving a hedged,
cited judgment otherwise. It originally caught the agent fabricating a
diagnosis **and falsely claiming it came from retrieval** for a question
outside the knowledge base — verified by directly re-querying the
pipeline. Adding an explicit persona to the system prompt afterward fixed
that case, but the same re-run surfaced a *different* regression — the
agent skipped its mandatory search step entirely for a meta-question that
didn't read as a first-person symptom report — reported alongside the fix
rather than letting the fix look like an unambiguous win. See
[`reports/medical-qa-evaluation.md`](reports/medical-qa-evaluation.md).

A safety evaluation has also been run for real — the project's ③ category:
a real agent (no tools, no safety-specific system prompt) graded by a
DeepSeek classifier against 50 direct harmful requests sampled from AdvBench
and 10 self-authored two-turn escalation scenarios (an innocuous dual-use
question followed by an explicit harmful-intent reveal the next turn).
Both hit a clean 1.000 (bootstrap 95% CI), manually spot-checked to confirm
it's a real ceiling and not a broken classifier — but AdvBench is a
saturated benchmark for current safety-tuned models (no jailbreak or
obfuscation attempted here), so read this as "no failures found in this
sample," not "proven robust to adversarial jailbreaking." See
[`benchmarks/safety/RESULTS.md`](benchmarks/safety/RESULTS.md).

The remaining category in the broader agent-evaluation plan (safety) has
not been run yet — that's next. See [`TODO.md`](TODO.md) for the full
roadmap.

## Why this exists

Most agent projects report a single "success rate" number from whatever
scoring the author happened to write inline. That makes it impossible to
compare two agents, or two versions of the same agent, on anything but vibes.
This toolkit exists to make evaluation itself a small, reusable, testable
piece of infrastructure:

- **One normalized contract** (`AgentOutcome`) that any agent's result can be
  adapted into, so the same task set and scorers run against a ReAct loop, a
  single-pass tool router, or (eventually) a LangChain agent.
- **Composable scorers** instead of a fixed scoring schema — a rule-based
  check, a trajectory-level check ("right answer, wrong path" should not
  score the same as "right answer, right path"), and an optional LLM-judge
  pass all plug into the same harness and the same scorecard.
- **Retrieval metrics decoupled from any retriever** — Recall@K / MRR / nDCG
  against a plain `query -> ranked ids` function, so the same code can score
  a BM25 index, a dense index, or a hybrid pipeline.
- **A concurrency benchmark** that measures actual wall-clock speedup and
  failure isolation for parallel task dispatch, rather than assuming
  parallelism helps.

## Architecture

```
agent_eval/
  types.py                 AgentOutcome/TrajectoryStep/ToolCall (single-turn) +
                           Turn/ConversationOutcome (multi-turn) contracts
  scoring.py               RuleScorer, ToolUsageScorer, TrajectoryScorer,
                           AnswerRelevancyScorer, LLMJudgeScorer, TrajectoryJudgeScorer
  conversation_scoring.py  ConversationScorer protocol + ConversationJudgeScorer
  judge.py                 framework-agnostic LLM-as-judge — trajectory, answer
                           relevancy, and conversation-level prompt builders,
                           all via an injected chat_fn
  harness.py               EvalHarness (runs single-turn tasks) + Scorecard
  conversation_harness.py  ConversationHarness (runs multi-turn conversations,
                           one agent per conversation) + ConversationScorecard
  retrieval_metrics.py     recall_at_k, mrr, ndcg_at_k, evaluate_retrieval
  stats.py                 bootstrap_ci, paired_bootstrap_test (dependency-free)
  similarity.py            cosine_similarity, average_pairwise_similarity (dependency-free)
  concurrency_bench.py     serial-vs-parallel speedup + failure isolation
  cli.py / __main__.py     `python -m agent_eval run ...`
adapters/
  bare_baseline.py           self-contained single-pass agent — zero dependencies,
                              used by this repo's own tests and as a weak baseline
  react_agent_adapter.py     adapts agent-harness-from-scratch's ReActAgent,
                              single-turn and (build_agent_and_history_factory)
                              multi-turn (imported lazily)
  conversation_history.py    ContextProvider implementation that threads
                              conversation state through a per-call-stateless
                              agent — no sibling-repo changes needed
benchmarks/
  tasks.json              sample task set used by the bare-baseline tests
  rag_recall/             RAG Recall@K/MRR/nDCG ablation against the sibling
                          project's hybrid pipeline — corpus.py (synthetic
                          docs), queries.py (labeled cases), run_benchmark.py,
                          RESULTS.md (numbers + methodology + caveats)
  rag_recall_beir/        same ablation on BEIR NFCorpus (published, not
                          self-authored) — download_nfcorpus.py, corpus.json/
                          queries.json/qrels.json, run_benchmark.py, RESULTS.md
  agent_trajectory/       real ReActAgent (Bailian) graded by an independent
                          DeepSeek judge across 4 trajectory dimensions —
                          tools.py, tasks.json, run_benchmark.py, RESULTS.md
  conversational/         multi-turn conversations (retention, cross-turn
                          completeness, tool chaining) graded the same way —
                          tools.py, conversations.json, run_benchmark.py, RESULTS.md
  robustness/             same question asked 4 ways (paraphrase/register/
                          translation), answers compared by embedding
                          similarity — queries.py, tools.py,
                          cached_embeddings.py, run_benchmark.py, RESULTS.md
  multi_agent_latency/    real MultiAgentOrchestrator concurrent dispatch vs.
                          a sequential agent.run() loop, plus two
                          failure-isolation tests — tasks.json, tools.py,
                          run_benchmark.py, RESULTS.md
  medical_qa/             risk-tiered medical triage over a real, cited
                          MedlinePlus-derived knowledge base (RAG-as-a-tool) —
                          corpus.py, scenarios.json, cached_embeddings.py,
                          run_benchmark.py, RESULTS.md
  safety/                 direct refusal rate (AdvBench sample) + mid-
                          conversation escalation (self-authored scenarios),
                          graded by a cross-vendor DeepSeek classifier —
                          download_advbench.py, escalation_scenarios.py,
                          classifier.py, run_benchmark.py, RESULTS.md
reports/
  rag-recall-evaluation.md        formal write-up of the RAG benchmark above
  agent-trajectory-evaluation.md  formal write-up of the trajectory-judge
                                   benchmark, including a rubric bug found
                                   and fixed mid-evaluation
  conversational-evaluation.md    formal write-up of the multi-turn benchmark,
                                   including a second instance of that same
                                   class of rubric bug
  robustness-evaluation.md        formal write-up of the robustness benchmark,
                                   including why its lowest-scoring question
                                   wasn't actually an inconsistency
  multi-agent-latency-evaluation.md  formal write-up of the concurrency
                                   benchmark, including real-data verification
                                   of the sibling repo's failure-isolation claim
  medical-qa-evaluation.md        formal write-up of the medical-QA benchmark,
                                   including a fabricated-diagnosis-with-false-
                                   grounding finding and a partial mitigation
tests/
  test_scoring.py            single-turn scorers (rule/tool-usage/trajectory/
                              answer-relevancy/LLM-judge)
  test_conversation.py       multi-turn types, ConversationHarness, conversation
                              scoring — via a fake agent, no real LLM
  test_judge.py              judge prompt builders + judge-backed scorers
  test_harness.py            EvalHarness/Scorecard end-to-end via the bare baseline
  test_retrieval_metrics.py  Recall@K/MRR/nDCG against plain ranked-id lists
  test_concurrency_bench.py  serial-vs-parallel speedup + failure isolation
  test_similarity.py         cosine_similarity / average_pairwise_similarity
```

Every scorer and metric operates on plain data (`AgentOutcome`, ranked id
lists, zero-arg callables) — nothing in `agent_eval/` knows what a "tool
call" actually did, which is what keeps it reusable across agents.

## Quickstart

```bash
pip install -r requirements.txt
pytest -q

python -m agent_eval run \
  --tasks benchmarks/tasks.json \
  --agent adapters.bare_baseline:build_agent \
  --outcome-adapter adapters.bare_baseline:adapt \
  --dump results/bare_baseline.json
```

### Evaluating the sibling ReAct agent

```bash
pip install -e ../agent-harness-from-scratch
```

```python
from adapters.react_agent_adapter import adapt, build_agent_factory
from agent import MockLLM, ToolRegistry, tool  # from agent-harness-from-scratch
from agent_eval.harness import EvalHarness
from agent_eval.scoring import RuleScorer, ToolUsageScorer, TrajectoryScorer


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expression))  # the sibling repo ships a safe evaluator instead


harness = EvalHarness(
    build_agent=build_agent_factory(MockLLM, lambda: ToolRegistry([calculator])),
    outcome_adapter=adapt,
    tasks="benchmarks/tasks.json",
    scorers=[RuleScorer(), ToolUsageScorer(), TrajectoryScorer()],
)
print(harness.run_all().render())
```

## Design notes

**Why `AgentOutcome` instead of importing a specific agent's result type.**
The moment `agent_eval` imports `ReActAgent`, it stops being an evaluation
toolkit and becomes a test suite for one agent. Every scorer, metric, and the
harness itself take the normalized dataclass; translating a specific agent's
result into it is the ~15-line job of one adapter function.

**Why scorers return a dict instead of a fixed set of fields.** The original
inline harness this project generalizes from had a hardcoded `TaskResult`
dataclass with one field per metric — adding a metric meant editing that
class, the aggregator, and the renderer. Here, `Scorecard.aggregate()` and
`render()` are driven entirely by whatever keys the configured scorers
produce, so a new `Scorer` is a pure addition.

**Why retrieval metrics take a callable, not a retriever object.** A
`Callable[[str], List[str]]` is the smallest interface that can wrap a BM25
index, a dense index, an RRF-fused hybrid pipeline, or a single `lambda`
returning canned ids in a test — no adapter class hierarchy needed.

**Why the concurrency benchmark runs the same cases twice.** Measuring
serial and parallel dispatch against the *same* case set is what makes the
speedup number mean something; measuring parallel dispatch alone only tells
you it finished, not what it saved.

**What I'd add to make this a "full" toolkit** (see `TODO.md`): a
`Scorer`-shaped wrapper around the new `agent_eval.similarity` primitive for
grading open-ended answers against a reference without an LLM judge call,
and a LangChain adapter to prove the framework-agnostic claim against a
second real framework, not just an internal baseline.

## License

MIT
