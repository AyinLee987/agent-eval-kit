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
synthetic corpus — read it before citing a number from it).

The multi-agent latency/failure-isolation benchmark against
`MultiAgentOrchestrator` has not been run yet — that's next. See
[`TODO.md`](TODO.md) for the full roadmap.

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
  types.py              AgentOutcome, TrajectoryStep, ToolCall — the only contract
  scoring.py            RuleScorer, ToolUsageScorer, TrajectoryScorer, LLMJudgeScorer
  harness.py            EvalHarness (runs tasks) + Scorecard (aggregates + renders)
  retrieval_metrics.py  recall_at_k, mrr, ndcg_at_k, evaluate_retrieval
  concurrency_bench.py  serial-vs-parallel speedup + failure isolation
  cli.py / __main__.py  `python -m agent_eval run ...`
adapters/
  bare_baseline.py       self-contained single-pass agent — zero dependencies,
                          used by this repo's own tests and as a weak baseline
  react_agent_adapter.py adapts agent-harness-from-scratch's ReActAgent
                          (imported lazily; only needed if you use this adapter)
benchmarks/
  tasks.json              sample task set used by the bare-baseline tests
  rag_recall/             RAG Recall@K/MRR/nDCG ablation against the sibling
                          project's hybrid pipeline — corpus.py (synthetic
                          docs), queries.py (labeled cases), run_benchmark.py,
                          RESULTS.md (numbers + methodology + caveats)
tests/
  test_scoring.py, test_retrieval_metrics.py, test_concurrency_bench.py,
  test_harness.py        cover the toolkit end-to-end via the bare baseline
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

**What I'd add to make this a "full" toolkit** (see `TODO.md`): a semantic
similarity scorer for open-ended answers that doesn't require an LLM judge; a
LangChain adapter to prove the framework-agnostic claim against a second real
framework, not just an internal baseline; real Recall@K/MRR/nDCG numbers from
running this against the sibling project's hybrid RAG pipeline; and a real
concurrency benchmark against `MultiAgentOrchestrator` instead of synthetic
`time.sleep` cases.

## License

MIT
