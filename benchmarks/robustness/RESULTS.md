# Robustness (paraphrase/register/translation) — results

Run with `python benchmarks/robustness/run_benchmark.py`. Full per-phrasing
answers and per-case similarity are in `results.json`; the full write-up
(including why the lowest-scoring case isn't actually an inconsistency) is
`../../reports/robustness-evaluation.md`.

## Setup

- **Agent**: real `ReActAgent` (`agent-harness-from-scratch`), LLM = Bailian
  `qwen-plus`, same 3 tools as `agent_trajectory`/`conversational`
  (`calculator`, `lookup_fact`, `current_datetime`).
- **Embeddings**: real `qwen3.7-text-embedding` (same requirement as the RAG
  recall benchmark — refuses to run without `RAG_EMBEDDING_*` configured).
- **8 base questions** (`queries.py`), each asked 4 ways: the original
  Chinese wording, a same-language paraphrase, a formal/politeness-noise
  register shift, and an English translation — 32 tasks total, each run as
  an independent single-turn task (fresh agent per task).
- **Primary metric**: average pairwise cosine similarity across a
  question's 4 answers — higher means the agent held steady regardless of
  phrasing/language.
- **Secondary metric**: `RuleScorer` + `ToolUsageScorer` per phrasing —
  catches "answers look similar but the content or tool use was actually
  wrong," which similarity alone can't.

## Results (n=8 questions / 32 phrasings)

| Metric | Score |
|---|---|
| rule_pass (n=32) | 1.00 |
| used_expected_tool (n=20 tool-requiring phrasings) | 1.00 |
| avg pairwise answer similarity (n=8 questions) | 0.92 |

| Question | Avg similarity |
|---|---|
| calc-basic | 0.97 |
| lookup-ratelimit | 0.97 |
| fact-boiling-point | 0.95 |
| calc-compound | 0.91 |
| lookup-pto | 0.91 |
| datetime-check | 0.91 |
| fact-speed-of-light | 0.90 |
| chain-stipend | 0.80 |

Every one of the 32 phrasings got the content right (`rule_pass=1.00`) and
used the right tool when one was needed (`used_expected_tool=1.00`) — the
agent never got *confused* by rewording. The similarity spread is real, but
it's measuring something narrower than "did the agent get it right
consistently": see the report for what actually drives the lowest score.

## Why `chain-stipend` scored lowest — and it isn't an inconsistency

All four phrasings correctly computed the same answer (**$600**). The
`paraphrase` variant's answer is short — just the final total — while
`base`/`register` restate the looked-up intermediate value ($200) before
the total, and `translation` answers in English. The embedding model is
picking up on **verbosity and language**, not a difference in correctness.
This is the caveat this benchmark's plan flagged going in ("embedding
cosine similarity is coarse for short factual answers") — now with a
concrete example instead of a hypothetical one. See the full report for
three side-by-side examples across different questions showing the same
pattern.
