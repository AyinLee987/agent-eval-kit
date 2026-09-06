# Public dataset attribution and scope

These files are public benchmark subsets, not newly authored questions. The
repository's MIT license does not replace the dataset-specific terms below.

## GSM8K

- Creators: Karl Cobbe and coauthors; copyright (c) 2021 OpenAI.
- Source: https://github.com/openai/grade-school-math
- Pinned upstream commit: 3101c7d5072418e28b9008a6636bde82a006892c
- Paper: Training Verifiers to Solve Math Word Problems, 2021,
  https://arxiv.org/abs/2110.14168
- License: MIT; the complete upstream notice is in licenses/GSM8K-MIT.txt.
- Changes: deterministic subsampling, added provenance/split/family fields,
  separated the final numeric answer from the supplied reference solution.
  Question and reference-solution text are retained.
- Local development comes from upstream train. Local frozen test comes from
  upstream test. Upstream records lack an ID, so source_id records the original
  split and zero-based line index in the hash-pinned JSONL file.

## HotpotQA

- Creators: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
  Ruslan Salakhutdinov and Christopher D. Manning; context passages originate
  from Wikipedia and its contributors.
- Official source and license declaration: https://hotpotqa.github.io/
- Paper: HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering, EMNLP 2018, https://arxiv.org/abs/1809.09600
- Verified mirror: https://huggingface.co/datasets/hotpotqa/hotpot_qa
  at revision 1908d6afbbead072334abe2965f91bd2709910ab.
- The original CMU data endpoint timed out during preparation. The mirror
  retains the original question IDs, answers, passages and supporting facts.
- License: CC BY-SA 4.0. The HotpotQA-derived normalized data and additions to
  that data are distributed under the same license:
  https://creativecommons.org/licenses/by-sa/4.0/
  Complete legal terms: licenses/HOTPOTQA-CC-BY-SA-4.0.txt.
- Changes: deterministic subsampling, conversion of column-oriented context
  into a list of documents, added provenance/split/family fields, and exclusion
  of records whose supplied supporting facts are missing from the context or
  reference fewer than two distinct supporting documents. Passage sentences,
  answer text and valid supporting-fact labels are retained.
- Both local splits come from the official distractor development split.
  Local test is a frozen public holdout, **not HotpotQA's official hidden
  test**. The official test has no public gold answers/supporting facts.
- Each context document retains its original Wikipedia title. The canonical
  article address is https://en.wikipedia.org/wiki/ followed by the URL-encoded
  title. The actual passages are the frozen dataset snapshot, not current
  versions of those articles.

## Reproduction and interpretation

Install the preparation-only Parquet reader:

```text
python -m pip install -r benchmarks/public_data/requirements.txt
python -m benchmarks.public_data.prepare --offline
python -m benchmarks.public_data.prepare --verify
```

Omit --offline only when downloading missing sources. Downloads must match
source_lock.json. Raw files and download receipts are ignored in .cache/.
The default is 400 dev and 400 test cases per dataset, seed 20260905.
Rebuilding from the same cache/configuration preserves normalized file bytes.
A changed frozen test requires a different output directory or the explicit
--replace-frozen flag.

manifest.json records source URLs, exact upstream revisions, SHA-256 hashes,
download timestamps, original source split counts, rejected records, sampling
and grouping rules, per-file counts/hashes, and split-overlap audits.

GSM8K groups matching question wording after numeric values are masked.
HotpotQA groups supporting-title sets and additionally permits no context
title to appear across the two published splits. Both datasets also exclude
cross-split numeric-masked question duplicates. This is an explicit,
reproducible grouping policy; it cannot certify detection of every semantic
paraphrase. Public benchmark familiarity in model pretraining is a separate
limitation, not something a local split can remove.

No model was run to select these samples or manufacture their labels.
Scoring does not require an LLM judge. GSM8K uses final-number equality.
HotpotQA uses answer EM/F1, supporting-fact set metrics and joint metrics
following https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py .
The scorer is an independent implementation of those metric definitions.
