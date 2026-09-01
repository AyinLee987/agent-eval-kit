"""LLM-as-reranker: ask the chat model itself to score each candidate
passage's relevance to the query, instead of a cross-encoder or the
sibling repo's lexical-overlap HeuristicReranker.

Wired into RAGPipeline via the existing CallableReranker adapter
(agent/rag/rerank.py) -- no harness changes needed, the pipeline was
already designed to accept any ``scorer(query_text, passages) -> scores``
callable.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence


class RerankCache:
    """On-disk cache for LLM rerank scores, keyed by (model, query, passages).

    Same rationale as benchmarks/rag_recall/cached_embeddings.py: a scoring
    call is one real, paid LLM request, and a killed/interrupted run should
    be resumable from where it left off rather than re-paying for every
    query on restart -- this cost the earlier embedding benchmark real time
    once already (see rag_recall_beir/RESULTS.md's run history). Flushes
    every ``flush_every`` new entries rather than on every write, for the
    same O(n^2)-avoidance reason CachedEmbeddingProvider does.

    Keyed on the *model name* too, not just query+passages, so switching
    the reranker LLM (as this experiment already did once, Bailian ->
    DeepSeek) can't silently serve stale scores from a different model.
    Relies on temperature=0.0 (this project's LLM presets all default to
    it) for a cache hit to be a reasonable stand-in for a fresh call --
    not a hard guarantee of byte-identical output, but consistent with how
    every other cache in this toolkit treats temperature=0.0 determinism.
    """

    def __init__(self, cache_path: str, flush_every: int = 5) -> None:
        self._cache_path = cache_path
        self._flush_every = flush_every
        self._unflushed = 0
        self._cache: Dict[str, List[float]] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh)

    @staticmethod
    def _key(model: str, prompt_version: str, query_text: str, passages: Sequence[str]) -> str:
        # prompt_version (the caller passes the prompt *template* itself) is
        # part of the key so that editing the prompt -- exactly what
        # happened here, zero-shot -> one-shot after a 41% parse-failure
        # rate -- can't silently serve stale answers scored under the old
        # wording back for a "new prompt" run. Old entries just become
        # unreachable dead weight in the file rather than a correctness bug.
        payload = json.dumps([model, prompt_version, query_text, list(passages)], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt_version: str, query_text: str, passages: Sequence[str]) -> Optional[List[float]]:
        return self._cache.get(self._key(model, prompt_version, query_text, passages))

    def put(
        self, model: str, prompt_version: str, query_text: str, passages: Sequence[str], scores: List[float]
    ) -> None:
        self._cache[self._key(model, prompt_version, query_text, passages)] = scores
        self._unflushed += 1
        if self._unflushed >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        with open(self._cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        self._unflushed = 0

    def stats(self) -> Dict[str, int]:
        return {"cached_entries": len(self._cache)}

LLM_RERANK_PROMPT = """You are scoring how relevant each candidate passage is to a search query, for a retrieval system. Score every passage independently on a 0-10 scale (10 = directly and fully answers the query, 0 = completely irrelevant to it).

Respond with ONLY a JSON array of numbers, one real score per passage, in the same order as the passages -- no explanation, no markdown fences, no placeholder text describing the array, nothing else but the array of actual numbers itself.

### Example

Query: What foods lower cholesterol?

Candidate passages:
[1] Oatmeal contains soluble fiber that has been shown to reduce LDL cholesterol levels when eaten regularly.
[2] The Eiffel Tower was completed in 1889 and is located in Paris, France.
[3] Regular exercise improves cardiovascular health but this passage does not mention cholesterol specifically.
[4] Walnuts and almonds contain unsaturated fats that can help lower bad cholesterol.
[5] A study of urban transportation patterns in the 1990s found commuting times increased.

Your response:
[9, 0, 3, 8, 0]

### Now score this one

Query: {query}

Candidate passages:
{passages}

Your response (a JSON array of exactly {n} real numbers, nothing else):"""


def _parse_scores(text: str, expected_n: int) -> List[float]:
    """Extract a JSON array of exactly ``expected_n`` numbers from ``text``.

    Tolerant of the model wrapping the array in a markdown code fence or
    adding a stray sentence before/after it -- takes the substring between
    the first ``[`` and the last ``]`` rather than requiring the whole
    response to be valid JSON on its own.
    """

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in LLM response: {text[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list) or len(parsed) != expected_n:
        raise ValueError(f"Expected {expected_n} scores, got: {parsed!r}")
    return [float(x) for x in parsed]


def build_llm_rerank_scorer(
    llm,
    *,
    max_passage_chars: int = 300,
    failure_counter: Optional[List[int]] = None,
    cache: Optional[RerankCache] = None,
) -> Callable[[str, Sequence[str]], Sequence[float]]:
    """Build a ``CallableReranker``-shaped scorer backed by ``llm``.

    ``max_passage_chars`` truncates each candidate before it goes in the
    prompt -- NFCorpus abstracts run long (see RESULTS.md), and up to
    ~30 full candidates in one prompt would be an unnecessarily large
    (and slow, and expensive) request when the first few hundred
    characters are almost always enough to judge topical relevance.

    On an unparseable response, raises rather than silently returning a
    fabricated score list -- ``CallableReranker.rerank`` propagates that,
    and ``RAGPipeline._retrieve_single`` already catches a reranker
    exception and falls back to plain RRF order for that one query (see
    ``degraded_components``), which is the right behavior: one bad LLM
    response degrades gracefully instead of corrupting that query's
    ranking with made-up numbers. ``failure_counter``, if given a
    single-element list, is incremented on every such failure so the
    caller can report how often it happened. A cache hit obviously can't
    fail to parse (it was already validated once, at write time), so it
    never touches ``failure_counter``.

    Pass ``cache`` (a :class:`RerankCache`) to make an interrupted run
    resumable -- see that class's docstring for why this exists.
    """

    model_id = getattr(llm, "model", llm.__class__.__name__)
    truncated = lambda passages: [p[:max_passage_chars] for p in passages]  # noqa: E731

    def scorer(query_text: str, passages: Sequence[str]) -> Sequence[float]:
        passages_for_cache = truncated(passages)  # cache on what actually reaches the prompt
        if cache is not None:
            cached = cache.get(model_id, LLM_RERANK_PROMPT, query_text, passages_for_cache)
            if cached is not None:
                return cached

        numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages_for_cache))
        prompt = LLM_RERANK_PROMPT.format(query=query_text, passages=numbered, n=len(passages))
        response = llm.chat([{"role": "user", "content": prompt}])
        try:
            scores = _parse_scores(response.content or "", len(passages))
        except ValueError:
            if failure_counter is not None:
                failure_counter[0] += 1
            raise

        if cache is not None:
            cache.put(model_id, LLM_RERANK_PROMPT, query_text, passages_for_cache, scores)
        return scores

    return scorer
