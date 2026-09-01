"""On-disk cache wrapper for a real (paid) embedding provider.

Real embedding calls cost money and hit the network; without a cache,
re-running this benchmark while iterating on queries would re-embed the
unchanged 36-chunk corpus every single time. Caches by
``sha256(model_id + text)`` so a change of embedding model invalidates
naturally instead of silently reusing stale vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Sequence


class CachedEmbeddingProvider:
    """Wraps any EmbeddingProvider-shaped object with a JSON file cache.

    ``flush_every`` batches disk writes: flushing the whole cache after
    *every* new vector is O(1) at rag_recall's 36-chunk corpus, but at
    rag_recall_beir's ~4,000-vector scale a naive per-call flush degrades
    into O(n^2) total I/O (each flush re-serializes everything written so
    far). Flushing every ``flush_every`` new vectors instead, plus a final
    ``flush()`` the caller is responsible for calling once done, keeps that
    linear -- at the cost of losing at most ``flush_every`` vectors of
    progress if the process is killed mid-run (re-embedding a few dozen
    texts is cheap; O(n^2) disk writes on a multi-thousand-vector run is not).
    """

    def __init__(self, inner: Any, cache_path: str, flush_every: int = 25) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._dim: int | None = None
        self._cache: Dict[str, List[float]] = {}
        self._flush_every = flush_every
        self._unflushed = 0
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh)

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise ValueError("Embedding dimension is unknown until the first embedding call.")
        return self._dim

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        key = hashlib.sha256(f"{self.model_id}\0{text}".encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            cached = list(self._inner.embed_query(text))
            self._cache[key] = cached
            self._unflushed += 1
            if self._unflushed >= self._flush_every:
                self.flush()
        if self._dim is None:
            self._dim = len(cached)
        return cached

    def flush(self) -> None:
        """Write the in-memory cache to disk now. Safe to call any time,
        including redundantly (e.g. once more at the end of a run to catch
        whatever's accumulated since the last periodic flush)."""

        with open(self._cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        self._unflushed = 0

    def stats(self) -> Dict[str, int]:
        return {"cached_vectors": len(self._cache)}
