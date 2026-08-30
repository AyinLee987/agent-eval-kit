"""On-disk cache wrapper for a real (paid) embedding provider.

Duplicated from benchmarks/rag_recall (same convention as tools.py in the
other benchmarks — each folder stays self-contained). Real embedding calls
cost money and hit the network; without a cache, re-running this benchmark
while iterating on scenarios would re-embed the unchanged corpus every
single time. Caches by ``sha256(model_id + text)`` so a change of
embedding model invalidates naturally instead of silently reusing stale
vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Sequence


class CachedEmbeddingProvider:
    """Wraps any EmbeddingProvider-shaped object with a JSON file cache."""

    def __init__(self, inner: Any, cache_path: str) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._dim: int | None = None
        self._cache: Dict[str, List[float]] = {}
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
            self._flush()
        if self._dim is None:
            self._dim = len(cached)
        return cached

    def _flush(self) -> None:
        with open(self._cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)

    def stats(self) -> Dict[str, int]:
        return {"cached_vectors": len(self._cache)}
