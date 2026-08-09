"""Retrieval over chunks: BM25, dense bi-encoder, and hybrid (RRF fusion).

All retrievers share the interface:
    retrieve(query: str, top_k: int) -> list[tuple[chunk_id, score]]
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .chunking import Chunk


# --------------------------------------------------------------------- BM25
class BM25Retriever:
    """Okapi BM25 (self-contained; parameters match Pyserini defaults for
    MS MARCO: k1=0.9, b=0.4)."""

    def __init__(self, chunks: list[Chunk], k1: float = 0.9, b: float = 0.4, **_):
        self.k1, self.b = k1, b
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.docs = [c.text.lower().split() for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        q = query.lower().split()
        scores = defaultdict(float)
        for t in q:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.ids[i], s) for i, s in ranked]


# -------------------------------------------------------------------- Dense
class DenseRetriever:
    """Bi-encoder dense retrieval (sentence-transformers + cosine sim).

    Embeddings are computed once at index time; queries at search time.
    """

    def __init__(self, chunks: list[Chunk],
                 model: str = "sentence-transformers/all-mpnet-base-v2",
                 batch_size: int = 64, **_):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.model = SentenceTransformer(model)
        self.index = self.model.encode(
            [c.text for c in chunks], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=True,
        )
        self._np = np

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.index @ q
        top = self._np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in top]


# ------------------------------------------------------------------- Hybrid
class HybridRetriever:
    """Reciprocal Rank Fusion of BM25 + dense rankings.

    RRF(d) = sum over systems of 1 / (k + rank_system(d)); k=60 per
    Cormack et al. (2009). Rank-based fusion sidesteps score-scale mismatch.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, **_):
        self.bm25, self.dense, self.k = bm25, dense, rrf_k

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        fused = defaultdict(float)
        for system in (self.bm25, self.dense):
            for rank, (cid, _) in enumerate(system.retrieve(query, top_k * 2), start=1):
                fused[cid] += 1.0 / (self.k + rank)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return ranked


# ------------------------------------------------------------------ Rerank
class CrossEncoderReranker:
    """Cross-encoder reranking of the top_k_in candidates."""

    def __init__(self, chunks: list[Chunk],
                 model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_k_in: int = 50, **_):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.top_k_in = top_k_in
        self.lookup = {c.chunk_id: c.text for c in chunks}

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        cands = candidates[: self.top_k_in]
        pairs = [(query, self.lookup[cid]) for cid, _ in cands]
        scores = self.model.predict(pairs)
        return sorted(zip((c for c, _ in cands), map(float, scores)), key=lambda x: -x[1])
