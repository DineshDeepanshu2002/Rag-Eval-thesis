"""Three chunking strategies: fixed, sliding-window, semantic.

Chunks carry provenance (source doc_id) so IR metrics can map a retrieved
chunk back to its parent document for qrel lookup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str      # parent document — needed for qrels
    text: str


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer as a proxy for token counts.

    For the thesis, note this approximates tiktoken counts within ~15%;
    switching to tiktoken only changes chunk boundaries, not the design.
    """
    return text.split()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_fixed(doc_id: str, text: str, chunk_size: int = 256, overlap: int = 0, **_) -> list[Chunk]:
    """Fixed-size, non-overlapping token windows (overlap=0)."""
    return chunk_sliding(doc_id, text, chunk_size=chunk_size, overlap=overlap)


def chunk_sliding(doc_id: str, text: str, chunk_size: int = 256, overlap: int = 64, **_) -> list[Chunk]:
    """Sliding window over tokens with configurable overlap."""
    toks = _tokenize(text)
    if not toks:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(toks), step)):
        window = toks[start:start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, " ".join(window)))
        if start + chunk_size >= len(toks):
            break
    return chunks


def chunk_semantic(doc_id: str, text: str, max_chunk_size: int = 384,
                   similarity_threshold: float = 0.55,
                   embedder=None, **_) -> list[Chunk]:
    """Semantic chunking: greedily merge adjacent sentences while their
    embedding similarity to the running chunk stays above the threshold.

    `embedder` is a callable: list[str] -> np.ndarray (n, d), injected so the
    model is loaded once per run, not per document. If None, falls back to a
    sentence-boundary + size heuristic (useful for dry runs/tests).
    """
    import numpy as np

    sents = _sentences(text)
    if not sents:
        return []

    if embedder is None:
        # Fallback: sentence-packed chunks bounded by max size
        chunks, cur, n = [], [], 0
        for s in sents:
            t = len(_tokenize(s))
            if cur and n + t > max_chunk_size:
                chunks.append(" ".join(cur)); cur, n = [], 0
            cur.append(s); n += t
        if cur:
            chunks.append(" ".join(cur))
    else:
        embs = embedder(sents)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        chunks, cur, cur_vec, n = [], [], None, 0
        for s, e in zip(sents, embs):
            t = len(_tokenize(s))
            if cur:
                sim = float(np.dot(cur_vec / (np.linalg.norm(cur_vec) + 1e-9), e))
                if sim < similarity_threshold or n + t > max_chunk_size:
                    chunks.append(" ".join(cur)); cur, cur_vec, n = [], None, 0
            cur.append(s); n += t
            cur_vec = e.copy() if cur_vec is None else cur_vec + e
        if cur:
            chunks.append(" ".join(cur))

    return [Chunk(f"{doc_id}::c{i}", doc_id, c) for i, c in enumerate(chunks)]


CHUNKERS = {"fixed": chunk_fixed, "sliding": chunk_sliding, "semantic": chunk_semantic}
