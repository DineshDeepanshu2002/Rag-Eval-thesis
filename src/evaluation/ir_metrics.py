"""Traditional IR metrics: Precision@k, MRR, nDCG@k.

Retrieval returns chunk_ids; qrels are at document level. We map each chunk
back to its parent doc and dedupe (first occurrence keeps the rank) —
standard practice for passage-chunked corpora; document this choice in the
methodology chapter.
"""
from __future__ import annotations

import math


def chunks_to_docs(ranked_chunks: list[tuple[str, float]]) -> list[str]:
    """chunk_id 'doc::cN' -> ordered, deduped doc list."""
    seen, docs = set(), []
    for cid, _ in ranked_chunks:
        doc = cid.split("::")[0]
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def precision_at_k(ranked_docs: list[str], qrel: dict[str, int], k: int) -> float:
    top = ranked_docs[:k]
    if not top:
        return 0.0
    rel = sum(1 for d in top if qrel.get(d, 0) > 0)
    return rel / k


def reciprocal_rank(ranked_docs: list[str], qrel: dict[str, int]) -> float:
    for i, d in enumerate(ranked_docs, start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_docs: list[str], qrel: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        (2 ** qrel.get(d, 0) - 1) / math.log2(i + 1)
        for i, d in enumerate(ranked_docs[:k], start=1)
    )
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(ranked_chunks: list[tuple[str, float]],
                   qrel: dict[str, int],
                   k_values: list[int] = (1, 3, 5, 10)) -> dict[str, float]:
    docs = chunks_to_docs(ranked_chunks)
    out = {f"P@{k}": precision_at_k(docs, qrel, k) for k in k_values}
    out["MRR"] = reciprocal_rank(docs, qrel)
    out["nDCG@10"] = ndcg_at_k(docs, qrel, 10)
    return out
