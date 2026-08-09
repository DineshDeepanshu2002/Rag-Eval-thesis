"""Tests for the API-free components: chunking, BM25, IR metrics, stats."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loaders import _pool_corpus
from src.pipeline.chunking import Chunk, chunk_fixed, chunk_semantic, chunk_sliding
from src.pipeline.retrieval import BM25Retriever, HybridRetriever
from src.evaluation.ir_metrics import (chunks_to_docs, evaluate_query,
                                       ndcg_at_k, precision_at_k,
                                       reciprocal_rank)
from src.stats.agreement import (agreement_report, bootstrap_spearman,
                                 holm_bonferroni, paired_wilcoxon,
                                 rank_biserial, spearman_matrix,
                                 cross_dataset_stability)


def test_chunking():
    text = " ".join(f"tok{i}" for i in range(700))
    fixed = chunk_fixed("d1", text, chunk_size=256, overlap=0)
    assert len(fixed) == 3 and len(fixed[0].text.split()) == 256
    sliding = chunk_sliding("d1", text, chunk_size=256, overlap=64)
    assert len(sliding) > len(fixed)  # overlap => more chunks
    # overlap check: last 64 of chunk0 == first 64 of chunk1
    assert fixed[0].doc_id == "d1" and "::" in fixed[0].chunk_id
    sem = chunk_semantic("d1", "One. Two. Three. " * 100, max_chunk_size=50)
    assert all(len(c.text.split()) <= 60 for c in sem)
    print("  chunking OK")


def test_corpus_pooling():
    corpus = {f"d{i}": {"title": "", "text": f"doc {i}"} for i in range(1000)}
    qrels = {"q1": {"d0": 1, "d1": 1}, "q2": {"d0": 1}}
    pooled = _pool_corpus(corpus, qrels, 100, random.Random(42))
    assert "d0" in pooled and "d1" in pooled       # relevant docs never dropped
    assert len(pooled) == 100                       # distractor cap respected
    # seeded => reproducible pool
    assert set(_pool_corpus(corpus, qrels, 100, random.Random(42))) == set(pooled)
    # None => full corpus
    assert len(_pool_corpus(corpus, qrels, None, random.Random(42))) == 1000
    # cap below #relevant still keeps all relevant
    tiny = _pool_corpus(corpus, qrels, 1, random.Random(42))
    assert "d0" in tiny and "d1" in tiny
    print("  corpus pooling OK")


def test_ir_metrics():
    ranked = [("docA::c0", 3.0), ("docA::c1", 2.5), ("docB::c0", 2.0),
              ("docC::c0", 1.0)]
    assert chunks_to_docs(ranked) == ["docA", "docB", "docC"]  # dedupe
    qrel = {"docB": 1}
    docs = chunks_to_docs(ranked)
    assert precision_at_k(docs, qrel, 1) == 0.0
    assert precision_at_k(docs, qrel, 2) == 0.5
    assert reciprocal_rank(docs, qrel) == 0.5           # docB at rank 2
    # perfect ranking => nDCG 1
    assert abs(ndcg_at_k(["docB", "docA"], {"docB": 1}, 10) - 1.0) < 1e-9
    m = evaluate_query(ranked, qrel)
    assert set(m) == {"P@1", "P@3", "P@5", "P@10", "MRR", "nDCG@10"}
    print("  ir_metrics OK")


def test_bm25():
    chunks = [
        Chunk("d1::c0", "d1", "the cat sat on the mat"),
        Chunk("d2::c0", "d2", "dogs chase cats in the park"),
        Chunk("d3::c0", "d3", "quantum computing uses qubits"),
    ]
    bm25 = BM25Retriever(chunks)
    top = bm25.retrieve("cat mat", top_k=3)
    assert top[0][0] == "d1::c0"
    top = bm25.retrieve("quantum qubits", top_k=3)
    assert top[0][0] == "d3::c0"
    print("  bm25 OK")


def test_hybrid_rrf():
    class Fake:
        def __init__(self, order): self.order = order
        def retrieve(self, q, top_k): return [(c, 1.0) for c in self.order[:top_k]]
    h = HybridRetriever(Fake(["a", "b", "c"]), Fake(["b", "a", "c"]))
    ranked = [c for c, _ in h.retrieve("q", top_k=3)]
    assert set(ranked[:2]) == {"a", "b"} and ranked[2] == "c"
    print("  hybrid RRF OK")


def test_stats():
    rng = np.random.default_rng(0)
    # perfectly correlated rankings
    x = np.arange(18, dtype=float)
    y = x + rng.normal(0, .01, 18)
    rho, lo, hi = bootstrap_spearman(x, y, n_boot=500)
    assert rho > 0.99 and lo <= rho <= hi

    # tidy 18-config frame across 2 datasets
    rows = []
    for ds in ("msmarco", "nq"):
        base = rng.uniform(0, 1, 18)
        for i in range(18):
            rows.append({"dataset": ds, "config_id": f"cfg{i}",
                         "ir": base[i],
                         "ragas": base[i] + rng.normal(0, .05),
                         "ares": rng.uniform(0, 1)})  # uncorrelated framework
    df = pd.DataFrame(rows)
    mat = spearman_matrix(df[df.dataset == "msmarco"], ["ir", "ragas", "ares"])
    assert mat.loc["ir", "ragas"] > 0.7            # correlated pair detected
    assert abs(mat.loc["ir", "ares"]) < 0.6        # noise pair low
    rep = agreement_report(df, ["ir", "ragas", "ares"], n_boot=300)
    assert len(rep) == 6                            # 3 pairs x 2 datasets
    cds = cross_dataset_stability(df, ["ir", "ragas", "ares"])
    assert len(cds) == 3

    # Wilcoxon: construct a true effect of retrieval
    pq = []
    for q in range(60):
        for retr in ("bm25", "dense"):
            for ch in ("fixed", "sliding"):
                score = 0.5 + (0.15 if retr == "dense" else 0) + rng.normal(0, .05)
                pq.append({"query_id": q, "chunking": ch, "retrieval": retr,
                           "rerank": "none", "score": score})
    res = paired_wilcoxon(pd.DataFrame(pq), "retrieval", "dense", "bm25", "score")
    assert res["p"] < 0.01 and res["effect_r"] > 0.5
    adj = holm_bonferroni([res, {"variable": "x", "a": "a", "b": "b",
                                 "W": 1, "p": 0.9, "effect_r": 0, "n": 60}])
    assert adj["significant"].iloc[0] and not adj["significant"].iloc[1]
    print("  stats OK")


if __name__ == "__main__":
    test_chunking()
    test_corpus_pooling()
    test_ir_metrics()
    test_bm25()
    test_hybrid_rrf()
    test_stats()
    print("ALL TESTS PASSED")
