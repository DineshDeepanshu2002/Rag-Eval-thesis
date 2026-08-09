"""Load MS MARCO and Natural Questions in BEIR format.

BEIR gives us a unified schema across both datasets:
  corpus : {doc_id: {"title": str, "text": str}}
  queries: {query_id: str}
  qrels  : {query_id: {doc_id: relevance}}

We sample a fixed subset of queries (seeded) to keep the 108-run budget
tractable — justify this in the methodology chapter (cf. ARES paper, which
also evaluates on query subsets).

Corpus pooling: the full BEIR corpora are far too large to index on a laptop
(MS MARCO ~8.8M passages, NQ ~2.6M). We therefore evaluate against a *pooled*
corpus: every ground-truth-relevant document for the sampled queries (these can
never be dropped, or IR metrics become undefined) plus a seeded random sample
of non-relevant documents that act as distractors, capped at ``max_corpus``.
This is the standard subsampled-retrieval compromise; document it as a
threat to validity (retrieval difficulty is reduced relative to the full index)
and keep the cap identical across configs so comparisons stay fair. Set
``max_corpus=None`` to index the full corpus.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict]          # doc_id -> {title, text}
    queries: dict[str, str]          # query_id -> query text
    qrels: dict[str, dict[str, int]] # query_id -> {doc_id: rel}


def _pool_corpus(corpus: dict[str, dict], qrels: dict[str, dict[str, int]],
                 max_corpus: int | None, rng: random.Random) -> dict[str, dict]:
    """Relevant docs (always kept) + seeded distractor sample, capped at max_corpus."""
    relevant = {doc_id for rels in qrels.values() for doc_id, r in rels.items()
                if r > 0 and doc_id in corpus}
    if max_corpus is None:
        return corpus
    distractor_ids = sorted(set(corpus) - relevant)
    n_distractor = max(0, max_corpus - len(relevant))
    keep_distractor = set(rng.sample(distractor_ids, min(n_distractor, len(distractor_ids))))
    keep = relevant | keep_distractor
    return {doc_id: corpus[doc_id] for doc_id in corpus if doc_id in keep}


def load_beir(hf_id: str, split: str, n_queries: int, seed: int, name: str,
              max_corpus: int | None = 50_000) -> BeirDataset:
    """Download and subsample a BEIR dataset via the `beir` package.

    Requires: pip install beir

    max_corpus: cap on pooled corpus size (relevant docs + distractors). Keeps
        indexing tractable on a laptop. None => full corpus (not laptop-feasible
        for MS MARCO/NQ).
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    dataset = hf_id.split("/")[-1]
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "datasets/")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # Seeded subsample: only queries that have qrels (else IR metrics undefined)
    valid = sorted(q for q in queries if q in qrels and qrels[q])
    rng = random.Random(seed)
    keep = set(rng.sample(valid, min(n_queries, len(valid))))

    queries = {q: t for q, t in queries.items() if q in keep}
    qrels = {q: r for q, r in qrels.items() if q in keep}
    corpus = _pool_corpus(corpus, qrels, max_corpus, rng)
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
