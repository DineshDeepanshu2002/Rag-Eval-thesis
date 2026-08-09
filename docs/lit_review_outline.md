# Literature Review — outline (~2,500 words target, per your Card 6 question)

Structure follows your Card 3: RAG foundations → retrieval → eval frameworks → cross-dataset → gap.

## 2.1 RAG foundations (~400 words)
- Lewis et al. (2020): retriever + generator architecture; why grounding reduces hallucination.
- Brief: how the retrieval side (chunking, retriever, reranker) became the main tuning surface once generators commoditised.

## 2.2 Retrieval methods (~500 words)
- Sparse: BM25 as the persistent baseline.
- Karpukhin et al. (2020), DPR: dense retrieval; bi-encoder vs cross-encoder distinction sets up reranking.
- Hybrid fusion (RRF, Cormack et al. 2009 — add to reading list) as the practical default.
- Chunking as an under-theorised variable (find 1–2 recent chunking studies — gap in your current six papers; ask William for pointers → Card 6).

## 2.3 Evaluation frameworks (~700 words) ← core section
- Traditional IR metrics: what they measure (retrieval only), what they miss (generation quality).
- Es et al. (2024), RAGAS: reference-free LLM-based metrics; strengths (no gold answers needed) and critiques (judge sensitivity).
- Saad-Falcon et al. (2024), ARES: trained judges + PPI calibration; contrast with RAGAS on the calibration question.
- Key synthesis move: all three families were validated *in isolation*, on different datasets, against different ground truths.

## 2.4 Cross-dataset generalisation (~450 words)
- Thakur et al. (2021), BEIR: retrieval effectiveness does not transfer across domains — the direct precedent for asking whether *evaluation* verdicts transfer either.
- Gao et al. (2023) survey: names evaluation as an open problem; use to frame the field-level view.

## 2.5 The gap (~300 words)
- No systematic head-to-head comparison of RAGAS, ARES and IR metrics on identical configurations.
- No evidence that framework agreement (or disagreement) is stable across datasets.
- Restate RQ as the logical consequence of 2.3 + 2.4.

## Reading-list candidates to raise with William
- Cormack et al. 2009 (RRF) — needed to cite the hybrid method.
- A recent chunking-strategy paper (2024–25) — currently the weakest citation area.
- Angelopoulos et al. 2023 (PPI) — needed to cite the ARES calibration math.
- Anything he recommends on LLM-judge reliability (e.g. judge bias / position bias literature).
