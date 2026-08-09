# Methodology (working draft — rewrite in your own voice before submission)

## 3.1 Research design

This study adopts an empirical, comparative, mixed-methods design within the Design Science Research (DSR) paradigm. The artefact is a reproducible evaluation harness that subjects an identical set of RAG configurations to three evaluation frameworks; the knowledge contribution is evidence about whether those frameworks agree and whether their agreement generalises across datasets.

*(If William pushes back on DSR — one of your Card 6 questions — the fallback framing is a purely empirical comparative study; only this section changes, not the design itself.)*

## 3.2 Experimental factors

A full-factorial design over three RAG design variables yields 18 configurations: chunking strategy (fixed 256-token, sliding 256/64 overlap, semantic with similarity threshold 0.55), retrieval method (BM25 with k1=0.9, b=0.4; dense bi-encoder all-mpnet-base-v2; hybrid via reciprocal rank fusion, k=60), and cross-encoder reranking (present/absent, ms-marco-MiniLM-L-6-v2 over the top 50 candidates). The generator is held constant (gpt-4o-2024-08-06, temperature 0, fixed seed) so that variance is attributable to the retrieval-side variables the frameworks are meant to discriminate.

## 3.3 Datasets

MS MARCO (passage) and Natural Questions, both in BEIR format, giving a shared corpus/queries/qrels schema. A fixed, seeded sample of 500 qrel-bearing queries per dataset bounds the evaluation budget; ARES itself evaluates on query subsets, and BEIR (Thakur et al., 2021) motivates the two-dataset generalisation test.

## 3.4 Evaluation frameworks

1. **Traditional IR metrics**: Precision@{1,3,5,10}, MRR, nDCG@10, computed against document-level qrels with chunk-to-document mapping (first-occurrence dedup).
2. **RAGAS** (Es et al., 2024): faithfulness, answer relevancy, context relevancy, with the library version pinned (prompt templates vary across versions — noted under threats to validity).
3. **ARES-adapted** (Saad-Falcon et al., 2024): a zero-shot GPT-4o judge scoring context relevance, answer faithfulness and answer relevance, calibrated with Prediction-Powered Inference against ~200 hand-labelled examples per dataset, yielding 95% confidence intervals. This is a stated adaptation of full ARES (which fine-tunes classifier judges) justified by the 6-month scope.

## 3.5 Statistical analysis

Framework agreement is measured as Spearman's rho between each pair of frameworks' rankings of the 18 configurations, per dataset, with paired-bootstrap 95% CIs (10,000 resamples). Cross-dataset generalisation is tested two ways: (a) each framework's own ranking stability between MS MARCO and NQ, and (b) replication of the pairwise agreement pattern, judged conservatively by CI overlap. Variable attribution uses paired Wilcoxon signed-rank tests on per-query scores (levels compared within query, other factors marginalised), with rank-biserial effect sizes and Holm–Bonferroni correction across the test family.

## 3.6 Reproducibility

Pinned model and library versions, fixed seeds throughout, prompts under version control, per-call response caching, experiment tracking in Weights & Biases, and a public repository.

## 3.7 Threats to validity

- **Construct**: RAGAS/ARES scores depend on judge prompts; version pinning mitigates but does not remove this.
- **Internal**: chunk-to-document mapping for IR metrics may favour finer chunking; the dedup rule is documented and constant across configs.
- **External**: two datasets constrain generalisation claims — precisely why cross-dataset stability is measured rather than assumed.
- **Statistical**: n=18 configurations limits Spearman power; bootstrap CIs make that uncertainty explicit rather than hidden.
