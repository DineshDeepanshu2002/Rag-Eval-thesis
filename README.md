# RAG Evaluation Framework Agreement — Thesis Codebase

Do RAGAS, ARES and traditional IR metrics rank the same 18 RAG configurations
consistently — and does that agreement generalise from MS MARCO to Natural
Questions? 18 configs × 2 datasets × 3 frameworks = 108 evaluation runs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required for generation, RAGAS, ARES
python tests/test_core.py           # should print ALL TESTS PASSED (no API needed)
```

## Repo map

```
configs/experiment_matrix.yaml   the 3×3×2 design, seeds, judges, stats params
src/config.py                    matrix expansion → 18 RAGConfig objects
src/data/loaders.py              BEIR loaders (MS MARCO, NQ), seeded sampling
src/pipeline/chunking.py         fixed / sliding / semantic
src/pipeline/retrieval.py        BM25 / dense / hybrid-RRF + cross-encoder rerank
src/pipeline/generation.py       pinned GPT-4o, caching, cost(€) + latency
src/evaluation/ir_metrics.py     P@k, MRR, nDCG@10          [tested]
src/evaluation/ragas_eval.py     RAGAS wrapper
src/evaluation/ares_eval.py      LLM judge + PPI calibration [PPI math self-contained]
src/stats/agreement.py           Spearman+bootstrap, cross-dataset, Wilcoxon [tested]
scripts/run_experiment.py        orchestrator + --analyse
docs/                            methodology draft, lit review outline
```

## Run order (maps to the JUL→OCT timeline)

```bash
# JUL — smoke test one cell end-to-end (cheapest config)
python scripts/run_experiment.py --dataset msmarco --config fixed__bm25__none

# AUG — all 18 on MS MARCO
python scripts/run_experiment.py --dataset msmarco --all

# SEP — all 18 on NQ, then hand-label ~200 ARES calibration examples
python scripts/run_experiment.py --dataset nq --all
#   put labels in results/human_labels/{dataset}.csv (query_id + 3 binary cols)

# SEP — statistics: produces the four results-chapter tables
python scripts/run_experiment.py --analyse
```

## Cost control

- All GPT-4o calls are cached by (model, temperature, question, contexts) hash —
  reruns are free; configs sharing retrieval results share generations.
- Rough budget at 500 queries/dataset: estimate after the JUL smoke test by
  multiplying its measured cost; verify current per-token pricing in
  `src/pipeline/generation.py` first.

## Honest status

- **Tested and working**: config matrix, chunking, BM25, hybrid RRF, IR
  metrics, full statistics pipeline (see tests/test_core.py and the dry run).
- **Written but needs your environment to verify**: dense retrieval, reranker,
  generation, RAGAS and ARES wrappers — they need GPU-optional model downloads
  and an OpenAI key. Expect minor version-drift fixes in the RAGAS wrapper
  (its API moves between releases; pin the version that works and record it).
