# Automated RAG Evaluation Frameworks — Thesis Codebase

Reproducible code and results for the MSc dissertation *"Automated RAG Evaluation
Frameworks: A Cross-Framework and Cross-Dataset Comparative Study"* (Gisma University).

**Research question.** When the same Retrieval-Augmented Generation (RAG) systems are
scored by three different evaluation frameworks — traditional IR metrics, RAGAS, and
ARES — do the frameworks agree on which configuration is best, and does that agreement
hold across datasets?

**Design.** 12 RAG configurations (2 chunking × 3 retrieval × 2 reranking) × 2 datasets
(MS MARCO, Natural Questions) × 500 queries each = **24 evaluation runs**.

## Headline finding

Judge **calibration changes the conclusion**. Using Prediction-Powered Inference (PPI)
with 200 human labels per dataset:

| Comparison | Raw ARES | PPI-calibrated ARES |
|---|---|---|
| IR vs ARES agreement (MS MARCO) | ρ = 0.951 | **ρ = −0.490** |
| IR vs ARES agreement (NQ) | ρ = 0.865 | **ρ = −0.641** |
| ARES cross-dataset stability | ρ = 0.851 | **ρ = −0.189** |

The agreement between the LLM judge and IR metrics is highly sensitive to calibration.
Because the calibration here was derived from a single reference configuration and applied
as one global rectifier, the inversion is interpreted cautiously, as a limitation of the
calibration design as much as a property of the judge: a single global correction compresses
ARES scores into a narrow band, after which they no longer track IR metrics or transfer
across datasets. The practical caution is that uncalibrated LLM-judge scores should not be
treated as a proxy for retrieval quality without calibration across configurations. IR and
RAGAS, by contrast, agree moderately and transfer stably (cross-dataset ρ ≈ 0.94–0.95).
Retrieval method and reranking dominate performance; chunking has the smallest effect.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required for generation, RAGAS, ARES
python tests/test_core.py           # offline sanity check → ALL TESTS PASSED
```

## Reproduce the study

```bash
# 1. Run all 12 configs on each dataset (LLM calls are cached; reruns are free)
python scripts/run_experiment.py --dataset msmarco --all
python scripts/run_experiment.py --dataset nq --all

# 2. (Optional) regenerate the ARES human-label sheets, then hand-label them
python scripts/make_annotation_sheet.py            # → annotation/{dataset}_to_label.csv
python scripts/label_cli.py msmarco                # resumable terminal labeller
python scripts/label_cli.py nq                     # → results/human_labels/{dataset}.csv

# 3. Run the statistics (PPI calibration fires automatically if labels exist)
python scripts/run_experiment.py --analyse         # → the results-chapter tables
```

Human labels for calibration are already included under `results/human_labels/`, so
step 3 reproduces the calibrated results directly.

## Repository map

```
configs/experiment_matrix.yaml   the 2×3×2 design, seeds, judges, statistics params
src/config.py                    matrix expansion → 12 RAGConfig objects
src/data/loaders.py              BEIR loaders (MS MARCO, NQ), seeded sampling
src/pipeline/chunking.py         fixed / semantic chunking
src/pipeline/retrieval.py        BM25 / dense / hybrid-RRF + cross-encoder rerank
src/pipeline/generation.py       gpt-4o-mini generation, caching, cost (€) + latency
src/evaluation/ir_metrics.py     Precision@k, MRR, nDCG@10
src/evaluation/ragas_eval.py     RAGAS wrapper (faithfulness, answer/context relevancy)
src/evaluation/ares_eval.py      zero-shot LLM judge + self-contained PPI calibration
src/stats/agreement.py           Spearman + bootstrap CIs, cross-dataset, Wilcoxon
scripts/run_experiment.py        orchestrator (--dataset / --config / --all / --analyse)
scripts/make_annotation_sheet.py builds the human-label sheets
scripts/label_cli.py             resumable annotation tool
results/                         all outputs: per-config runs, human labels, final tables
docs/THESIS_WORKING_DRAFT.md     the dissertation
```

## Key result files

| File | Contents |
|---|---|
| `results/tidy_config_scores.csv` | per-configuration scores, incl. PPI-calibrated ARES + CIs |
| `results/framework_agreement.csv` | SQ1 — pairwise Spearman ρ + bootstrap CIs |
| `results/cross_dataset_stability.csv` | SQ2 — per-framework cross-dataset ρ |
| `results/variable_attribution.csv` | SQ3 — Wilcoxon + Holm–Bonferroni effect sizes |

## Configuration & reproducibility

- **Model:** gpt-4o-mini (generation and judging), temperature 0.0, seed 42.
- **Chunking:** 256 tokens (fixed) / 384 (semantic); top-5 passages as context.
- **Determinism:** fixed seeds throughout; LLM calls cached by content hash.
- Pinned package versions are in `requirements.txt`.

## Datasets

MS MARCO and Natural Questions in BEIR format. These are public, anonymised datasets;
no private or personally identifiable data is used. An `OPENAI_API_KEY` is required to
regenerate LLM-based scores; all cached outputs are included so the analysis reproduces
without new API calls.
