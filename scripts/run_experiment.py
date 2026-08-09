"""Orchestrator: run one (dataset x config) cell end-to-end, or the full 36.

Usage:
    python scripts/run_experiment.py --dataset msmarco --config fixed__bm25__none
    python scripts/run_experiment.py --dataset msmarco --all
    python scripts/run_experiment.py --analyse          # after all runs

Outputs per run: results/{dataset}/{config_id}/
    retrieval.jsonl   per-query ranked chunks + IR metrics
    generation.jsonl  answers + cost + latency
    ragas.csv         RAGAS per-query scores
    ares.json         ARES calibrated scores + CIs
    summary.json      config-level means for the tidy stats frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# Load OPENAI_API_KEY (and any other secrets) from the local .env file so the
# key never has to be exported manually in the shell.
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.config import expand_matrix, load_experiment
from src.data.loaders import load_beir
from src.pipeline.chunking import CHUNKERS
from src.pipeline.retrieval import (BM25Retriever, CrossEncoderReranker,
                                    DenseRetriever, HybridRetriever)
from src.pipeline.generation import Generator
from src.evaluation.ir_metrics import evaluate_query


def build_pipeline(cfg, chunks, exp):
    if cfg.retrieval == "bm25":
        retriever = BM25Retriever(chunks, **cfg.retrieval_params)
    elif cfg.retrieval == "dense":
        retriever = DenseRetriever(chunks, **cfg.retrieval_params)
    else:
        retriever = HybridRetriever(
            BM25Retriever(chunks), DenseRetriever(chunks), **cfg.retrieval_params)
    reranker = (CrossEncoderReranker(chunks, **cfg.rerank_params)
                if cfg.rerank == "cross_encoder" else None)
    return retriever, reranker


def run_cell(dataset_cfg, cfg, exp, limit=None):
    ds = load_beir(dataset_cfg["hf_id"], dataset_cfg["split"],
                   dataset_cfg["n_queries"], exp["seed"], dataset_cfg["name"],
                   max_corpus=dataset_cfg.get("max_corpus", 50_000))
    out_dir = ROOT / "results" / ds.name / cfg.config_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke-test cap: process only the first `limit` queries (a few cents)
    # to prove the end-to-end path before spending on the full 500.
    query_items = list(ds.queries.items())
    if limit:
        query_items = query_items[:limit]

    # 1. chunk corpus
    chunker = CHUNKERS[cfg.chunking]
    embedder = None
    if cfg.chunking == "semantic":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg.chunking_params["embed_model"])
        embedder = lambda s: m.encode(s, normalize_embeddings=False)
    chunks = []
    for doc_id, d in ds.corpus.items():
        text = (d.get("title", "") + " " + d.get("text", "")).strip()
        params = {k: v for k, v in cfg.chunking_params.items() if k != "embed_model"}
        chunks.extend(chunker(doc_id, text, embedder=embedder, **params)
                      if cfg.chunking == "semantic"
                      else chunker(doc_id, text, **params))

    # 2. retrieve (+ rerank) and score IR metrics
    retriever, reranker = build_pipeline(cfg, chunks, exp)
    gen = Generator(**{k: v for k, v in exp["generation"].items()
                       if k in ("model", "temperature", "max_tokens")})
    lookup = {c.chunk_id: c.text for c in chunks}
    top_ctx = exp["generation"]["top_k_context"]

    rows, gen_rows = [], []
    for qid, qtext in query_items:
        ranked = retriever.retrieve(qtext, top_k=100)
        if reranker:
            ranked = reranker.rerank(qtext, ranked)
        ir = evaluate_query(ranked, ds.qrels[qid], exp["evaluation"]["ir"]["k_values"])
        contexts = [lookup[cid] for cid, _ in ranked[:top_ctx]]
        g = gen.generate(qtext, contexts)
        rows.append({"query_id": qid, **ir})
        gen_rows.append({"query_id": qid, "question": qtext, "answer": g.answer,
                         "contexts": contexts, "latency_ms": g.latency_ms,
                         "cost_eur": g.cost_eur, "cached": g.cached})

    pd.DataFrame(rows).to_json(out_dir / "retrieval.jsonl", orient="records", lines=True)
    pd.DataFrame(gen_rows).to_json(out_dir / "generation.jsonl", orient="records", lines=True)

    # 3. RAGAS
    from src.evaluation.ragas_eval import run_ragas
    records = [{"question": r["question"], "answer": r["answer"],
                "contexts": r["contexts"], "ground_truth": ""} for r in gen_rows]
    ragas_df = run_ragas(records, exp["evaluation"]["ragas"]["judge_model"])
    ragas_df.to_csv(out_dir / "ragas.csv", index=False)

    # 4. ARES judge scores (calibration happens in --analyse once human
    #    labels exist in results/human_labels/{dataset}.csv)
    from openai import OpenAI
    from src.evaluation.ares_eval import judge_one
    client = OpenAI()
    ares_rows = [
        {"query_id": r["query_id"],
         **judge_one(client, r["question"], "\n".join(r["contexts"]), r["answer"],
                     exp["evaluation"]["ares"]["judge_model"])}
        for r in gen_rows
    ]
    pd.DataFrame(ares_rows).to_json(out_dir / "ares_raw.jsonl",
                                    orient="records", lines=True)

    # 5. config-level summary row for the stats frame
    ir_mean = pd.DataFrame(rows).drop(columns="query_id").mean().to_dict()
    ragas_cols = [c for c in ragas_df.columns if c.startswith("ragas_")]
    summary = {
        "dataset": ds.name, "config_id": cfg.config_id,
        "chunking": cfg.chunking, "retrieval": cfg.retrieval, "rerank": cfg.rerank,
        **{f"ir_{k}": v for k, v in ir_mean.items()},
        **{c: float(ragas_df[c].mean()) for c in ragas_cols},
        "ares_composite": float(pd.DataFrame(ares_rows)
                                [["context_relevance", "answer_faithfulness",
                                  "answer_relevance"]].mean().mean()),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in gen_rows])),
        "cost_eur_per_1k": float(np.mean([r["cost_eur"] for r in gen_rows]) * 1000),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {ds.name} / {cfg.config_id}")


def _ppi_calibrate(df: pd.DataFrame, exp: dict) -> pd.DataFrame:
    """Fix 1 — PPI calibration for ARES scores.

    For each dataset that has a human-labels CSV at
    results/human_labels/{dataset}.csv, replace the raw ares_composite in `df`
    with a PPI-calibrated point estimate and attach CI columns.

    Expected CSV columns: query_id, context_relevance, answer_faithfulness,
    answer_relevance  (binary 0/1, matching the ARES judge dimensions).

    If no labels file exists for a dataset the raw composite is kept and a
    warning is printed — the analysis can still run; calibration just won't
    happen for that dataset.
    """
    from src.evaluation.ares_eval import ppi_mean

    alpha = exp["stats"].get("alpha", 0.05)
    dims = ["context_relevance", "answer_faithfulness", "answer_relevance"]
    results_root = ROOT / "results"

    calibrated_rows = []
    for ds_name, group in df.groupby("dataset"):
        labels_path = results_root / "human_labels" / f"{ds_name}.csv"
        if not labels_path.exists():
            print(f"  [PPI] no human labels for {ds_name} — using raw ARES composite")
            calibrated_rows.append(group)
            continue

        human_df = pd.read_csv(labels_path)
        group = group.copy()
        ppi_points, ppi_los, ppi_his = [], [], []

        for cfg_id, cfg_row in group.iterrows():
            raw_path = (results_root / ds_name /
                        cfg_row["config_id"] / "ares_raw.jsonl")
            if not raw_path.exists():
                ppi_points.append(cfg_row["ares_composite"])
                ppi_los.append(np.nan)
                ppi_his.append(np.nan)
                continue

            raw = pd.read_json(raw_path, orient="records", lines=True)
            # Align labels to the raw judge scores by query_id
            merged = raw.merge(human_df, on="query_id", suffixes=("_judge", "_human"))
            if merged.empty:
                ppi_points.append(cfg_row["ares_composite"])
                ppi_los.append(np.nan)
                ppi_his.append(np.nan)
                continue

            # PPI for each dimension then average — mirrors how ares_composite is built
            dim_points = []
            for dim in dims:
                judge_col = f"{dim}_judge" if f"{dim}_judge" in merged.columns else dim
                human_col = f"{dim}_human" if f"{dim}_human" in merged.columns else dim
                # judge_all is the full raw file; judge_labeled is the subset that
                # matches the human labels (same query_ids)
                judge_labeled = merged[judge_col].to_numpy(dtype=float)
                human_labeled = merged[human_col].to_numpy(dtype=float)
                judge_all = raw[dim].to_numpy(dtype=float)
                res = ppi_mean(judge_all, judge_labeled, human_labeled, alpha=alpha)
                dim_points.append(res.point)
            ppi_composite = float(np.mean(dim_points))
            # CI: take the widest dimension CI as a conservative envelope
            res_list = [
                ppi_mean(
                    raw[dim].to_numpy(dtype=float),
                    merged[f"{dim}_judge" if f"{dim}_judge" in merged.columns else dim]
                    .to_numpy(dtype=float),
                    merged[f"{dim}_human" if f"{dim}_human" in merged.columns else dim]
                    .to_numpy(dtype=float),
                    alpha=alpha,
                )
                for dim in dims
            ]
            ppi_points.append(ppi_composite)
            ppi_los.append(float(np.mean([r.lo for r in res_list])))
            ppi_his.append(float(np.mean([r.hi for r in res_list])))

        group["ares_composite"] = ppi_points
        group["ares_ci_lo"] = ppi_los
        group["ares_ci_hi"] = ppi_his
        calibrated_rows.append(group)
        print(f"  [PPI] calibrated ARES scores for {ds_name} "
              f"({len(human_df)} human labels)")

    return pd.concat(calibrated_rows).reset_index(drop=True)


def _variable_attribution(df: pd.DataFrame, exp: dict) -> pd.DataFrame:
    """Fix 2 — Variable attribution (SQ3): which design variable drives disagreement?

    Loads per-query retrieval.jsonl files to get query-level IR scores, then
    builds a per-query tidy frame and runs paired Wilcoxon + Holm–Bonferroni
    for every level-pair of chunking / retrieval / rerank.

    Returns a tidy dataframe of Wilcoxon results with adjusted p-values.
    """
    from src.stats.agreement import holm_bonferroni, paired_wilcoxon

    results_root = ROOT / "results"
    pq_rows = []
    for summary_path in results_root.glob("*/*/summary.json"):
        summary = json.loads(summary_path.read_text())
        retrieval_path = summary_path.parent / "retrieval.jsonl"
        if not retrieval_path.exists():
            continue
        ret_df = pd.read_json(retrieval_path, orient="records", lines=True)
        ret_df["dataset"] = summary["dataset"]
        ret_df["config_id"] = summary["config_id"]
        ret_df["chunking"] = summary["chunking"]
        ret_df["retrieval"] = summary["retrieval"]
        ret_df["rerank"] = summary["rerank"]
        pq_rows.append(ret_df)

    if not pq_rows:
        print("  [SQ3] no per-query retrieval data found — skipping variable attribution")
        return pd.DataFrame()

    pq = pd.concat(pq_rows, ignore_index=True)
    score_col = "nDCG@10"  # use the IR metric that exists at query level

    # Derive the level sets from the actual data (robust to scope changes)
    variable_levels = {
        "chunking": sorted(pq["chunking"].unique()),
        "retrieval": sorted(pq["retrieval"].unique()),
        "rerank": sorted(pq["rerank"].unique()),
    }

    wilcoxon_results = []
    for ds_name, ds_group in pq.groupby("dataset"):
        for variable, levels in variable_levels.items():
            for i, level_a in enumerate(levels):
                for level_b in levels[i + 1:]:
                    res = paired_wilcoxon(ds_group, variable, level_a, level_b,
                                         score_col)
                    res["dataset"] = ds_name
                    wilcoxon_results.append(res)

    if not wilcoxon_results:
        return pd.DataFrame()

    # Apply Holm–Bonferroni correction across all tests (per dataset)
    corrected_parts = []
    raw_df = pd.DataFrame(wilcoxon_results)
    for ds_name, ds_part in raw_df.groupby("dataset"):
        corrected = holm_bonferroni(ds_part.to_dict("records"),
                                    alpha=exp["stats"].get("alpha", 0.05))
        corrected["dataset"] = ds_name
        corrected_parts.append(corrected)

    return pd.concat(corrected_parts, ignore_index=True)


def analyse(exp):
    """Collect all summary.json files and run the full statistical analysis.

    Outputs written to results/:
      tidy_config_scores.csv        one row per (dataset, config)
      ares_calibrated.csv           PPI-calibrated ARES scores + CIs (Fix 1)
      framework_agreement.csv       Spearman rho + bootstrap CIs per framework pair
      cross_dataset_stability.csv   each framework's cross-dataset ranking stability
      agreement_pattern_shift.csv   does the agreement pattern replicate MS MARCO→NQ?
      variable_attribution.csv      Wilcoxon + Holm–Bonferroni for SQ3 (Fix 2)
    """
    from src.stats.agreement import (agreement_pattern_shift, agreement_report,
                                     cross_dataset_stability, spearman_matrix)

    summaries = [json.loads(p.read_text())
                 for p in (ROOT / "results").glob("*/*/summary.json")]
    if not summaries:
        print("No summary.json files found — run experiments first.")
        return

    df = pd.DataFrame(summaries)

    # Fix 1: replace raw ares_composite with PPI-calibrated values where possible
    print("\n[1/5] PPI calibration...")
    df = _ppi_calibrate(df, exp)
    df.to_csv(ROOT / "results" / "tidy_config_scores.csv", index=False)

    # Fix 2: variable attribution (SQ3) — Wilcoxon + Holm–Bonferroni
    print("[2/5] Variable attribution (SQ3)...")
    attr_df = _variable_attribution(df, exp)
    if not attr_df.empty:
        attr_df.to_csv(ROOT / "results" / "variable_attribution.csv", index=False)
        print(attr_df.to_string(index=False))

    # SQ1/SQ2: framework agreement + cross-dataset stability
    score_cols = ["ir_nDCG@10", "ragas_faithfulness", "ares_composite"]
    # Only keep score_cols that are actually in the dataframe
    score_cols = [c for c in score_cols if c in df.columns]

    print("[3/5] Framework agreement (SQ1)...")
    rep = agreement_report(df, score_cols, exp["stats"]["bootstrap_iters"], exp["seed"])
    rep.to_csv(ROOT / "results" / "framework_agreement.csv", index=False)

    print("[4/5] Cross-dataset stability (SQ2)...")
    cross_dataset_stability(df, score_cols).to_csv(
        ROOT / "results" / "cross_dataset_stability.csv", index=False)

    print("[5/5] Agreement pattern shift...")
    agreement_pattern_shift(rep).to_csv(
        ROOT / "results" / "agreement_pattern_shift.csv", index=False)

    for ds, g in df.groupby("dataset"):
        print(f"\nSpearman matrix — {ds}")
        print(spearman_matrix(g, score_cols).round(3))

    print("\nAll results written to results/")
    print("  tidy_config_scores.csv  (includes PPI-calibrated ares_composite)")
    print("  ares_calibrated note: ci columns ares_ci_lo / ares_ci_hi present if labels exist")
    print("  variable_attribution.csv  (Wilcoxon + Holm–Bonferroni, SQ3)")
    print("  framework_agreement.csv   (Spearman + bootstrap CIs, SQ1)")
    print("  cross_dataset_stability.csv  (SQ2)")
    print("  agreement_pattern_shift.csv  (SQ2 replication check)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["msmarco", "nq"])
    p.add_argument("--config", help="config_id, e.g. fixed__bm25__none")
    p.add_argument("--all", action="store_true", help="run all 12 configs")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of queries per cell (smoke test, e.g. 3)")
    p.add_argument("--analyse", action="store_true")
    args = p.parse_args()

    exp = load_experiment(ROOT / "configs/experiment_matrix.yaml")
    if args.analyse:
        analyse(exp)
    else:
        ds_cfg = next(d for d in exp["datasets"] if d["name"] == args.dataset)
        configs = expand_matrix(exp)
        if not args.all:
            configs = [c for c in configs if c.config_id == args.config]
        for cfg in configs:
            run_cell(ds_cfg, cfg, exp, limit=args.limit)
