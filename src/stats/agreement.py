"""Statistical analysis answering the RQ directly.

Inputs: a tidy dataframe with one row per (dataset, config_id) and one
column per framework score, e.g.:

    dataset  config_id                 ir_ndcg  ragas_composite  ares_composite
    msmarco  fixed__bm25__none         0.41     0.63             0.58
    ...

Three analyses:
  1. Framework agreement  — Spearman rho between framework rankings of the
     18 configs, per dataset, with bootstrap CIs.
  2. Cross-dataset generalisation — does each framework rank configs the
     same way on MS MARCO vs NQ? And do the *agreement patterns* replicate?
  3. Variable attribution — which design variable (chunking / retrieval /
     rerank) drives disagreement? Wilcoxon on per-query paired scores +
     rank-biserial effect size.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------ 1. framework agreement
def spearman_matrix(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Pairwise Spearman rho between framework rankings of configs."""
    out = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
    for a, b in itertools.product(score_cols, repeat=2):
        rho, _ = stats.spearmanr(df[a], df[b])
        out.loc[a, b] = rho
    return out


def bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                       n_boot: int = 10_000, seed: int = 42,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired bootstrap CI for Spearman rho (resampling configs jointly)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    point = stats.spearmanr(x, y).statistic
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # degenerate resamples (all-tied) yield nan; keep and drop later
        boots[i] = stats.spearmanr(x[idx], y[idx]).statistic
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def agreement_report(df: pd.DataFrame, score_cols: list[str],
                     n_boot: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Long-form table: framework pair, rho, CI — per dataset."""
    rows = []
    for ds, g in df.groupby("dataset"):
        for a, b in itertools.combinations(score_cols, 2):
            rho, lo, hi = bootstrap_spearman(
                g[a].to_numpy(), g[b].to_numpy(), n_boot=n_boot, seed=seed)
            rows.append({"dataset": ds, "framework_a": a, "framework_b": b,
                         "spearman_rho": rho, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# --------------------------------------- 2. cross-dataset generalisation
def cross_dataset_stability(df: pd.DataFrame, score_cols: list[str],
                            ds_a: str = "msmarco", ds_b: str = "nq") -> pd.DataFrame:
    """For each framework: Spearman between its config ranking on dataset A
    vs dataset B. High rho = the framework's verdicts generalise."""
    a = df[df.dataset == ds_a].set_index("config_id")
    b = df[df.dataset == ds_b].set_index("config_id")
    common = a.index.intersection(b.index)
    rows = []
    for col in score_cols:
        rho, lo, hi = bootstrap_spearman(
            a.loc[common, col].to_numpy(), b.loc[common, col].to_numpy())
        rows.append({"framework": col, "rho_cross_dataset": rho,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def agreement_pattern_shift(report: pd.DataFrame,
                            ds_a: str = "msmarco", ds_b: str = "nq") -> pd.DataFrame:
    """Does the *pattern* of pairwise agreement replicate across datasets?
    Returns per framework-pair: rho on A, rho on B, delta, and whether the
    95% CIs overlap (a conservative replication check)."""
    a = report[report.dataset == ds_a].set_index(["framework_a", "framework_b"])
    b = report[report.dataset == ds_b].set_index(["framework_a", "framework_b"])
    rows = []
    for key in a.index:
        ra, rb = a.loc[key], b.loc[key]
        overlap = not (ra.ci_hi < rb.ci_lo or rb.ci_hi < ra.ci_lo)
        rows.append({"framework_a": key[0], "framework_b": key[1],
                     f"rho_{ds_a}": ra.spearman_rho, f"rho_{ds_b}": rb.spearman_rho,
                     "delta": rb.spearman_rho - ra.spearman_rho,
                     "ci_overlap": overlap})
    return pd.DataFrame(rows)


# ------------------------------------------------ 3. variable attribution
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial effect size for paired samples: r = 1 - 2W_minus/W_total.
    Range [-1, 1]; |r| ~ .1 small, .3 medium, .5 large (rough guide)."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_tot = ranks.sum()
    return float(2 * w_pos / w_tot - 1)


def paired_wilcoxon(per_query: pd.DataFrame, variable: str, level_a: str,
                    level_b: str, score_col: str) -> dict:
    """Paired Wilcoxon signed-rank across queries, holding other variables
    fixed by averaging within query over the remaining matrix cells.

    per_query columns: query_id, chunking, retrieval, rerank, <score_col>
    """
    ga = (per_query[per_query[variable] == level_a]
          .groupby("query_id")[score_col].mean())
    gb = (per_query[per_query[variable] == level_b]
          .groupby("query_id")[score_col].mean())
    common = ga.index.intersection(gb.index)
    x, y = ga.loc[common].to_numpy(), gb.loc[common].to_numpy()
    if np.allclose(x, y):
        return {"variable": variable, "a": level_a, "b": level_b,
                "W": np.nan, "p": 1.0, "effect_r": 0.0, "n": len(common)}
    w, p = stats.wilcoxon(x, y)
    return {"variable": variable, "a": level_a, "b": level_b,
            "W": float(w), "p": float(p), "effect_r": rank_biserial(x, y),
            "n": int(len(common))}


def holm_bonferroni(results: list[dict], alpha: float = 0.05) -> pd.DataFrame:
    """Holm–Bonferroni correction over the family of Wilcoxon tests."""
    df = pd.DataFrame(results).sort_values("p").reset_index(drop=True)
    m = len(df)
    df["p_adj"] = [min(1.0, p * (m - i)) for i, p in enumerate(df["p"])]
    df["p_adj"] = df["p_adj"].cummax()  # enforce monotonicity
    df["significant"] = df["p_adj"] < alpha
    return df
