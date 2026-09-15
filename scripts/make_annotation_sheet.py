"""Build human-annotation sheets for PPI calibration of the ARES judge.

For each dataset we draw a fixed random sample of ~200 queries from the
FLAGSHIP configuration (fixed__hybrid__cross_encoder) and lay out the exact
(question, retrieved context, generated answer) triple the ARES judge saw, with
three empty label columns for a human to fill in.

Why one reference config: PPI estimates the judge's systematic leniency from a
labelled sample and corrects the whole set. Labelling per-config would be
~200 x 12 x 2 labels; instead we label one representative config per dataset and
apply the correction dataset-wide, assuming the judge's bias is approximately
constant across configs (same model, same rubric). This is stated as a
limitation in the methodology.

Usage:
    python scripts/make_annotation_sheet.py

Output (one CSV per dataset):
    annotation/{dataset}_to_label.csv

Fill in the three binary columns (0/1), then move the finished file to
    results/human_labels/{dataset}.csv
and rerun:  python scripts/run_experiment.py --analyse
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]

DATASETS = ["msmarco", "nq"]
REFERENCE_CONFIG = "fixed__hybrid__cross_encoder"
SAMPLE_N = 200
SEED = 42  # matches the seed used throughout the study

# Columns the human fills in — binary 0/1, mirroring the ARES judge dimensions.
LABEL_COLS = ["context_relevance", "answer_faithfulness", "answer_relevance"]


def _format_contexts(contexts) -> str:
    """Render the retrieved passages as one readable, numbered block."""
    if not isinstance(contexts, (list, tuple)):
        return str(contexts)
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


def build_sheet(dataset: str) -> Path:
    gen_path = ROOT / "results" / dataset / REFERENCE_CONFIG / "generation.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"missing {gen_path} — has the flagship config run?")

    gen = pd.read_json(gen_path, orient="records", lines=True)
    sample = gen.sample(n=min(SAMPLE_N, len(gen)), random_state=SEED)

    sheet = pd.DataFrame({
        "query_id": sample["query_id"].values,
        "question": sample["question"].values,
        "contexts_shown": [_format_contexts(c) for c in sample["contexts"].values],
        "answer": sample["answer"].values,
    })
    # Empty columns for the human annotator (kept blank on purpose — blind labelling).
    for col in LABEL_COLS:
        sheet[col] = ""

    out_dir = ROOT / "annotation"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{dataset}_to_label.csv"
    sheet.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    print(f"Sampling {SAMPLE_N} queries from '{REFERENCE_CONFIG}' (seed {SEED})\n")
    for dataset in DATASETS:
        out_path = build_sheet(dataset)
        n = len(pd.read_csv(out_path))
        print(f"  {dataset:8s} -> {out_path.relative_to(ROOT)}  ({n} rows)")
    print(
        "\nNext: fill the three 0/1 columns, then move each file to\n"
        "  results/human_labels/{dataset}.csv\n"
        "and rerun:  python scripts/run_experiment.py --analyse"
    )


if __name__ == "__main__":
    main()
