"""Fast, resumable terminal tool for hand-labelling the PPI annotation sheets.

Shows one example at a time (question / context / answer) and records three
binary human judgements. Writes incrementally to results/human_labels/{dataset}.csv
so you can stop and resume any time — already-labelled query_ids are skipped.

Usage:
    python scripts/label_cli.py msmarco
    python scripts/label_cli.py nq

At each example enter three 0/1 values (context_relevance, answer_faithfulness,
answer_relevance). Shortcuts:
    just press Enter after a digit sequence like "1 0 1"
    b  = go back one (re-label the previous example)
    s  = skip this example for now
    q  = save and quit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
LABEL_COLS = ["context_relevance", "answer_faithfulness", "answer_relevance"]


def _load_done(out_path: Path) -> dict:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(out_path)
    except pd.errors.EmptyDataError:
        return {}
    return {row["query_id"]: row for _, row in df.iterrows()}


def _save(out_path: Path, done: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"query_id": qid,
             **{c: int(rec[c]) for c in LABEL_COLS}}
            for qid, rec in done.items()]
    pd.DataFrame(rows, columns=["query_id", *LABEL_COLS]).to_csv(out_path, index=False)


def _prompt(i: int, total: int, row) -> list | str:
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{total}]  query_id={row['query_id']}")
    print("-" * 70)
    print(f"QUESTION:\n  {row['question']}\n")
    print(f"CONTEXT:\n{row['contexts_shown']}\n")
    print(f"ANSWER:\n  {row['answer']}")
    print("-" * 70)
    print("Enter 3 binary labels: context_relevance answer_faithfulness answer_relevance")
    print("  e.g. '1 0 1'   |   b=back  s=skip  q=save&quit")
    while True:
        raw = input("> ").strip().lower()
        if raw in ("q", "b", "s"):
            return raw
        parts = raw.replace(",", " ").split()
        if len(parts) == 3 and all(p in ("0", "1") for p in parts):
            return [int(p) for p in parts]
        print("  ! need exactly three 0/1 values (or b/s/q)")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("msmarco", "nq"):
        print("usage: python scripts/label_cli.py [msmarco|nq]")
        sys.exit(1)
    dataset = sys.argv[1]

    sheet_path = ROOT / "annotation" / f"{dataset}_to_label.csv"
    out_path = ROOT / "results" / "human_labels" / f"{dataset}.csv"
    sheet = pd.read_csv(sheet_path)
    done = _load_done(out_path)

    print(f"Dataset: {dataset} | {len(sheet)} examples | {len(done)} already labelled")
    i = 0
    while i < len(sheet):
        row = sheet.iloc[i]
        qid = row["query_id"]
        if qid in done:
            i += 1
            continue
        result = _prompt(i, len(sheet), row)
        if result == "q":
            break
        if result == "s":
            i += 1
            continue
        if result == "b":
            i = max(0, i - 1)
            # allow re-labelling the previous one
            done.pop(sheet.iloc[i]["query_id"], None)
            continue
        done[qid] = dict(zip(LABEL_COLS, result))
        _save(out_path, done)  # save after every label — crash-safe
        i += 1

    _save(out_path, done)
    remaining = len(sheet) - len([q for q in done if q in set(sheet["query_id"])])
    print(f"\nSaved {len(done)} labels to {out_path.relative_to(ROOT)}")
    print(f"Remaining unlabelled: {remaining}")
    if remaining == 0:
        print("✅ dataset complete!")


if __name__ == "__main__":
    main()
