# Thesis Plan & To-Do — Automated RAG Evaluation Frameworks

**Student:** Dinesh · **Supervisor:** William Baker Morrison · **Institution:** Gisma University
**Draft dated:** 24 June 2026 · **Plan created:** 13 July 2026 · **Self-audit folded in:** 13 July 2026
**Target:** experiments done by end Sep 2026, writing Sep–Oct 2026, viva after.
**Internal submission target: 24 October 2026** (one week of slack before any hard deadline).

> This file is the single source of truth for what's left. It merges (a) the
> code/draft audit and (b) the "only you can do" items. Check boxes as you go.

---

## 0. Status at a glance

| Area | State |
|---|---|
| Chapters 1–4 (Gap, RQ, Lit Review, Methodology) | Draft exists (scaffolding, not submittable prose) |
| Chapters 5–7 (Results, Discussion, Conclusion) | Not started |
| Reference list | Incomplete (8+ in-text citations missing) |
| Codebase — offline parts | Done & tested (config, chunking, BM25, hybrid, IR metrics, stats) |
| Codebase — API parts | Written, **never run against live APIs** |
| Experiments (108 runs) | **None run — every number so far is synthetic** |
| ARES PPI calibration | `_ppi_calibrate()` wired into `analyse()` — **done 2026-08-02** |
| SQ3 variable-attribution | `_variable_attribution()` wired into `analyse()` — **done 2026-08-02** |
| Human calibration labels (~200×2) | Not created |
| W&B logging | Promised, not implemented |

**Critical path:** smoke-test one cell → fix live-API bugs → run all 108 →
hand-label + wire PPI → interpret → write up.

---

## Supervisor feedback — Will (received 2026-07-17)

**Nature of the ask: iterative — revise the current Ch.1–4 draft and RESEND an improved
draft. NOT the final thesis (final due end Oct; Results/Discussion/Conclusion don't exist yet).**

1. **Gap needs cited evidence, not assertion.** Replace "to the best of my knowledge / no
   comparison exists" with 3–4 cited 2024–2026 papers that point to the gap.
2. **More critical lit review.** For each paper: strength + weakness + synthesis (how it relates
   to others). Fill Will's table (Paper | Dataset | Methodology | Evaluation | Gaps), 10–15 rows.
3. **Methodology replicability detail.** Exact model versions, chunk sizes/overlap, k values,
   temperature/top-p/max-tokens, seeds, pinned software versions, hardware, prompt appendix ref.
4. **Scale scope down** (Will says 18×2×3 may be unfeasible). Options: 18→12 configs (2 chunking
   × 3 retrieval × 2 rerank), OR keep 18 on MS MARCO but run only top-8 on NQ. **← DECISION NEEDED.**
5. **Add recent 2025/2026 papers.** Search Scholar/arXiv; only cite what you actually read.

Cross-cutting (already in this plan): complete reference list, threats-to-validity paragraph,
pin versions, fix PPI bug, add W&B, `--dry-run`, retry logic, Zotero.
**Write in your own voice — Will wants your writing, not AI drafts (I coach, you write).**

---

## Phase 1 — De-risk the pipeline (target: mid–late July)

Goal: prove the API path works end-to-end and get a real cost number before
committing budget. *Cheapest first.*

- [ ] Pin exact package versions after a clean `pip install` (esp. `ragas`, `datasets`, `langchain-openai`); record them in `requirements.txt` and the thesis.
- [ ] Set `OPENAI_API_KEY`; confirm `python tests/test_core.py` still prints `ALL TESTS PASSED`.
- [ ] **Smoke-test one cell:** `python scripts/run_experiment.py --dataset msmarco --config fixed__bm25__none`.
- [ ] Fix what breaks (expected): dense retriever model download, cross-encoder reranker, RAGAS wrapper (version drift is likely), ARES `judge_one` JSON parsing.
- [ ] Verify GPT-4o per-token pricing in `src/pipeline/generation.py` (`PRICE_IN_PER_M`, `PRICE_OUT_PER_M`, `USD_TO_EUR`); record the date checked.
- [ ] Extrapolate total budget from the smoke-test's measured cost × ~108. Decide if 500 queries/dataset is affordable; adjust `n_queries` if not.

## Phase 2 — Code fixes (can run in parallel with Phase 1)

Offline / compute-free — I can help with all of these.

- [x] **Fix corpus-scale bug** (done 2026-07-17). `load_beir` previously loaded the full corpus (~8.8M passages MS MARCO / ~2.6M NQ) while sampling only queries — infeasible to index on a laptop. Now pools relevant docs + seeded distractors, capped at `max_corpus` (default 50k, set per-dataset in YAML). Unit-tested (`test_corpus_pooling`). **Document as a threat to validity** (reduced retrieval difficulty vs full index) in methodology.
- [x] **Wire PPI into `analyse()`.** Done 2026-08-02. `_ppi_calibrate()` loads `results/human_labels/{dataset}.csv`, calls `ppi_mean()` per dimension per config, writes `ares_ci_lo`/`ares_ci_hi`; falls back gracefully when labels absent. Dry-run verified.
- [x] **Wire the variable-attribution analysis into `analyse()` (SQ3).** Done 2026-08-02. `_variable_attribution()` loads per-query `retrieval.jsonl`, runs `paired_wilcoxon()` for every level-pair of chunking/retrieval/rerank, applies `holm_bonferroni()`, saves `variable_attribution.csv`. Dry-run verified.
- [ ] **Set up a reference manager (Zotero + Harvard style) today.** Import every cited paper now, not at the end. Harvard formatting is ~10% of the grade (Handbook §4). ~1 hour that saves ~20 later. *(Audit #4)*
- [ ] Implement **Weights & Biases logging** (config + per-run metrics) — promised in methodology and `configs/experiment_matrix.yaml` but absent from code.
- [ ] Add a `--dry-run` / offline fallback path so full orchestration can be tested without burning API calls.
- [ ] Confirm `chunks_to_docs` dedupe + qrel mapping behaves on real BEIR data (not just the unit test).

## Phase 3 — Run the experiments (target: Aug → mid-Sep) — **only you can do**

Needs your API key, compute, money, and time.

- [ ] **Aug:** all 18 configs on MS MARCO — `--dataset msmarco --all`.
- [ ] **Early Sep:** all 18 configs on NQ — `--dataset nq --all`.
- [ ] Monitor cost/latency as you go (caching makes reruns free — lean on it).
- [ ] **Add failure recovery to the orchestrator** *(audit #7)*: per-call retry-with-backoff on rate-limits/OOM/timeouts; after 3 failed retries, log the failed (dataset, config, query) and continue rather than crash. Report any failed configs transparently in the results chapter. *(Currently `run_cell` has no retry — one bad query aborts the whole run.)*
- [ ] **3-2-1 backup of `results/`** *(audit #9)*: primary local disk + cloud (Drive/OneDrive) + third copy (GitHub, LFS for large files); run the copy after each batch. Raw jsonl/csv/json is irreplaceable once the API spend is gone.
- [ ] **Pin dataset versions** *(audit #10)*: record the exact BEIR snapshot used (your loader already freezes it via the `thakur/BEIR/datasets/{name}.zip` URL — capture that URL + access date). Note the NQ variant (short vs long answer) in a `DATASET_VERSIONS.md`.

## Phase 4 — Human annotation for ARES (target: early Sep) — **only you can do**

A few days of your own work; needed for PPI calibration.

- [ ] Write the **annotation guideline** (binary: context_relevance, answer_faithfulness, answer_relevance) → thesis appendix.
- [ ] Hand-label **~200 examples for MS MARCO** → `results/human_labels/msmarco.csv`.
- [ ] Hand-label **~200 examples for NQ** → `results/human_labels/nq.csv`.
- [ ] **Inter-annotator agreement** *(audit #6)*: recruit a second annotator (lab-mate/friend who follows the guideline) to double-label ~40 examples per dataset; report **Cohen's kappa** in the appendix. If impossible, explicitly frame the set as "author-annotated, single-rater" in threats-to-validity.
- [ ] Re-run `--analyse` so PPI-calibrated ARES scores + CIs flow through.

## Phase 5 — Analysis & interpretation (target: mid-Sep) — **mostly you**

- [ ] `python scripts/run_experiment.py --analyse` → produces the 4 results tables (tidy scores, framework agreement, cross-dataset stability, agreement-pattern shift).
- [ ] **Interpret** (the real intellectual contribution — cannot be pre-written):
  - [ ] SQ1: Do RAGAS / ARES / IR agree on rankings on MS MARCO? Why / why not?
  - [ ] SQ2: Do the agreement patterns generalise to NQ?
  - [ ] SQ3: Which variable (chunking / retrieval / rerank) drives the most disagreement?
  - [ ] Practitioner guidance on framework selection (a stated deliverable).
- [ ] **Apply Holm–Bonferroni to all pairwise framework significance tests** *(audit #5)* — reporting raw p-values across 108 runs is a viva vulnerability. (Function exists; ensure it's actually run — see Phase 2 wiring task.)
- [ ] **Sensitivity analysis (optional, if time)** *(audit #11)*: for the best-performing config, sweep `top_k_context` (currently 5) and chunk size (currently 256 fixed / 384 semantic — *not* 512) to pre-empt "why these values?" at viva.

## Phase 6 — Write the thesis (target: Sep → Oct) — **only you can do**

Realistically 12,000–20,000 words. Draft chapters are scaffolding, not prose.

- [ ] Introduction (distinct from the Gap section).
- [ ] Full literature review (~2,500 words from `docs/lit_review_outline.md`).
- [ ] Methodology in your own words (rewrite `docs/methodology_draft.md`).
- [ ] Results chapter (tables + figures from Phase 5).
- [ ] Discussion (ties interpretation back to the RQ and the literature).
- [ ] Conclusion + practitioner guidance + future work.
- [ ] Abstract (write last).
- [ ] **Appendix A — all prompts verbatim** *(audit #8)*: generation (`PROMPT_TEMPLATE`), ARES judge (`JUDGE_PROMPT`), and the pinned RAGAS prompt versions. Verify they match what actually ran (reference the commit hash).

### Supervisor draft-review milestones *(audit #2 — Handbook requires 6 supervision meetings + draft review)*
- [ ] Methodology chapter → to supervisor for review (~mid-Aug).
- [ ] Results chapter draft → to supervisor (~mid-Sep).
- [ ] **Full draft → supervisor sign-off ≥10 days before submission** (Handbook §7) — i.e. by ~14 Oct given the 24 Oct internal target.

### Draft fixes to fold in while writing
- [ ] **Complete the reference list.** Missing from §5 but cited in text: Lewis et al. (2020), Karpukhin et al. (2020), Khattab & Zaharia (2020), Reimers & Gurevych (2019), Robertson & Zaragoza (2009), Thakur et al. (2021), Saad-Falcon et al. (2024), Zheng et al. (2023).
- [ ] Add **construct-mismatch framing** (IR = retrieval-only vs RAGAS/ARES = end-to-end; some disagreement is expected by design — reframe into comparable pairings).
- [ ] Add **sparse-qrels caveat** for MS MARCO / NQ (recall & nDCG unreliable under sparse judgments) to threats-to-validity.
- [ ] Add **self-preference bias** note (GPT-4o as both generator and RAGAS judge).
- [ ] Note **n=18 statistical power** limit on the rank-correlation headline (report CIs on the correlations).
- [ ] Add recent (2024–2025) papers once verified (see Phase 0-side task below).

## Phase 7 — Reading & viva prep (ongoing) — **only you can do**

- [ ] Read for real (Card 7 "never bluff"): the 6 Card-3 papers + RRF (Cormack 2009) + PPI (Angelopoulos 2023) + a chunking paper.
- [ ] Be able to defend: DSR framing, statistical choices, ARES adaptation (zero-shot judge + PPI vs full trained judges), dataset choice.
- [ ] Viva rehearsal.

## Phase 8 — Supervisor process (ongoing) — **only you can do**

- [ ] Meeting with William; bring your Card 6 questions (gap sharpness, scope feasibility, citation style, recommended recent papers).
- [ ] Incorporate his feedback — **he may change scope / DSR framing / word count**; update this plan accordingly.
- [ ] **Submit the Ethical Approval Form by end of July** *(audit #1)* — Handbook §5 allows ~2 weeks processing even for secondary-data category; submitting late blocks Phase 3.
- [ ] Track the **6 required supervision meetings** (Handbook minimum) — log dates here.

## Phase 9 — Buffer & stretch goals

- [ ] **Buffer week: 17–24 Oct** *(audit #3)* — absorb slippage (API failures, ethics delay, scope changes, illness). Do **not** plan work into it.
- [ ] **(Stretch) Workshop paper** *(audit #12)*: a 4-page write-up is achievable from this work and strengthens CV + viva. Candidate 2027 venues: RAG-adjacent workshops at ACL/EMNLP, BIRDS, GEM.

---

## Side task (compute-free, can start now)
- [ ] Web search for verified recent (2024–2025) RAG-evaluation papers to strengthen §3 (you asked for these; must be real, not from memory).

## Risk register
| Risk | Mitigation |
|---|---|
| RAGAS API drift breaks the wrapper | Pin working version immediately after smoke test; record it |
| Budget overrun on 108 runs | Estimate from smoke test first; reduce `n_queries` before scaling |
| ARES calibration slips (labels are slow) | Start the 200×2 labels early Sep, don't leave to the end |
| Supervisor changes scope late | Keep experiments modular; this plan is editable |
| n=18 → weak agreement significance | Report CIs honestly; frame as exploratory where needed |

## Where things live
```
configs/experiment_matrix.yaml   design, seeds, judges, stats params
scripts/run_experiment.py        orchestrator (--dataset/--config/--all/--analyse)
src/...                          pipeline + evaluation + stats
results/                         (created on first run — nothing yet)
docs/                            lit_review_outline.md, methodology_draft.md
THESIS_PLAN.md                   this file
```
