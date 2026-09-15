# Supervision Call — Prep Sheet (18 Aug 2026, with William Baker Morrison)

## 0. Open with a 30-second status line
"Since your feedback I've finished all the experiments and revised Chapters 1–4.
All 24 runs are done, and I have a full working draft through Chapter 8. I'd like
to walk you through how I addressed your five points, then ask a few questions."

---

## 1. How I addressed your five feedback points
- **Gap → cited, not asserted.** Now grounded in Gan et al. (2025) and Brown, Roman
  & Devereux (2025), who both explicitly call for cross-framework comparison.
- **Critical lit review + synthesis.** Strength / weakness / synthesis per framework;
  filled the summary table (your "B — Literature Summary Table" template).
- **Replicable methodology.** Pinned package versions, seed = 42 throughout, exact chunk
  sizes (256 fixed / 384 semantic), temp 0.0, max_tokens, all prompts in Appendix A.
- **Scaled scope down.** 18 → **12 configs** (2 chunking × 3 retrieval × 2 rerank),
  kept both datasets. 24 runs total.
- **Recent papers.** Added Gan (2025), Brown/Roman/Devereux (2025), Yu (2024).
- **GenAI note.** Understood — thesis reflects my own work; happy to talk through any part.

## 2. Headline results (substance to show)
- **SQ1 (MS MARCO):** IR vs RAGAS agree strongly (ρ = 0.951), but one framework pair
  only reaches ρ = 0.587 with a CI crossing zero → frameworks *do* disagree = the core point.
- **SQ2:** agreement patterns stable across datasets (ρ ≈ 0.94–0.95).
- **SQ3:** retrieval method + reranking dominate (large effect sizes, p ≈ 1e-45);
  chunking significant but weakest.
- **Cost:** whole study ran for cents per 1k queries on gpt-4o-mini.

## 3. Questions / doubts to raise (my agenda)
1. **ARES PPI (most important):** no human labels collected, so ARES runs zero-shot
   *without* PPI. OK to frame as an adapted lightweight variant + limitation, or should
   I hand-label ~200 examples per dataset?
2. **Construct mismatch:** IR = retrieval-only vs RAGAS/ARES = end-to-end. Is comparing
   their rankings fair, or should I only compare like-for-like pairs?
3. **Statistical power:** at n = 12 the correlation CIs are wide (one crosses zero).
   Is an exploratory framing acceptable?
4. **Scope check:** is 12 × 2 enough, or do you want the full 18 back now that it runs?
5. **Next deliverable:** full draft next, or chapter-by-chapter? Target for end-Oct?
6. **Your two docs:** confirm I applied "A — Concept Development" and "B — Literature
   Summary Table" the way you intended.

## 4. Be ready to explain in my own words
Spearman ρ · 10k-iteration bootstrap CIs · Wilcoxon signed-rank + Holm–Bonferroni ·
PPI (Angelopoulos 2023) · why retrieval/rerank beat chunking.

## 5. Close
Confirm next meeting date + what to send before it. (Log this as one of the 6 required
supervision meetings.)
