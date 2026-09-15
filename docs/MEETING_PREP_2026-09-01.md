# Supervision Meeting — Final Speech (Tue 2 Sep 2026, with William Baker Morrison)

> Speak naturally, pause at the paragraph breaks. Target ~8–10 minutes, then discussion.
> The one number to land slowly: **ρ = 0.59 with a CI crossing zero** — that IS the thesis.
> Keep the ACL example in your back pocket for the ARES question.

---

## 1. Opening (30 sec)

"Thanks for making the time, Will. I've got a real update today. Since your last
feedback, the experiments are completely finished — 24 runs, 12 configurations across
two datasets — and I now have a full working draft through Chapter 8, about 12,000
words. So I'd like to walk you through how I addressed your five points, show you what
I actually found, then raise a few honest doubts — and finally ask you about the
submission date."

## 2. How I addressed your five points (tight)

"On your five points:

- **The gap** is now cited, not asserted — grounded in Gan et al. (2025) and Brown,
  Roman & Devereux (2025), who both explicitly call for cross-framework comparison.
- **Critical lit review** — strength, weakness, and synthesis per framework, and I
  filled in your Literature Summary Table template.
- **Replicability** — everything's pinned: package versions, seed 42 throughout,
  chunk sizes 256 fixed and 384 semantic, temperature 0.0, and all prompts in Appendix A.
- **Scope** — I scaled from 18 down to 12 configs: two chunking, three retrieval, two
  rerank, keeping both datasets. 24 runs total.
- **Recent papers** — added Gan (2025), Brown/Roman/Devereux (2025), and Yu (2024),
  all actually read.

And on the GenAI note — understood; the thinking and the writing are mine, and I'm
happy to talk through any part of it live."

## 3. What I found

"Now the substance.

**SQ1 — do the frameworks agree?** This is the core finding. On MS MARCO, IR metrics
and ARES agree very strongly — Spearman ρ of 0.95, CI from 0.77 to 1.0. But IR metrics
and RAGAS faithfulness agree only moderately — ρ of 0.59, and the confidence interval
actually crosses zero, from minus 0.03 up to 0.92. RAGAS and ARES sit in between at 0.73.

The takeaway is the whole point of the thesis: the evaluation framework you choose
materially changes the ranking you get. Someone using RAGAS alone could recommend a
different RAG configuration than someone using IR metrics.

**SQ2 — does that hold on a second dataset?** Yes. On Natural Questions the pattern
replicates — cross-dataset stability of 0.94 for IR, 0.95 for RAGAS, 0.85 for ARES —
and the ordering is preserved: IR–ARES strongest, IR–RAGAS weakest. So the disagreement
is a systematic property of the frameworks, not dataset noise.

**SQ3 — which design choice matters most?** I ran paired Wilcoxon signed-rank tests
with Holm–Bonferroni correction at the per-query level, n=500. Retrieval method
dominates — effect size 0.97 for BM25 versus hybrid, p around ten to the minus 45.
Reranking is second at 0.91. Chunking is the weakest lever at about 0.34 — and,
against the usual assumption, fixed chunking beat semantic chunking on both datasets.
So my practical guidance is: invest in retrieval and reranking, don't over-engineer chunking.

**Cost** — the whole study ran on gpt-4o-mini for cents per thousand queries, so it's a
genuinely reproducible, low-budget protocol."

## 4. How much is drafted

"On the writing: the draft is complete end to end — introduction, gap, research
questions, the critical lit review with the table, full methodology, a results chapter
with all four analysis tables, discussion, and conclusion with limitations and future
work. Appendix A has every prompt verbatim, Appendix B the annotation guideline, and
the reference list is complete. So what's left is polishing prose and fixing
consistency — not new research or new writing."

## 5. My doubts — where I want your steer

"Now the honest part.

**First, and most important — ARES and PPI.** I ran ARES as a zero-shot gpt-4o-mini
judge, without the human-calibration step called PPI. Let me be concrete about why that
matters. Take a real case from my data: the question 'how long to recover from ACL
surgery.' The passage says patients return to *light activity* in 2–3 weeks. The model's
answer said *full recovery* in 2–3 weeks. The LLM judge marked that faithful — a human
would mark it unfaithful, because the answer overstates the source. LLM judges are
reliably too lenient on that kind of subtle overstatement. PPI fixes it: you hand-label
about 200 examples per dataset, measure how generous the judge is, and correct the whole
set — so a raw ARES score of, say, 0.82 might become 0.72 with a proper confidence
interval.

My question: is it acceptable to submit ARES as an honest, clearly-labelled lightweight
variant *without* PPI and note it as a limitation — or do you want me to do the 200
hand-labels per dataset so it's fully calibrated? My own view is the lightweight version
is defensible for a 6-month MSc, but I want your call.

**Second — construct mismatch.** IR metrics measure retrieval only, but RAGAS and ARES
measure the whole end-to-end system. So some disagreement is expected by design. Is it
fair to compare their rankings head-to-head, or should I frame it strictly as comparable
pairings?

**Third — statistical power.** At n=12 configurations the correlation CIs are wide, and
one crosses zero. Is it acceptable to frame the agreement analysis as exploratory?

**Fourth — scope.** The pipeline runs cleanly now. Are you happy with 12 configs across
two datasets, or do you want the full 18 back?"

## 6. The ask — submission date

"There's one more thing I want to raise directly. The submission deadline is
September 27th, and I'd like your permission to aim for that date. My reasoning: the
experiments are done, the full draft already exists, and the references are complete —
so I'm polishing, not writing from scratch. My plan is to get you a clean full draft by
September 17th, which leaves the ten days the handbook asks for you to review and sign
off before the 27th. To keep that realistic, I'd propose we lock the ARES approach as
the lightweight no-PPI variant with an honest limitation, rather than adding a labelling
round — that's the one thing that could push me past the deadline. Does that plan work
for you, and are you comfortable signing off toward the 27th?"

## 7. Close

"That's my update. If we can settle the ARES question and the timeline today, I'll know
exactly what to do this week. Can we also set the next meeting date, confirm what I send
you beforehand, and log this as one of my six required supervision meetings? Thank you."

---

## Cue card — numbers only (glance during the call)

| Item | Value |
|---|---|
| SQ1 MS MARCO: IR vs RAGAS | ρ = 0.59, CI [−0.03, 0.92]  ← the headline |
| SQ1 MS MARCO: IR vs ARES | ρ = 0.95, CI [0.77, 1.0] |
| SQ1 MS MARCO: RAGAS vs ARES | ρ = 0.73 |
| SQ2 cross-dataset stability | IR 0.94 · RAGAS 0.95 · ARES 0.85 |
| SQ3 effect sizes | retrieval 0.97 · rerank 0.91 · chunking 0.34 |
| SQ3 significance | all pairs, Holm–Bonferroni, p < 10⁻¹⁰ |
| Cost | cents per 1k queries, gpt-4o-mini |
| Draft | ~12,000 words, all 8 chapters + appendices |
| Deadline ask | full draft to Will 17 Sep → submit 27 Sep |

## Be ready to explain in my own words
Spearman ρ · 10k-iteration bootstrap CIs · Wilcoxon signed-rank + Holm–Bonferroni ·
PPI (Angelopoulos 2023) · the ACL faithfulness example · why retrieval/rerank beat chunking.

## If he pushes back
- **"Why no PPI?"** → time/scope for a 6-month MSc; it's flagged as a limitation, code is
  wired to run it if labels are added later.
- **"n=12 is thin."** → agreed; framed exploratory, all CIs reported, per-query tests (n=500)
  carry the SQ3 claims.
- **"18 configs?"** → pipeline supports it; can add if he wants, but it risks the 27 Sep date.
