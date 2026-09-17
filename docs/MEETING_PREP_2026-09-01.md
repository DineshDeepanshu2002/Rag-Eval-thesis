# Supervision Meeting — Final Speech (with William Baker Morrison)

> Speak naturally, pause at the paragraph breaks. Target ~8–10 minutes, then discussion.
> The moment to land slowly: **raw ARES ρ = 0.95 with IR → after calibration ρ = −0.49.**
> That reversal IS the thesis now. Keep the ACL example ready to explain *why* the judge was biased.

---

## 1. Opening (30 sec)

"Thanks for making the time, Will. Big update since we last spoke. The experiments are
finished — 24 runs, 12 configurations across two datasets — I completed the ARES human
calibration, and I have a full working draft through Chapter 8. And the calibration changed
my central finding in a way I think is genuinely more interesting. Let me walk you through
how I addressed your five points, then the results, then a couple of things I'd like your
steer on — including the submission date."

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

## 3. What I found — the calibration result (this is the headline)

"Here's the substance, and the headline changed for the better.

**SQ1 — do the frameworks agree?** Before calibration, raw ARES agreed with IR metrics
*very strongly* — Spearman ρ of 0.95. Taken at face value that would say a cheap zero-shot
LLM judge is a great proxy for traditional IR evaluation. But once I calibrated ARES against
my 200 human labels per dataset, that agreement didn't just weaken — it *reversed*. IR versus
calibrated ARES is now minus 0.49 on MS MARCO and minus 0.64 on NQ. Meanwhile IR versus
RAGAS stays a moderate, positive 0.59 to 0.64.

The mechanism is score compression: once you debias the judge against the human labels, all
twelve configurations land in a tiny band between 0.77 and 0.84 — so ARES can barely tell
them apart, and its ranking becomes essentially noise. So the real finding is a cautionary
one: **the apparent agreement between an uncalibrated LLM judge and IR metrics was an
artefact of the judge's bias — and calibration is what exposed it.** That's a useful warning,
because a lot of people are now using zero-shot LLM judges with no calibration at all.

**SQ2 — does it hold on a second dataset?** It sharpens. IR and RAGAS rankings are highly
stable across datasets — 0.94 and 0.95. But calibrated ARES has *no* cross-dataset stability
— minus 0.19, with a confidence interval spanning zero — so its ranking doesn't transfer.
That's a second, independent line of evidence that the raw agreement wasn't real signal.

**SQ3 — which design choice matters most?** Unchanged — it's per-query, n=500, untouched by
calibration. Retrieval method dominates, effect size 0.97; reranking second at 0.91; chunking
weakest at 0.34, and fixed chunking beat semantic on both datasets. Practical guidance:
invest in retrieval and reranking, don't over-engineer chunking.

**Cost** — the whole study ran on gpt-4o-mini for cents per thousand queries — a genuinely
reproducible, low-budget protocol."

## 4. How much is drafted

"On the writing: the draft is complete end to end — introduction, gap, research questions,
the critical lit review with the table, full methodology, a results chapter with all four
analysis tables, discussion, and conclusion with limitations and future work. Appendix A has
every prompt verbatim, Appendix B the annotation guideline, and the reference list is
complete. I've just finished rewriting the results, discussion and conclusion around the
calibrated numbers. So what's left is polishing, not new research."

## 5. What I'd like your steer on

"A few things I want your judgement on.

**First — is the calibration story framed right?** My reading is that the reversal is the
real contribution: it's direct evidence that an uncalibrated LLM judge can agree spuriously
with an established metric. I want to check you're comfortable with me making that the
headline, rather than the old 'frameworks agree' framing.

**Second — the negative sign.** I want to be honest that at n=12, with the scores compressed
into a narrow band, the negative correlation is partly noise — the CI crosses zero on MS
MARCO, though on NQ it's significantly negative. I frame it as 'agreement collapses to
near-zero / negative,' not 'the frameworks are strongly anti-correlated.' Does that read as
appropriately cautious to you?

**Third — single annotator.** The 400 calibration labels are mine alone. I've flagged the
lack of a second rater and Cohen's kappa as a limitation. Is that acceptable, or would you
want a second annotator on a subset?

**Fourth — construct mismatch.** IR is retrieval-only; RAGAS and ARES are end-to-end. I treat
some IR–RAGAS gap as expected by design — is that the right framing?"

## 6. The ask — submission date

"One more thing directly. The deadline is September 27th, and I'd like your permission to aim
for it. The experiments are done, the draft exists, and I've already folded in the calibrated
results — so I'm polishing, not writing from scratch. My plan is to get you a clean full draft
by September 17th, leaving the ten days the handbook asks for you to review and sign off
before the 27th. Does that work, and are you comfortable signing off toward the 27th?"

## 7. Close

"That's my update. If we can confirm the framing and the timeline today, I'll know exactly
what to do this week. Can we set the next meeting date, agree what I send beforehand, and log
this as one of my six required supervision meetings? Thank you."

---

## Cue card — numbers only (glance during the call)

| Item | Value |
|---|---|
| SQ1 IR vs ARES — **raw → calibrated** | **0.95 → −0.49** (MS MARCO) · 0.87 → **−0.64** (NQ) ← the story |
| SQ1 IR vs RAGAS (stable, positive) | 0.587 [−0.03, 0.92] · 0.643 [0.13, 0.89] |
| SQ1 RAGAS vs ARES (calibrated) | 0.266 (MSM) · −0.497 (NQ) |
| Calibrated ARES score band | 0.77–0.84 across all 12 configs (compressed) |
| SQ2 cross-dataset stability | IR 0.944 · RAGAS 0.951 · **ARES −0.189** (CI spans 0) |
| SQ3 effect sizes | retrieval 0.97 · rerank 0.91 · chunking 0.34 |
| SQ3 significance | all pairs, Holm–Bonferroni, p < 10⁻¹⁰ |
| Calibration labels | 200 MS MARCO + 200 NQ = 400, author-annotated |
| Cost | cents per 1k queries, gpt-4o-mini |
| Deadline ask | full draft to Will 17 Sep → submit 27 Sep |

## Be ready to explain in my own words
Spearman ρ · 10k-iteration bootstrap CIs · Wilcoxon signed-rank + Holm–Bonferroni ·
PPI (Angelopoulos 2023) · the ACL faithfulness example (why the judge was too lenient) ·
score compression → noisy ranking · why retrieval/rerank beat chunking.

## If he pushes back
- **"Is the sign flip a bug?"** → No. Labels have real variance (means 0.70–0.91). The
  mechanism is score compression: debiased scores sit in a 0.77–0.84 band, so ranking is
  noise-dominated → near-zero/negative correlation + no cross-dataset transfer.
- **"Why is ARES so compressed?"** → A zero-shot binary judge, once debiased, saturates near
  the top for twelve already-competent RAG configs.
- **"n=12 is thin."** → Agreed; framed exploratory, all CIs reported; SQ3 rests on per-query
  tests (n=500).
- **"Single annotator?"** → Flagged as a limitation; second-rater kappa is proposed future work.
