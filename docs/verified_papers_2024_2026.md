# Verified Recent Papers (2024–2026) for Thesis Literature Review
# Source: deep-research workflow, adversarially verified, 2026-08-02
# IMPORTANT: Read the abstract + conclusion of each before citing (Will's instruction).

## KEY FINDING: Gap is NOT pre-empted
No paper found conducts a systematic head-to-head comparison of RAGAS, ARES,
and IR metrics on the SAME RAG configurations to test ranking agreement across
datasets. Your research question is open and defensible as of August 2026.

---

## Group 1: Cite to replace gap assertions with evidence

### Gan et al. (2025) — Comprehensive RAG Evaluation Survey
- arXiv: https://arxiv.org/abs/2504.14891
- Citable quote (gap evidence): "Despite the abundant evaluation frameworks at
  present, individual ones are somewhat limited in their metrics and methods of
  evaluation."
- Citable quote (open problem): "Determining an efficient method for system
  evaluation, or striking a balance between cost and effectiveness, is one of
  the directions for future research."
- Strength: 2025 survey explicitly names evaluation methodology as open problem.
- Weakness: Purely descriptive — no head-to-head empirical comparison.
- How to use: Replace "to the best of my knowledge, no comparison exists" with
  "As Gan et al. (2025) document, individual RAG evaluation frameworks are each
  limited in scope, and balanced system evaluation remains an open research
  direction."

### Brown, Roman & Devereux (2025) — Systematic Literature Review of RAG Evaluation
- arXiv: https://arxiv.org/abs/2508.06401
- Citable quote: "This juxtaposition highlights the trade-off between detailed,
  qualitative insights and streamlined quantitative evaluation, prompting
  critical questions about whether future frameworks might integrate the
  strengths of both methods to achieve a balanced, robust evaluation strategy."
- Also: Characterises ARES as quantitative/annotation-dependent and RAGAS as
  reference-free/prompt-sensitive — framing them as separately-designed with
  differing reliability properties, never mutually validated.
- Strength: MOST directly supports your gap — poses cross-framework comparison
  as explicit future work.
- Weakness: No experiments; purely conceptual review.
- How to use: "Brown et al. (2025) identify cross-framework reconciliation as an
  open question, noting that RAGAS and ARES were designed with contrasting
  reliability assumptions and have not been empirically compared on the same
  configurations."

### "Auepora" Survey (2024) — LLM-as-Judge Standardisation
- arXiv: https://arxiv.org/abs/2405.07437
- Citable quote: "there's no universally applicable grading scale and prompting
  text, complicating the standardization of 'LLM as a Judge'"
- Categorises RAGAS (LLM-as-judge/cosine-sim) and ARES (LLM+classifier) side-
  by-side in Table 1, showing they are the two recognised approaches — but
  never tested against each other.
- Strength: Establishes RAGAS and ARES as the comparable frameworks; names lack
  of standardised methodology as an open problem.
- Weakness: No cross-framework empirical comparison.
- How to use: Cite alongside the gap to show the field recognises the two main
  approaches but has not reconciled them.

---

## Group 2: Critical literature review (strength / weakness / synthesis)

### Saad-Falcon et al. (2024) — ARES (already in your draft)
- Venue: NAACL 2024 — https://aclanthology.org/2024.naacl-long.20
- Strength: First framework to use fine-tuned, domain-specific LLM judges per
  RAG component; claims +59.3 and +14.4 pp over RAGAS on context/answer
  relevance.
- Weakness: Validated only on KILT and SuperGLUE — "unable to generalise when
  making more drastic shifts in domain, such as switching languages." Comparison
  to RAGAS uses different datasets/configs, not a controlled common ground.
- Synthesis: ARES outperforms RAGAS on its OWN test sets, but this is not a
  neutral comparison — exactly the problem your study addresses by using the
  same 18 configurations for both.

### RAGChecker — Shi et al. (2024)
- arXiv: https://arxiv.org/abs/2408.08067
- Strength: Proposes fine-grained claim-level evaluation; claims better human-
  judgment correlation than existing metrics via meta-evaluation.
- Weakness: Another framework validated in isolation — adds to the fragmentation
  it diagnoses. Does not compare against RAGAS/ARES on the same data.
- Synthesis: RAGChecker explicitly states "comprehensive evaluation of RAG
  systems is still challenging due to the modular nature of RAG, evaluation of
  long-form responses and reliability of measurements" — supports your gap.

---

## Group 3: LLM-as-judge bias (threats-to-validity section)

### LLM Familiarity/Perplexity Bias (verified claim)
- Finding: LLM judges systematically assign higher scores to lower-perplexity
  (more stylistically familiar) outputs — driven by familiarity, not quality.
  Holds even when the output was NOT self-generated.
- How to use in threats-to-validity: "GPT-4o acting as both generator and RAGAS
  judge introduces a known self-preference bias — LLM judges have been shown to
  favour stylistically familiar outputs independent of actual quality."
- NOTE: Verify the specific paper title/authors before citing — the claim was
  confirmed 3-0 but the synthesis was cut off before recording the full citation.
  Search: "LLM judge perplexity bias" OR "LLM evaluator familiarity bias" 2024.

---

## Still needed (search yourself)
- A chunking paper (e.g. comparing fixed vs semantic chunking strategies)
- A paper on cross-dataset generalisation for dense retrievers (BEIR paper
  already cited; look for 2024–2025 follow-ups)
- The specific paper behind the perplexity-bias claim (see above)

## Harvard format stubs (complete after reading)
Gan, [first initial], et al. (2025) '[title TBC]', arXiv preprint arXiv:2504.14891.
Brown, [first initial], Roman, [first initial] and Devereux, [first initial] (2025)
  '[title TBC]', arXiv preprint arXiv:2508.06401.
Shi, [first initial], et al. (2024) 'RAGChecker: A Fine-grained Framework for
  Diagnosing Retrieval-Augmented Generation', arXiv preprint arXiv:2408.08067.
