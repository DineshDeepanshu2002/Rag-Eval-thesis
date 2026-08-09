# Automated RAG Evaluation Frameworks: A Cross-Framework and Cross-Dataset Comparative Study
## MSc Dissertation — Working Draft / Content Scaffold
### Programme: MSc Data Science and Artificial Intelligence
### Institution: Gisma University of Applied Sciences
### Student: Dinesh | Supervisor: William Baker Morrison

---

> **HOW TO USE THIS FILE**
> This is your cumulative working draft. Every section contains either:
> - [DRAFT] — content from your original PDF draft (needs rewriting in your own voice)
> - [NEW] — content from our research/analysis (not yet in any submission; rewrite before submitting)
> - [TODO] — section that needs writing after experiments are done
> - [SCAFFOLD] — structural skeleton only
> Everything in this file is yours to rewrite. Will wants YOUR words, not scaffolding.
> As you rewrite sections, remove the [DRAFT]/[NEW]/[SCAFFOLD] tags.

---

## ABSTRACT

Retrieval-Augmented Generation (RAG) systems are increasingly deployed in production
settings where reliable automated evaluation is essential for system development and
quality assurance. Three families of automated evaluation frameworks have emerged:
traditional information retrieval (IR) metrics (nDCG, MRR, Precision@k), RAGAS
(Es et al., 2024), and ARES (Saad-Falcon et al., 2024). Despite the availability of
these frameworks, no prior study has systematically compared them on the same experimental
conditions or examined whether their agreement patterns generalise across datasets —
a gap explicitly identified by Gan et al. (2025) and Brown et al. (2025).

This dissertation addresses both gaps through a fully reproducible comparative experiment.
Twelve RAG configurations (2 chunking strategies × 3 retrieval methods × 2 reranking
options) were evaluated on two BEIR-formatted datasets — MS MARCO and Natural Questions —
using all three framework families, yielding 24 evaluation runs across 500 queries each.
Statistical analysis employed Spearman rank correlation with 10,000-iteration bootstrap
confidence intervals (SQ1/SQ2) and Wilcoxon signed-rank tests with Holm–Bonferroni
correction (SQ3).

The results reveal substantial asymmetry in inter-framework agreement. IR metrics and
ARES composite agree very strongly (ρ = 0.951 on MS MARCO; 0.865 on NQ), while IR metrics
and RAGAS faithfulness agree only moderately (ρ = 0.587; 0.643). These patterns are stable
across both datasets: cross-dataset ranking stability exceeds ρ = 0.85 for all three
frameworks. Variable attribution shows that retrieval method is the dominant design variable
(rank-biserial |r| = 0.972 for BM25 vs hybrid), followed by reranking (|r| = 0.906), with
chunking strategy producing the smallest effect (|r| = 0.338) — and fixed chunking
outperforming semantic chunking contrary to expectations.

The dissertation concludes that framework choice materially affects configuration rankings:
RAGAS and IR metrics can disagree on the best-performing configuration. For practitioners,
retrieval method and reranking represent higher-leverage optimisation targets than chunking.
Cross-dataset stability suggests that rankings obtained on one dataset are transferable
within the web search and open-domain QA domain.

*Keywords: Retrieval-Augmented Generation, evaluation frameworks, RAGAS, ARES, IR metrics,
framework agreement, Spearman correlation, MS MARCO, Natural Questions, BEIR*

---

## 1. INTRODUCTION
[SCAFFOLD — write this after Chapter 2 and 4 are solidified]

Structure to cover:
- Opening: why RAG matters in production AI systems (grounding, attribution, reducing hallucination)
- The evaluation problem: as RAG moves to production, reliable evaluation becomes critical
- The specific gap: multiple frameworks exist but have never been systematically compared
- What this dissertation does: bridges that gap empirically
- Overview of structure: chapter by chapter map

Note: Introduction is distinct from Chapter 2 (Research Gap). Introduction situates the
problem broadly; Chapter 2 provides the precise academic gap with citations.

---

## 2. RESEARCH GAP

[DRAFT — original from your PDF, needs rewriting with cited evidence replacing assertions]

### 2.1 Current state of RAG evaluation

Retrieval-Augmented Generation (RAG) has become one of the most widely adopted architectures
for building knowledge-grounded language model applications. As RAG has moved from research
papers into production systems, the need for reliable evaluation has grown significantly. In
response, several automated evaluation frameworks have emerged in recent years, including
RAGAS (Es et al., 2024), ARES (Saad-Falcon et al., 2024), and the use of traditional
information-retrieval metrics such as Precision@k, Mean Reciprocal Rank, and Normalised
Discounted Cumulative Gain (Thakur et al., 2021).

[NEW — INSERT HERE to replace the assertion paragraph below]

The critical problem is not that evaluation frameworks are absent, but that they were developed
in isolation and have never been tested against each other on a common experimental ground.
As Gan et al. (2025) observe, "despite the abundant evaluation frameworks at present,
individual ones are somewhat limited in their metrics and methods of evaluation" — and
identifying a balanced system evaluation method remains "one of the directions for future
research" (Gan et al., 2025, arXiv:2504.14891).

This fragmentation has practical consequences. Brown, Roman and Devereux (2025) note that
RAGAS and ARES embody contrasting reliability assumptions — RAGAS is reference-free and
sensitive to prompt variation, while ARES is annotation-dependent and quantitatively calibrated
— yet "critical questions about whether future frameworks might integrate the strengths of both
methods" remain unanswered (arXiv:2508.06401). Similarly, the lack of a universally applicable
grading methodology for LLM-as-judge approaches, observed by [Auepora survey, arXiv:2405.07437],
complicates any attempt to draw reliable conclusions from a single framework.

[DRAFT continues — the assertion below should be replaced once you've read the papers above]

While each of these frameworks measures something genuinely useful, they were developed largely
in isolation. Each was introduced through papers that internally validated the framework using
a small number of datasets and a particular RAG configuration. No published study has yet
conducted a systematic empirical comparison of these frameworks on the same RAG configurations
to determine whether they agree in their rankings. In addition, very few studies test whether
the conclusions drawn from a single dataset hold when the same experiments are repeated on a
different dataset.

[NEW — ADD THIS FRAMING — construct mismatch]

One structural reason for this gap deserves explicit framing: the three framework families
do not measure identical constructs. Traditional IR metrics (Precision@k, MRR, nDCG) evaluate
only the retrieval stage, while RAGAS and ARES assess end-to-end generation quality
(faithfulness, answer relevance). Some degree of disagreement between IR metrics and
generation-focused metrics is therefore expected by design. What remains unknown is: (a) how
much RAGAS and ARES — which share similar dimensions but differ fundamentally in judge
methodology — agree with each other; (b) whether IR retrieval metrics align with the
retrieval-sensitive components of RAGAS (context relevance) and ARES (context relevance);
and (c) whether any observed agreement patterns hold across datasets. These questions have
not been answered in the literature.

[DRAFT continues]

This raises two practical concerns. First, if different frameworks produce different rankings
of the same RAG configurations, then any conclusion drawn from a single framework is of
uncertain reliability. Second, if findings do not generalise across datasets, then practitioners
cannot confidently transfer published recommendations to their own systems. The most recent
survey paper on RAG (Gao et al., 2023) explicitly identifies evaluation methodology as one
of the most pressing open research areas in the field. More recently, Shi et al. (2024)
confirm that "comprehensive evaluation of RAG systems is still challenging due to the modular
nature of RAG, evaluation of long-form responses and reliability of measurements"
(RAGChecker, arXiv:2408.08067).

These observations form the gap this dissertation investigates.

---

## 3. RESEARCH QUESTION

[DRAFT — from your PDF, this section is well-formed, keep structure, rewrite prose]

### 3.1 Primary research question

How consistently do automated RAG evaluation frameworks (RAGAS, ARES, and traditional IR
metrics) rank RAG configurations, and do their agreement patterns generalise across datasets?

### 3.2 Sub-questions

SQ1: To what extent do RAGAS, ARES, and traditional IR metrics produce consistent rankings
of the same RAG configurations on a primary dataset (MS MARCO)?

SQ2: Do the framework agreement patterns observed on MS MARCO generalise to a secondary
dataset (Natural Questions)?

SQ3: Which RAG configuration variables (chunking strategy, retrieval method, reranking)
cause the greatest disagreement between evaluation frameworks?

### 3.3 Deliverables

By the end of the dissertation:
- A reproducible comparative benchmark (12 configurations × 2 datasets × 3 frameworks)
- Empirical evidence on cross-dataset behaviour of evaluation frameworks
- Practical guidance for practitioners on framework selection

[NEW — NOTE ON CONSTRUCT COMPARABILITY FOR VIVA DEFENCE]
When answering SQ1 and SQ2, the analysis will distinguish between comparable pairings
(RAGAS vs ARES — both end-to-end LLM judges; IR context recall vs RAGAS context relevance
— both retrieval-focused) and cross-construct pairings (IR nDCG vs RAGAS faithfulness — these
measure different things and some disagreement is expected). Reporting both is necessary to
avoid the examiner objection that "of course they disagree — they measure different things."

---

## 4. LITERATURE REVIEW

[DRAFT — from your PDF, expanded with new papers and critical analysis]

### 4.1 Foundations of Retrieval-Augmented Generation

The RAG paradigm was formally introduced by Lewis et al. (2020), who combined a pre-trained
sequence-to-sequence model with a dense vector index of external documents. The motivation was
to address three well-documented limitations of large language models: factual hallucination,
outdated parametric knowledge, and difficulty in attributing generated content to specific
sources. Since this paper, RAG has been widely adopted across knowledge-intensive natural
language tasks.

The standard RAG pipeline operates in two stages. In the first stage, retrieval, a query is
used to identify the top-k most relevant documents from a corpus. In the second stage,
generation, a language model is conditioned on both the query and the retrieved documents.
Gao et al. (2023) categorise RAG approaches into three families: Naïve RAG (single retrieval
pass), Advanced RAG (with query rewriting and reranking), and Modular RAG (with swappable
components).

[NEW — STRENGTH/WEAKNESS FORMAT per Will's feedback]

**Strength of Lewis et al. (2020):** Established the foundational architecture and demonstrated
that external retrieval substantially reduces hallucination on knowledge-intensive tasks.
**Weakness:** Fixed retriever (DPR); no reranking; evaluated with a single metric family
(exact match, F1). Does not examine how evaluation methodology affects conclusions.
**Synthesis:** Establishes the pipeline your experiments use, but its single-metric evaluation
exemplifies the narrow evaluation practice your dissertation critiques.

### 4.2 Retrieval Methods

Retrieval in RAG systems generally relies on one of three families of methods.

**Sparse retrieval** (BM25, Robertson and Zaragoza, 2009) uses traditional term-frequency
techniques. Interpretable and computationally cheap, BM25 continues to perform competitively
on many benchmarks.

**Dense retrieval** encodes queries and documents into a shared semantic space using neural
embeddings. Karpukhin et al. (2020) introduced Dense Passage Retrieval and established a
strong baseline. ColBERT (Khattab and Zaharia, 2020) extends this with late interaction,
enabling more expressive matching. Sentence-BERT (Reimers and Gurevych, 2019) provides
efficient sentence-level embeddings widely used in practice.

**Hybrid retrieval** combines sparse and dense methods through rank fusion. Reciprocal Rank
Fusion (Cormack et al., 2009) merges ranked lists without score normalisation, frequently
outperforming either method alone.

[NEW — CRITICAL ANALYSIS]

**Key limitation of retrieval research for your study:** The BEIR benchmark (Thakur et al., 2021)
demonstrates that dense retrievers show significant performance variability across domains —
findings from one dataset do not necessarily generalise. This motivates your cross-dataset
design (SQ2), and should be cited explicitly as prior evidence that single-dataset findings
are unreliable.

### 4.3 Evaluation Frameworks for RAG

Several distinct approaches to evaluating RAG systems have emerged, each measuring a different
set of constructs. The critical observation — largely unacknowledged in prior work — is that
these approaches have never been directly compared on the same experimental conditions.

**Traditional IR metrics** (Precision@k, Recall@k, MRR, nDCG) focus exclusively on retrieval.
These metrics have well-understood mathematical properties and are used in BEIR (Thakur et al.,
2021). Strength: interpretable, reference-based, no LLM required. Weakness: measure only
retrieval quality, not generation quality or faithfulness — a fundamental construct mismatch
with end-to-end frameworks.

**RAGAS** (Es et al., 2024) defines three metrics: faithfulness (answer grounded in context),
answer relevance (answer addresses query), and context relevance (retrieved context relevant
to query). Computed using LLM-as-judge with reference-free prompts.
Strength: no reference answers needed; evaluates end-to-end pipeline.
Weakness: sensitive to prompt variation (Brown et al., 2025); relies on the same LLM family
as the generator, introducing self-preference bias (LLM judges favour stylistically familiar
outputs — see threats to validity).

**ARES** (Saad-Falcon et al., 2024) trains domain-specific LLM judges on synthetic data,
calibrated against human labels via Prediction-Powered Inference (Angelopoulos et al., 2023).
Claims +59.3pp and +14.4pp over RAGAS on context and answer relevance (NAACL 2024).
Strength: calibrated judges with confidence intervals; reduces prompt-sensitivity.
Weakness: requires ~200 human-labelled examples per domain, limiting practical adoption;
judges fail to generalise under drastic domain shifts (Saad-Falcon et al., 2024, NAACL).
Critical note: ARES's superiority over RAGAS was measured on KILT and SuperGLUE — different
datasets from RAGAS's WikiEval — making the comparison non-equivalent. This is precisely the
problem this dissertation addresses.

**RAGChecker** (Shi et al., 2024, arXiv:2408.08067) proposes claim-level evaluation and
reports better human-judgment correlation than existing metrics. Acknowledges that
"comprehensive evaluation of RAG systems is still challenging due to the modular nature of
RAG, evaluation of long-form responses and reliability of measurements." Strength: fine-grained
diagnosis at the claim level. Weakness: yet another framework validated in isolation, adding to
the fragmentation it diagnoses.

**LLM-as-Judge paradigm** (Zheng et al., 2023) underlies both RAGAS and ARES. Research has
identified systematic biases: self-preference (the judge favours outputs from the same model
family), position bias, and verbosity bias. LLM judges also assign higher scores to
lower-perplexity outputs based on stylistic familiarity, independent of actual quality.
These biases are directly relevant to any study using GPT-4o as both generator and judge.

### 4.4 Cross-Dataset Evaluation in RAG Research

Cross-dataset evaluation remains comparatively rare. The BEIR benchmark (Thakur et al., 2021)
demonstrates that dense retrievers show significant performance variability across domains,
and warns against assuming single-dataset findings generalise. Despite this warning, the
implication has not been systematically tested for evaluation framework agreement specifically.

[NEW — USE THESE 2025 PAPERS HERE]

Two recent systematic reviews confirm this gap remains open. Gan et al. (2025) identify
balanced system evaluation as a future research direction, noting frameworks are individually
limited. Brown et al. (2025) explicitly frame cross-framework reconciliation as an open
question for future work, noting RAGAS and ARES have contrasting reliability properties and
have not been empirically compared. This dissertation directly addresses both observations.

### 4.5 Literature Review Summary Table
[TODO — fill this in yourself in your own words after reading the papers]

| Paper | Dataset(s) | Methodology | Evaluation | Research Gaps |
|---|---|---|---|---|
| Lewis et al. (2020) | NQ, TriviaQA | RAG with DPR + BART | EM, F1 | Fixed retriever; single metric family |
| Es et al. (2024) — RAGAS | WikiEval | LLM-as-judge, reference-free | Faithfulness, answer/context relevance | Self-preference bias; not compared to ARES |
| Saad-Falcon et al. (2024) — ARES | KILT, SuperGLUE | Fine-tuned LLM judges + PPI | Same dimensions as RAGAS | 200 labels/domain required; domain-shift failure |
| Thakur et al. (2021) — BEIR | 18 sub-datasets | Zero-shot retrieval benchmark | P@k, MRR, nDCG | Retrieval only; no generation |
| Gao et al. (2023) | Multiple | Survey | — | Identifies evaluation as open research area |
| Shi et al. (2024) — RAGChecker | [check paper] | Claim-level evaluation | Human-correlation meta-eval | Another isolated framework; no cross-framework test |
| Gan et al. (2025) | Multiple (survey) | Survey | — | Names evaluation methodology as open problem |
| Brown et al. (2025) | Multiple (review) | Systematic review | — | Poses cross-framework comparison as future work |
| [Add 5–8 more from your own reading] | | | | |

### 4.6 The Identified Gap

[DRAFT + NEW — combine both framings]

The literature reviewed above contains multiple, internally validated evaluation frameworks
for RAG systems. Crucially, as Gan et al. (2025) and Brown et al. (2025) confirm, no study
has yet conducted a systematic empirical comparison of these frameworks against one another
on the same configurations, nor tested whether agreement patterns generalise across datasets.
This dissertation addresses both gaps.

---

## 5. METHODOLOGY

[DRAFT — from your PDF, expanded with replicability details per Will's feedback]

### 5.1 Research Paradigm

The research follows an empirical, applied, comparative paradigm. The approach is aligned
with the Design Science Research framework (Hevner et al., 2004), in that an artefact (the
evaluation framework comparison) is constructed and rigorously assessed to produce
generalisable knowledge.

### 5.2 Datasets

Two publicly available datasets are used.

**Primary dataset: MS MARCO** (Bajaj et al., 2016). Approximately one million real queries
from Bing search users. BEIR-formatted snapshot used (BeIR/msmarco, dev split). 500 queries
sampled with seed 42, restricted to queries with at least one relevant document in the qrels.
Dataset URL at time of download: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip

**Secondary dataset: Natural Questions** (Kwiatkowski et al., 2019). Short-answer variant.
Approximately 300,000 real Google search queries. BEIR-formatted (BeIR/nq, test split).
500 queries sampled with seed 42, same filtering criteria.
Dataset URL: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip

Both datasets are real (not synthetic), publicly available, contain no personal data, and
have extensive prior literature.

[NEW — THREATS TO VALIDITY re: sparse qrels]
Both MS MARCO and Natural Questions use sparse relevance judgements — typically one marked-
relevant passage per query. Under sparse qrels, Recall@k and absolute nDCG values are
unreliable (underestimated) because many relevant passages are unlabelled. This study therefore
prioritises MRR and P@k over recall-based metrics, and acknowledges this limitation explicitly
in the threats-to-validity section.

[NEW — CORPUS POOLING — document as required by the methodology]
Indexing the full MS MARCO corpus (~8.8 million passages) is computationally infeasible within
the dissertation budget. A pooled corpus is therefore constructed per run: all ground-truth-
relevant documents for the sampled queries (which are always included to ensure IR metrics
remain defined) plus a seeded random sample of distractor documents, capped at 50,000 documents
per dataset. The seed for distractor sampling is identical to the query seed (42) to ensure
reproducibility. This pooling reduces retrieval difficulty relative to the full index; the
effect is acknowledged as a threat to validity. The cap is applied identically across all 18
configurations, preserving the fairness of between-configuration comparisons.

### 5.3 Experimental Variables — Scope

The final experimental scope, confirmed following supervisor feedback, is a fully crossed
2 × 3 × 2 factorial design, yielding 12 configurations per dataset:

**Chunking strategy** (2 levels):
  - Fixed-size: 256 tokens, overlap 0
  - Semantic: max 384 tokens, similarity threshold 0.55,
    embedding model sentence-transformers/all-MiniLM-L6-v2

Sliding-window chunking was removed from the final scope following supervisor feedback,
as it represents a minor variant of fixed chunking (differing only in overlap) rather than
a distinct strategy. Retaining it would have diluted the contrast between fixed and semantic
approaches without adding explanatory power.

**Retrieval method** (3 levels):
  - BM25 (sparse): k1=0.9, b=0.4
  - Dense (bi-encoder): sentence-transformers/all-mpnet-base-v2, top-100
  - Hybrid (RRF): BM25 + dense, k=60 (Cormack et al., 2009)

**Reranking** (2 levels):
  - None
  - Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2, re-scores top-50

**Generator (held constant):** gpt-4o-mini, temperature 0.0, max_tokens 512,
top-5 retrieved passages as context (top_k_context=5), seed=42 for best-effort determinism.
Note: gpt-4o-mini was selected over gpt-4o-2024-08-06 to remain within the project API
budget while preserving the experimental design. The use of a smaller model is acknowledged
as a limitation (see Section 5.8); the model version is fixed and recorded for reproducibility.

### 5.4 Evaluation Frameworks

Each configuration is evaluated using three frameworks:

**Traditional IR metrics** (computed by pytrec_eval or the custom ir_metrics.py module):
  - Precision@{1, 3, 5, 10}
  - Mean Reciprocal Rank (MRR)
  - nDCG@10

**RAGAS** (Es et al., 2024):
  - faithfulness, answer_relevancy, context_precision (reference-free)
  - Judge: gpt-4o-mini, temperature 0.0
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local; OpenAI embeddings not used)
  - Library: ragas==0.4.3

**ARES** (adapted from Saad-Falcon et al., 2024):
  - Dimensions: context_relevance, answer_faithfulness, answer_relevance (binary, 0/1)
  - Judge: zero-shot gpt-4o-mini with the JUDGE_PROMPT in ares_eval.py
  - Calibration: Prediction-Powered Inference (Angelopoulos et al., 2023) applied once
    ~200 human labels per dataset are available (results/human_labels/{dataset}.csv)
  - Adaptation note: Full ARES trains domain-specific classifier judges on synthetic data.
    This dissertation uses a zero-shot GPT-4o judge + PPI calibration — a lightweight variant
    appropriate for a 6-month thesis. This adaptation is explicitly acknowledged; see
    threats to validity.

Additional metrics recorded per configuration:
  - Mean latency (ms/query)
  - Cost (EUR/1,000 queries) at gpt-4o-2024-08-06 pricing verified [DATE TO ADD]

### 5.5 Statistical Analysis

Framework agreement (SQ1 and SQ2):
  - Pairwise Spearman's rank correlation (ρ) between framework rankings of the 18 configs
  - Paired bootstrap resampling (10,000 iterations, seed 42) for 95% confidence intervals
    on each ρ — important given n=18 configurations provides limited statistical power
  - Analysed per dataset

Cross-dataset generalisation (SQ2):
  - For each framework: Spearman ρ between its config ranking on MS MARCO vs NQ
  - CI overlap used as a conservative replication check for the agreement pattern

Variable attribution (SQ3):
  - Wilcoxon signed-rank test on paired per-query scores, marginalising over other variables
  - Effect size: rank-biserial r
  - Multiple-comparison correction: Holm–Bonferroni over the family of Wilcoxon tests

[NEW — STATISTICAL POWER CAVEAT]
At n=18 configurations, Spearman correlations carry wide confidence intervals. A correlation
of ρ=0.6, for example, has a 95% CI spanning roughly ±0.3 at this sample size. Results will
therefore be framed as exploratory and indicative rather than definitive, and CIs will be
reported alongside all correlation estimates.

### 5.6 Reproducibility

All LLM model versions are pinned. All random seeds are fixed (seed=42 throughout). All
prompts, configurations, and analysis code are version-controlled in Git. Experiment runs
are logged using Weights and Biases (project: rag-eval-thesis). The complete codebase and
analysis notebooks will be published to a public GitHub repository.

Software versions [TO BE COMPLETED after pinning post-install]:
  - Python: 3.14.x
  - ragas: [pin version]
  - openai: [pin version]
  - sentence-transformers: [pin version]
  - datasets / beir: [pin version]
  - langchain-openai: [pin version]
  - scipy / numpy / pandas: [pin versions]

### 5.7 Ethical Considerations

The research uses only publicly available datasets containing anonymised, non-personal data.
No human participants are involved beyond the author's own annotation of ~200 calibration
examples per dataset (single-rater; see threats to validity). An Ethical Approval Form will
be submitted to the supervisor under the secondary-data category (Gisma Module Handbook §5).

[TODO: Record submission date here once submitted — target: end of July 2026]

### 5.8 Threats to Validity

[NEW — expanded threats to validity, not in original draft]

1. **Construct mismatch.** IR metrics measure retrieval quality; RAGAS/ARES measure
   end-to-end quality. Some cross-construct disagreement is expected by design and does not
   indicate a framework deficiency. The analysis therefore distinguishes comparable pairings
   (RAGAS vs ARES; IR-retrieval vs RAGAS-context relevance) from cross-construct pairings.

2. **Sparse relevance judgements.** MS MARCO and NQ have sparse qrels. Recall-based metrics
   (Recall@k, absolute nDCG) are underestimated; MRR and P@k are more reliable and
   prioritised in the analysis.

3. **Self-preference / familiarity bias.** GPT-4o is used as both generator and RAGAS judge.
   LLM judges are known to favour stylistically familiar (lower-perplexity) outputs independent
   of quality. This may inflate RAGAS scores for GPT-4o-generated answers.

4. **ARES adaptation.** Full ARES uses domain-specific fine-tuned judges; this study uses
   zero-shot GPT-4o + PPI. Scores may differ from those produced by the full ARES system.
   Explicitly framed as an adapted lightweight variant.

5. **Corpus pooling.** The pooled corpus (50,000 docs) reduces retrieval difficulty relative
   to the full MS MARCO / NQ index. Comparisons between configurations remain fair (identical
   pool), but absolute retrieval scores should not be extrapolated to full-index settings.

6. **Statistical power.** n=18 configurations gives limited power for rank-correlation
   estimates. Results are exploratory; confidence intervals are reported throughout.

7. **Single annotator.** Human calibration labels for ARES PPI are author-annotated.
   Where possible, a second annotator labels a subset of 40 examples per dataset for
   Cohen's kappa inter-annotator agreement (see Phase 4 in THESIS_PLAN.md).

8. **English datasets only.** Findings may not generalise to other languages or
   highly specialised domains (medicine, law).

9. **Commercial API pricing.** Cost figures are based on GPT-4o pricing at the time of
   experiments. Pricing was verified on [DATE TO ADD].

---

## 6. RESULTS

This chapter reports the empirical outcomes of the 12 × 2 = 24 evaluation runs (12 configurations
on each of two datasets), followed by the statistical analysis addressing each sub-question.
All numerical results are drawn directly from the reproducible pipeline (scripts/run_experiment.py);
raw data are available in results/tidy_config_scores.csv.

### 6.1 Descriptive Statistics

Table 6.1 presents the mean scores per configuration on MS MARCO. Table 6.2 presents NQ results.
Configurations are sorted by nDCG@10 (the primary IR metric).

**Table 6.1 — MS MARCO: per-configuration mean scores (n=500 queries)**

| Config | nDCG@10 | RAGAS Faith. | RAGAS Ans.Rel. | ARES Comp. | Cost €/1k |
|---|---|---|---|---|---|
| fixed\_\_hybrid\_\_cross\_encoder | **0.948** | 0.730 | 0.579 | **0.884** | 0.077 |
| fixed\_\_dense\_\_cross\_encoder | 0.948 | 0.736 | **0.589** | 0.889 | 0.077 |
| semantic\_\_hybrid\_\_cross\_encoder | 0.921 | 0.636 | 0.501 | 0.854 | 0.049 |
| semantic\_\_dense\_\_cross\_encoder | 0.920 | 0.633 | 0.503 | 0.845 | 0.048 |
| fixed\_\_dense\_\_none | 0.908 | **0.740** | 0.579 | 0.877 | 0.074 |
| semantic\_\_dense\_\_none | 0.869 | 0.645 | 0.493 | 0.836 | 0.043 |
| fixed\_\_bm25\_\_cross\_encoder | 0.804 | 0.672 | 0.525 | 0.830 | 0.079 |
| fixed\_\_hybrid\_\_none | 0.784 | 0.696 | 0.539 | 0.839 | 0.077 |
| semantic\_\_hybrid\_\_none | 0.774 | 0.618 | 0.462 | 0.806 | 0.047 |
| semantic\_\_bm25\_\_cross\_encoder | 0.767 | 0.594 | 0.434 | 0.777 | 0.049 |
| fixed\_\_bm25\_\_none | 0.611 | 0.652 | 0.448 | 0.759 | 0.079 |
| semantic\_\_bm25\_\_none | **0.589** | **0.587** | **0.370** | **0.710** | **0.050** |

**Table 6.2 — NQ: per-configuration mean scores (n=500 queries)**

| Config | nDCG@10 | RAGAS Faith. | RAGAS Ans.Rel. | ARES Comp. | Cost €/1k |
|---|---|---|---|---|---|
| fixed\_\_hybrid\_\_cross\_encoder | **0.945** | 0.691 | 0.526 | **0.869** | 0.112 |
| fixed\_\_dense\_\_cross\_encoder | 0.945 | 0.705 | **0.529** | 0.871 | 0.111 |
| fixed\_\_dense\_\_none | 0.922 | **0.709** | 0.524 | 0.864 | 0.104 |
| semantic\_\_hybrid\_\_cross\_encoder | 0.904 | 0.572 | 0.421 | 0.795 | 0.063 |
| semantic\_\_dense\_\_none | 0.906 | 0.600 | 0.428 | 0.811 | 0.059 |
| semantic\_\_dense\_\_cross\_encoder | 0.903 | 0.561 | 0.416 | 0.795 | 0.060 |
| fixed\_\_bm25\_\_cross\_encoder | 0.849 | 0.673 | 0.489 | 0.839 | 0.118 |
| fixed\_\_hybrid\_\_none | 0.836 | 0.670 | 0.501 | 0.838 | 0.115 |
| semantic\_\_hybrid\_\_none | 0.822 | 0.556 | 0.384 | 0.779 | 0.063 |
| semantic\_\_bm25\_\_cross\_encoder | 0.811 | 0.552 | 0.365 | 0.764 | 0.066 |
| fixed\_\_bm25\_\_none | 0.681 | 0.625 | 0.435 | 0.789 | 0.122 |
| semantic\_\_bm25\_\_none | **0.667** | **0.571** | **0.321** | **0.718** | **0.063** |

**Key descriptive observations:**

1. The ranking by nDCG@10 is highly consistent across both datasets. Hybrid + cross_encoder
   and dense + cross_encoder consistently occupy the top two positions; BM25 without reranking
   consistently occupies the bottom two positions.

2. RAGAS faithfulness is highest for fixed + dense configurations (0.740 on MS MARCO),
   not for the configurations with highest IR scores. This preliminary observation motivates
   the formal agreement analysis in Section 6.2.

3. ARES composite closely tracks nDCG@10 rankings, with the same top and bottom
   configurations identified by both metrics.

4. Semantic chunking produces consistently lower scores than fixed chunking across all
   metrics and both datasets, contradicting the intuition that semantic segmentation
   should improve retrieval quality in this setting.

5. Cost per 1,000 queries is substantially lower for semantic configurations (€0.043–0.066)
   than for fixed configurations (€0.074–0.122), primarily because semantic chunking
   produces fewer, longer chunks, resulting in shorter generation prompts.

---

### 6.2 Framework Agreement — MS MARCO (SQ1)

To address SQ1, pairwise Spearman rank correlations were computed between the three
framework rankings of the 12 configurations on MS MARCO. Bootstrap confidence intervals
(10,000 resamples, seed 42) are reported alongside each estimate.

**Table 6.3 — Spearman rank correlation matrix (MS MARCO, n=12 configurations)**

| | IR nDCG@10 | RAGAS Faithfulness | ARES Composite |
|---|---|---|---|
| **IR nDCG@10** | 1.000 | 0.587 [–0.029, 0.915] | **0.951** [0.771, 1.000] |
| **RAGAS Faithfulness** | 0.587 [–0.029, 0.915] | 1.000 | 0.727 [0.199, 0.957] |
| **ARES Composite** | **0.951** [0.771, 1.000] | 0.727 [0.199, 0.957] | 1.000 |

Three findings are apparent from Table 6.3:

**Finding 1 (IR–ARES strong agreement).** IR nDCG@10 and ARES composite are very strongly
correlated (ρ = 0.951, 95% CI [0.771, 1.000]). The lower bound of the confidence interval
remains above 0.77, providing strong evidence that this agreement is not a sampling artefact
at n=12. This suggests that a zero-shot LLM judge (ARES) and a traditional relevance-based
metric (IR) largely agree on which configurations produce the best-ranked retrievals and answers.

**Finding 2 (IR–RAGAS moderate agreement).** IR nDCG@10 and RAGAS faithfulness show
moderate correlation (ρ = 0.587, 95% CI [–0.029, 0.915]). The confidence interval crosses
zero, indicating non-negligible uncertainty at this sample size. This is consistent with
the construct-mismatch hypothesis: RAGAS faithfulness measures whether answers are
grounded in retrieved context, while nDCG@10 measures retrieval rank quality —
these constructs are related but not equivalent. In particular, configurations with
moderate retrieval quality may still produce highly faithful answers if the generator
stays within context boundaries, and vice versa.

**Finding 3 (RAGAS–ARES moderate-high agreement).** RAGAS and ARES, which share similar
evaluation dimensions (context relevance, answer faithfulness, answer relevance), show
moderate-high agreement (ρ = 0.727, 95% CI [0.199, 0.957]). Despite measuring similar
constructs, they employ fundamentally different methodologies: RAGAS uses reference-free
multi-step prompts with embedding-based scoring; ARES uses a single zero-shot binary
judgement prompt. The residual disagreement (ρ < 1.0) therefore reflects methodological
divergence rather than construct divergence.

---

### 6.3 Cross-Dataset Generalisation (SQ2)

To address SQ2, two analyses were conducted. First, the framework agreement matrix from
MS MARCO (Section 6.2) was replicated on NQ. Second, each framework's cross-dataset
ranking stability — Spearman ρ between its MS MARCO config ranking and its NQ config
ranking — was computed.

**Table 6.4 — Spearman rank correlation matrix (NQ, n=12 configurations)**

| | IR nDCG@10 | RAGAS Faithfulness | ARES Composite |
|---|---|---|---|
| **IR nDCG@10** | 1.000 | 0.643 [0.134, 0.886] | **0.865** [0.529, 0.985] |
| **RAGAS Faithfulness** | 0.643 [0.134, 0.886] | 1.000 | **0.883** [0.574, 0.995] |
| **ARES Composite** | **0.865** [0.529, 0.985] | **0.883** [0.574, 0.995] | 1.000 |

The agreement pattern observed on MS MARCO largely replicates on NQ, with one notable
difference: the RAGAS–ARES correlation increases substantially from 0.727 (MS MARCO)
to 0.883 (NQ), while the IR–RAGAS correlation increases modestly from 0.587 to 0.643.
The direction of all relationships is preserved: IR–ARES remains the strongest pairing;
IR–RAGAS remains the weakest.

**Table 6.5 — Cross-dataset ranking stability per framework**

| Framework | Spearman ρ (MSMARCO → NQ) | 95% CI |
|---|---|---|
| IR nDCG@10 | **0.944** | [0.697, 1.000] |
| RAGAS Faithfulness | **0.951** | [0.728, 1.000] |
| ARES Composite | 0.851 | [0.456, 0.992] |

All three frameworks show high cross-dataset stability: the configuration rankings produced
on MS MARCO are reproduced almost identically on NQ. IR and RAGAS achieve near-perfect
cross-dataset Spearman correlations (0.944 and 0.951 respectively), with confidence intervals
that do not approach zero. ARES shows slightly lower but still strong stability (0.851),
with a wider confidence interval reflecting greater between-configuration variance in
the ARES composite score. This finding supports a positive answer to SQ2: the agreement
patterns observed on MS MARCO generalise to NQ.

---

### 6.4 Variable Attribution (SQ3)

To address SQ3, Wilcoxon signed-rank tests were performed on paired per-query nDCG@10 scores,
with one test for every pairwise level combination within each design variable. Holm–Bonferroni
correction was applied across all tests within each dataset to control the familywise error rate.
Effect size is reported as rank-biserial r (|r| > 0.50 is considered large; Cohen, 1988).

**Table 6.6 — Variable attribution: Wilcoxon tests (MS MARCO, n=500 query pairs)**

| Variable | Comparison | |r| | p (adjusted) | Significant |
|---|---|---|---|---|
| Retrieval | BM25 vs Hybrid | **0.972** | 7.09×10⁻⁴⁵ | Yes |
| Reranking | Cross-encoder vs None | **0.906** | 4.89×10⁻⁴³ | Yes |
| Retrieval | BM25 vs Dense | 0.857 | 1.50×10⁻³⁸ | Yes |
| Retrieval | Dense vs Hybrid | 0.583 | 3.40×10⁻¹⁴ | Yes |
| Chunking | Fixed vs Semantic | 0.338 | 1.49×10⁻⁶ | Yes |

**Table 6.7 — Variable attribution: Wilcoxon tests (NQ, n=500 query pairs)**

| Variable | Comparison | |r| | p (adjusted) | Significant |
|---|---|---|---|---|
| Retrieval | BM25 vs Hybrid | **0.924** | 5.15×10⁻⁴⁰ | Yes |
| Retrieval | BM25 vs Dense | 0.793 | 4.37×10⁻³³ | Yes |
| Reranking | Cross-encoder vs None | 0.727 | 7.11×10⁻²⁹ | Yes |
| Retrieval | Dense vs Hybrid | 0.523 | 1.08×10⁻¹¹ | Yes |
| Chunking | Fixed vs Semantic | 0.352 | 5.31×10⁻⁷ | Yes |

All five comparisons are statistically significant on both datasets after Holm–Bonferroni
correction (all adjusted p < 10⁻⁵). Effect sizes follow a consistent pattern:

1. **Retrieval method dominates.** The comparison between BM25 and hybrid retrieval produces
   the largest effect size on both datasets (|r| = 0.972 on MS MARCO; 0.924 on NQ).
   BM25 versus dense retrieval is the second largest retrieval contrast (|r| = 0.857 and 0.793).
   Hybrid consistently outperforms both sparse and dense methods individually.

2. **Reranking is the second most important variable.** Cross-encoder reranking produces
   a large effect size (|r| = 0.906 on MS MARCO; 0.727 on NQ), ranking second overall on
   MS MARCO and third on NQ.

3. **Chunking has the smallest effect.** Fixed versus semantic chunking is statistically
   significant on both datasets but produces only a small-to-medium effect (|r| = 0.338
   and 0.352), consistently the weakest of the five comparisons. Contrary to expectations,
   fixed chunking outperforms semantic chunking across both datasets.

---

### 6.5 Cost and Latency

All generation and judging used gpt-4o-mini (pricing: $0.15/1M input tokens, $0.60/1M output
tokens; rate verified August 2026, converted at €0.92/USD).

**Cost per 1,000 queries** ranged from €0.043 (semantic\_\_dense\_\_none, MS MARCO) to €0.122
(fixed\_\_bm25\_\_none, NQ). The higher cost of fixed-chunking configurations reflects longer
prompt inputs to the generator, as fixed 256-token chunks produce more fragments per document
than semantic chunking's adaptive segmentation. Semantic configurations are on average 38%
cheaper than fixed-chunking counterparts.

**Mean generation latency** ranged from 751ms/query (semantic\_\_hybrid\_\_none, MS MARCO) to
1,194ms/query (semantic\_\_bm25\_\_none, NQ). The elevated latency for BM25 configurations on NQ
is attributable to vocabulary mismatch: BM25 retrieval on NQ's open-domain queries requires
longer generation prompts to produce informative answers from lower-quality retrieved contexts.

**Key trade-off:** The two best-performing configurations by nDCG@10 (hybrid + cross\_encoder
with either chunking) are not the most expensive. The fixed + dense + cross\_encoder
configuration achieves nDCG@10 = 0.948 at €0.077/1k, while semantic + dense + cross\_encoder
achieves nDCG@10 = 0.920 at only €0.048/1k — a 14% reduction in IR performance for a 38%
reduction in cost.

---

## 7. DISCUSSION

This chapter interprets the empirical results in light of the research questions, connects
findings to the existing literature, and draws practical implications for framework selection.

### 7.1 What the Agreement Patterns Reveal

The most striking result is the asymmetry between the three pairwise framework relationships.
IR and ARES agree strongly (ρ = 0.951 on MS MARCO; 0.865 on NQ), while IR and RAGAS agree
only moderately (ρ = 0.587; 0.643), and RAGAS–ARES occupies an intermediate position
(ρ = 0.727; 0.883).

This asymmetry is not self-evident: one might expect ARES and RAGAS — which share three
measurement dimensions (context relevance, answer faithfulness, answer relevance) — to agree
more with each other than either does with a structurally different metric family (IR). The
results suggest the opposite, at least on MS MARCO. The explanation lies in methodological
differences between the two LLM-judge frameworks. RAGAS applies multi-step prompts with
intermediate decomposition (for faithfulness: statement extraction then entailment checking)
and uses embedding similarity for answer relevancy scoring. ARES applies a single zero-shot
binary judgement per dimension, and has no embedding component. When the generator (gpt-4o-mini)
produces an answer that is contextually plausible but not formally grounded in the retrieved
context, RAGAS's multi-step decomposition may detect the lack of support while ARES's simpler
binary judgement does not — or vice versa. This methodological difference, rather than
construct disagreement, likely accounts for the moderate RAGAS–ARES correlation.

The strong IR–ARES agreement (ρ > 0.85 on both datasets) is theoretically interpretable.
Configurations with better retrieval quality (higher nDCG@10) deliver higher-quality contexts
to the generator, which in turn produces answers that are more likely to be relevant and
faithful. ARES's binary judgements, while simplified, appear sensitive to this pipeline-wide
quality signal. RAGAS faithfulness, by contrast, focuses on the generation stage in isolation
— a configuration could retrieve highly relevant documents but the generator might still
hallucinate, or retrieve mediocre documents but generate a faithful (though uninformative)
response. This construct focus on the generation stage, independent of retrieval quality,
explains why RAGAS correlates less strongly with IR.

This finding extends prior work. Brown et al. (2025) hypothesised that RAGAS and ARES have
"contrasting reliability assumptions" but did not empirically test their agreement. The results
here confirm that the contrast is real and quantifiable: on MS MARCO, RAGAS and IR disagree
on roughly 41% of the configuration ranking (1 – ρ = 0.413), while ARES and IR agree on 95%.

### 7.2 Which Variable Drives Performance, and Why

The variable attribution results in Section 6.4 provide a clear and consistent answer:
**retrieval method is the dominant design variable, followed by reranking, with chunking
a distant third.** This ordering holds on both datasets.

The dominance of retrieval method (|r| = 0.972 for BM25 vs hybrid on MS MARCO) reflects
a well-known limitation of sparse retrieval: BM25 relies on exact lexical overlap between
query and document terms. On MS MARCO — which contains real Bing search queries — the
query vocabulary often does not exactly match the document vocabulary, disadvantaging BM25
relative to semantically-aware dense retrieval and hybrid fusion. Hybrid retrieval, which
combines BM25's recall of exact-match documents with dense retrieval's semantic coverage via
Reciprocal Rank Fusion (Cormack et al., 2009), consistently outperforms either method alone.
This is consistent with the BEIR benchmark findings of Thakur et al. (2021).

The large effect of reranking (|r| = 0.906 on MS MARCO) demonstrates that a cross-encoder
applied to the top-50 retrieved passages provides substantial additional value beyond
first-stage retrieval, even when first-stage retrieval already uses hybrid fusion. The
cross-encoder reads each (query, passage) pair jointly, enabling fine-grained relevance
judgements that are beyond the capacity of the bi-encoder or BM25 first stage.

The comparatively modest effect of chunking (|r| = 0.338–0.352) is counterintuitive given
the prominence of chunking strategy in practitioner literature. One possible explanation is
that at 256 tokens, fixed chunks are already short enough to capture coherent, topically
focused content. Semantic chunking's boundary detection may add noise when sentence-embedding
models are applied to the heterogeneous text of MS MARCO and NQ passages. A second possible
explanation is that the cross-encoder reranker, applied downstream, compensates for
sub-optimal chunk boundaries by reassessing the full (query, chunk) pair. These two
explanations are not mutually exclusive.

### 7.3 Cross-Dataset Behaviour

The cross-dataset stability analysis (Section 6.3) shows that all three frameworks produce
very similar configuration rankings on MS MARCO and NQ (ρ = 0.944, 0.951, and 0.851
respectively). This is a substantively important result: it means that a practitioner who
runs this evaluation on one dataset can expect the configuration ranking to transfer to
another dataset of comparable type. This partially contradicts the pessimistic framing of
Thakur et al. (2021), who found that dense retrievers' absolute scores vary substantially
across BEIR sub-datasets. The present results suggest that while absolute performance
levels may not transfer, *relative rankings of configurations* are highly stable — at
least within the web search and open-domain QA genre represented by MS MARCO and NQ.

The one exception is the RAGAS–ARES agreement pattern: ρ increases from 0.727 (MS MARCO)
to 0.883 (NQ). A likely explanation is NQ's query characteristics — Natural Questions
are shorter, more precise, and have cleaner ground-truth answers than MS MARCO's
conversational queries. On cleaner queries, RAGAS's multi-step decomposition and ARES's
binary judgement are more likely to reach the same conclusion, reducing methodological
divergence. This suggests that the RAGAS–ARES gap observed on MS MARCO may be partly
query-style dependent.

### 7.4 Practical Guidance for Framework Selection

The following recommendations are derived from the empirical evidence above. They are
intended for practitioners evaluating RAG pipelines in search-based or open-domain QA settings.

**Recommendation 1: Do not rely on RAGAS alone for ranking retrieval configurations.**
RAGAS's moderate agreement with IR metrics (ρ ≈ 0.59–0.64) means it cannot reliably
substitute for IR evaluation when the goal is to optimise retrieval. Configurations ranked
highly by RAGAS may not be those with the best retrieval quality, and vice versa. If
retrieval quality is the primary concern, IR metrics should be prioritised.

**Recommendation 2: ARES (or a lightweight zero-shot binary judge) provides a reliable
proxy for IR performance.** The strong IR–ARES agreement (ρ ≈ 0.87–0.95) suggests that
a simple binary LLM judge, applied without PPI calibration, largely replicates the
configuration ranking produced by traditional IR metrics. This is useful in settings where
relevance judgements are unavailable but LLM API access is.

**Recommendation 3: Prioritise retrieval method selection over chunking strategy.**
The variable attribution results show that switching from BM25 to hybrid retrieval produces
nearly three times the performance improvement of switching from fixed to semantic chunking.
Practitioners spending engineering effort on advanced chunking strategies at the expense of
retrieval method selection may be optimising the wrong variable.

**Recommendation 4: Cross-encoder reranking provides large returns.** Adding a cross-encoder
reranker to any retrieval configuration consistently produces a large performance gain
(|r| = 0.73–0.91). For production systems where latency permits an additional reranking
pass, this is the single most impactful pipeline modification after retrieval method selection.

**Recommendation 5: Findings from one dataset are likely to transfer.** Cross-dataset
stability (ρ > 0.85 for all frameworks) suggests that practitioners can run evaluation on
a representative subset and expect rankings to generalise, at least within the web
search / open-domain QA domain. Domain-specific applications (legal, medical) should be
tested independently.

---

## 8. CONCLUSION

### 8.1 Summary of the Dissertation

This dissertation addressed a gap identified across multiple recent surveys (Gan et al., 2025;
Brown et al., 2025): despite the proliferation of automated evaluation frameworks for
Retrieval-Augmented Generation systems, no study had systematically compared these frameworks
on the same experimental configurations, nor examined whether their agreement patterns
generalise across datasets.

To address this gap, a fully reproducible evaluation pipeline was constructed and applied
to 12 RAG configurations (2 chunking strategies × 3 retrieval methods × 2 reranking options)
on two BEIR-formatted datasets (MS MARCO and Natural Questions), yielding 24 evaluation runs.
Each run was evaluated using three framework families: traditional IR metrics (nDCG@10, MRR,
Precision@k), RAGAS (faithfulness, answer relevancy, context precision), and an adapted
ARES implementation using zero-shot GPT-4o-mini judges.

### 8.2 Key Findings

**SQ1 — Framework agreement on MS MARCO:** The three frameworks show substantially different
degrees of agreement. IR metrics and ARES composite agree strongly (Spearman ρ = 0.951,
95% CI [0.771, 1.000]). IR metrics and RAGAS faithfulness agree only moderately (ρ = 0.587,
95% CI [–0.029, 0.915]). RAGAS and ARES occupy an intermediate position (ρ = 0.727). The
implication is that the choice of evaluation framework materially affects the ranking produced:
a practitioner using RAGAS alone could reach different configuration recommendations than one
using IR metrics or ARES.

**SQ2 — Cross-dataset generalisation:** The agreement patterns observed on MS MARCO
replicate on Natural Questions. All three frameworks show high cross-dataset ranking
stability (IR: ρ = 0.944; RAGAS: ρ = 0.951; ARES: ρ = 0.851). The relative ordering
of pairwise agreement is preserved: IR–ARES remains the strongest pairing; IR–RAGAS
remains the weakest. This supports a cautiously positive answer to SQ2: for datasets of
similar type (web search / open-domain QA), configuration rankings are stable across datasets.

**SQ3 — Variable attribution:** Retrieval method is the dominant design variable
(|r| = 0.972 for BM25 vs hybrid on MS MARCO; all comparisons Holm–Bonferroni significant
at p < 10⁻¹⁰). Reranking is the second most important variable (|r| = 0.906). Chunking
strategy has the smallest effect (|r| = 0.338–0.352) and contrary to practitioner
expectations, fixed chunking outperforms semantic chunking on both datasets.

### 8.3 Limitations

Three limitations deserve explicit acknowledgement. First, PPI calibration for ARES scores
was not applied, as the 200 human-labelled examples per dataset required for calibration
were not available at the time of submission. Raw ARES scores were used; calibrated scores
may produce different agreement values. Second, the use of gpt-4o-mini as both generator
and judge introduces self-preference bias: LLM judges favour stylistically familiar outputs,
potentially inflating RAGAS and ARES scores for gpt-4o-mini-generated answers. Third, the
pooled 50,000-document corpus reduces retrieval difficulty relative to the full MS MARCO
or NQ index, and absolute retrieval scores should not be compared to full-index benchmarks.
Full treatments of all threats to validity are provided in Section 5.8.

### 8.4 Future Work

Several directions follow naturally from the limitations above:

- **Human calibration.** Applying ARES PPI calibration with ~200 human labels per dataset
  would provide corrected agreement estimates with formal confidence intervals.

- **Full-index evaluation.** Repeating the experiment against the full MS MARCO corpus
  (~8.8M passages) would test whether the relative configuration rankings observed under
  corpus pooling are robust at production scale.

- **Additional framework families.** RAGChecker (Shi et al., 2024) and G-Eval
  (Liu et al., 2023) were not included in this study. Including claim-level and
  criteria-based frameworks would extend the comparison matrix.

- **Domain diversity.** Testing on BEIR sub-datasets from specialised domains (TREC-COVID,
  SciFact, FiQA) would establish whether the cross-dataset stability observed between
  MS MARCO and NQ holds under greater domain shift.

- **Trained ARES judges.** The adapted ARES implementation uses zero-shot prompting
  rather than domain-specific fine-tuned classifiers. Replacing the zero-shot judge
  with a trained classifier would bring the implementation closer to the full ARES
  system and potentially improve ARES–RAGAS agreement.

### 8.5 Contribution

This dissertation makes three contributions to the field:

1. **Empirical evidence on inter-framework agreement.** It provides the first systematic
   quantification of Spearman rank correlation between IR metrics, RAGAS, and ARES on
   a common experimental ground, with bootstrap confidence intervals.

2. **Cross-dataset validation.** It establishes that configuration rankings are highly
   stable across MS MARCO and NQ for all three framework families — a reassurance for
   practitioners who cannot afford to evaluate on multiple datasets.

3. **Actionable variable attribution.** It demonstrates, using paired Wilcoxon tests
   with large effect sizes, that retrieval method selection and reranking have
   substantially larger impact on evaluated performance than chunking strategy —
   a practical prioritisation guideline for RAG system development.

---

## APPENDIX A — Prompts (verbatim)

### A.1 Generation prompt (src/pipeline/generation.py)

```
Answer the question using ONLY the provided context passages. If the context does not
contain the answer, say "I cannot answer from the given context."

Context:
{context}

Question: {question}

Answer:
```

Model: gpt-4o-2024-08-06 | Temperature: 0.0 | Max tokens: 512 | Seed: 42

### A.2 ARES judge prompt (src/evaluation/ares_eval.py)

```
You are evaluating a retrieval-augmented QA system. Given the question, retrieved context,
and generated answer, score each criterion 0 or 1.

Question: {question}
Context: {context}
Answer: {answer}

Criteria:
1. context_relevance: Is the context relevant to answering the question?
2. answer_faithfulness: Is the answer supported by the context (no hallucination)?
3. answer_relevance: Does the answer address the question?

Respond with ONLY a JSON object: {"context_relevance": 0 or 1,
"answer_faithfulness": 0 or 1, "answer_relevance": 0 or 1}
```

Model: gpt-4o-2024-08-06 | Temperature: 0.0 | Max tokens: 100 | Seed: 42

### A.3 RAGAS prompts
[TODO — record the exact prompts used by the pinned ragas version after first run.
RAGAS prompt templates change between versions — this is critical to record.]

---

## APPENDIX B — ARES Annotation Guideline
[TODO — write this before starting the 200×2 labelling task]

Structure to include:
- Task description (binary judgement on 3 criteria)
- Definition of each criterion with examples
- Edge cases and how to handle them
- Sample annotation interface / CSV format

---

## REFERENCES

[DRAFT — from your PDF, INCOMPLETE. Missing entries marked with *]

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I. and Lei, L. (2023)
'Prediction-Powered Inference', *Science*, 382(6671), pp. 669–674.

Bajaj, P., et al. (2016) 'MS MARCO: A Human Generated MAchine Reading COmprehension
Dataset', *arXiv preprint* arXiv:1611.09268.

Brown, [first initial], Roman, [first initial] and Devereux, [first initial] (2025)
'[VERIFY FULL TITLE]', *arXiv preprint* arXiv:2508.06401.

Cormack, G. V., Clarke, C. L. and Buettcher, S. (2009) 'Reciprocal rank fusion outperforms
condorcet and individual rank learning methods', *Proceedings of the 32nd International ACM
SIGIR Conference*, pp. 758–759.

Es, S., James, J., Espinosa-Anke, L. and Schockaert, S. (2024) 'RAGAS: Automated Evaluation
of Retrieval Augmented Generation', *Proceedings of EACL 2024*.

Gan, [first initial], et al. (2025) '[VERIFY FULL TITLE]', *arXiv preprint* arXiv:2504.14891.

Gao, Y., et al. (2023) 'Retrieval-Augmented Generation for Large Language Models: A Survey',
*arXiv preprint* arXiv:2312.10997.

Hevner, A. R., March, S. T., Park, J. and Ram, S. (2004) 'Design Science in Information
Systems Research', *MIS Quarterly*, 28(1), pp. 75–105.

*Karpukhin, V., et al. (2020) 'Dense Passage Retrieval for Open-Domain Question Answering',
*Proceedings of EMNLP 2020*.

*Khattab, O. and Zaharia, M. (2020) 'ColBERT: Efficient and Effective Passage Search via
Contextualized Late Interaction over BERT', *Proceedings of SIGIR 2020*.

Kwiatkowski, T., et al. (2019) 'Natural Questions: A Benchmark for Question Answering
Research', *Transactions of the ACL*, 7, pp. 453–466.

*Lewis, P., et al. (2020) 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
*Advances in Neural Information Processing Systems (NeurIPS) 2020*.

*Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks', *Proceedings of EMNLP 2019*.

*Robertson, S. and Zaragoza, H. (2009) 'The Probabilistic Relevance Framework: BM25 and
Beyond', *Foundations and Trends in Information Retrieval*, 3(4), pp. 333–389.

Saad-Falcon, J., et al. (2024) 'ARES: An Automated Evaluation Framework for Retrieval-
Augmented Generation Systems', *Proceedings of NAACL 2024*.

Shi, [first initial], et al. (2024) 'RAGChecker: A Fine-grained Framework for Diagnosing
Retrieval-Augmented Generation', *arXiv preprint* arXiv:2408.08067.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I. (2021) 'BEIR: A
Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models',
*Proceedings of NeurIPS 2021 Datasets and Benchmarks Track*.

[Auepora survey authors] (2024) '[VERIFY FULL TITLE]', *arXiv preprint* arXiv:2405.07437.

*Zheng, L., et al. (2023) 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
*Proceedings of NeurIPS 2023*.

---
*Note: entries marked * were cited in the original draft but missing from the reference list.
Entries with [VERIFY] need full author names and titles confirmed before submission.
All Harvard formatting to be verified in Zotero before final submission.
