"""ARES-style evaluation (Saad-Falcon et al., 2024).

ARES = LLM judges scoring (context relevance, answer faithfulness, answer
relevance) + Prediction-Powered Inference (PPI) to calibrate judge scores
against a small human-labelled set, yielding confidence intervals.

Full ARES trains classifier judges on synthetic data; for a 6-month thesis
the accepted lightweight variant is a zero-shot GPT-4o judge + PPI
calibration — state this adaptation explicitly in the methodology and cite
the original. You need ~200 human labels per dataset (you can label these
yourself; log the annotation guideline in the appendix).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

JUDGE_PROMPT = """You are evaluating a retrieval-augmented QA system. Given the
question, retrieved context, and generated answer, score each criterion 0 or 1.

Question: {question}
Context: {context}
Answer: {answer}

Criteria:
1. context_relevance: Is the context relevant to answering the question?
2. answer_faithfulness: Is the answer supported by the context (no hallucination)?
3. answer_relevance: Does the answer address the question?

Respond with ONLY a JSON object: {{"context_relevance": 0 or 1,
"answer_faithfulness": 0 or 1, "answer_relevance": 0 or 1}}"""


def judge_one(client, question: str, context: str, answer: str,
              model: str = "gpt-4o-mini") -> dict[str, int]:
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=100, seed=42,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=context, answer=answer)}],
    )
    text = resp.choices[0].message.content or "{}"
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {k: int(d.get(k, 0)) for k in
            ("context_relevance", "answer_faithfulness", "answer_relevance")}


@dataclass
class PPIResult:
    point: float      # calibrated mean
    lo: float         # CI lower
    hi: float         # CI upper


def ppi_mean(judge_all: np.ndarray, judge_labeled: np.ndarray,
             human_labeled: np.ndarray, alpha: float = 0.05) -> PPIResult:
    """Prediction-Powered Inference for a mean (Angelopoulos et al., 2023),
    as used by ARES.

    judge_all     : judge scores on ALL n queries
    judge_labeled : judge scores on the m human-labelled queries
    human_labeled : human labels on those same m queries
    """
    from scipy import stats

    n, m = len(judge_all), len(human_labeled)
    rectifier = human_labeled - judge_labeled          # judge bias estimate
    point = judge_all.mean() + rectifier.mean()

    var = judge_all.var(ddof=1) / n + rectifier.var(ddof=1) / m
    z = stats.norm.ppf(1 - alpha / 2)
    half = z * np.sqrt(var)
    return PPIResult(point=float(point),
                     lo=float(point - half), hi=float(point + half))
