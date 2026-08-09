"""RAGAS evaluation (Es et al., 2024): faithfulness, answer relevancy,
context relevancy — matching Card 4.

Pin the ragas version in requirements.txt and record it in the thesis;
RAGAS prompt templates change between versions and that materially affects
scores (a point worth a paragraph in your threats-to-validity section).
"""
from __future__ import annotations

import pandas as pd


def run_ragas(records: list[dict], judge_model: str = "gpt-4o-2024-08-06") -> pd.DataFrame:
    """records: [{question, answer, contexts: list[str], ground_truth}]
    Returns per-query dataframe with ragas_* columns.
    """
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    ds = Dataset.from_list(records)
    # Use local sentence-transformers embeddings — OpenAI embedding models are
    # not available on this project. Cost: free; model downloads on first run.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ChatOpenAI(model=judge_model, temperature=0.0),
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return df.rename(columns={
        "faithfulness": "ragas_faithfulness",
        "answer_relevancy": "ragas_answer_relevancy",
        "context_precision": "ragas_context_relevancy",
    })
