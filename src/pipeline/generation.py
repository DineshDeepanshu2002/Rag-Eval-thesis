"""Answer generation with a pinned model, response caching, and
cost/latency accounting (Card 4: cost €/1k queries + latency ms).

Caching matters: 18 configs share many (query, contexts) pairs after
identical retrieval — the cache prevents paying twice for the same call
and makes reruns free.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_TEMPLATE = """Answer the question using ONLY the provided context passages. \
If the context does not contain the answer, say "I cannot answer from the given context."

Context:
{context}

Question: {question}

Answer:"""

# USD per 1M tokens for gpt-4o-mini — verified 2026-08-08
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60
USD_TO_EUR = 0.92  # record the rate + date used in the thesis


@dataclass
class GenResult:
    answer: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cached: bool = False

    @property
    def cost_usd(self) -> float:
        return (self.tokens_in * PRICE_IN_PER_M + self.tokens_out * PRICE_OUT_PER_M) / 1e6

    @property
    def cost_eur(self) -> float:
        return self.cost_usd * USD_TO_EUR


@dataclass
class Generator:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    cache_dir: Path = field(default_factory=lambda: Path("results/gen_cache"))

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from openai import OpenAI  # requires OPENAI_API_KEY in env
        self.client = OpenAI()

    def _key(self, question: str, contexts: list[str]) -> Path:
        payload = json.dumps({"m": self.model, "t": self.temperature,
                              "q": question, "c": contexts}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(payload.encode()).hexdigest() + ".json")

    def generate(self, question: str, contexts: list[str]) -> GenResult:
        key = self._key(question, contexts)
        if key.exists():
            d = json.loads(key.read_text())
            return GenResult(**d, cached=True)

        prompt = PROMPT_TEMPLATE.format(
            context="\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)),
            question=question,
        )
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=42,  # best-effort determinism
        )
        latency = (time.perf_counter() - t0) * 1000
        result = GenResult(
            answer=resp.choices[0].message.content or "",
            latency_ms=latency,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
        )
        key.write_text(json.dumps({
            "answer": result.answer, "latency_ms": result.latency_ms,
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
        }))
        return result
