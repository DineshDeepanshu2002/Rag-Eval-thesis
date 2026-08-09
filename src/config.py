"""Experiment configuration: loads YAML and expands the 2x3x2 matrix."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RAGConfig:
    """One concrete RAG configuration (a cell of the 2x3x2 matrix)."""
    chunking: str
    chunking_params: dict = field(hash=False)
    retrieval: str
    retrieval_params: dict = field(hash=False)
    rerank: str
    rerank_params: dict = field(hash=False)

    @property
    def config_id(self) -> str:
        return f"{self.chunking}__{self.retrieval}__{self.rerank}"


def load_experiment(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_matrix(exp: dict[str, Any]) -> list[RAGConfig]:
    """2 chunking x 3 retrieval x 2 rerank -> 12 RAGConfig objects."""
    configs = []
    for c, r, rr in itertools.product(exp["chunking"], exp["retrieval"], exp["rerank"]):
        configs.append(RAGConfig(
            chunking=c["name"], chunking_params=c["params"],
            retrieval=r["name"], retrieval_params=r["params"],
            rerank=rr["name"], rerank_params=rr["params"],
        ))
    assert len(configs) == 12, f"expected 12 configs, got {len(configs)}"
    return configs


if __name__ == "__main__":
    exp = load_experiment(Path(__file__).parents[1] / "configs/experiment_matrix.yaml")
    for cfg in expand_matrix(exp):
        print(cfg.config_id)
