from __future__ import annotations

from typing import Any, Callable

from retrievers.base import StrategyRunner, UnimplementedStrategyRunner
from retrievers.simple_sparse import SimpleSparseRunner
from retrievers.simple_vector import SimpleVectorRunner

RunnerFactory = Callable[[Any, int, tuple[str, ...] | None, str | None], StrategyRunner]

SUPPORTED_STRATEGIES = (
    "simple_vector",
    "simple_sparse",
    "hybrid",
    "hybrid_rerank",
    "contextual_retrieval",
    "parent_child",
)

_RUNNER_FACTORIES: dict[str, RunnerFactory] = {
    "simple_vector": SimpleVectorRunner,
    "simple_sparse": SimpleSparseRunner,
}


def build_runner(
    strategy_name: str,
    cfg: Any,
    top_k: int,
    collections: tuple[str, ...] | None,
    dense_vector_name: str | None,
    sparse_vector_name: str | None,
) -> StrategyRunner:
    factory = _RUNNER_FACTORIES.get(strategy_name)
    if factory is None:
        return UnimplementedStrategyRunner(strategy_name)

    vector_name = sparse_vector_name if strategy_name == "simple_sparse" else dense_vector_name
    return factory(cfg, top_k, collections, vector_name)
