from __future__ import annotations

from typing import Any, Protocol


class StrategyRunner(Protocol):
    vector_name: str | None

    def retrieve(self, question: str) -> list[Any]:
        ...


class UnimplementedStrategyRunner:
    vector_name: str | None = None

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    def retrieve(self, question: str) -> list[Any]:
        raise NotImplementedError(
            f"Retrieval strategy {self.strategy_name!r} is not implemented in CanaR yet. "
            "TODO: add the strategy to canar.app.retrieval and wire it into this benchmark."
        )
