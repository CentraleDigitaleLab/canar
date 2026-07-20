from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

from canar.app.retrieval.models import RetrievalHit

logger = logging.getLogger(__name__)

GENERAL_KNOWLEDGE_WARNING = (
    "La réponse à cette question repose sur les connaissances de base du LLM et non sur "
    "des documents précis issus de notre base de données. Vérifiez bien les informations "
    "fournies ici"
)
MISSING_SCORE_WARNING = (
    "Le système RAG n’a obtenu aucun score de similarité exploitable pour cette recherche."
)


class ResponseMode(str, Enum):
    RAG_ONLY = "rag_only"
    RAG_WITH_GENERAL_KNOWLEDGE = "rag_with_general_knowledge"
    GENERAL_KNOWLEDGE_ONLY = "general_knowledge_only"


@dataclass(frozen=True)
class ResponseStrategyConfig:
    enabled: bool = True
    general_knowledge_threshold: float = 0.60
    rag_only_threshold: float = 0.85

    @classmethod
    def from_env(cls) -> ResponseStrategyConfig:
        return cls(
            enabled=_read_bool_env("RESPONSE_STRATEGY_ENABLED", default=True),
            general_knowledge_threshold=_read_float_env(
                "RESPONSE_GENERAL_KNOWLEDGE_THRESHOLD",
                default=0.60,
            ),
            rag_only_threshold=_read_float_env(
                "RESPONSE_RAG_ONLY_THRESHOLD",
                default=0.85,
            ),
        )

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("Invalid RESPONSE_STRATEGY_ENABLED: expected a boolean value")
        general = _validated_threshold(
            "RESPONSE_GENERAL_KNOWLEDGE_THRESHOLD",
            self.general_knowledge_threshold,
        )
        rag_only = _validated_threshold(
            "RESPONSE_RAG_ONLY_THRESHOLD",
            self.rag_only_threshold,
        )
        if general >= rag_only:
            raise ValueError(
                "Invalid response thresholds: "
                "RESPONSE_GENERAL_KNOWLEDGE_THRESHOLD must be strictly lower than "
                "RESPONSE_RAG_ONLY_THRESHOLD "
                f"(received {general} and {rag_only})"
            )


@dataclass(frozen=True)
class ResponseDecision:
    mode: ResponseMode
    confidence_score: float | None
    confidence_available: bool
    strategy_enabled: bool = True

    @property
    def uses_retrieved_context(self) -> bool:
        return self.mode in {
            ResponseMode.RAG_ONLY,
            ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE,
        }

    @property
    def warning(self) -> str | None:
        if not self.strategy_enabled or self.mode is not ResponseMode.GENERAL_KNOWLEDGE_ONLY:
            return None
        if not self.confidence_available:
            return f"{MISSING_SCORE_WARNING}\n\n{GENERAL_KNOWLEDGE_WARNING}"
        return GENERAL_KNOWLEDGE_WARNING


def select_response_mode(
    hits: list[RetrievalHit],
    config: ResponseStrategyConfig,
) -> ResponseDecision:
    """Classify the first final result using its preserved dense cosine confidence."""
    config.validate()

    if not config.enabled:
        decision = ResponseDecision(
            mode=ResponseMode.RAG_ONLY,
            confidence_score=None,
            confidence_available=False,
            strategy_enabled=False,
        )
        _log_decision(decision)
        return decision

    confidence = _final_response_confidence(hits)
    if confidence is None:
        decision = ResponseDecision(
            mode=ResponseMode.GENERAL_KNOWLEDGE_ONLY,
            confidence_score=None,
            confidence_available=False,
        )
    elif confidence > config.rag_only_threshold:
        decision = ResponseDecision(
            mode=ResponseMode.RAG_ONLY,
            confidence_score=confidence,
            confidence_available=True,
        )
    elif confidence >= config.general_knowledge_threshold:
        decision = ResponseDecision(
            mode=ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE,
            confidence_score=confidence,
            confidence_available=True,
        )
    else:
        decision = ResponseDecision(
            mode=ResponseMode.GENERAL_KNOWLEDGE_ONLY,
            confidence_score=confidence,
            confidence_available=True,
        )

    _log_decision(decision)
    return decision


def stream_response(
    chat_client: Any,
    messages: list[dict],
    decision: ResponseDecision,
    **generation_settings: Any,
) -> Iterable[str]:
    """Stream a guaranteed fallback warning followed by the model response."""
    if decision.warning:
        yield f"{decision.warning}\n\n"
    yield from chat_client.stream_chat(messages, **generation_settings)


def _final_response_confidence(hits: list[RetrievalHit]) -> float | None:
    if not hits:
        return None
    confidence = hits[0].response_confidence
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        return None
    cosine_confidence = float(confidence)
    if not math.isfinite(cosine_confidence) or not -1.0 <= cosine_confidence <= 1.0:
        return None
    return cosine_confidence


def _validated_threshold(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Invalid {name}: expected a numeric value between 0 and 1")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Invalid {name}: expected a finite value between 0 and 1, received {value!r}"
        )
    return threshold


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {name}: expected one of true/false, 1/0, yes/no, or on/off; "
        f"received {raw_value!r}"
    )


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {name}: expected a numeric value, received {raw_value!r}"
        ) from exc


def _log_decision(decision: ResponseDecision) -> None:
    logger.info(
        "Selected response mode=%s response_confidence=%s confidence_available=%s "
        "strategy_enabled=%s",
        decision.mode.value,
        decision.confidence_score,
        decision.confidence_available,
        decision.strategy_enabled,
    )
