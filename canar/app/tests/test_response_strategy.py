from __future__ import annotations

import logging

import pytest

from canar.app.agents import r_helpdesk
from canar.app.config import AppConfig
from canar.app.response_strategy import (
    GENERAL_KNOWLEDGE_WARNING,
    MISSING_SCORE_WARNING,
    ResponseMode,
    ResponseStrategyConfig,
    select_response_mode,
    stream_response,
)
from canar.app.retrieval.models import RetrievalHit


def hit(response_confidence, *, score_norm=1.0) -> RetrievalHit:
    return RetrievalHit(
        text="Contexte documentaire",
        collection="docs",
        score=0.5,
        score_norm=score_norm,
        response_confidence=response_confidence,
        source_url="https://example.test",
        section="Section",
    )


@pytest.mark.parametrize(
    ("score", "expected_mode"),
    [
        (0.86, ResponseMode.RAG_ONLY),
        (0.85, ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE),
        (0.70, ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE),
        (0.60, ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE),
        (0.59, ResponseMode.GENERAL_KNOWLEDGE_ONLY),
        (0.0, ResponseMode.GENERAL_KNOWLEDGE_ONLY),
        (-0.1, ResponseMode.GENERAL_KNOWLEDGE_ONLY),
    ],
)
def test_select_response_mode_uses_explicit_boundaries(score, expected_mode):
    decision = select_response_mode([hit(score)], ResponseStrategyConfig())

    assert decision.mode is expected_mode
    assert decision.confidence_score == score
    assert decision.confidence_available is True


def test_select_response_mode_uses_first_final_result_after_reranking():
    final_hits = [
        hit(0.72),
        hit(0.99),
    ]

    decision = select_response_mode(final_hits, ResponseStrategyConfig())

    assert decision.mode is ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE
    assert decision.confidence_score == 0.72


@pytest.mark.parametrize(
    "invalid_score",
    [None, float("nan"), float("inf"), 1.1, -1.1, True, "0.9"],
)
def test_missing_or_invalid_final_score_uses_general_knowledge(invalid_score):
    decision = select_response_mode([hit(invalid_score)], ResponseStrategyConfig())

    assert decision.mode is ResponseMode.GENERAL_KNOWLEDGE_ONLY
    assert decision.confidence_score is None
    assert decision.confidence_available is False
    assert MISSING_SCORE_WARNING in decision.warning


def test_no_retrieved_results_uses_general_knowledge_with_missing_score_warning():
    decision = select_response_mode([], ResponseStrategyConfig())

    assert decision.mode is ResponseMode.GENERAL_KNOWLEDGE_ONLY
    assert decision.confidence_score is None
    assert decision.warning == f"{MISSING_SCORE_WARNING}\n\n{GENERAL_KNOWLEDGE_WARNING}"


def test_disabled_strategy_preserves_legacy_rag_prompt_selection():
    decision = select_response_mode([], ResponseStrategyConfig(enabled=False))

    assert decision.mode is ResponseMode.RAG_ONLY
    assert decision.strategy_enabled is False
    assert decision.warning is None


@pytest.mark.parametrize(
    "config",
    [
        ResponseStrategyConfig(enabled="yes"),
        ResponseStrategyConfig(general_knowledge_threshold=0.85, rag_only_threshold=0.85),
        ResponseStrategyConfig(general_knowledge_threshold=0.90, rag_only_threshold=0.85),
        ResponseStrategyConfig(general_knowledge_threshold=-0.1),
        ResponseStrategyConfig(rag_only_threshold=1.1),
        ResponseStrategyConfig(general_knowledge_threshold=float("nan")),
        ResponseStrategyConfig(rag_only_threshold=float("inf")),
        ResponseStrategyConfig(general_knowledge_threshold=True),
    ],
)
def test_response_strategy_rejects_invalid_thresholds(config):
    with pytest.raises(ValueError, match="RESPONSE_"):
        config.validate()


def test_app_config_validation_reports_threshold_ordering():
    config = AppConfig(
        qdrant_collections=("docs",),
        response_strategy=ResponseStrategyConfig(
            general_knowledge_threshold=0.85,
            rag_only_threshold=0.60,
        ),
    )

    with pytest.raises(ValueError, match="strictly lower"):
        config.validate()


def test_response_strategy_reads_environment_configuration(monkeypatch):
    monkeypatch.setenv("RESPONSE_STRATEGY_ENABLED", "off")
    monkeypatch.setenv("RESPONSE_GENERAL_KNOWLEDGE_THRESHOLD", "0.25")
    monkeypatch.setenv("RESPONSE_RAG_ONLY_THRESHOLD", "0.75")

    config = ResponseStrategyConfig.from_env()

    assert config == ResponseStrategyConfig(
        enabled=False,
        general_knowledge_threshold=0.25,
        rag_only_threshold=0.75,
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RESPONSE_STRATEGY_ENABLED", "sometimes", "true/false"),
        ("RESPONSE_GENERAL_KNOWLEDGE_THRESHOLD", "high", "numeric value"),
    ],
)
def test_response_strategy_reports_invalid_environment_values(
    monkeypatch,
    name,
    value,
    message,
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        ResponseStrategyConfig.from_env()


def test_response_mode_selection_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="canar.app.response_strategy"):
        select_response_mode([hit(0.70)], ResponseStrategyConfig())

    assert "mode=rag_with_general_knowledge" in caplog.text
    assert "response_confidence=0.7" in caplog.text


class FakeChatClient:
    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, **settings):
        self.calls.append((messages, settings))
        yield "Réponse du modèle"


def test_general_knowledge_stream_guarantees_warning_and_reuses_generation_settings():
    chat = FakeChatClient()
    decision = select_response_mode([hit(0.40)], ResponseStrategyConfig())
    messages, sources = r_helpdesk.build_messages(
        "Question générale",
        [hit(0.40)],
        response_mode=decision.mode,
    )

    answer = "".join(
        stream_response(
            chat,
            messages,
            decision,
            temperature=0.3,
            max_tokens=512,
        )
    )

    assert answer == f"{GENERAL_KNOWLEDGE_WARNING}\n\nRéponse du modèle"
    assert "Contexte documentaire" not in messages[1]["content"]
    assert sources == []
    assert chat.calls == [(messages, {"temperature": 0.3, "max_tokens": 512})]


def test_missing_score_stream_guarantees_both_warnings():
    decision = select_response_mode([], ResponseStrategyConfig())

    answer = "".join(stream_response(FakeChatClient(), [], decision))

    assert answer.startswith(f"{MISSING_SCORE_WARNING}\n\n{GENERAL_KNOWLEDGE_WARNING}\n\n")


def test_rag_stream_has_no_fallback_warning():
    decision = select_response_mode([hit(0.90)], ResponseStrategyConfig())

    answer = "".join(stream_response(FakeChatClient(), [], decision))

    assert answer == "Réponse du modèle"


def test_score_norm_is_never_used_for_response_mode_selection():
    high_cosine = select_response_mode(
        [hit(0.90, score_norm=0.0)],
        ResponseStrategyConfig(),
    )
    missing_cosine = select_response_mode(
        [hit(None, score_norm=1.0)],
        ResponseStrategyConfig(),
    )

    assert high_cosine.mode is ResponseMode.RAG_ONLY
    assert high_cosine.confidence_score == 0.90
    assert missing_cosine.mode is ResponseMode.GENERAL_KNOWLEDGE_ONLY
    assert missing_cosine.confidence_available is False
