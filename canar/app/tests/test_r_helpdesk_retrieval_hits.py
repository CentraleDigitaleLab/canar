from __future__ import annotations

from canar.app.agents import r_helpdesk
from canar.app.response_strategy import ResponseMode
from canar.app.retrieval.models import RetrievalHit


def test_r_helpdesk_builds_context_from_retrieval_hits():
    hits = [
        RetrievalHit(
            text="Utilise filter() pour garder des lignes.",
            collection="utilitr",
            score=0.9,
            score_norm=1.0,
            source="utilitr",
            source_url="https://example.test/filter",
            section="Filtrer",
        )
    ]

    messages, sources = r_helpdesk.build_messages("Comment filtrer ?", hits)

    assert "[S1] Filtrer" in messages[1]["content"]
    assert "Utilise filter() pour garder des lignes." in messages[1]["content"]
    assert sources == [
        {
            "label": "S1",
            "url": "https://example.test/filter",
            "section": "Filtrer",
            "collection": "utilitr",
        }
    ]


def test_r_helpdesk_prefers_generation_text_when_available():
    hits = [
        RetrievalHit(
            text="Résumé court retrouvé.",
            generation_text="Texte original plus complet pour la génération.",
            collection="utilitr",
            score=0.9,
            score_norm=1.0,
            section="Résumé",
        )
    ]

    messages, _ = r_helpdesk.build_messages("Comment filtrer ?", hits)

    assert "Texte original plus complet pour la génération." in messages[1]["content"]
    assert "Résumé court retrouvé." not in messages[1]["content"]


def test_rag_only_prompt_uses_context_and_forbids_general_knowledge():
    hits = [
        RetrievalHit(
            text="Contexte fiable.",
            collection="utilitr",
            score=0.9,
            score_norm=0.9,
            section="Source",
        )
    ]

    messages, sources = r_helpdesk.build_messages(
        "Question",
        hits,
        response_mode=ResponseMode.RAG_ONLY,
    )

    assert "Contexte fiable." in messages[1]["content"]
    assert "uniquement à partir des extraits" in messages[1]["content"]
    assert "aucune information issue de tes connaissances générales" in messages[1]["content"]
    assert len(sources) == 1


def test_mixed_prompt_separates_dataset_and_general_knowledge_content():
    hits = [
        RetrievalHit(
            text="Contexte partiel.",
            collection="utilitr",
            score=0.7,
            score_norm=0.7,
            section="Source",
        )
    ]

    messages, sources = r_helpdesk.build_messages(
        "Question",
        hits,
        response_mode=ResponseMode.RAG_WITH_GENERAL_KNOWLEDGE,
    )
    prompt = messages[1]["content"]

    assert "Contexte partiel." in prompt
    assert "Informations issues de la base documentaire" in prompt
    assert "Complément fondé sur les connaissances générales du modèle" in prompt
    assert "les extraits documentaires prévalent" in prompt
    assert "aucune citation documentaire" in prompt
    assert len(sources) == 1


def test_general_knowledge_prompt_excludes_context_and_citations():
    hits = [
        RetrievalHit(
            text="Ce document ne doit pas être transmis.",
            collection="utilitr",
            score=0.4,
            score_norm=0.4,
            section="Source",
        )
    ]

    messages, sources = r_helpdesk.build_messages(
        "Question",
        hits,
        response_mode=ResponseMode.GENERAL_KNOWLEDGE_ONLY,
    )
    prompt = messages[1]["content"]

    assert "Ce document ne doit pas être transmis." not in prompt
    assert "Contexte (extraits documentaires)" not in prompt
    assert "connaissances générales" in prompt
    assert "n'ajoute aucune citation" in prompt
    assert sources == []
