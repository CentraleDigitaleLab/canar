from __future__ import annotations

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
)

AGENT_RETRIEVAL_PROFILES: dict[str, str | None] = {
    "r_helpdesk": "simple_vector",
    "sas_to_r": None,
}


def build_retrieval_profiles(
    collections: tuple[str, ...],
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> dict[str, RetrievalProfile]:
    return {
        "simple_vector": RetrievalProfile(
            name="simple_vector",
            strategy="simple_vector",
            collections=collections,
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=3,
            vector_name=dense_vector_name or None,
        ),
        "simple_sparse": RetrievalProfile(
            name="simple_sparse",
            strategy="simple_sparse",
            collections=collections,
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=3,
            vector_name=sparse_vector_name or None,
        ),
        "hybrid": RetrievalProfile(
            name="hybrid",
            strategy="hybrid",
            collections=collections,
            top_k=5,
            score_threshold=0.35,
            source_filter="utilitr",
            fallback_top_k=5,
            vector_name=sparse_vector_name or None,
            dense=DenseRetrievalParams(
                top_k=10,
                min_score=0.35,
                max_kept=None,
            ),
            sparse=SparseRetrievalParams(
                top_k=10,
                min_score_ratio=0.35,
                gap_ratio=None,
                max_kept=None,
            ),
            fusion=FusionRetrievalParams(
                method="rrf",
                rrf_k=60,
                weights={"dense": 1.0, "sparse": 1.0},
                final_top_k=5,
            ),
        ),
    }
