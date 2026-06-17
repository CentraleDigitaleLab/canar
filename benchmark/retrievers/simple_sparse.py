from __future__ import annotations

from dataclasses import replace
from typing import Any


class SimpleSparseRunner:
    def __init__(
        self,
        cfg: Any,
        top_k: int,
        collections: tuple[str, ...] | None = None,
        vector_name: str | None = None,
    ):
        from canar.app.api.embed_client import FastEmbedClient
        from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
        from canar.app.retrieval.models import RetrievalQuery
        from canar.app.retrieval.profiles import build_retrieval_profiles
        from canar.app.retrieval.strategies.simple_sparse import SimpleSparseStrategy

        sparse_model = cfg.fastembed_sparse_model
        if not sparse_model:
            raise ValueError(
                "simple_sparse benchmark retrieval requires FASTEMBED_SPARSE_MODEL "
                "in .env or the shell."
            )

        self.sparse_embed_client = FastEmbedClient(sparse_model)
        profile_collections = collections or tuple(cfg.qdrant_collections)
        sparse_vector_name = vector_name or cfg.qdrant_sparse_vector_name or None
        profile = build_retrieval_profiles(
            profile_collections,
            sparse_vector_name=sparse_vector_name,
        )["simple_sparse"]
        self.vector_name = sparse_vector_name
        self.profile = replace(profile, top_k=top_k, vector_name=sparse_vector_name)
        self.strategy = SimpleSparseStrategy(
            self.profile,
            QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key),
        )
        self.retrieval_query_cls = RetrievalQuery

    def retrieve(self, question: str) -> list[Any]:
        sparse_vector = self.sparse_embed_client.embed_query(question)
        query = self.retrieval_query_cls(
            text=question,
            profile_name=self.profile.name,
            sparse_vector=sparse_vector,
        )
        return self.strategy.search(query)
