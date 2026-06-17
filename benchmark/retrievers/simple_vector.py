from __future__ import annotations

from dataclasses import replace
from typing import Any


class SimpleVectorRunner:
    def __init__(
        self,
        cfg: Any,
        top_k: int,
        collections: tuple[str, ...] | None = None,
        vector_name: str | None = "dense",
    ):
        from canar.app.api.embed_client import EmbedClient
        from canar.app.retrieval.adapters.qdrant import QdrantRetrievalAdapter
        from canar.app.retrieval.models import RetrievalQuery
        from canar.app.retrieval.profiles import build_retrieval_profiles
        from canar.app.retrieval.strategies.simple_vector import SimpleVectorStrategy

        self.embed_client = EmbedClient(cfg.embed_base, cfg.embed_model, cfg.embed_key)
        profile_collections = collections or tuple(cfg.qdrant_collections)
        profile = build_retrieval_profiles(profile_collections)["simple_vector"]
        self.vector_name = vector_name
        self.profile = replace(profile, top_k=top_k, vector_name=vector_name)
        self.strategy = SimpleVectorStrategy(
            self.profile,
            QdrantRetrievalAdapter(cfg.qdrant_url, cfg.qdrant_api_key),
        )
        self.retrieval_query_cls = RetrievalQuery

    def retrieve(self, question: str) -> list[Any]:
        dense_vector = self.embed_client.embed_query(question)
        query = self.retrieval_query_cls(
            text=question,
            profile_name=self.profile.name,
            dense_vector=dense_vector,
        )
        return self.strategy.search(query)
