import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from canar.app.response_strategy import ResponseStrategyConfig

load_dotenv(override=True)


@dataclass
class AppConfig:
    llm_base: str = os.getenv("LLM_API_BASE", "")
    llm_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_thinking: str = os.getenv("LLM_THINKING", "")  # none, low, medium, high

    embed_base: str = os.getenv("EMBED_API_BASE", "")
    embed_key: str = os.getenv("EMBED_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "")

    fastembed_sparse_model: str = os.getenv("FASTEMBED_SPARSE_MODEL", "")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_dense_vector_name: str = os.getenv("QDRANT_DENSE_VECTOR_NAME", "")
    qdrant_sparse_vector_name: str = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "")
    qdrant_collections: list[str] = tuple(
        c.strip() for c in os.getenv("QDRANT_COLLECTIONS", "").split(",") if c.strip()
    )

    db_path: str = os.getenv("APP_DB", "data/app.db")
    response_strategy: ResponseStrategyConfig = field(
        default_factory=ResponseStrategyConfig.from_env
    )

    def validate(self) -> None:
        if not self.qdrant_collections:
            raise ValueError("QDRANT_COLLECTIONS cannot be empty")
        self.response_strategy.validate()
