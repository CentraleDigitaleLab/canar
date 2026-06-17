from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = "utilitr_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "qdrant_payload_export.jsonl"


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY") or None

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = export_payloads(
        client=client,
        collection=args.collection,
        output_path=output_path,
        batch_size=args.batch_size,
    )
    print(f"Exported {exported} payloads to {output_path}")
    return 0


def export_payloads(
    client: QdrantClient,
    collection: str,
    output_path: Path,
    batch_size: int,
) -> int:
    next_page_offset: Any = None
    exported = 0

    with output_path.open("w", encoding="utf-8") as handle:
        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                handle.write(json.dumps(point_to_row(point), ensure_ascii=False) + "\n")
                exported += 1

            if next_page_offset is None:
                break

    return exported


def point_to_row(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "point_id": str(point.id),
        "file_path": payload.get("file_path"),
        "chunk_index": payload.get("chunk_index"),
        "text": payload.get("text"),
        "section": payload.get("section"),
        "breadcrumbs": payload.get("breadcrumbs"),
        "source_url": payload.get("source_url") or payload.get("url"),
        "token_count": payload.get("token_count"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qdrant point payloads for benchmark ground-truth inspection."
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
