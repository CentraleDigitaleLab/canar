from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
from typing import Any

from io_utils import ensure_dir, read_dataset, read_json, write_json, write_jsonl
from metrics import compute_and_write_metrics
from retrievers import SUPPORTED_STRATEGIES, build_runner
from retrievers.base import StrategyRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANAR_ROOT = PROJECT_ROOT


def main() -> int:
    args = _parse_args()
    _add_package_paths()
    _load_canar_env()

    dataset = read_dataset(args.dataset)
    output_dir = ensure_dir(args.output)
    results_dir = ensure_dir(args.results_output)

    cfg = _build_config(args.strategies)
    collections = tuple(args.collections) if args.collections else None
    if not collections and not tuple(cfg.qdrant_collections):
        dataset_collections = tuple(
            sorted({str(row["collection"]) for row in dataset if row.get("collection")})
        )
        collections = dataset_collections or None

    all_records: list[dict[str, Any]] = []
    for strategy_name in args.strategies:
        runner = build_runner(
            strategy_name,
            cfg,
            args.top_k,
            collections,
            args.vector_name,
            args.sparse_vector_name,
        )
        strategy_records = _run_strategy(dataset, strategy_name, runner, args.top_k)
        all_records.extend(strategy_records)
        strategy_path = output_dir / f"{strategy_name}.jsonl"
        write_jsonl(strategy_path, strategy_records)
        print(f"Wrote {len(strategy_records)} rows to {strategy_path}")

    raw_results_path = results_dir / "raw_results.jsonl"
    write_jsonl(raw_results_path, all_records)
    print(f"Wrote combined raw results to {raw_results_path}")

    compute_and_write_metrics(all_records, str(results_dir), args.top_k)
    print(f"Wrote metric CSVs to {results_dir}")

    manifest_path = results_dir / "run_manifest.json"
    write_json(
        manifest_path,
        _build_run_manifest(
            args=args,
            cfg=cfg,
            dataset=dataset,
            collections=collections,
            output_dir=output_dir,
            results_dir=results_dir,
            record_count=len(all_records),
        ),
    )
    print(f"Wrote run manifest to {manifest_path}")
    return 0


def _run_strategy(
    dataset: list[dict[str, Any]],
    strategy_name: str,
    runner: StrategyRunner,
    top_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in dataset:
        started = time.perf_counter()
        error: str | None = None
        hits: list[Any] = []
        try:
            hits = runner.retrieve(item["question"])[:top_k]
        except NotImplementedError as exc:
            error = str(exc)
            print(f"Warning: {error}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"Warning: retrieval failed for {item['id']} with {strategy_name}: {error}")
        latency_ms = (time.perf_counter() - started) * 1000

        records.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "question_category": item.get("question_category"),
                "strategy": strategy_name,
                "top_k": top_k,
                "vector_name": getattr(runner, "vector_name", None),
                "retrieved_chunk_ids": [_chunk_id(hit, rank) for rank, hit in enumerate(hits, start=1)],
                "retrieved_document_ids": [_document_id(hit) for hit in hits],
                "retrieval_scores": [_score(hit) for hit in hits],
                "latency_ms": latency_ms,
                "context_tokens": _context_tokens(hits),
                "raw_retrieved_metadata": [_metadata(hit) for hit in hits],
                "expected_chunks": item.get("expected_chunks") or [],
                "expected_documents": item.get("expected_documents") or [],
                "answer_present": item.get("answer_present"),
                "difficulty": item.get("difficulty"),
                "collection": item.get("collection"),
                "error": error,
            }
        )
    return records


def _build_config(strategies: list[str]) -> Any:
    from canar.app.config import AppConfig

    cfg = AppConfig()
    missing = []
    if "simple_vector" in strategies and not cfg.embed_base:
        missing.append("EMBED_API_BASE")
    if "simple_vector" in strategies and not cfg.embed_model:
        missing.append("EMBED_MODEL")
    if not cfg.qdrant_url:
        missing.append("QDRANT_URL")
    if missing:
        raise ValueError(
            "Missing required CanaR retrieval configuration: "
            + ", ".join(missing)
            + ". Set these in canar/.env or the shell before running the benchmark."
        )
    return cfg


def _build_run_manifest(
    args: argparse.Namespace,
    cfg: Any,
    dataset: list[dict[str, Any]],
    collections: tuple[str, ...] | None,
    output_dir: Path,
    results_dir: Path,
    record_count: int,
) -> dict[str, Any]:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    categories = sorted({str(row.get("question_category")) for row in dataset if row.get("question_category")})
    difficulties = sorted({str(row.get("difficulty")) for row in dataset if row.get("difficulty")})
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config": args.experiment_config,
        "dataset": {
            "path": args.dataset,
            "question_count": len(dataset),
            "categories": categories,
            "difficulties": difficulties,
            "questions_with_ground_truth": sum(1 for row in dataset if row.get("expected_chunks")),
        },
        "retrieval": {
            "strategies": args.strategies,
            "collections": list(collections or tuple(cfg.qdrant_collections)),
            "top_k": args.top_k,
            "vector_name": args.vector_name,
            "sparse_vector_name": args.sparse_vector_name or cfg.qdrant_sparse_vector_name or None,
            "embed_model": cfg.embed_model,
            "fastembed_sparse_model": cfg.fastembed_sparse_model or None,
            "qdrant_url": cfg.qdrant_url,
        },
        "artifacts": {
            "runs_dir": str(output_dir),
            "results_dir": str(results_dir),
            "raw_results": str(results_dir / "raw_results.jsonl"),
            "metrics_by_strategy": str(results_dir / "metrics_by_strategy.csv"),
            "metrics_by_category": str(results_dir / "metrics_by_category.csv"),
            "metrics_by_strategy_and_category": str(
                results_dir / "metrics_by_strategy_and_category.csv"
            ),
        },
        "record_count": record_count,
    }


def _load_canar_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(CANAR_ROOT / ".env", override=False)


def _add_package_paths() -> None:
    for path in (str(CANAR_ROOT), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _metadata(hit: Any) -> dict[str, Any]:
    metadata = getattr(hit, "metadata", None) or {}
    return dict(metadata)


def _chunk_id(hit: Any, rank: int) -> str | None:
    metadata = _metadata(hit)
    for key in (
        "chunk_id",
        "chunk_uuid",
        "point_id",
        "id",
        "_id",
        "uuid",
        "hash",
        "chunk_hash",
    ):
        if metadata.get(key) is not None:
            return str(metadata[key])

    document_id = _document_id(hit)
    chunk_index = _first_present(metadata, ("chunk_index", "chunk_idx", "part", "offset"))
    if document_id is not None and chunk_index is not None:
        return f"{document_id}#{chunk_index}"
    if document_id is not None:
        section = getattr(hit, "section", None) or metadata.get("section")
        if section:
            return f"{document_id}#{section}"
    return None


def _document_id(hit: Any) -> str | None:
    metadata = _metadata(hit)
    for key in (
        "document_id",
        "doc_id",
        "file_id",
        "source_id",
        "document",
        "file_path",
        "url",
        "source_url",
    ):
        if metadata.get(key) is not None:
            return str(metadata[key])

    source_url = getattr(hit, "source_url", None)
    if source_url:
        return str(source_url)
    source = getattr(hit, "source", None)
    return str(source) if source else None


def _score(hit: Any) -> float | None:
    score = getattr(hit, "score", None)
    if score is None:
        return None
    return float(score)


def _context_tokens(hits: list[Any]) -> int | None:
    total = 0
    found = False
    for hit in hits:
        metadata = _metadata(hit)
        token_value = _first_present(
            metadata,
            ("context_tokens", "token_count", "tokens", "n_tokens", "num_tokens"),
        )
        if token_value is None:
            continue
        try:
            total += int(token_value)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def _first_present(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if metadata.get(key) is not None:
            return metadata[key]
    return None


def _parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--experiment-config")
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = _load_experiment_config(pre_args.experiment_config)

    parser = argparse.ArgumentParser(
        description="Run standalone retrieval benchmarks against the CanaR/AgoRa RAG pipeline."
    )
    parser.add_argument(
        "--experiment-config",
        default=pre_args.experiment_config,
        help="Optional JSON experiment config. CLI flags override config values.",
    )
    parser.add_argument(
        "--run-id",
        default=experiment_config.get("run_id"),
        help="Optional stable id written to run_manifest.json.",
    )
    parser.add_argument(
        "--dataset",
        default=_config_path(
            experiment_config.get("dataset"),
            PROJECT_ROOT / "benchmark" / "benchmark_dataset.example.json",
        ),
        help="Path to the benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=experiment_config.get("strategies", ["simple_vector"]),
        choices=SUPPORTED_STRATEGIES,
        help="Retrieval strategies to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=experiment_config.get("top_k", 10),
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--vector-name",
        default=experiment_config.get("vector_name"),
        help="Qdrant named dense vector. Defaults to dense for simple_vector.",
    )
    parser.add_argument(
        "--sparse-vector-name",
        default=experiment_config.get("sparse_vector_name"),
        help="Qdrant named sparse vector. Defaults to QDRANT_SPARSE_VECTOR_NAME for simple_sparse.",
    )
    parser.add_argument(
        "--output",
        default=_config_path(
            experiment_config.get("output"),
            PROJECT_ROOT / "benchmark" / "runs",
        ),
        help="Directory for per-strategy JSONL run files.",
    )
    parser.add_argument(
        "--results-output",
        default=_config_path(
            experiment_config.get("results_output"),
            PROJECT_ROOT / "benchmark" / "results",
        ),
        help="Directory for combined raw results and metrics CSV files.",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=experiment_config.get("collections"),
        help="Optional Qdrant collection names. Defaults to QDRANT_COLLECTIONS, then dataset collection values.",
    )
    return parser.parse_args()


def _load_experiment_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return read_json(_config_path(path, path))


def _config_path(value: str | None, fallback: str | Path) -> str:
    if value is None:
        return str(fallback)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
