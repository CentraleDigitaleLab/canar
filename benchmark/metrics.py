from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Iterable

from io_utils import write_csv


MetricRows = dict[str, list[dict[str, Any]]]


def compute_and_write_metrics(
    records: list[dict[str, Any]],
    results_dir: str,
    top_k: int,
) -> MetricRows:
    ranx_available = _try_ranx(records, top_k)
    if not ranx_available and _has_ground_truth(records):
        print("Warning: ranx is not installed or could not evaluate this run; using manual metrics.")
    elif not _has_ground_truth(records):
        print("No expected_chunks found; retrieval metrics are blank and latency metrics were computed.")

    by_strategy = _aggregate(records, top_k, lambda row: (row.get("strategy") or ""))
    by_category = _aggregate(records, top_k, lambda row: (row.get("question_category") or ""))
    by_strategy_category = _aggregate(
        records,
        top_k,
        lambda row: (
            row.get("strategy") or "",
            row.get("question_category") or "",
        ),
    )

    strategy_rows = [_strategy_row(key, metrics, top_k) for key, metrics in by_strategy.items()]
    category_rows = [_category_row(key, metrics, top_k) for key, metrics in by_category.items()]
    strategy_category_rows = [
        _strategy_category_row(key, metrics, top_k) for key, metrics in by_strategy_category.items()
    ]

    write_csv(
        f"{results_dir}/metrics_by_strategy.csv",
        strategy_rows,
        [
            "Retrieval Strategy",
            "Questions",
            f"Hit@{top_k}",
            f"Recall@{top_k}",
            f"Precision@{top_k}",
            "MRR",
            f"nDCG@{top_k}",
            "Latency Avg(ms)",
            "Latency Median(ms)",
            "Latency p95(ms)",
            "Token Cost",
        ],
    )
    write_csv(
        f"{results_dir}/metrics_by_category.csv",
        category_rows,
        [
            "Category",
            "Questions",
            f"Hit@{top_k}",
            f"Recall@{top_k}",
            f"Precision@{top_k}",
            "MRR",
            f"nDCG@{top_k}",
            "Latency Avg(ms)",
            "Latency Median(ms)",
            "Latency p95(ms)",
            "Token Cost",
        ],
    )
    write_csv(
        f"{results_dir}/metrics_by_strategy_and_category.csv",
        strategy_category_rows,
        [
            "Category",
            "Retrieval Strategy",
            f"Hit@{top_k}",
            f"Recall@{top_k}",
            "MRR",
            "Latency(ms)",
            f"Precision@{top_k}",
            "nDCG",
            "Token Cost",
            "Qualitative Notes",
        ],
    )

    return {
        "by_strategy": strategy_rows,
        "by_category": category_rows,
        "by_strategy_and_category": strategy_category_rows,
    }


def _aggregate(
    records: Iterable[dict[str, Any]],
    top_k: int,
    key_fn: Callable[[dict[str, Any]], Any],
) -> dict[Any, dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)

    return {key: _aggregate_group(rows, top_k) for key, rows in sorted(grouped.items())}


def _aggregate_group(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    per_question = [_question_metrics(record, top_k) for record in records]
    latencies = [_as_float(record.get("latency_ms")) for record in records]
    latencies = [value for value in latencies if value is not None]
    token_costs = [_as_float(record.get("context_tokens")) for record in records]
    token_costs = [value for value in token_costs if value is not None]

    return {
        "questions": len(records),
        "hit": _mean_or_none(row["hit"] for row in per_question),
        "recall": _mean_or_none(row["recall"] for row in per_question),
        "precision": _mean_or_none(row["precision"] for row in per_question),
        "mrr": _mean_or_none(row["mrr"] for row in per_question),
        "ndcg": _mean_or_none(row["ndcg"] for row in per_question),
        "latency_avg_ms": _mean_or_none(latencies),
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95) if latencies else None,
        "token_cost": _mean_or_none(token_costs),
    }


def _question_metrics(record: dict[str, Any], top_k: int) -> dict[str, float | None]:
    expected = [str(item) for item in record.get("expected_chunks") or []]
    if not expected:
        return {"hit": None, "recall": None, "precision": None, "mrr": None, "ndcg": None}

    expected_set = set(expected)
    retrieved = [
        str(item)
        for item in (record.get("retrieved_chunk_ids") or [])[:top_k]
        if item is not None
    ]
    relevant_at_rank = [chunk_id in expected_set for chunk_id in retrieved]
    relevant_found = {chunk_id for chunk_id in retrieved if chunk_id in expected_set}

    first_relevant_rank = next(
        (rank for rank, is_relevant in enumerate(relevant_at_rank, start=1) if is_relevant),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevant_at_rank, start=1)
        if is_relevant
    )
    ideal_relevant_count = min(len(expected_set), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    return {
        "hit": 1.0 if relevant_found else 0.0,
        "recall": len(relevant_found) / len(expected_set),
        "precision": len(relevant_found) / top_k if top_k else 0.0,
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def _try_ranx(records: list[dict[str, Any]], top_k: int) -> bool:
    if not _has_ground_truth(records):
        return False

    try:
        from ranx import Qrels, Run, evaluate
    except ImportError:
        return False

    try:
        qrels_data: dict[str, dict[str, int]] = {}
        run_data: dict[str, dict[str, float]] = {}
        for record in records:
            qid = str(record["question_id"])
            expected = record.get("expected_chunks") or []
            if expected:
                qrels_data[qid] = {str(chunk_id): 1 for chunk_id in expected}
                run_data[qid] = {
                    str(chunk_id): float(score)
                    for chunk_id, score in zip(
                        record.get("retrieved_chunk_ids") or [],
                        record.get("retrieval_scores") or [],
                    )
                    if chunk_id is not None and score is not None
                }

        if not qrels_data or not run_data:
            return False

        metrics = [f"hit_rate@{top_k}", f"recall@{top_k}", f"precision@{top_k}", "mrr", f"ndcg@{top_k}"]
        scores = evaluate(Qrels(qrels_data), Run(run_data), metrics)
        print(f"ranx overall metrics: {scores}")
        return True
    except Exception as exc:
        print(f"Warning: ranx evaluation failed ({exc}); using manual metrics.")
        return False


def _has_ground_truth(records: list[dict[str, Any]]) -> bool:
    return any(record.get("expected_chunks") for record in records)


def _strategy_row(strategy: str, metrics: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "Retrieval Strategy": strategy,
        "Questions": metrics["questions"],
        f"Hit@{top_k}": metrics["hit"],
        f"Recall@{top_k}": metrics["recall"],
        f"Precision@{top_k}": metrics["precision"],
        "MRR": metrics["mrr"],
        f"nDCG@{top_k}": metrics["ndcg"],
        "Latency Avg(ms)": metrics["latency_avg_ms"],
        "Latency Median(ms)": metrics["latency_median_ms"],
        "Latency p95(ms)": metrics["latency_p95_ms"],
        "Token Cost": metrics["token_cost"],
    }


def _category_row(category: str, metrics: dict[str, Any], top_k: int) -> dict[str, Any]:
    row = _strategy_row("", metrics, top_k)
    row.pop("Retrieval Strategy")
    row["Category"] = category
    return row


def _strategy_category_row(
    key: tuple[str, str],
    metrics: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    strategy, category = key
    return {
        "Category": category,
        "Retrieval Strategy": strategy,
        f"Hit@{top_k}": metrics["hit"],
        f"Recall@{top_k}": metrics["recall"],
        "MRR": metrics["mrr"],
        "Latency(ms)": metrics["latency_avg_ms"],
        f"Precision@{top_k}": metrics["precision"],
        "nDCG": metrics["ndcg"],
        "Token Cost": metrics["token_cost"],
        "Qualitative Notes": "",
    }


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
