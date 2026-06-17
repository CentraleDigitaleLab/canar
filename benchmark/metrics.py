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
    metric_engine = _select_metric_engine(records)
    if metric_engine == "ranx":
        print("Using ranx as the primary retrieval metric engine.")
    elif _has_ground_truth(records):
        print("Warning: ranx is not installed or failed to initialize; using manual metrics.")
    else:
        print("No expected_chunks found; retrieval metrics are blank and latency metrics were computed.")

    by_strategy = _aggregate(records, top_k, lambda row: (row.get("strategy") or ""), metric_engine)
    by_category = _aggregate(records, top_k, lambda row: (row.get("question_category") or ""), metric_engine)
    by_strategy_category = _aggregate(
        records,
        top_k,
        lambda row: (
            row.get("strategy") or "",
            row.get("question_category") or "",
        ),
        metric_engine,
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
            "Metric Engine",
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
            "Metric Engine",
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
            "Metric Engine",
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


def _select_metric_engine(records: list[dict[str, Any]]) -> str:
    if not _has_ground_truth(records):
        return "manual"
    try:
        import ranx  # noqa: F401
    except ImportError:
        return "manual"
    return "ranx"


def _aggregate(
    records: Iterable[dict[str, Any]],
    top_k: int,
    key_fn: Callable[[dict[str, Any]], Any],
    metric_engine: str,
) -> dict[Any, dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)

    return {
        key: _aggregate_group(rows, top_k, metric_engine)
        for key, rows in sorted(grouped.items())
    }


def _aggregate_group(
    records: list[dict[str, Any]],
    top_k: int,
    metric_engine: str,
) -> dict[str, Any]:
    retrieval_metrics = _empty_retrieval_metrics()
    engine_used = "none"

    if _has_ground_truth(records):
        if metric_engine == "ranx":
            try:
                retrieval_metrics = _ranx_group_metrics(records, top_k)
                engine_used = "ranx"
            except Exception as exc:
                print(f"Warning: ranx group evaluation failed ({exc}); using manual metrics.")
                retrieval_metrics = _manual_group_metrics(records, top_k)
                engine_used = "manual"
        else:
            retrieval_metrics = _manual_group_metrics(records, top_k)
            engine_used = "manual"

    latency_metrics = _latency_metrics(records)
    return {
        "questions": len(records),
        "metric_engine": engine_used,
        **retrieval_metrics,
        **latency_metrics,
    }


def _manual_group_metrics(records: list[dict[str, Any]], top_k: int) -> dict[str, float | None]:
    per_question = [_question_metrics(record, top_k) for record in records]
    return {
        "hit": _mean_or_none(row["hit"] for row in per_question),
        "recall": _mean_or_none(row["recall"] for row in per_question),
        "precision": _mean_or_none(row["precision"] for row in per_question),
        "mrr": _mean_or_none(row["mrr"] for row in per_question),
        "ndcg": _mean_or_none(row["ndcg"] for row in per_question),
    }


def _ranx_group_metrics(records: list[dict[str, Any]], top_k: int) -> dict[str, float | None]:
    from ranx import Qrels, Run, evaluate

    qrels_data: dict[str, dict[str, int]] = {}
    run_data: dict[str, dict[str, float]] = {}

    for index, record in enumerate(records):
        expected = [str(chunk_id) for chunk_id in record.get("expected_chunks") or []]
        if not expected:
            continue

        qid = _ranx_qid(record, index)
        qrels_data[qid] = {chunk_id: 1 for chunk_id in expected}
        run_data[qid] = _ranx_run_items(record)

    if not qrels_data:
        return _empty_retrieval_metrics()

    metric_names = [
        f"hit_rate@{top_k}",
        f"recall@{top_k}",
        f"precision@{top_k}",
        "mrr",
        f"ndcg@{top_k}",
    ]
    scores = evaluate(Qrels(qrels_data), Run(run_data), metric_names)
    return {
        "hit": _score_value(scores, f"hit_rate@{top_k}"),
        "recall": _score_value(scores, f"recall@{top_k}"),
        "precision": _score_value(scores, f"precision@{top_k}"),
        "mrr": _score_value(scores, "mrr"),
        "ndcg": _score_value(scores, f"ndcg@{top_k}"),
    }


def _ranx_qid(record: dict[str, Any], index: int) -> str:
    strategy = record.get("strategy") or "unknown_strategy"
    question_id = record.get("question_id") or f"question_{index}"
    return f"{strategy}:{question_id}:{index}"


def _ranx_run_items(record: dict[str, Any]) -> dict[str, float]:
    run_items: dict[str, float] = {}
    retrieved = record.get("retrieved_chunk_ids") or []
    scores = record.get("retrieval_scores") or []
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id is None:
            continue
        score = scores[rank - 1] if rank - 1 < len(scores) else None
        numeric_score = _as_float(score)
        run_items[str(chunk_id)] = (
            numeric_score if numeric_score is not None else float(len(retrieved) - rank + 1)
        )
    return run_items


def _score_value(scores: Any, key: str) -> float | None:
    if isinstance(scores, dict):
        return _as_float(scores.get(key))
    return _as_float(scores)


def _latency_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    latencies = [_as_float(record.get("latency_ms")) for record in records]
    latencies = [value for value in latencies if value is not None]
    token_costs = [_as_float(record.get("context_tokens")) for record in records]
    token_costs = [value for value in token_costs if value is not None]
    return {
        "latency_avg_ms": _mean_or_none(latencies),
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95) if latencies else None,
        "token_cost": _mean_or_none(token_costs),
    }


def _empty_retrieval_metrics() -> dict[str, None]:
    return {"hit": None, "recall": None, "precision": None, "mrr": None, "ndcg": None}


def _question_metrics(record: dict[str, Any], top_k: int) -> dict[str, float | None]:
    expected = [str(item) for item in record.get("expected_chunks") or []]
    if not expected:
        return _empty_retrieval_metrics()

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


def _has_ground_truth(records: list[dict[str, Any]]) -> bool:
    return any(record.get("expected_chunks") for record in records)


def _strategy_row(strategy: str, metrics: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "Retrieval Strategy": strategy,
        "Questions": metrics["questions"],
        "Metric Engine": metrics["metric_engine"],
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
        "Metric Engine": metrics["metric_engine"],
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
