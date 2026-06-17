from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Benchmark dataset must be a JSON list: {dataset_path}")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object")
        for required_key in ("id", "question", "question_category"):
            if required_key not in item:
                raise ValueError(f"Dataset item {index} is missing {required_key!r}")
        item.setdefault("expected_chunks", [])
        item.setdefault("expected_documents", [])
        if not isinstance(item["expected_chunks"], list):
            raise ValueError(f"Dataset item {item['id']!r} expected_chunks must be a list")
        if not isinstance(item["expected_documents"], list):
            raise ValueError(f"Dataset item {item['id']!r} expected_documents must be a list")

    return data


def read_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {input_path}")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    if not input_path.exists():
        return rows

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}") from exc
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value
