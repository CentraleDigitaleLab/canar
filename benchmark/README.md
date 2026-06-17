# CanaR/AgoRa Retrieval Benchmark

This directory contains a standalone benchmark runner for the CanaR/AgoRa RAG retrieval pipeline. It lives outside the chatbot UI and calls retrieval code programmatically, so retrieval strategies can be evaluated without changing the app flow.

The current benchmark can evaluate dense and sparse Qdrant retrieval over the same benchmark dataset.

## Current Strategies

Implemented benchmark strategies:

- `simple_vector`: dense vector retrieval through CanaR `SimpleVectorStrategy`
- `simple_sparse`: sparse vector retrieval through CanaR `SimpleSparseStrategy`

Accepted but not implemented yet:

- `hybrid`
- `hybrid_rerank`
- `contextual_retrieval`
- `parent_child`

Those placeholder strategies write clear errors until the corresponding CanaR retrieval strategies are available.

## Run Dense And Sparse

From the `/home/cereq/opt/sacha/canar` project root:

```bash
python benchmark/run_benchmark.py \
  --dataset benchmark/benchmark_dataset.example.json \
  --strategies simple_vector simple_sparse \
  --collections utilitr_v2 \
  --top-k 10 \
  --vector-name dense \
  --sparse-vector-name sparse \
  --output benchmark/runs \
  --results-output benchmark/results
```

This compares dense and sparse retrieval against the same questions and ground truth.

## Required Configuration

The runner loads CanaR configuration from `.env` or the current shell.

Dense retrieval needs:

- `EMBED_API_BASE`
- `EMBED_MODEL`
- `QDRANT_URL`
- `QDRANT_COLLECTIONS`, or pass `--collections`

Sparse retrieval needs:

- `FASTEMBED_SPARSE_MODEL`
- `QDRANT_URL`
- `QDRANT_SPARSE_VECTOR_NAME`, or pass `--sparse-vector-name`
- `QDRANT_COLLECTIONS`, or pass `--collections`

For named-vector Qdrant collections, pass `--vector-name` for dense retrieval and `--sparse-vector-name` for sparse retrieval. Dense retrieval defaults to `dense` when omitted. Sparse retrieval defaults to `QDRANT_SPARSE_VECTOR_NAME` when omitted.

For a Qdrant collection like `utilitr_v2` with vectors named `dense`, `sparse`, and `multi`, the benchmark currently supports:

- `dense` through `simple_vector`
- `sparse` through `simple_sparse`
- `multi` is not supported yet

## Experiment Configs

Experiment configs are saved benchmark run recipes. They are not datasets and not results.

Run a dense recipe:

```bash
python benchmark/run_benchmark.py \
  --experiment-config benchmark/experiments/utilitr_simple_vector_dense.example.json
```

Run a sparse recipe:

```bash
python benchmark/run_benchmark.py \
  --experiment-config benchmark/experiments/utilitr_simple_sparse.example.json
```

Config example:

```json
{
  "run_id": "utilitr_simple_vector_dense_example",
  "dataset": "benchmark/benchmark_dataset.example.json",
  "collections": ["utilitr_v2"],
  "strategies": ["simple_vector"],
  "top_k": 10,
  "vector_name": "dense",
  "output": "benchmark/runs",
  "results_output": "benchmark/results"
}
```

CLI flags override values from the config. Each benchmark execution writes `benchmark/results/run_manifest.json` so runs can be compared later by a dashboard or database layer.

## Dataset Format

The benchmark dataset is shared by all retrieval methods so comparisons are fair. The current dataset lives at:

```text
benchmark/benchmark_dataset.example.json
```

It must be a JSON array:

```json
[
  {
    "id": "q_exact_001",
    "question": "What does melt() do in data.table?",
    "question_category": "Exact Lookup",
    "expected_chunks": [
      "03_Fiches_thematiques/Fiche_datatable.qmd#6"
    ],
    "expected_documents": [
      "03_Fiches_thematiques/Fiche_datatable.qmd"
    ],
    "answer_present": true,
    "difficulty": "easy",
    "collection": "utilitr_v2"
  }
]
```

`expected_chunks` should use this format:

```text
file_path#chunk_index
```

If a question has no answer in the corpus, use:

```json
"expected_chunks": [],
"expected_documents": [],
"answer_present": false
```

When `expected_chunks` is empty, retrieval relevance metrics are blank for that question, but raw retrieval outputs and latency metrics are still saved.

## Qdrant Payload Export

Use this script to export Qdrant payloads without vectors:

```bash
python benchmark/export_qdrant_payloads.py \
  --collection utilitr_v2 \
  --output benchmark/qdrant_payload_export.jsonl
```

The export is useful for building ground truth because each row includes fields such as:

- `point_id`
- `file_path`
- `chunk_index`
- `text`
- `section`
- `breadcrumbs`
- `source_url`
- `token_count`

Vectors are not exported.

## Outputs

Per-strategy retrieval outputs are written as JSONL:

- `benchmark/runs/simple_vector.jsonl`
- `benchmark/runs/simple_sparse.jsonl`

Each line represents one question result and includes:

- question id, question text, and category
- strategy name
- vector name used by that strategy
- retrieved chunk ids
- retrieved document ids when available
- retrieval scores
- latency in milliseconds
- context token count when available
- raw retrieved metadata when available
- expected chunks/documents from the dataset
- error message if retrieval failed

Combined results and metric tables are written to:

- `benchmark/results/run_manifest.json`
- `benchmark/results/raw_results.jsonl`
- `benchmark/results/metrics_by_strategy.csv`
- `benchmark/results/metrics_by_category.csv`
- `benchmark/results/metrics_by_strategy_and_category.csv`

The strategy-by-category CSV is shaped for spreadsheet use:

```text
Category, Retrieval Strategy, Metric Engine, Hit@10, Recall@10, MRR, Latency(ms), Precision@10, nDCG, Token Cost, Qualitative Notes
```

## Metrics

Manual metrics are computed first:

- `Hit@k`
- `Recall@k`
- `Precision@k`
- `MRR`
- `nDCG@k`
- average latency
- median latency
- p95 latency

Metrics are aggregated by:

- strategy
- question category
- strategy and question category

If a question has no `expected_chunks`, `Hit@k`, `Recall@k`, `Precision@k`, `MRR`, and `nDCG@k` are blank for that question's aggregate groups. Latency is always computed.

## ranx Metrics

`ranx` is the default retrieval metric engine when it is installed and at least one question has `expected_chunks`. The benchmark evaluates each aggregate group with ranx, then writes the same CSV outputs.

The ranx-backed metrics are:

- `hit_rate@k`, written as `Hit@k`
- `recall@k`
- `precision@k`
- `mrr`
- `ndcg@k`

If `ranx` is not installed, if a ranx evaluation fails, or if no ground-truth chunks exist, the benchmark falls back to the manual metric implementation. CSV outputs include a `Metric Engine` column so you can see whether each group used `ranx`, `manual`, or no retrieval metrics.

## Generated Files

These files are benchmark outputs or labeling aids rather than framework code:

- `benchmark/qdrant_payload_export.jsonl`
- `benchmark/runs/*.jsonl`
- `benchmark/results/*.jsonl`
- `benchmark/results/*.csv`
- `benchmark/results/run_manifest.json`

They can be regenerated. Commit them only when you intentionally want to version a specific export or benchmark result.

## Future Work

Likely next steps:

- add `failures.csv` for question-level debugging
- add Qdrant collection validation before running retrieval
- add hybrid and rerank strategies once available in CanaR
- later, add DuckDB/dashboard support for comparing many runs
- later, add answer-level evaluation such as RAGAS
