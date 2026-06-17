# CanaR/AgoRa Retrieval Benchmark

This directory contains a standalone benchmark runner for the CanaR/AgoRa RAG retrieval pipeline. It lives outside the chatbot UI and calls retrieval code programmatically, so retrieval strategies can be evaluated without changing the app flow.

## Run

From the `/home/cereq/opt/sacha/canar` project root:

```bash
python benchmark/run_benchmark.py \
  --dataset benchmark/benchmark_dataset.example.json \
  --strategies simple_vector hybrid hybrid_rerank \
  --top-k 10 \
  --vector-name dense \
  --output benchmark/runs/
```

The runner loads CanaR configuration from `.env` or the current shell. At minimum, retrieval needs:

- `EMBED_API_BASE`
- `EMBED_MODEL`
- `QDRANT_URL`
- `QDRANT_COLLECTIONS`, or pass `--collections`

For named-vector Qdrant collections, pass `--vector-name`. The current default is `dense`.

Only `simple_vector` is currently implemented in CanaR. The other accepted strategy names (`hybrid`, `hybrid_rerank`, `contextual_retrieval`, `parent_child`) are explicit placeholders and will write clear error messages until those strategies are added to `canar.app.retrieval`.

## Experiment Configs

CLI flags can be replaced with an experiment config:

```bash
python benchmark/run_benchmark.py \
  --experiment-config benchmark/experiments/utilitr_simple_vector_dense.example.json
```

Config example:

```json
{
  "run_id": "utilitr_simple_vector_dense_example",
  "dataset": "benchmark/benchmark_dataset.example.json",
  "collections": ["utilitr_v1"],
  "strategies": ["simple_vector"],
  "top_k": 10,
  "vector_name": "dense",
  "output": "benchmark/runs",
  "results_output": "benchmark/results"
}
```

CLI flags override values from the config. Each benchmark execution writes `benchmark/results/run_manifest.json` so runs can be compared later by a dashboard or database layer.

## Dataset Format

`benchmark_dataset.example.json` is a JSON list:

```json
[
  {
    "id": "q_exact_001",
    "question": "What does variable PS010 mean?",
    "question_category": "Exact Lookup",
    "expected_chunks": [],
    "expected_documents": [],
    "answer_present": true,
    "difficulty": "easy",
    "collection": "variables"
  }
]
```

`expected_chunks` can be empty while the gold dataset is still being built. In that case, the benchmark skips retrieval relevance metrics for those questions but still saves the retrieved chunks, scores, latency, metadata, and CSV latency summaries.

## Outputs

Per-strategy retrieval outputs are written as JSONL:

- `benchmark/runs/simple_vector.jsonl`
- `benchmark/runs/hybrid.jsonl`
- `benchmark/runs/hybrid_rerank.jsonl`

Each line represents one question result and includes:

- question id, question text, and category
- strategy name
- retrieved chunk ids
- retrieved document ids when available
- retrieval scores
- latency in milliseconds
- context token count when available
- raw retrieved metadata when available
- expected chunks/documents from the dataset

Combined results and metric tables are written to:

- `benchmark/results/run_manifest.json`
- `benchmark/results/raw_results.jsonl`
- `benchmark/results/metrics_by_strategy.csv`
- `benchmark/results/metrics_by_category.csv`
- `benchmark/results/metrics_by_strategy_and_category.csv`

The strategy-by-category CSV is shaped for spreadsheet use:

```text
Category, Retrieval Strategy, Hit@10, Recall@10, MRR, Latency(ms), Precision@10, nDCG, Token Cost, Qualitative Notes
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

If a question has no `expected_chunks`, `Hit@k`, `Recall@k`, `Precision@k`, `MRR`, and `nDCG@k` are blank for that question's aggregate groups. Latency is always computed.

Metrics are aggregated by:

- strategy
- question category
- strategy and question category

## Adding Ground Truth

When official gold labels are available, add chunk ids to each question:

```json
"expected_chunks": ["variables:PS010:definition"]
```

The benchmark will start computing relevance metrics automatically. The retrieved chunk ids are extracted from common metadata fields such as `chunk_id`, `id`, `uuid`, `document_id` plus chunk index, or document plus section. If the current Qdrant payload does not include stable chunk ids, add them during indexing before using relevance metrics seriously.

## Optional ranx Evaluation

If `ranx` is installed and at least one question has `expected_chunks`, the benchmark checks that the saved qrels/runs can be evaluated with:

- `hit_rate@10`
- `recall@10`
- `precision@10`
- `mrr`
- `ndcg@10`

If `ranx` is not installed or cannot evaluate the run, the benchmark prints a warning and falls back to the manual metrics.

## Future Answer Evaluation

This module only evaluates retrieval. It intentionally does not implement RAGAS yet, but the JSONL records keep enough question, context, metadata, and strategy information to add answer-quality evaluation later without changing the chatbot UI.
