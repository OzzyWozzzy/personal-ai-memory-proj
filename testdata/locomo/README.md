# LoCoMo evaluation for the Graphiti-native engine

This harness evaluates the current memory engine against the official
`locomo10.json` release. It deliberately does not use `benchmarks/runner.py`:
that runner targets deleted pre-Graphiti modules and does not implement the
official LoCoMo answer metric.

## Methodology

- Dataset: SNAP Research `locomo10.json`, pinned SHA-256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Isolation: one tenant/graph per conversation.
- Ingestion: one native Graphiti `message` episode per dated session, in
  chronological order. Speaker names and released BLIP captions are retained.
- Retrieval: the hybrid graph, lexical, and semantic context assembler,
  fixed 1,500-token budget, historical facts enabled for every question.
- Answers: generated from blind `{question, context}` records; gold answers,
  category labels, and evidence IDs are kept in a separate private scoring file.
- Primary score: official normalized lexical F1 on categories 1-4 (`n=1540`).
  Adversarial abstention is reported separately because the upstream category-5
  data/runner contract is inconsistent.
- Retrieval score: annotated evidence-dialog recall and complete-evidence rate.

The default Graphiti mode is explicitly a **session-episode ablation**, not the
local API's actor-turn granularity. A faithful 5,882-turn Graphiti
extraction run would require roughly 47,000 structured LLM responses with the
current bridge, versus about 2,000 for the 272-session ablation. The
`--direct-raw` mode below loads all 5,882 turns without extraction and isolates
the source retrieval/RAG path.

## Run

```bash
mkdir -p testdata/locomo/data
curl -fsSL \
  https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json \
  -o testdata/locomo/data/locomo10.json

MEMORY_GRAPHITI_LLM=record \
MEMORY_GRAPH_DIR=.graphdb-locomo \
MEMORY_BRIDGE_DIR=.bridge-locomo \
.venv/bin/python testdata/locomo/run_locomo.py --run-id graphiti-session-v1
```

The driver is resumable and writes ignored artifacts beneath
`testdata/locomo/runs/<run-id>/`. `--scripted --max-sessions 1 --max-questions 5`
is an infrastructure smoke test only; it returns no extracted graph facts and
must never be reported as a Graphiti quality score.

For the full direct-ingest source/RAG baseline (all 5,882 actor turns, no
Graphiti LLM extraction), run:

```bash
MEMORY_GRAPH_DIR=.graphdb-locomo \
.venv/bin/python testdata/locomo/run_locomo.py \
  --run-id direct-raw-v1 --direct-raw
```

This is useful for isolating retrieval quality, but it must be labeled
`direct-raw`: it does not measure graph extraction or traversal.

After blind answers have been written to `predictions.jsonl`:

```bash
.venv/bin/python testdata/locomo/answer_with_codex.py \
  testdata/locomo/runs/graphiti-session-v1 --model gpt-5.6-sol

.venv/bin/python testdata/locomo/score_locomo.py \
  testdata/locomo/runs/graphiti-session-v1
```

## Standard binary LLM J-score

For an apples-to-apples Mem0-style evaluation, freeze a category-stratified
sample, pass it to `run_locomo.py` with `--qid-file`, then run the isolated
answer and judge phases:

```bash
.venv/bin/python testdata/locomo/prepare_jscore_sample.py \
  testdata/locomo/data/locomo10.json \
  testdata/locomo/runs/<new-run-id>/sample_manifest.json

MEMORY_RAW_RERANK_POOL=200 \
MEMORY_RAW_FINAL_CAP=50 \
MEMORY_GRAPH_DIR=.graphdb-locomo-<new-run-id> \
.venv/bin/python testdata/locomo/run_locomo.py \
  --run-id <new-run-id> \
  --qid-file testdata/locomo/runs/<new-run-id>/sample_manifest.json \
  --token-budget 7000 --direct-raw

.venv/bin/python testdata/locomo/jscore_with_codex.py \
  testdata/locomo/runs/<new-run-id> --phase answer
.venv/bin/python testdata/locomo/jscore_with_codex.py \
  testdata/locomo/runs/<new-run-id> --phase judge
```

This excludes category 5, uses the pinned Mem0 answer and binary judge prompts,
and never passes annotated evidence to the judge. Generated run artifacts stay
under the ignored `testdata/locomo/runs/` directory.
