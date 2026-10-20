# Evaluation

The primary quality result is a frozen one-third LoCoMo evaluation of the full
graph-memory lane: Graphiti extraction, temporal graph retrieval, source
hydration, answer generation, and binary J-score judging.

## Results

| Category | Graph memory | Wilson 95% CI | Direct-source ablation* |
|---|---:|---:|---:|
| Multi-hop | 79/94 · 84.04% | 75.33–90.08 | 86.17% |
| Temporal | 97/107 · 90.65% | 83.65–94.84 | 89.72% |
| Open-domain | 19/32 · 59.38% | 42.26–74.48 | 68.75% |
| Single-hop | 249/280 · 88.93% | 84.71–92.09 | 94.29% |
| **Overall** | **444/513 · 86.55%** | **83.32–89.23** | **90.25%** |

\* The direct-source result is a retrieval/RAG ablation with no graph
extraction. It used a different answerer model, so the comparison is diagnostic,
not a controlled claim that one memory representation is better.

## Protocol

- Dataset: official LoCoMo10 release, SHA-256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Sample: deterministic category-stratified one-third of categories 1–4,
  yielding 513 questions.
- Ingestion: 272 dated conversation sessions across 10 conversations as native
  Graphiti message episodes.
- Retrieval: the repository's hybrid graph/source context assembler with a
  fixed token budget and historical facts enabled.
- Scoring: isolated blind answer and judge phases using the pinned Mem0-style
  `CORRECT` / `WRONG` J-score prompt. Gold answers and evidence IDs were not
  supplied to retrieval or answer generation.

The scripts that freeze the sample, run retrieval, generate answers, and score
the results live under `testdata/locomo/`. Generated run artifacts are ignored.

## What the result shows

- Temporal performance is the strongest category: native validity windows and
  invalidations retain useful historical state without application-side date
  repair.
- Multi-hop performance is solid but remains below single-hop, leaving graph
  expansion and evidence packing as the clearest research direction.
- Open-domain has only 32 questions and a wide confidence interval; it should
  not drive architecture conclusions.
- The source lane remains important. Exact episode hydration recovers details
  that graph extraction may omit, while the graph supplies temporal precision
  and structured traversal.

## Limitations

- This run uses session-level Graphiti ingestion rather than the more expensive
  5,882 actor-turn extraction run.
- Retrieval was iterated against LoCoMo, so this is development-on-test rather
  than a sealed external benchmark.
- The direct-source and graph-memory rows have an answer-model confound.
- No latency or ingest-cost claim is attached to these accuracy numbers.

For reproducibility commands and scorer details, see
[`testdata/locomo/README.md`](../testdata/locomo/README.md).
