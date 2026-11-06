# Evaluation results

LLM `gemini:gemini-3.5-flash-lite` · embeddings `huggingface:sentence-transformers/all-MiniLM-L6-v2` · 20 probes over 1 run(s)

| Metric | With memory | Without memory |
|---|---|---|
| Continuity (probes answered from earlier sessions) | 100.0% | 10.0% |
| Chat latency, mean | 941.1 ms | 864.5 ms |
| Chat latency, p95 | 2278.1 ms | 1658.4 ms |

Memory recall step alone: mean 150.2 ms, p95 430.7 ms.

## Failed probes (with memory)

None.
