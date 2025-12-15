Practical tips for row‑wise enrichment
•
Concurrency: Use a small worker pool (e.g., 2–6 workers) when calling a local model; find the sweet spot where GPU/CPU
stays busy without OOM.
•
Batching: If your framework supports it, send small batches to reduce overhead.
•
Retries & timeouts: Add exponential backoff and timeouts for robustness.
•
Determinism/quality: Set low temperature, seed (if supported), and use few‑shot examples in your prompt for consistency.
•
Schema validation: If you need structured outputs (JSON), use Guidance/Outlines or LangChain/LlamaIndex structured
output helpers.
•
Idempotency: Write intermediate checkpoints (every N rows) to handle restarts; keep an id column to join with originals.
•
Caching: Enable LangChain/LlamaIndex caching to avoid recompute when iterating.
•
Observability: Use tracing (LangSmith for LangChain, Haystack telemetry, basic logging) to diagnose prompt/latency.