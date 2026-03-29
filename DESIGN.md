# Design

The system is a single FastAPI service with one public runtime surface: `POST /runs/stream`. Each request creates a fresh ADK session, a run workspace on disk, and a constrained six-tool registry that owns all invoice access. The server, not ADK, owns the public SSE contract and maps internal planner/tool events into the assignment-defined event names.

The planner and execution loop are separated. ADK `LlmAgent` plus `Runner` handle the turn-by-turn agent loop, while the tool registry owns invoice discovery, extraction, normalization, categorization, aggregation, and report generation. In mock mode the agent uses a deterministic custom `BaseLlm` planner that still chooses the next tool from intermediate results, including a retry branch when extraction misses critical fields. In live mode the same agent, tool registry, state model, and SSE adapter stay in place, but the planner model can switch to Gemini through ADK.

Session state is the execution scratchpad. `load_images` registers invoice refs, later tools enrich per-invoice state, and `generate_report` converts that accumulated state into the final structured output. This keeps the model from receiving raw invoice data outside the tools and makes it straightforward to emit `invoice_result` events once categorization finishes.

Tracing is intentionally simple for the timebox. Every run writes JSONL traces plus a saved final report in the run directory. The trace captures run start, planner progress narration, tool calls, tool results, invoice-level completions, and the final result. A deterministic checked-in sample run under `sample_traces/` makes compliance quick to review.

I chose Google ADK because it satisfies the locked architectural direction while still letting the server control the reviewer-facing SSE wire format. The mock-first path keeps tests and artifacts deterministic without replacing the real planner loop, session handling, or streaming pipeline.

For production, I would replace the in-memory session service with a persistent store, add stronger JSON Schema validation on streamed payloads, move trace storage behind a real observability backend, support real multimodal extraction for live mode, and tighten authorization around allowed local folders.
