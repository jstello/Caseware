# Codex Session Log

This is the implementation log captured from the Codex build session for this repository.

## Kickoff

- User input: `documentation/implementation-kickoff-prompt.md`
- First actions:
  - Read `AGENTS.md`
  - Read `documentation/assignment.md`
  - Read `documentation/adr/ADR-001-adopt-google-adk-mock-first-custom-sse-contract.md`
  - Read `.codex/config.toml`
  - Read `.codex/agents/*.toml`

## Observations

- The repository initially contained only the assignment, ADR, and agent configuration documents.
- The source of truth locked the stack to Python + FastAPI + Google ADK, mock-first execution, and a custom SSE contract.
- The local system `python3` was `3.9.6`, but the current `google-adk` package requires Python `>=3.10`.

## Decisions

- Created a local Python 3.12 virtual environment with `uv`.
- Installed `google-adk==1.28.0` and inspected the ADK runtime.
- Ran a minimal custom-LLM ADK experiment to confirm how `Runner.run_async()` emits:
  - model tool-call events
  - tool response events
  - final agent response events
- Implemented a deterministic mock planner as a custom ADK `BaseLlm` so mock mode still exercises a real planner loop.

## Implemented Work

- FastAPI application with:
  - `POST /runs/stream`
  - JSON folder-path input
  - multipart upload input
  - custom SSE event mapping
- Constrained six-tool registry:
  - `load_images`
  - `extract_invoice_fields`
  - `normalize_invoice`
  - `categorize_invoice`
  - `aggregate_invoices`
  - `generate_report`
- Synthetic invoice fixture pack with six varied cases
- Deterministic sample trace and final report artifacts
- Contract tests for:
  - required SSE event presence and ordering
  - prompt-influenced categorization
  - multipart support
  - extraction retry behavior

## Validation Commands

```bash
uv sync --group dev
uv run pytest -q
```

## Generated Reviewer Artifacts

- `sample_traces/mock_folder_run/trace.jsonl`
- `sample_traces/mock_folder_run/sse_events.json`
- `sample_traces/mock_folder_run/final_result.json`
