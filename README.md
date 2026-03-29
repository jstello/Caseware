# Invoice Agent

This repository contains a local invoice-processing agent built for the Caseware take-home assignment. The implementation uses Python, FastAPI, Google ADK, a server-owned SSE adapter, and a deterministic mock-first tool pipeline.

## What It Does

- Exposes `POST /runs/stream`
- Accepts either a local folder path or multipart invoice images
- Runs a planner loop through a constrained six-tool registry
- Streams `run_started`, `progress`, `tool_call`, `tool_result`, `invoice_result`, `final_result`, and `error`
- Produces JSONL traces plus a saved final report for every run
- Ships with six synthetic invoice fixtures and a checked-in sample trace

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --group dev
```

## Run The API

Mock mode is the default and does not require any secrets.

```bash
uv run uvicorn invoice_agent.app:app --reload
```

The API starts on [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Mock Mode

Mock mode is the intended development and test path.

- It uses the real FastAPI endpoint, ADK runner, session state, and SSE mapping.
- It keeps invoice access constrained to the registered tools.
- It uses deterministic synthetic fixtures so traces and tests are reviewable.
- One fixture intentionally triggers a second extraction attempt to prove the planner makes a tool decision from intermediate results.

## Optional Live Planner Mode

The planner can be switched to Gemini through ADK without reading any `.env` file:

```bash
export INVOICE_AGENT_PLANNER_MODE=live
export INVOICE_AGENT_LIVE_MODEL=gemini-2.5-flash
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_API_KEY=REDACTED_VALUE
uv run uvicorn invoice_agent.app:app --reload
```

The repository intentionally does not read `.env` or `.env.*` files.

## cURL Example: Folder Input

```bash
curl -N \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/runs/stream \
  -d '{
    "folder_path": "/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices",
    "prompt": "Use conservative categorization and flag unusual invoices."
  }'
```

## cURL Example: Multipart Input

```bash
curl -N \
  -X POST http://127.0.0.1:8000/runs/stream \
  -F "prompt=Flag unusual invoices." \
  -F "files=@/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices/acme-air-travel-001.svg;type=image/svg+xml" \
  -F "files=@/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices/team-bistro-ambiguous.svg;type=image/svg+xml"
```

## Example SSE Output

```text
event: run_started
data: {"run_id":"sample-mock-run","mode":"mock",...}

event: progress
data: {"run_id":"sample-mock-run","message":"I need to discover the local invoice images before I can inspect any invoice details.","agent":"invoice_agent"}

event: tool_call
data: {"run_id":"sample-mock-run","tool_name":"load_images","tool_call_id":"load_images-2752","stage":"loading","args":{}}

event: tool_result
data: {"run_id":"sample-mock-run","tool_name":"load_images","tool_call_id":"load_images-2752","stage":"loading","result":{"invoice_count":6,...}}

event: final_result
data: {"run_id":"sample-mock-run","report":{"run_summary":{"total_spend":1508.32,...}},...}
```

## Validation

Run the test suite:

```bash
uv run pytest -q
```

Current coverage includes:

- SSE event ordering and presence
- Deterministic final totals for the folder path flow
- Prompt influence on ambiguous categorization
- Multipart upload support
- Retry behavior for the noisy invoice fixture

## Review Artifacts

- Contract tests: [`tests/test_streaming_endpoint.py`](/Users/juan_tello/Documents/Caseware/Caseware/tests/test_streaming_endpoint.py)
- Synthetic fixtures: [`fixtures/invoices`](/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices)
- Sample trace: [`sample_traces/mock_folder_run/trace.jsonl`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/trace.jsonl)
- Sample SSE log: [`sample_traces/mock_folder_run/sse_events.json`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/sse_events.json)
- Sample final result: [`sample_traces/mock_folder_run/final_result.json`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/final_result.json)
- AI session log: [`transcripts/codex-session-2026-03-28.md`](/Users/juan_tello/Documents/Caseware/Caseware/transcripts/codex-session-2026-03-28.md)
