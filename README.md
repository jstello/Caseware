# Invoice Agent

This repository contains a local invoice-processing agent built for the Caseware take-home assignment. The implementation uses Python, FastAPI, Google ADK, a server-owned SSE adapter, a deterministic mock-first tool pipeline, YAML-backed runtime configuration, and MLflow-backed run observability.

## What It Does

- Exposes `POST /runs/stream`
- Accepts either a local folder path or multipart invoice images
- Runs a planner loop through a constrained six-tool registry
- Uses Gemini-backed extraction and categorization in live mode while keeping normalization, aggregation, and reporting deterministic
- Streams `run_started`, `progress`, `tool_call`, `tool_result`, `invoice_result`, `final_result`, and `error`
- Produces JSONL traces plus a saved final report for every run
- Logs MLflow experiments, run params, metrics, and trace artifacts for each run
- Captures live Gemini thought summaries in internal traces and MLflow while keeping the public SSE contract unchanged
- Keeps prompts, model selection, tool stages, and tracing settings in [`config/invoice_agent.yaml`](/Users/juan_tello/Documents/Caseware/Caseware/config/invoice_agent.yaml)
- Ships with six synthetic invoice fixtures and a checked-in sample trace

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --group dev
```

## Configuration

Non-secret runtime behavior lives in [`config/invoice_agent.yaml`](/Users/juan_tello/Documents/Caseware/Caseware/config/invoice_agent.yaml).

- `runtime`: app name, planner mode, extraction retry limit, trace directory, and local MLflow storage path
- `agent`: root agent name, description, planner prompt, live extraction prompt, live categorization prompt, allowed categories, and tool stage labels
- `tracing`: MLflow enablement, experiment name, tracking URI override, and artifact logging options

Environment variables can still override selected values without reading any `.env` file:

```bash
export INVOICE_AGENT_PLANNER_MODE=live
export INVOICE_AGENT_LIVE_MODEL=gemini-2.5-flash
export INVOICE_AGENT_MLFLOW_EXPERIMENT_NAME=invoice-agent-live
```

## Run The API

Mock mode is the default and does not require any secrets.

```bash
uv run uvicorn invoice_agent.app:app --reload
```

The API starts on [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Test In ADK Web

You can also exercise the ADK-native agent directly in the Web UI:

```bash
uv run adk web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), choose `invoice_agent`, and start a session.

- The checked-in config currently defaults to the live planner, so the UI will call Gemini unless you override it.
- If your message includes an absolute local folder path, `load_images` will use that folder.
- If you upload invoice images directly in the ADK Web chat, the app now materializes those uploads into a run-local folder and `load_images` will use them instead of silently falling back to bundled fixtures.
- If you want the deterministic fixture-backed UI flow instead, run `INVOICE_AGENT_PLANNER_MODE=mock uv run adk web`. With no folder path, ADK Web falls back to the bundled fixture folder at [`fixtures/invoices`](/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices).
- `adk` does not need to be installed globally, `uv run adk web` is enough.

## Mock Mode

Mock mode is the intended development and test path.

Override into mock mode explicitly when you want deterministic local runs:

```bash
INVOICE_AGENT_PLANNER_MODE=mock uv run uvicorn invoice_agent.app:app --reload
```

- It uses the real FastAPI endpoint, ADK runner, session state, and SSE mapping.
- It keeps invoice access constrained to the registered tools.
- It uses deterministic synthetic fixtures so traces and tests are reviewable.
- One fixture intentionally triggers a second extraction attempt to prove the planner makes a tool decision from intermediate results.
- The runtime blocks premature `aggregate_invoices` or `generate_report` calls until every loaded invoice has been categorized.
- In live mode, SDK-exposed Gemini thought summaries are preserved in internal trace artifacts and tool-response history so later planner turns can see earlier tool reasoning without exposing those summaries on the public SSE stream.

## Live Planner Mode

The checked-in config now defaults to Gemini through ADK. If you need to set the runtime explicitly, you can still do it without reading any `.env` file:

```bash
export INVOICE_AGENT_PLANNER_MODE=live
export INVOICE_AGENT_LIVE_MODEL=gemini-2.5-flash
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_API_KEY=REDACTED_VALUE
uv run uvicorn invoice_agent.app:app --reload
```

The repository intentionally does not read `.env` or `.env.*` files.
If `INVOICE_AGENT_PLANNER_MODE=live` is set without either `GOOGLE_API_KEY` or a valid Vertex setup (`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`), the stream emits `run_started` followed by a `LiveConfigurationError`.

In live mode:

- The planner stays on ADK `Gemini`
- `extract_invoice_fields` uses a multimodal Gemini API call for raster images, and falls back to parsing raw SVG source for `.svg` invoices because Gemini rejects `image/svg+xml`
- `categorize_invoice` uses a text-only Gemini API call with schema-constrained JSON output
- Final categories are still clamped in code to the assignment taxonomy so unsupported model labels gracefully fall back to `Other`

## MLflow Tracing

MLflow tracing is enabled by default and writes to a local SQLite-backed store under [`artifacts/mlflow`](/Users/juan_tello/Documents/Caseware/Caseware/artifacts/mlflow).

- Each run logs flattened config params, tags, metrics, the effective YAML config artifact, the request prompt artifact, the JSONL trace, the SSE log, and the final report.
- ADK Web invocations now emit the same MLflow run and local trace artifacts as the custom `/runs/stream` endpoint.
- The saved request prompt artifact captures the effective planner prompt the model saw after template expansion, not just the optional reviewer hint.
- The run directory also keeps `prompts/system_instruction.txt` and `prompts/request_prompt.txt` so prompt review still works when MLflow is disabled.
- Tool execution is traced with lightweight decorator-based MLflow spans while the existing JSONL trace remains the reviewer-friendly source of truth.
- Live Gemini planner decisions and live extraction/categorization tool calls also capture thought summaries, signature presence, and reasoning-token counts in MLflow spans plus a `thought_ledger.json` artifact.
- You can point to another MLflow backend by setting `INVOICE_AGENT_MLFLOW_TRACKING_URI`.
- Repo-local `sitecustomize.py` now exports matching `MLFLOW_BACKEND_STORE_URI` and `MLFLOW_TRACKING_URI` defaults, so plain `uv run mlflow ui` opens the same SQLite store the app writes to.

### Version Tracking

Git-linked version tracking is enabled before each run starts so the active `LoggedModel` is associated with the planner trace and the run tags.

- The recorder calls `mlflow.genai.enable_git_model_versioning()` before the agent loop starts.
- Each run logs `mlflow.active_model_id` plus the Git branch, commit, and dirty-state tags when Git metadata is available.
- The SSE `run_started` and `final_result` payloads include the active version metadata so the stream itself is self-describing.
- To inspect the version in the UI, open the experiment, click the Logged Model entry, and then use the Traces tab for that model.
- To query the traces programmatically:

```python
import mlflow

model_id = "<model_id from the run_started payload or MLflow run tag>"
traces = mlflow.search_traces(model_id=model_id, return_type="list")
```

If Git metadata cannot be detected, the run still proceeds normally and version tracking is skipped gracefully.

If you want to inspect the local MLflow runs in the UI:

```bash
uv run mlflow ui
```

If you launch MLflow from outside the repo root, or you intentionally want an explicit command, use:

```bash
uv run mlflow ui --backend-store-uri sqlite:////Users/juan_tello/Documents/Caseware/Caseware/artifacts/mlflow/mlflow.db
```

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
- ADK Web agent discovery plus end-to-end mock execution
- Deterministic final totals for the folder path flow
- Prompt influence on ambiguous categorization
- Multipart upload support
- Retry behavior for the noisy invoice fixture
- Guardrails that reject early finalization before all invoices are processed
- Live extraction request shaping and schema parsing
- Live categorization request shaping and category clamping
- Deterministic live-mode runs using mocked Gemini responses

## Review Artifacts

- Contract tests: [`tests/test_streaming_endpoint.py`](/Users/juan_tello/Documents/Caseware/Caseware/tests/test_streaming_endpoint.py)
- ADK pattern audit: [`documentation/adk_pattern_audit.md`](/Users/juan_tello/Documents/Caseware/Caseware/documentation/adk_pattern_audit.md)
- YAML runtime config: [`config/invoice_agent.yaml`](/Users/juan_tello/Documents/Caseware/Caseware/config/invoice_agent.yaml)
- Synthetic fixtures: [`fixtures/invoices`](/Users/juan_tello/Documents/Caseware/Caseware/fixtures/invoices)
- Sample trace: [`sample_traces/mock_folder_run/trace.jsonl`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/trace.jsonl)
- Sample SSE log: [`sample_traces/mock_folder_run/sse_events.json`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/sse_events.json)
- Sample final result: [`sample_traces/mock_folder_run/final_result.json`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/mock_folder_run/final_result.json)
- Simulated live trace: [`sample_traces/live_mode_simulated/trace.jsonl`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/live_mode_simulated/trace.jsonl)
- Simulated live final result: [`sample_traces/live_mode_simulated/final_result.json`](/Users/juan_tello/Documents/Caseware/Caseware/sample_traces/live_mode_simulated/final_result.json)
- AI session log: [`transcripts/codex-session-2026-03-28.md`](/Users/juan_tello/Documents/Caseware/Caseware/transcripts/codex-session-2026-03-28.md)
