# `/runs/stream` Contract

## Endpoint

- Method: `POST`
- Path: `/runs/stream`
- Response: `text/event-stream`

## Request Shapes

### JSON

```json
{
  "folder_path": "/absolute/or/local/path/to/invoice/folder",
  "prompt": "Optional reviewer guidance"
}
```

### Multipart

- Content type: `multipart/form-data`
- Fields:
  - `files`: one or more invoice images
  - `prompt`: optional string

Supported image suffixes in the current implementation: `.png`, `.jpg`, `.jpeg`, `.svg`, `.webp`

## Runtime Configuration

Non-secret runtime behavior is defined in [`config/invoice_agent.yaml`](/Users/juan_tello/Documents/Caseware/Caseware/config/invoice_agent.yaml).

- `runtime`: app name, planner mode, live model, extraction retry cap, local trace directory, and local MLflow storage path
- `agent`: root instruction, request prompt template, allowed categories, and tool-to-stage mapping
- `tracing`: MLflow experiment settings and artifact logging options

## SSE Event Schema

### `run_started`

```json
{
  "run_id": "string",
  "mode": "mock|live",
  "input_source": {
    "source_type": "folder|upload_dir",
    "path": "string"
  },
  "prompt": "string|null"
}
```

### `progress`

```json
{
  "run_id": "string",
  "message": "planner narration",
  "agent": "invoice_agent"
}
```

### `tool_call`

```json
{
  "run_id": "string",
  "tool_name": "load_images|extract_invoice_fields|normalize_invoice|categorize_invoice|aggregate_invoices|generate_report",
  "tool_call_id": "string",
  "stage": "loading|extraction|normalization|categorization|aggregation|reporting",
  "args": {}
}
```

### `tool_result`

```json
{
  "run_id": "string",
  "tool_name": "string",
  "tool_call_id": "string",
  "stage": "string",
  "result": {}
}
```

### `invoice_result`

```json
{
  "run_id": "string",
  "invoice_id": "string",
  "invoice": {
    "invoice_id": "string",
    "filename": "string",
    "vendor": "string|null",
    "invoice_date": "YYYY-MM-DD|null",
    "invoice_number": "string|null",
    "total": 0.0,
    "assigned_category": "Travel|Meals & Entertainment|Software / Subscriptions|Professional Services|Office Supplies|Shipping / Postage|Utilities|Other",
    "confidence": 0.0,
    "notes": ["string"]
  }
}
```

### `final_result`

```json
{
  "run_id": "string",
  "report": {
    "run_summary": {
      "total_spend": 0.0,
      "spend_by_category": {
        "Travel": 0.0
      },
      "invoice_count": 0
    },
    "invoices": [],
    "issues_and_assumptions": ["string"]
  },
  "trace_path": "string",
  "sse_path": "string",
  "report_path": "string"
}
```

### `error`

```json
{
  "run_id": "string",
  "error_type": "string",
  "message": "string"
}
```

## Ordering Rules

- The stream always begins with `run_started`.
- `progress`, `tool_call`, `tool_result`, and `invoice_result` may repeat.
- `invoice_result` is emitted immediately after a successful `categorize_invoice` result.
- `aggregate_invoices` and `generate_report` are valid only after `load_images` has run and every loaded invoice has reached `invoice_result`.
- `final_result` is emitted once, after the agent loop finishes and the report has been saved.
- In live mode, missing provider configuration emits `error` after `run_started` without attempting model execution.
- `error` terminates the run without `final_result`.

## Observability Outputs

Each run produces:

- JSONL execution trace at `trace_path`
- JSONL SSE log at `sse_path`
- Final structured report at `report_path`
- Prompt artifacts under `run_dir/prompts/`, including `system_instruction.txt` and `request_prompt.txt`
- An MLflow run in the local SQLite-backed experiment store containing flattened config params, tags, summary metrics, the effective config artifact, the request prompt artifact when present, and the saved run artifacts

## Tool Registry

The agent is limited to six tools and may only access invoice data through them.

### `load_images()`

- Purpose: discover the run-local invoice image files
- Inputs: none
- Output:

```json
{
  "invoice_count": 6,
  "invoice_refs": [
    {
      "invoice_id": "string",
      "filename": "string",
      "path": "string",
      "fixture_key": "string|null"
    }
  ],
  "notes": ["string"]
}
```

### `extract_invoice_fields(invoice_id, focus_hint?)`

- Purpose: extract raw invoice fields for one invoice
- Dynamic behavior: may be retried with a focused hint when critical fields are missing
- Output:

```json
{
  "invoice_id": "string",
  "attempt": 1,
  "vendor": "string|null",
  "invoice_date": "YYYY-MM-DD|null",
  "invoice_number": "string|null",
  "total": 0.0,
  "currency": "USD",
  "raw_category_hint": "string|null",
  "extraction_confidence": 0.0,
  "notes": ["string"],
  "missing_critical_fields": ["vendor", "total"],
  "needs_retry": true,
  "retry_focus_hint": "vendor, total"
}
```

### `normalize_invoice(invoice_id)`

- Purpose: convert extracted fields into one stable record
- Output:

```json
{
  "invoice_id": "string",
  "vendor": "string|null",
  "invoice_date": "YYYY-MM-DD|null",
  "invoice_number": "string|null",
  "total": 0.0,
  "currency": "USD",
  "notes": ["string"],
  "assumptions": ["string"]
}
```

### `categorize_invoice(invoice_id)`

- Purpose: map the normalized invoice to one allowed category
- Output:

```json
{
  "invoice_id": "string",
  "decision": {
    "invoice_id": "string",
    "category": "string",
    "confidence": 0.0,
    "notes": ["string"]
  },
  "invoice_result": {
    "invoice_id": "string",
    "filename": "string",
    "vendor": "string|null",
    "invoice_date": "YYYY-MM-DD|null",
    "invoice_number": "string|null",
    "total": 0.0,
    "assigned_category": "string",
    "confidence": 0.0,
    "notes": ["string"]
  }
}
```

### `aggregate_invoices()`

- Purpose: compute run-level totals
- Guardrail: fails if `load_images()` has not run or if any loaded invoice is still missing `invoice_result`
- Output:

```json
{
  "total_spend": 0.0,
  "spend_by_category": {
    "Travel": 0.0
  },
  "invoice_count": 0,
  "issues_and_assumptions": ["string"]
}
```

### `generate_report()`

- Purpose: build the final submission payload
- Guardrail: fails if `load_images()` has not run or if any loaded invoice is still missing `invoice_result`
- Output: same `report` object described in `final_result`

## Final Output Schema

The final structured output contains:

- `run_summary.total_spend`
- `run_summary.spend_by_category`
- `run_summary.invoice_count`
- `invoices[]`
  - `vendor`
  - `invoice_date`
  - `invoice_number`
  - `total`
  - `assigned_category`
  - `confidence`
  - `notes`
- `issues_and_assumptions[]`
