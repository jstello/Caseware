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
- `final_result` is emitted once, after the agent loop finishes and the report has been saved.
- `error` terminates the run without `final_result`.

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
