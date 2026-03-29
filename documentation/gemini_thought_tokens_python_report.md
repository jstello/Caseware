# Gemini Python: Surfacing Thought / Reasoning Tokens

Date: 2026-03-29

This note summarizes how the official Gemini API docs expose model reasoning in Python, based on the current Google AI for Developers documentation.

## What the Python SDK exposes

The docs split reasoning into three related but distinct surfaces:

1. **Thought summaries**: human-readable summaries of the model's internal reasoning.
2. **Thought token counts**: usage metadata that reports how many reasoning tokens were generated.
3. **Thought signatures**: cryptographic markers that preserve reasoning context across turns, especially during tool calling.

That distinction matters because you can surface summaries without counting tokens, count tokens without printing summaries, and preserve multi-turn reasoning without exposing raw chain-of-thought. The docs also note that thinking budgets and levels apply to the model's raw thoughts, not to the summaries.

## How to surface thought summaries in Python

For the `generate_content` flow, the docs show `thinking_config.include_thoughts=True`:

```python
from google import genai
from google.genai import types

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What is the sum of the first 50 prime numbers?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
)

for part in response.candidates[0].content.parts:
    if getattr(part, "thought", False):
        print(part.text)
```

The documentation notes that these are **thought summaries**, not raw internal thoughts. In practice, that means:

- `part.thought` tells you the part is a reasoning summary.
- `part.text` contains the readable summary text.
- Summaries may be absent for very simple prompts or if thinking summaries are disabled.

## How to count reasoning tokens in Python

The official token-counting docs say that the `response.usage_metadata` object includes `thoughts_token_count` for thinking models:

```python
print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
```

That is the clearest Python-side way to report reasoning-token usage. The same usage metadata also includes the usual prompt/output counts, so you can log all three together:

- `prompt_token_count`
- `candidates_token_count`
- `thoughts_token_count`

If you need a fuller accounting view, the token guide also describes `total_token_count`.

## How to preserve reasoning across turns

If you are using multi-turn chat or function calling, the docs say Gemini may return a `thought_signature` / `thoughtSignature` inside response parts. That signature must be passed back exactly as received on the next turn, otherwise reasoning quality can degrade or the request can fail for some Gemini 3 function-calling flows.

The SDK usually handles this automatically if you keep and resend the full response history. You only need to manage signatures manually if you are rewriting history or using REST directly.

## Model-specific configuration

The docs also distinguish how you control the amount of thinking:

- **Gemini 2.5 models** use `thinking_budget`.
- **Gemini 3 models** use `thinking_level`.

So if the goal is to surface reasoning tokens cleanly, the practical recipe is:

1. Enable thinking with the right model setting.
2. Request thought summaries if you want readable reasoning traces.
3. Read `usage_metadata.thoughts_token_count` if you want token accounting.
4. Preserve thought signatures if the conversation continues across turns.

## Recommended use in this repository

For the invoice agent, the safest pattern is to log `thoughts_token_count` to tracing, and optionally store thought summaries only in developer-facing artifacts such as traces or MLflow runs. That gives reviewers observability into reasoning cost and planner behavior without depending on hidden model internals.

## Current repository implementation

This repository now follows that pattern in live Gemini mode:

- The root ADK planner enables `include_thoughts=True`.
- Direct Gemini extraction and categorization calls also enable `include_thoughts=True`.
- SDK-exposed thought summaries, signature presence, and reasoning-token counts are captured into internal JSONL traces, MLflow spans, and `thought_ledger.json`.
- Live tool responses keep an internal `reasoning` envelope so later planner turns can see earlier tool reasoning through normal function-response history.
- Public SSE payloads deliberately redact that reasoning metadata, so the reviewer-facing stream contract stays unchanged.

## Sources

- [Gemini thinking docs](https://ai.google.dev/gemini-api/docs/thinking)
- [Thought signatures docs](https://ai.google.dev/gemini-api/docs/thought-signatures)
- [Gemini token counting docs](https://ai.google.dev/gemini-api/docs/tokens?lang=python)
- [Gemini interactions docs](https://ai.google.dev/gemini-api/docs/interactions)
