# ADR-001: Adopt Google ADK with Mock-First Execution and a Custom SSE Contract

- Status: Accepted
- Date: 2026-03-28
- Related source of truth: `documentation/assignment.md`

## Context

The take-home assignment requires a local invoice-processing agent exposed through `POST /runs/stream`, with a true LLM-as-planner loop, a constrained 4-6 tool registry, Server-Sent Events for progress and results, and reviewable execution traces.

The assignment is explicitly time-boxed to 2-3 hours and recommends mock mode. That creates a tension we want to resolve early:

1. We need enough realism to prove this is a genuine agent system rather than a scripted demo.
2. We need enough determinism to make development, testing, and reviewer validation credible within the timebox.
3. We need a transport and tracing strategy that satisfies the assignment without introducing avoidable framework complexity.

We also want the decision to be specific enough that future subagent definitions can be derived from it without reopening stack-level debates.

## Decision

We will implement the challenge using **Python + FastAPI + Google ADK for Python**, with a **custom SSE adapter** that emits the exact event contract required by the assignment.

We will use a **dual-mode execution model**:

- **Mock mode is the default** for development, testing, and documented examples.
- **Live mode is optional**, using **Gemini through Google ADK**, enabled only when credentials are provided outside the repository.

We will keep the **planner loop, orchestration, tool registry, state transitions, and SSE streaming logic real in both modes**. In mock mode, we will replace provider-facing and tool-edge behavior with deterministic stubs so that runs are repeatable.

We will validate the streaming contract with **both schema-level checks and at least one golden end-to-end transcript**.

We will keep tracing simple for v1:

- **Structured JSON traces/logs are required**
- **OpenTelemetry is optional if it stays light**
- **MLflow is explicitly not on the critical path**

We will not use **AG-UI** in v1 unless the Google ADK event model proves too awkward to map cleanly to the assignment's required SSE contract.

## Rationale

### Why Google ADK for Python

Google ADK gives us a Python-first agent runtime that can support a real planner loop and streaming event emission. It aligns with the stack direction we already explored and with the optional Google-backed live mode we may want for demonstration.

The important caveat is that ADK is **not** a drop-in match for the assignment wire protocol. Its runtime events still need to be translated into the exact SSE event names required by the assignment:

- `run_started`
- `progress`
- `tool_call`
- `tool_result`
- `invoice_result`
- `final_result`
- `error`

That translation is acceptable and preferable to introducing a second event standard just to satisfy a reviewer-facing protocol.

### Why mock-first, but not fake

The assignment explicitly allows mocked OCR/VLM/LLM usage and recommends mock mode. We are taking that recommendation seriously, but we are not planning a transcript replay system.

In mock mode, the system must still prove:

- the planner loop exists
- the agent selects tools through a constrained registry
- the server emits the proper SSE sequence
- the final output is computed from tool results and agent decisions
- the trace explains what happened

That means the architecture stays real while external uncertainty is removed.

### Why synthetic invoice fixtures in the repository

The repository currently contains no invoice fixtures. To make the submission reproducible, we will check in a curated synthetic invoice set. This is better than relying on ad hoc local files because it gives us deterministic examples for tests, transcripts, and reviewer walkthroughs.

We will treat **invoices** as the core fixture type because that is what the assignment requires. We may include one adversarial or malformed input only if it strengthens the "issues and assumptions" story, but we will not broaden the scope into a receipt-processing project.

### Why schema checks plus a golden transcript

Schema-only checks are not enough to demonstrate that the streamed trajectory feels coherent, and a transcript alone is too brittle unless the runtime is deterministic. Because we are choosing a mock-first deterministic mode, we can afford both:

- schema and ordering checks for each SSE event
- one saved end-to-end transcript representing a canonical run

This directly supports the deliverables in the assignment, especially the sample run trace and the requirement for observable, reviewable execution.

### Why not AG-UI or MLflow first

AG-UI is a credible fallback for event transport, but the assignment already defines the stream contract. Adding AG-UI in v1 would increase surface area without clearly improving our reviewer story unless ADK event mapping becomes painful in practice.

MLflow is also credible, especially for richer tracing and observability, but it is not required to satisfy the assignment. Structured JSON traces are faster to implement, easier to inspect locally, and more compatible with the timebox. MLflow remains a future enhancement rather than a foundational dependency.

## Implementation Consequences

The implementation must honor these constraints:

1. `POST /runs/stream` is the only required runtime surface and must immediately open an SSE response.
2. The server must own the mapping from internal agent/runtime events to the assignment-defined SSE events.
3. Mock mode and live mode must share the same planner and streaming pipeline. The mode switch may change providers and tool-edge behavior, but not the overall architecture.
4. Automated tests must run without secrets and must not depend on a real model provider.
5. Live mode must assume credentials are supplied externally; the repository must remain usable without reading any local secret files.
6. Tracing must be written in a way that a reviewer can follow step boundaries, tool calls, summarized inputs and outputs, planner intent, timings, and errors.

## Fixture and Test Strategy

We will create a small but deliberately varied synthetic fixture set, targeting at least six invoice cases:

1. Clean happy-path invoice
2. Invoice with missing or partial fields
3. Ambiguous category assignment
4. Multi-line or nontrivial totals structure
5. Low-confidence or noisy extraction case
6. Odd or conflicting details that should surface issues or assumptions

Validation will include:

- unit tests for event shaping and tool outputs where appropriate
- contract tests for the SSE event schema and ordering
- at least one golden transcript from a deterministic mock run

## Alternatives Considered

### AG-UI as the primary streaming layer

Rejected for v1. It is a reasonable fallback if event mapping from ADK proves awkward, but it introduces another protocol layer when the assignment already gives us the public contract we must emit.

### MLflow as a required tracing dependency

Rejected for v1. It is useful, but heavier than necessary for a short challenge whose tracing needs can be satisfied with structured logs and a saved run trace.

### Mock-only architecture with no live path

Rejected. It would simplify implementation, but it weakens the architectural story and makes the design feel more like a simulation than a real system.

### Live-first architecture with real provider calls as the default

Rejected. It increases volatility, secret handling complexity, and debugging cost, while reducing determinism for the exact reviewer artifacts we need to produce.

## Implications for Subagent Definitions

Any subagent definitions created after this ADR should assume:

- the runtime center of gravity is **Google ADK + FastAPI**
- the public transport contract is the assignment-defined SSE schema, not a framework-native event schema
- mock mode is first-class and must remain truthful to the real architecture
- fixture quality and trace readability are core success criteria, not secondary polish
- MLflow and AG-UI are optional exploration lanes, not primary implementation ownership

This ADR is the baseline for defining the coordinator, SDK-focused specialist, backend implementation role, trace/contract verifier, and documentation or submission roles.
