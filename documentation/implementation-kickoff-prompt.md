# Implementation Kickoff Prompt

Use the following prompt to kick off implementation in this repository:

```text
You are implementing the invoice-agent take-home challenge in this repository.

Before making changes, read these files in this order:
1. AGENTS.md
2. documentation/assignment.md
3. documentation/adr/ADR-001-adopt-google-adk-mock-first-custom-sse-contract.md
4. .codex/config.toml
5. .codex/agents/*.toml

Treat `documentation/assignment.md` as the source of truth. If anything conflicts with prior assumptions or summaries, the assignment wins.

Architectural decisions that are already locked:
- Use Python + FastAPI + Google ADK for Python
- Use a server-owned custom SSE adapter for the assignment-defined event contract
- Keep mock mode as the default development and test path
- Keep live mode optional via Gemini through ADK
- Keep one architecture across mock and live mode
- Keep the existing MLflow-backed tracing and verification path unless you discover a concrete blocker and explain why

Use the repo-local subagents intentionally:
- `assignment_guardian` for strict requirement-by-requirement compliance checks, missing evidence review, and helper-agent prompt upgrades
- `agent_sdk_expert` for Google ADK behavior, planner-loop integrity, SSE mapping risks, and agent-pattern review
- `trace_fixture_auditor` for fixture coverage, golden transcripts, SSE traces, MLflow observability quality, and reviewer-trust gaps
- `backend_builder` for bounded implementation tasks after the review agents produce patch-ready recommendations

Keep the main thread as the conductor. Use subagents for bounded specialized work, especially read-heavy review and verification. Prefer only one write-capable implementation agent at a time. Any ad hoc spawned implementation or verification agent for this repository must use `gpt-5.4` with extra high reasoning effort unless the user explicitly changes that policy.

Default reflection workflow for helper-agent and workflow-guidance changes:
1. Run `assignment_guardian` first for compliance framing
2. Run `agent_sdk_expert` for ADK, planner-loop, and orchestrator review
3. Run `trace_fixture_auditor` for evidence, traces, fixtures, and MLflow review
4. Hand the resulting patch-ready actions to `backend_builder`
5. Re-run `assignment_guardian` and `trace_fixture_auditor` for final review

Implementation goals:
- Build `POST /runs/stream`
- Accept either multipart invoice image input or JSON with a local folder path
- Implement a true LLM-as-planner loop, not a fixed DAG
- Constrain invoice access through a 4-6 tool registry
- Stream the required SSE events:
  - `run_started`
  - `progress`
  - `tool_call`
  - `tool_result`
  - `invoice_result`
  - `final_result`
  - `error`
- Produce the required run-level and per-invoice output
- Produce reviewable traces
- Keep the implementation local and mock-first

Validation goals:
- Add a deterministic mock-mode path that is suitable for tests
- Create synthetic invoice fixtures in the repo
- Add SSE schema and ordering validation
- Add at least one golden transcript / sample run trace
- Make it possible to review compliance quickly

Execution instructions:
1. Start with a brief implementation plan based on the assignment and ADR.
2. Identify the smallest vertical slice that proves the architecture end to end.
3. Implement incrementally, validating as you go.
4. Re-read `documentation/assignment.md` before considering the work complete.
5. Perform a strict compliance pass before finishing.

Important constraints:
- Do not read `.env`, `.env.*`, or any secret-bearing local env file.
- Do not ask for secret values such as `GOOGLE_API_KEY`.
- If config details are needed, ask for redacted variable names or expected shape only.
- Avoid unnecessary scope expansion.
- Prefer changes that make assignment compliance easy to verify.

When you finish, provide:
- a short implementation summary
- a compliance check using `Met`, `Partially met`, or `Not met`
- assumptions, open questions, or known gaps

Begin by reading the source-of-truth files, summarizing the implementation checklist, and then starting the first vertical slice.
```

## Notes

This prompt is designed to work with the repo-local subagent setup under `.codex/agents/` and the stack decision recorded in `documentation/adr/ADR-001-adopt-google-adk-mock-first-custom-sse-contract.md`.

It is intentionally specific about the locked decisions so a future implementation session does not re-open the stack choice, mock-first strategy, MLflow-backed verification path, or the custom SSE contract unless a real blocker is discovered.
