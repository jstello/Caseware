# Repository Instructions

This repository contains the take-home assignment source of truth in `documentation/assignment.md`.

If implementation ideas, assumptions, or prior summaries conflict with `documentation/assignment.md`, the assignment markdown wins.

Secret-handling rule:

- Treat `.env`, `.env.*`, and any local secret-bearing environment files as off-limits.
- Do not read, print, grep, cat, source, parse, summarize, or inspect those files.
- Do not print or request secret values such as `GOOGLE_API_KEY`.
- If configuration details are needed, ask the user for the non-secret shape, expected variable names, or a redacted example instead of opening the secret file.

Testing and verification policy:

- All testing, validation, and live runtime verification should register MLflow traces in experiments and runs.
- Prefer a single shared MLflow experiment for now unless the user asks for a different layout.
- Keep generated MLflow runs, artifacts, and local trace outputs out of the repository unless the user explicitly asks to commit them.

Subagent policy for implementation tasks:

- Do not use `gpt-5.4-mini` subagents for implementation work in this repository.
- Always use `gpt-5.4` subagents with extra high reasoning effort for implementation tasks.
- When spawning subagents on the fly for repository work, default to `gpt-5.4` with extra high reasoning effort unless the user explicitly overrides that policy.
- Treat the repo-local `.codex/agents/*.toml` agents and any ad hoc spawned implementation or verification agents as subject to the same model policy.

Repo-local subagent roster:

- `assignment_guardian`: strict requirement-by-requirement compliance reviewer; spawn when you need a `Met` / `Partially met` / `Not met` pass against `documentation/assignment.md` and the ADR, or when you need missing evidence called out clearly.
- `agent_sdk_expert`: ADK, planner-loop, SSE-mapping, and agent-pattern reviewer; spawn when you need architectural feedback on orchestrator shape, tool boundaries, or event mapping.
- `trace_fixture_auditor`: evidence-quality reviewer; spawn when you need trace quality, fixture coverage, MLflow observability, transcript quality, or sample-run trustworthiness reviewed.
- `backend_builder`: the default write-capable implementation agent; spawn when bounded code, helper-agent, or workflow-doc changes are needed after review agents produce patch-ready recommendations.

Subagent dispatch guidance:

- Keep the main thread as the conductor.
- Prefer the read-only reviewers first for analysis and verification, then hand concrete patch-ready actions to `backend_builder`.
- Prefer only one write-capable implementation agent at a time unless the user explicitly asks for parallel write work with disjoint scopes.

Required workflow for every implementation task:

1. Read `documentation/assignment.md` before making changes.
2. Extract the explicit requirements, constraints, and deliverables into working notes or a mental checklist.
3. Implement only what the assignment requires unless the user explicitly asks for additional scope.
4. Before considering the work complete, reread `documentation/assignment.md`.
5. Perform a strict compliance pass against every explicit requirement in the markdown file.
6. If any requirement is unmet or ambiguous, either fix it or clearly call it out.

Do not rely on memory alone, especially after a long session or once the context window has shifted.

Final responses for implementation work should include:

- A short summary of what was implemented
- A compliance check against the assignment, using `Met`, `Partially met`, or `Not met` for each relevant requirement group
- Any assumptions, open questions, or known gaps

Keep the implementation faithful to the assignment, avoid unnecessary scope expansion, and prefer changes that make compliance easy to verify.
