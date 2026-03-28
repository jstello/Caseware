# Repository Instructions

This repository contains the take-home assignment source of truth in `documentation/assignment.md`.

If implementation ideas, assumptions, or prior summaries conflict with `documentation/assignment.md`, the assignment markdown wins.

Secret-handling rule:

- Treat `.env`, `.env.*`, and any local secret-bearing environment files as off-limits.
- Do not read, print, grep, cat, source, parse, summarize, or inspect those files.
- Do not print or request secret values such as `GOOGLE_API_KEY`.
- If configuration details are needed, ask the user for the non-secret shape, expected variable names, or a redacted example instead of opening the secret file.

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
