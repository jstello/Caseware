# Codex Raw Session Logs

This folder contains the raw Codex Desktop JSONL session logs that correspond to the development and verification work for this repository.

Why this location:

- The assignment explicitly asks for a `/transcripts/` folder containing raw AI coding tool interaction logs.
- Keeping the raw logs under `transcripts/raw/codex/` keeps them close to the existing human-readable summary at `transcripts/codex-session-2026-03-28.md`.
- The dated subfolders preserve the original session provenance and filenames.

Selection policy:

- Included: milestone sessions that directly map to repository implementation, ADK Web support, MLflow tracing, live-mode compliance, and follow-up requirement analysis.
- Excluded: clearly redundant fork-only child sessions, user-level tool installation work, and small administrative sessions that do not add much reviewer value.
- Source set size across both days was about 38 MB.
- Included subset size is about 20.5 MB, which keeps the repository reasonable while still providing raw coverage of the major development threads.

Included files:

| File | Topic | Size |
| --- | --- | ---: |
| `2026-03-28/rollout-2026-03-28T14-40-14-019d35f6-1b88-7642-8edd-2602a86aa9da.jsonl` | Implementation kickoff from the repository prompt and assignment/ADR grounding | 0.85 MB |
| `2026-03-28/rollout-2026-03-28T21-09-17-019d375a-49cf-7483-aec4-2a47040e1661.jsonl` | ADK Web compatibility investigation and implementation thread | 3.89 MB |
| `2026-03-28/rollout-2026-03-28T22-07-27-019d378f-8bd6-71b2-ad1b-309dfeee509c.jsonl` | End-to-end MLflow experiment and curl-driven tracing work | 4.98 MB |
| `2026-03-28/rollout-2026-03-28T22-24-55-019d379f-878a-7261-b585-deb0260633d8.jsonl` | Real API-based LLM execution preparation, prompt review, and compliance review | 2.01 MB |
| `2026-03-29/rollout-2026-03-29T10-54-47-019d3a4e-0e7a-76e0-b924-fed73afd61e9.jsonl` | Gemini thought-token research and report generation | 1.28 MB |
| `2026-03-29/rollout-2026-03-29T11-06-53-019d3a59-2276-7841-9f05-6689377c277f.jsonl` | MLflow version-tracking research thread | 1.07 MB |
| `2026-03-29/rollout-2026-03-29T14-13-07-019d3b03-a3fd-7451-ac16-0f1736625aad.jsonl` | Continued ADK Web verification and wiring validation | 5.00 MB |
| `2026-03-29/rollout-2026-03-29T14-35-32-019d3b18-2971-7a50-baf5-89d5cf996e76.jsonl` | ADK Web MLflow trace visibility debugging | 0.68 MB |
| `2026-03-29/rollout-2026-03-29T17-01-16-019d3b9d-95c0-7db3-9b52-85a9f836272a.jsonl` | Requirement analysis for the curl/live-mode testing story | 0.78 MB |

Notes for reviewers:

- These are copied verbatim from the local Codex session store under `~/.codex/sessions/...`.
- The existing markdown file `transcripts/codex-session-2026-03-28.md` is a curated summary, while the JSONL files in this folder are the raw logs.
- The logs show tool calls, intermediate commentary, and repository file references for the work that landed in this repo on March 28-29, 2026.
