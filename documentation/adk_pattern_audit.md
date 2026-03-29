# ADK Pattern Audit

## What I Compared

I compared the current invoice runtime against the local Google ADK samples in `/Users/juan_tello/Repos/Google/adk-samples/python`, with emphasis on:

- `SequentialAgent` examples such as `llm-auditor` and other pipeline-oriented samples
- `ParallelAgent` examples such as `parallel_task_decomposition_execution`
- `LoopAgent` examples such as `image-scoring`
- The coordinator-plus-specialist layouts used in larger multi-agent samples

## What The Samples Suggest

- `SequentialAgent` is strongest when the order is fixed and each stage is intentionally deterministic.
- `ParallelAgent` only pays off when branches are independent and do not compete for shared mutable state or reviewer-facing ordering.
- `LoopAgent` is a good fit for bounded refinement and retry behavior where a checker can stop the cycle.
- Larger ADK examples keep the coordinator logic explicit and keep sub-agents narrowly scoped.

## Recommendation For This Invoice Workflow

This invoice workflow should stay **planner-centered** rather than becoming a rigid DAG.

- The root runtime should remain a single planner-driven `LlmAgent` so the model chooses the next tool from intermediate results.
- `SequentialAgent` is reasonable only as a future readability aid for coarse phases, not as the primary control mechanism for the whole run.
- `LoopAgent` is the best conceptual fit for bounded retry behavior, especially extraction retry, but the top-level control still needs to remain planner-led to satisfy the assignment.
- `ParallelAgent` is not the right default for v1 because invoice records share run state, trace ordering matters, and the assignment prioritizes reviewer readability over fan-out throughput.

## Bottom Line

For this repository, the strongest ADK pattern is a **single planner-driven root agent with explicit tool boundaries, YAML-backed configuration, and bounded retry behavior**, not a fully fixed workflow graph.
