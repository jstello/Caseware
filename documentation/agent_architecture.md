# Agent Architecture and Data Flow

This document details the layout of the LLM-as-Planner agent architecture and tool availability mapping for the local invoice-processing pipeline.

## Architectural Diagram

```mermaid
graph TD
    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef tool fill:#bbf,stroke:#333,stroke-width:1px;
    classDef endpoint fill:#bfb,stroke:#333,stroke-width:1px;
    classDef stream fill:#fbb,stroke:#333,stroke-width:1px;
    classDef source fill:#eee,stroke:#333,stroke-width:1px;

    Client((Client)):::endpoint
    API["API Endpoint<br/>POST /runs/stream"]:::endpoint
    Planner{"LLM-as-Planner<br/>(invoice-agent)"}:::llm
    SSE[/"Server-Sent Events<br/>(SSE Stream)"/]:::stream

    subgraph "Constrained Tool Registry Sandbox"
        T1["🛠️ load_images(input)<br/>-> [image_refs]"]:::tool
        T2["🛠️ extract_invoice_fields(image_ref)<br/>-> structured_fields"]:::tool
        T3["🛠️ normalize_invoice(fields)<br/>-> normalized_fields"]:::tool
        T4["🛠️ categorize_invoice(fields, categories)<br/>-> {category, conf, notes}"]:::tool
        T5["🛠️ aggregate_invoices(invoices)<br/>-> totals"]:::tool
        T6["🛠️ generate_report(aggregates, invoices, issues)<br/>-> final_output"]:::tool
    end

    FileSystem[("File System<br/>(Invoice Images)")]:::source

    %% Request flow
    Client -- "Payload: Folder Path / Images<br/>(+ Optional Prompt)" --> API
    API -- "Starts processing run" --> Planner

    %% Tool execution logic & data flow
    Planner == "1. decides to fetch" ==> T1
    T1 -. "Loads data" .-> FileSystem
    T1 -. "Returns refs" .-> Planner

    Planner == "2. decides to extract (per ref)" ==> T2
    T2 -. "Returns raw fields" .-> Planner

    Planner == "3. decides to format" ==> T3
    T3 -. "Returns normalized fields" .-> Planner

    Planner == "4. decides to classify" ==> T4
    T4 -. "Returns {category, notes}" .-> Planner

    Planner == "5. decides to sum (all invoices)" ==> T5
    T5 -. "Returns totals" .-> Planner

    Planner == "6. decides to compile" ==> T6
    T6 -. "Returns final summary" .-> Planner

    %% Streaming Output Data Flow
    API -. "Initializes TCP stream" .-> SSE
    Planner -. "Yields trajectory runtime events<br/>(tool_call, tool_result, run_started, progress)" .-> SSE
    SSE -. "Streams asynchronous chunks" .-> Client
```

## Component Details

*   **LLM-as-Planner**: The orchestrating model (`invoice-agent`) dynamically evaluates intermediate tool results to determine the next action. It is NOT a static DAG pipeline, so if an extraction step fails or misses a field, the planner can voluntarily attempt a retry before progressing.
*   **Tool Registry Constraints**: The agent itself does not possess direct file I/O capabilities. It strictly acts through defined Python tools ensuring predictability and sandboxed operations.
*   **Data Flow & Real-Time Feedback**: Instead of generating an upfront batch response, the backend synchronously streams agent trajectory logs via `Server-Sent Events` (SSE). This includes every active step ranging from `run_started` to granular `tool_result` items, yielding the final payloads on stream completion.
