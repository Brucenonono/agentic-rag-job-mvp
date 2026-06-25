# Architecture

## Goal

Build an Agentic RAG system for AI and algorithm internship applications that can:

1. parse JD and resume text into normalized schemas
2. retrieve supporting evidence with hybrid retrieval
3. score match quality with explainable signals
4. orchestrate reusable tools in an end-to-end pipeline
5. evaluate retrieval, matching, and generation quality across baselines

## Why This Is the Minimum Useful Framework

### 1. `schemas`

Why it exists:

- JD, Resume, Skill, Evidence, and MatchScore are the shared language of the whole system.
- Without a normalized schema layer, retrieval, scoring, rewrite, and evaluation would all invent their own field names and become hard to compare.
- Pydantic is the right base because the system begins from messy text but needs strict downstream contracts.

### 2. `parsers`

Why it exists:

- JD parsing and resume parsing are separate concerns from retrieval and scoring.
- This keeps the system provider-agnostic: a parser can later be LLM-based, rule-based, or hybrid without changing the rest of the pipeline.

### 3. `retrieval`

Why it exists:

- BM25 covers explicit keywords, tool names, and phrase overlap.
- BGE embeddings cover semantic similarity and hidden requirement alignment.
- FAISS gives an efficient vector index for dense retrieval.
- Cross-encoder reranking improves top candidate evidence quality before scoring.

### 4. `scoring`

Why it exists:

- One score is not enough for recruiting decisions.
- We need a staged scoring layer so keyword coverage, semantic similarity, evidence relevance, and judge-style reasoning stay separable and explainable.

### 5. `tools`

Why it exists:

- JD parsing, retrieval, scoring, rewriting, and interview preparation should be reusable capabilities, not hardcoded function chains.
- A tool layer gives us a stable interface that can later be exposed through an MCP-compatible server boundary.
- For the MVP, we keep the abstraction but avoid committing to a heavy agent framework too early.

### 6. `pipelines`

Why it exists:

- The business workflow is predictable: parse -> retrieve -> score -> rewrite -> interview prep.
- A pipeline layer lets us run end-to-end tasks with explicit state transitions while keeping each tool independently testable.

### 7. `evaluation`

Why it exists:

- Without a shared experiment harness, the project becomes a demo instead of a system.
- The evaluation layer lets us compare keyword-only, dense, hybrid, rerank, and full-agent variants under the same metrics and datasets.

## Chosen Frameworks

### Python 3.11+

- Strong ecosystem support for Pydantic, FAISS bindings, sentence-transformers, and API tooling.
- Easy path from scripts to services to evaluation harnesses.

### Pydantic v2

- Best fit for strongly typed schemas over partially structured LLM outputs.
- Useful for validation, serialization, and contract enforcement between modules.

### FastAPI

- Minimal API surface for local service endpoints and tool invocation wrappers.
- Pairs naturally with Pydantic models for request and response contracts.

### Typer

- Fastest way to expose a local CLI for ingestion, indexing, matching, and evaluation commands.
- Keeps the MVP usable before we decide whether a UI is necessary.

### `rank-bm25` + BGE via `sentence-transformers` + `faiss-cpu`

- Covers lexical retrieval, semantic retrieval, and vector indexing with a simple local stack.
- This is enough to validate the hybrid retrieval hypothesis before considering heavier infrastructure.

### Optional MCP SDK

- The project needs MCP-style tool boundaries, but not necessarily a full distributed tool network on day one.
- We define the internal tool shape first and keep an optional dependency path for a true MCP adapter.

## Intentionally Deferred

These are deliberately not included in the first framework:

- no frontend application
- no database
- no workflow scheduler
- no heavy agent orchestration framework
- no online serving infrastructure beyond a local API shell

Reason:

- The core risk is retrieval, matching, and evidence quality, not UI or distributed systems.
- A lighter skeleton makes the first milestone easier to debug and evaluate.
