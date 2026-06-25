# Package Layout

This directory is reserved for future implementation code.

Current subpackages:

- `schemas/`: Pydantic domain models such as JD, Resume, Skill, Evidence, and MatchScore
- `parsers/`: JD and resume parsing interfaces
- `retrieval/`: lexical, dense, hybrid, and rerank components
- `scoring/`: multi-stage scoring and explanation logic
- `tools/`: tool registry and MCP-style adapters
- `pipelines/`: end-to-end task flows
- `evaluation/`: experiment runner and metrics
- `prompts/`: prompt templates for parsing, judging, rewriting, and interview prep
