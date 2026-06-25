# Resume Agent

Agentic RAG job decision system focused on AI and algorithm internship applications.

This stage contains the minimum runnable scaffold:

- dependency and environment skeleton
- module boundaries and architecture docs
- minimal Pydantic schemas
- minimal parser and CLI entrypoint
- experiment configuration templates
- data and output directories

It intentionally does not include retrieval, scoring, or agent logic yet.

## MVP Scope

The first executable milestone for this project is:

1. parse one JD and one resume into normalized schemas
2. retrieve candidate evidence with keyword and dense search
3. score match quality with multi-stage signals
4. output a structured match report, rewrite hints, and interview prep inputs
5. compare retrieval and pipeline variants under a shared evaluation harness

## Structure

- `docs/`: architecture, schema contracts, and MVP boundaries
- `configs/`: model and experiment templates
- `data/`: raw inputs, processed artifacts, indexes, and outputs
- `src/resume_agent/`: future package layout for schemas, retrieval, scoring, tools, pipelines, and evaluation
- `tests/`: future tests and golden cases

## Next Step

The next implementation step is to add:

1. retrieval interfaces
2. scoring interfaces
3. tool registry contracts
4. pipeline entrypoints
5. richer parsing strategies

## Current Run Path

Once Python is available locally, the first runnable commands will be:

```bash
python main.py parse-jd data/raw/jd/sample_jd.txt
python main.py parse-resume data/raw/resume/sample_resume.txt
python main.py parse-both --jd data/raw/jd/sample_jd.txt --resume data/raw/resume/sample_resume.txt
```
