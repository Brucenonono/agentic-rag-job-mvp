# Resume Agent

Agentic RAG job decision system focused on AI and algorithm internship applications.

This stage contains the minimum runnable scaffold:

- dependency and environment skeleton
- module boundaries and architecture docs
- minimal Pydantic schemas
- minimal parser and CLI entrypoint
- retrieval backends with heuristic rerank stage
- experiment configuration templates
- data and output directories

It intentionally does not include scoring or agent logic yet.

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

1. scoring interfaces
2. tool registry contracts
3. pipeline entrypoints
4. richer parsing strategies
5. evaluation and stronger rerankers

## Current Run Path

Once Python is available locally, the first runnable commands will be:

```bash
py main.py parse-jd data/raw/jd/sample_jd.txt
py main.py parse-resume data/raw/resume/sample_resume.txt
py main.py parse-both --jd data/raw/jd/sample_jd.txt --resume data/raw/resume/sample_resume.txt
py main.py retrieve --jd data/raw/jd/sample_jd.txt --resume data/raw/resume/sample_resume.txt --method bm25 --candidate-k 9
py main.py retrieve --jd data/raw/jd/sample_jd.txt --resume data/raw/resume/sample_resume.txt --method dense --dense-model-name BAAI/bge-small-zh-v1.5 --dense-index-dir data/indexes/faiss/sample
py main.py retrieve --jd data/raw/jd/sample_jd.txt --resume data/raw/resume/sample_resume.txt --method hybrid --dense-model-name BAAI/bge-small-zh-v1.5 --dense-index-dir data/indexes/faiss/sample
```

## Retrieval Notes

- Supported coarse retrieval backends: `lexical_overlap`, `bm25`, `dense`, `hybrid`
- `dense` uses sentence-transformers embeddings plus FAISS `IndexFlatIP`
- `hybrid` fuses BM25 and dense candidates with Reciprocal Rank Fusion
- The current rerank stage is `heuristic_rerank`
- The current rerank stage is not a Cross-Encoder and is intentionally named that way
