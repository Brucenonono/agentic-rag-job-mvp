# Retrieval

Minimal retrieval layer for the first runnable matching baseline.

Current scope:

- coarse lexical or BM25 retrieval over `Resume.evidence_pool`
- dense retrieval with sentence-transformers and FAISS
- hybrid retrieval with BM25 + dense RRF fusion
- thin rerank stage over retrieved candidates
- stable retrieval result contracts that expose both coarse and rerank scores

Current entrypoints:

- `retrieve_for_requirement()`
- `retrieve_for_jd()`

Design note:

- The current rerank stage is `heuristic_rerank`, not a Cross-Encoder.
- The interface is shaped so a future hybrid retriever or cross-encoder reranker can slot in without changing callers.
