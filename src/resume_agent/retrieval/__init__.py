"""Minimal retrieval exports."""

from .bm25 import retrieve_with_bm25
from .contracts import RequirementRetrievalResult, RetrievedEvidence, RetrievalReport
from .dense import DenseRetriever, EmbeddingEncoder, SentenceTransformerBGEEncoder
from .hybrid import HybridRetriever, rrf_fuse_candidates
from .rerank import rerank_candidates
from .simple import RetrievalMethod, retrieve_for_jd, retrieve_for_requirement

__all__ = [
    "DenseRetriever",
    "EmbeddingEncoder",
    "HybridRetriever",
    "RequirementRetrievalResult",
    "RetrievalMethod",
    "RetrievedEvidence",
    "RetrievalReport",
    "SentenceTransformerBGEEncoder",
    "retrieve_with_bm25",
    "rerank_candidates",
    "rrf_fuse_candidates",
    "retrieve_for_jd",
    "retrieve_for_requirement",
]
