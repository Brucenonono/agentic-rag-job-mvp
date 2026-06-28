"""Hybrid retrieval backend built from BM25, dense retrieval, and RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass

from resume_agent.retrieval.bm25 import retrieve_with_bm25
from resume_agent.retrieval.contracts import RetrievedEvidence
from resume_agent.retrieval.dense import DenseRetriever
from resume_agent.schemas import Evidence, Requirement


@dataclass
class HybridRetriever:
    """Hybrid retriever that fuses BM25 and dense candidates with RRF."""

    dense_retriever: DenseRetriever
    rrf_k: int = 60

    def search_text(
        self,
        query_text: str,
        evidence_pool: list[Evidence],
        *,
        candidate_k: int,
        requirement_skill_refs: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        query = query_text.strip()
        if not query:
            return []

        requirement = Requirement(
            text=query,
            skill_refs=requirement_skill_refs or [],
        )
        return self.retrieve(requirement, evidence_pool, candidate_k=candidate_k)

    def retrieve(
        self,
        requirement: Requirement,
        evidence_pool: list[Evidence],
        *,
        candidate_k: int,
    ) -> list[RetrievedEvidence]:
        bm25_candidates = retrieve_with_bm25(requirement, evidence_pool)[:candidate_k]
        dense_candidates = self.dense_retriever.retrieve(requirement, top_k=candidate_k)
        return rrf_fuse_candidates(
            [bm25_candidates, dense_candidates],
            rrf_k=self.rrf_k,
        )


def rrf_fuse_candidates(
    candidate_lists: list[list[RetrievedEvidence]],
    *,
    rrf_k: int = 60,
) -> list[RetrievedEvidence]:
    """Fuse ranked candidate lists with Reciprocal Rank Fusion."""

    fused: dict[str, dict[str, object]] = {}

    for ranked_candidates in candidate_lists:
        for rank, candidate in enumerate(ranked_candidates, start=1):
            entry = fused.setdefault(
                candidate.evidence_id,
                {
                    "score": 0.0,
                    "candidate": candidate,
                    "matched_terms": set(candidate.matched_terms),
                },
            )
            entry["score"] = float(entry["score"]) + (1.0 / (rrf_k + rank))
            entry["matched_terms"] = set(entry["matched_terms"]) | set(candidate.matched_terms)

    results: list[RetrievedEvidence] = []
    for entry in fused.values():
        candidate = entry["candidate"]
        fused_score = float(entry["score"])
        matched_terms = sorted(entry["matched_terms"])
        results.append(
            candidate.model_copy(
                update={
                    "score": round(fused_score, 6),
                    "retrieval_score": round(fused_score, 6),
                    "rerank_score": None,
                    "matched_terms": matched_terms,
                }
            )
        )

    return sorted(results, key=lambda item: item.score, reverse=True)
