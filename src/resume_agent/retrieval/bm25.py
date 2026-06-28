"""BM25 coarse retrieval backend."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from resume_agent.retrieval.contracts import RetrievedEvidence
from resume_agent.retrieval.text import token_set, tokenize_text
from resume_agent.schemas import Evidence, Requirement


def retrieve_with_bm25(requirement: Requirement, evidence_pool: list[Evidence]) -> list[RetrievedEvidence]:
    """Retrieve evidence candidates with BM25 over evidence text."""

    if not evidence_pool:
        return []

    corpus = [tokenize_text(evidence.text) for evidence in evidence_pool]
    query = tokenize_text(requirement.text)
    if not query:
        return []

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    candidates = [
        _build_candidate(requirement, evidence, float(score))
        for evidence, score in zip(evidence_pool, scores, strict=False)
    ]
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _build_candidate(
    requirement: Requirement,
    evidence: Evidence,
    retrieval_score: float,
) -> RetrievedEvidence:
    matched_terms = sorted(token_set(requirement.text) & token_set(evidence.text))
    if requirement.skill_refs:
        matched_terms = sorted(set(matched_terms) | (set(requirement.skill_refs) & set(evidence.skill_refs)))

    score = max(retrieval_score, 0.0)
    return RetrievedEvidence(
        evidence_id=evidence.id,
        score=round(score, 4),
        retrieval_score=round(score, 4),
        rerank_score=None,
        matched_terms=matched_terms,
        evidence=evidence,
    )
