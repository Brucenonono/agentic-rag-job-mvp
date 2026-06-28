"""Thin rerank layer for requirement-to-evidence matching."""

from __future__ import annotations

from resume_agent.retrieval.contracts import RetrievedEvidence
from resume_agent.schemas import Requirement


def rerank_candidates(
    requirement: Requirement,
    candidates: list[RetrievedEvidence],
    top_k: int,
) -> list[RetrievedEvidence]:
    """Rerank retrieved evidence with a tighter relevance heuristic.

    This is intentionally a lightweight stand-in for a future cross-encoder
    reranker. It keeps the interface stable while improving ranking quality
    over the initial coarse retrieval stage.
    """

    reranked = sorted(
        (_apply_rerank_score(requirement, candidate) for candidate in candidates),
        key=lambda item: item.score,
        reverse=True,
    )
    return reranked[:top_k]


def _apply_rerank_score(requirement: Requirement, candidate: RetrievedEvidence) -> RetrievedEvidence:
    requirement_terms = set(candidate.matched_terms)
    evidence_terms = _tokenize(candidate.evidence.text)
    requirement_all_terms = _tokenize(requirement.text)

    term_recall = 0.0
    if requirement_all_terms:
        term_recall = len(requirement_terms & requirement_all_terms) / len(requirement_all_terms)

    term_precision = 0.0
    if evidence_terms:
        term_precision = len(requirement_terms & evidence_terms) / len(evidence_terms)

    skill_overlap = 0.0
    if requirement.skill_refs:
        shared_skill_refs = set(requirement.skill_refs) & set(candidate.evidence.skill_refs)
        skill_overlap = len(shared_skill_refs) / len(requirement.skill_refs)
    else:
        shared_skill_refs = set()

    phrase_bonus = 0.0
    evidence_text = candidate.evidence.text.lower()
    for phrase in requirement.skill_refs:
        if phrase and phrase in evidence_text:
            phrase_bonus = 1.0
            break

    rerank_score = (
        0.45 * term_recall
        + 0.20 * term_precision
        + 0.25 * skill_overlap
        + 0.10 * phrase_bonus
    )
    matched_terms = sorted(set(candidate.matched_terms) | shared_skill_refs)

    return candidate.model_copy(
        update={
            "score": round(rerank_score, 4),
            "rerank_score": round(rerank_score, 4),
            "matched_terms": matched_terms,
        }
    )


def _tokenize(text: str) -> set[str]:
    tokens: list[str] = []
    current: list[str] = []

    for char in text.lower():
        if char.isalnum() or char == "+":
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return set(tokens)
