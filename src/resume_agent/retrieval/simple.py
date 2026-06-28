"""Retrieval pipeline with selectable coarse recall backends and rerank."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from resume_agent.retrieval.bm25 import retrieve_with_bm25
from resume_agent.retrieval.contracts import (
    RequirementRetrievalResult,
    RetrievedEvidence,
    RetrievalReport,
)
from resume_agent.retrieval.dense import (
    DenseRetriever,
    EmbeddingEncoder,
    SentenceTransformerBGEEncoder,
    load_or_build_dense_retriever,
)
from resume_agent.retrieval.hybrid import HybridRetriever
from resume_agent.retrieval.rerank import rerank_candidates
from resume_agent.retrieval.text import token_set
from resume_agent.schemas import Evidence, JD, Requirement, Resume

RetrievalMethod = Literal["lexical_overlap", "bm25", "dense", "hybrid"]


def retrieve_for_jd(
    jd: JD,
    resume: Resume,
    top_k: int = 3,
    candidate_k: int | None = None,
    use_rerank: bool = True,
    retrieval_method: RetrievalMethod = "lexical_overlap",
    dense_model_name: str | None = None,
    dense_index_dir: str | Path | None = None,
    dense_encoder: EmbeddingEncoder | None = None,
    hybrid_rrf_k: int = 60,
) -> RetrievalReport:
    """Retrieve evidence for every requirement in a JD."""

    candidate_limit = _resolve_candidate_k(top_k, candidate_k)
    dense_retriever: DenseRetriever | None = None
    hybrid_retriever: HybridRetriever | None = None

    if retrieval_method in {"dense", "hybrid"}:
        dense_retriever = _prepare_dense_retriever(
            resume.evidence_pool,
            dense_model_name=dense_model_name,
            dense_index_dir=dense_index_dir,
            dense_encoder=dense_encoder,
        )
        if retrieval_method == "hybrid":
            hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, rrf_k=hybrid_rrf_k)

    results = [
        retrieve_for_requirement(
            requirement,
            resume.evidence_pool,
            top_k=top_k,
            candidate_k=candidate_limit,
            requirement_index=index,
            use_rerank=use_rerank,
            retrieval_method=retrieval_method,
            dense_model_name=dense_model_name,
            dense_index_dir=dense_index_dir,
            dense_encoder=dense_encoder,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            hybrid_rrf_k=hybrid_rrf_k,
        )
        for index, requirement in enumerate(jd.requirements)
    ]
    return RetrievalReport(
        jd_title=jd.title,
        candidate_name=resume.candidate_name,
        top_k=top_k,
        candidate_k=candidate_limit,
        retrieval_method=retrieval_method,
        rerank_method="heuristic_rerank" if use_rerank else None,
        results=results,
    )


def retrieve_for_requirement(
    requirement: Requirement,
    evidence_pool: list[Evidence],
    top_k: int = 3,
    candidate_k: int | None = None,
    requirement_index: int = 0,
    use_rerank: bool = True,
    retrieval_method: RetrievalMethod = "lexical_overlap",
    dense_model_name: str | None = None,
    dense_index_dir: str | Path | None = None,
    dense_encoder: EmbeddingEncoder | None = None,
    dense_retriever: DenseRetriever | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    hybrid_rrf_k: int = 60,
) -> RequirementRetrievalResult:
    """Retrieve candidates and optionally rerank them."""

    candidate_limit = _resolve_candidate_k(top_k, candidate_k)
    resolved_dense_retriever = dense_retriever
    resolved_hybrid_retriever = hybrid_retriever

    if retrieval_method in {"dense", "hybrid"} and resolved_dense_retriever is None:
        resolved_dense_retriever = _prepare_dense_retriever(
            evidence_pool,
            dense_model_name=dense_model_name,
            dense_index_dir=dense_index_dir,
            dense_encoder=dense_encoder,
        )
    if retrieval_method == "hybrid" and resolved_hybrid_retriever is None:
        if resolved_dense_retriever is None:
            raise ValueError("Hybrid retrieval requires a dense retriever")
        resolved_hybrid_retriever = HybridRetriever(
            dense_retriever=resolved_dense_retriever,
            rrf_k=hybrid_rrf_k,
        )

    ranked = sorted(
        _retrieve_candidates(
            requirement,
            evidence_pool,
            retrieval_method,
            candidate_limit=candidate_limit,
            dense_retriever=resolved_dense_retriever,
            hybrid_retriever=resolved_hybrid_retriever,
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    candidates = _select_candidates(ranked, retrieval_method, candidate_limit)
    final_ranked = rerank_candidates(requirement, candidates, top_k=top_k) if use_rerank else candidates[:top_k]

    return RequirementRetrievalResult(
        requirement_index=requirement_index,
        requirement_text=requirement.text,
        candidate_count=len(candidates),
        retrieved=final_ranked,
    )


def _retrieve_candidates(
    requirement: Requirement,
    evidence_pool: list[Evidence],
    retrieval_method: RetrievalMethod,
    *,
    candidate_limit: int,
    dense_retriever: DenseRetriever | None,
    hybrid_retriever: HybridRetriever | None,
) -> list[RetrievedEvidence]:
    if retrieval_method == "lexical_overlap":
        return [_retrieve_candidate_with_overlap(requirement, evidence) for evidence in evidence_pool]
    if retrieval_method == "bm25":
        return retrieve_with_bm25(requirement, evidence_pool)
    if retrieval_method == "dense":
        if dense_retriever is None:
            raise ValueError("Dense retrieval requires a dense retriever")
        return dense_retriever.retrieve(requirement, top_k=candidate_limit)
    if retrieval_method == "hybrid":
        if hybrid_retriever is None:
            raise ValueError("Hybrid retrieval requires a hybrid retriever")
        return hybrid_retriever.retrieve(requirement, evidence_pool, candidate_k=candidate_limit)
    raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")


def _select_candidates(
    ranked: list[RetrievedEvidence],
    retrieval_method: RetrievalMethod,
    candidate_limit: int,
) -> list[RetrievedEvidence]:
    if retrieval_method in {"dense", "hybrid"}:
        return ranked[:candidate_limit]
    return [item for item in ranked if item.score > 0 or item.matched_terms][:candidate_limit]


def _retrieve_candidate_with_overlap(requirement: Requirement, evidence: Evidence) -> RetrievedEvidence:
    requirement_terms = token_set(requirement.text)
    evidence_terms = token_set(evidence.text)
    matched_terms = sorted(requirement_terms & evidence_terms)

    overlap_score = 0.0
    if requirement_terms:
        overlap_score = len(matched_terms) / len(requirement_terms)

    skill_bonus = 0.0
    if requirement.skill_refs:
        shared_skill_refs = set(requirement.skill_refs) & set(evidence.skill_refs)
        if shared_skill_refs:
            skill_bonus = len(shared_skill_refs) / max(len(requirement.skill_refs), 1)
            matched_terms = sorted(set(matched_terms) | shared_skill_refs)

    score = overlap_score + 0.5 * skill_bonus

    return RetrievedEvidence(
        evidence_id=evidence.id,
        score=round(score, 4),
        retrieval_score=round(score, 4),
        rerank_score=None,
        matched_terms=matched_terms,
        evidence=evidence,
    )


def _resolve_candidate_k(top_k: int, candidate_k: int | None) -> int:
    if candidate_k is not None:
        return max(top_k, candidate_k)
    return max(top_k * 3, top_k)


def _prepare_dense_retriever(
    evidence_pool: list[Evidence],
    *,
    dense_model_name: str | None,
    dense_index_dir: str | Path | None,
    dense_encoder: EmbeddingEncoder | None,
) -> DenseRetriever:
    resolved_model_name = _resolve_dense_model_name(dense_model_name, dense_encoder)
    encoder = dense_encoder or SentenceTransformerBGEEncoder(model_name=resolved_model_name)
    return load_or_build_dense_retriever(
        evidence_pool,
        model_name=resolved_model_name,
        encoder=encoder,
        index_dir=dense_index_dir,
    )


def _resolve_dense_model_name(
    dense_model_name: str | None,
    dense_encoder: EmbeddingEncoder | None,
) -> str:
    if dense_model_name:
        return dense_model_name
    if dense_encoder is not None and getattr(dense_encoder, "model_name", None):
        return str(dense_encoder.model_name)

    env_model_name = os.getenv("DENSE_MODEL_NAME") or os.getenv("EMBED_MODEL")
    if env_model_name:
        return env_model_name

    raise ValueError(
        "Dense retrieval requires a model name via --dense-model-name, DENSE_MODEL_NAME, "
        "EMBED_MODEL, or a preconfigured encoder."
    )
