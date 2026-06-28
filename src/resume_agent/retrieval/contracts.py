"""Typed retrieval and rerank result models for the minimal baseline."""

from __future__ import annotations

from pydantic import Field

from resume_agent.schemas.base import ResumeAgentModel
from resume_agent.schemas.common import Evidence


class RetrievedEvidence(ResumeAgentModel):
    """One retrieved evidence candidate for a requirement."""

    evidence_id: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    retrieval_score: float | None = Field(default=None, ge=0.0)
    rerank_score: float | None = Field(default=None, ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)
    evidence: Evidence


class RequirementRetrievalResult(ResumeAgentModel):
    """Top-k evidence candidates for a single requirement."""

    requirement_index: int = Field(ge=0)
    requirement_text: str = Field(min_length=1)
    candidate_count: int = Field(default=0, ge=0)
    retrieved: list[RetrievedEvidence] = Field(default_factory=list)


class RetrievalReport(ResumeAgentModel):
    """Retrieval output for all JD requirements against one resume."""

    jd_title: str = Field(min_length=1)
    candidate_name: str | None = None
    top_k: int = Field(ge=1)
    candidate_k: int = Field(ge=1)
    retrieval_method: str = Field(min_length=1)
    rerank_method: str | None = None
    results: list[RequirementRetrievalResult] = Field(default_factory=list)
