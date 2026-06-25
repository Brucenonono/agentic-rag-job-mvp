"""Match scoring schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import ResumeAgentModel

GapType = Literal["missing_skill", "missing_evidence", "weak_evidence", "implicit_requirement"]
RecoverabilityLevel = Literal["high", "medium", "low"]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


class GapAnalysis(ResumeAgentModel):
    """Structured missing-signal analysis."""

    missing_requirement: str = Field(min_length=1)
    gap_type: GapType = "missing_skill"
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    recoverability: RecoverabilityLevel = "medium"


class MatchDimension(ResumeAgentModel):
    """One explainable scoring dimension."""

    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("supporting_evidence_ids")
    @classmethod
    def dedupe_supporting_evidence_ids(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)


class MatchScore(ResumeAgentModel):
    """Top-level explainable match result."""

    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: list[MatchDimension] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    gaps: list[GapAnalysis] = Field(default_factory=list)
    summary: str | None = None

    @field_validator("strengths", "risks")
    @classmethod
    def dedupe_text_lists(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)

    @model_validator(mode="after")
    def ensure_unique_dimension_names(self) -> "MatchScore":
        names = [dimension.name for dimension in self.dimension_scores]
        if len(names) != len(set(names)):
            raise ValueError("dimension_scores must have unique dimension names")
        return self
