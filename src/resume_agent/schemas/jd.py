"""Job description schemas."""

from __future__ import annotations

from pydantic import Field, field_validator

from .base import ResumeAgentModel
from .common import Requirement, Skill


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


class JD(ResumeAgentModel):
    """Normalized job description object."""

    title: str = Field(min_length=1)
    company: str | None = None
    summary: str | None = None
    requirements: list[Requirement] = Field(default_factory=list)
    hard_skills: list[Skill] = Field(default_factory=list)
    soft_signals: list[str] = Field(default_factory=list)
    raw_text: str | None = None

    @field_validator("soft_signals")
    @classmethod
    def dedupe_soft_signals(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)
