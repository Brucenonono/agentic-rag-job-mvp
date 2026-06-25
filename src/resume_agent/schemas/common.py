"""Common domain models shared across JD, resume, and scoring."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import ResumeAgentModel

SkillCategory = Literal[
    "programming_language",
    "framework",
    "ml",
    "tool",
    "database",
    "cloud",
    "soft_skill",
    "domain",
    "other",
]
SkillSource = Literal["jd", "resume", "project", "inferred", "manual"]
RequirementCategory = Literal[
    "hard_skill",
    "responsibility",
    "qualification",
    "soft_skill",
    "experience",
    "education",
    "other",
]
ImportanceLevel = Literal["must", "preferred", "bonus"]
EvidenceSourceType = Literal["experience", "project", "education", "skill", "summary", "other"]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_skill_name(value: str) -> str:
    lowered = value.lower().strip()
    collapsed = re.sub(r"[^a-z0-9\+]+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


class Skill(ResumeAgentModel):
    """Normalized skill mention extracted from a JD or resume."""

    name: str = Field(min_length=1)
    category: SkillCategory = "other"
    normalized_name: str | None = None
    source: SkillSource = "manual"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def populate_normalized_name(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if data.get("name") and not data.get("normalized_name"):
            return {**data, "normalized_name": _normalize_skill_name(str(data["name"]))}
        return data


class Requirement(ResumeAgentModel):
    """Single requirement extracted from a job description."""

    text: str = Field(min_length=1)
    category: RequirementCategory = "other"
    importance: ImportanceLevel = "preferred"
    is_explicit: bool = True
    skill_refs: list[str] = Field(default_factory=list)

    @field_validator("skill_refs")
    @classmethod
    def dedupe_skill_refs(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)


class Evidence(ResumeAgentModel):
    """Atomic evidence unit that the system can cite downstream."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_type: EvidenceSourceType = "other"
    source_ref: str = Field(min_length=1)
    skill_refs: list[str] = Field(default_factory=list)
    project_ref: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

#去重
    @field_validator("skill_refs")
    @classmethod
    def dedupe_skill_refs(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)
