"""Resume and candidate profile schemas."""

from __future__ import annotations

from pydantic import Field

from .base import ResumeAgentModel
from .common import Evidence, Skill


class EducationEntry(ResumeAgentModel):
    """Minimal education record."""

    school: str = Field(min_length=1)
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    highlights: list[str] = Field(default_factory=list)


class ExperienceEntry(ResumeAgentModel):
    """Minimal work or internship experience record."""

    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    summary: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    highlights: list[str] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)


class ProjectEntry(ResumeAgentModel):
    """Minimal project record used as retrieval evidence source."""

    name: str = Field(min_length=1)
    summary: str | None = None
    highlights: list[str] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)


class Resume(ResumeAgentModel):
    """Normalized resume object."""

    candidate_name: str | None = None
    summary: str | None = None
    education: list[EducationEntry] = Field(default_factory=list)
    experiences: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    evidence_pool: list[Evidence] = Field(default_factory=list)
    raw_text: str | None = None
