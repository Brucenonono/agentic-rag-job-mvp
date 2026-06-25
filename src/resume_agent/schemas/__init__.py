"""Minimal schema exports for the MVP."""

from .common import Evidence, Requirement, Skill
from .jd import JD
from .match import GapAnalysis, MatchDimension, MatchScore
from .resume import EducationEntry, ExperienceEntry, ProjectEntry, Resume

__all__ = [
    "EducationEntry",
    "Evidence",
    "ExperienceEntry",
    "GapAnalysis",
    "JD",
    "MatchDimension",
    "MatchScore",
    "ProjectEntry",
    "Requirement",
    "Resume",
    "Skill",
]
