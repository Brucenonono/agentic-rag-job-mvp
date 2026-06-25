"""Minimal parser exports."""

from .simple import (
    parse_jd_file,
    parse_jd_text,
    parse_resume_file,
    parse_resume_text,
)

__all__ = [
    "parse_jd_file",
    "parse_jd_text",
    "parse_resume_file",
    "parse_resume_text",
]
