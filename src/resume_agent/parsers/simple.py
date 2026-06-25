"""Minimal rule-based parsers for the first runnable version."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from resume_agent.schemas import (
    EducationEntry,
    Evidence,
    ExperienceEntry,
    JD,
    ProjectEntry,
    Requirement,
    Resume,
    Skill,
)

SKILL_LIBRARY: dict[str, str] = {
    "python": "programming_language",
    "c++": "programming_language",
    "java": "programming_language",
    "sql": "database",
    "pytorch": "framework",
    "tensorflow": "framework",
    "scikit-learn": "ml",
    "sklearn": "ml",
    "numpy": "tool",
    "pandas": "tool",
    "docker": "tool",
    "git": "tool",
    "linux": "tool",
    "faiss": "tool",
    "bm25": "ml",
    "machine learning": "ml",
    "deep learning": "ml",
    "nlp": "domain",
    "rag": "ml",
    "llm": "ml",
    "transformers": "framework",
    "aws": "cloud",
    "gcp": "cloud",
}

SOFT_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "communication": ("communication", "communicate", "presentation"),
    "collaboration": ("collaboration", "collaborate", "cross-functional", "teamwork"),
    "ownership": ("ownership", "own", "end-to-end"),
    "fast_learner": ("fast learner", "learn quickly", "self-driven", "proactive"),
}

JD_SECTION_NAMES = {
    "requirements",
    "responsibilities",
    "qualifications",
    "preferred",
    "nice to have",
    "must have",
}
RESUME_SECTION_ALIASES = {
    "summary": "summary",
    "education": "education",
    "experience": "experience",
    "work experience": "experience",
    "projects": "projects",
    "skills": "skills",
}


def parse_jd_file(path: str | Path) -> JD:
    payload = _load_input(path)
    if isinstance(payload, dict):
        return JD.model_validate(_coerce_jd_payload(payload))
    return parse_jd_text(payload)


def parse_resume_file(path: str | Path) -> Resume:
    payload = _load_input(path)
    if isinstance(payload, dict):
        return Resume.model_validate(_coerce_resume_payload(payload))
    return parse_resume_text(payload)


def parse_jd_text(text: str) -> JD:
    lines = _non_empty_lines(text)
    title = lines[0] if lines else "Untitled JD"
    company = _extract_prefixed_value(lines, "company")
    summary = _extract_summary(lines)
    requirement_lines = _extract_jd_requirement_lines(text)

    requirements = [_build_requirement(line) for line in requirement_lines]
    hard_skills = _extract_skills(text, source="jd")
    soft_signals = _extract_soft_signals(text)

    return JD(
        title=title,
        company=company,
        summary=summary,
        requirements=requirements,
        hard_skills=hard_skills,
        soft_signals=soft_signals,
        raw_text=text,
    )


def parse_resume_text(text: str) -> Resume:
    sections = _split_sections(text, RESUME_SECTION_ALIASES)
    header_lines = [line for line in sections.get("header", []) if line]
    candidate_name = header_lines[0] if header_lines else None
    summary = _join_lines(sections.get("summary", [])) or None

    skills = _extract_skills("\n".join(sections.get("skills", [])) or text, source="resume")
    education = _parse_education_entries(sections.get("education", []))
    experiences = _parse_experience_entries(sections.get("experience", []))
    projects = _parse_project_entries(sections.get("projects", []))
    evidence_pool = _build_evidence_pool(sections)

    return Resume(
        candidate_name=candidate_name,
        summary=summary,
        education=education,
        experiences=experiences,
        projects=projects,
        skills=skills,
        evidence_pool=evidence_pool,
        raw_text=text,
    )


def _load_input(path: str | Path) -> dict[str, Any] | str:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        return json.loads(content)
    return content


def _coerce_jd_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if "requirements" in data:
        data["requirements"] = [
            item if isinstance(item, dict) else _build_requirement(str(item)).model_dump()
            for item in data["requirements"]
        ]
    if "hard_skills" in data:
        data["hard_skills"] = [
            item
            if isinstance(item, dict)
            else Skill(name=str(item), category=_infer_skill_category(str(item)), source="jd").model_dump()
            for item in data["hard_skills"]
        ]
    return data


def _coerce_resume_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if "skills" in data:
        data["skills"] = [
            item
            if isinstance(item, dict)
            else Skill(name=str(item), category=_infer_skill_category(str(item)), source="resume").model_dump()
            for item in data["skills"]
        ]
    if "evidence_pool" in data:
        evidence_items: list[dict[str, Any]] = []
        for index, item in enumerate(data["evidence_pool"], start=1):
            if isinstance(item, dict):
                evidence_items.append(item)
                continue
            text = str(item)
            evidence_items.append(
                Evidence(
                    id=f"evidence-{index}",
                    text=text,
                    source_type="other",
                    source_ref=f"manual:{index}",
                    skill_refs=[skill.normalized_name for skill in _extract_skills(text, source="resume")],
                ).model_dump()
            )
        data["evidence_pool"] = evidence_items
    return data


def _extract_prefixed_value(lines: list[str], key: str) -> str | None:
    prefix = f"{key.lower()}:"
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(prefix):
            return line.split(":", 1)[1].strip() or None
    return None


def _extract_summary(lines: list[str]) -> str | None:
    for line in lines[1:]:
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("company:"):
            continue
        if ":" in line and line.split(":", 1)[0].strip().lower() in JD_SECTION_NAMES:
            continue
        return line
    return None


def _extract_jd_requirement_lines(text: str) -> list[str]:
    section_lines = _split_sections(text, {name: name for name in JD_SECTION_NAMES})
    requirement_lines: list[str] = []
    for section_name in JD_SECTION_NAMES:
        requirement_lines.extend(_bullet_candidates(section_lines.get(section_name, [])))
    if requirement_lines:
        return requirement_lines

    fallback_lines = _bullet_candidates(text.splitlines())
    return fallback_lines


def _build_requirement(text: str) -> Requirement:
    requirement_text = _clean_bullet(text)
    skills = _extract_skills(requirement_text, source="jd")
    lowered = requirement_text.lower()

    importance = "preferred"
    if any(token in lowered for token in ("must", "required", "requirement")):
        importance = "must"
    elif any(token in lowered for token in ("bonus", "plus", "nice to have", "preferred")):
        importance = "bonus"

    category = "hard_skill" if skills else "responsibility"
    if any(token in lowered for token in ("communication", "collaboration", "team")):
        category = "soft_skill"

    return Requirement(
        text=requirement_text,
        category=category,
        importance=importance,
        is_explicit=True,
        skill_refs=[skill.normalized_name for skill in skills],
    )


def _extract_skills(text: str, source: str) -> list[Skill]:
    lowered = text.lower()
    skills: list[Skill] = []
    for skill_name, category in SKILL_LIBRARY.items():
        pattern = rf"(?<![a-z0-9]){re.escape(skill_name.lower())}(?![a-z0-9])"
        if re.search(pattern, lowered):
            skills.append(Skill(name=skill_name, category=category, source=source))
    return skills


def _extract_soft_signals(text: str) -> list[str]:
    lowered = text.lower()
    results: list[str] = []
    for label, patterns in SOFT_SIGNAL_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            results.append(label)
    return results


def _split_sections(text: str, aliases: dict[str, str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"header": []}
    current_section = "header"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = line.lower().rstrip(":")
        if normalized in aliases:
            current_section = aliases[normalized]
            sections.setdefault(current_section, [])
            continue
        sections.setdefault(current_section, []).append(line)

    return sections


def _parse_education_entries(lines: list[str]) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    for block in _split_blocks(lines):
        school = block[0]
        highlights = [_clean_bullet(line) for line in block[1:] if line]
        entries.append(EducationEntry(school=school, highlights=highlights))
    return entries


def _parse_experience_entries(lines: list[str]) -> list[ExperienceEntry]:
    entries: list[ExperienceEntry] = []
    for block in _split_blocks(lines):
        header = block[0]
        company, role = _split_header_pair(header, default_role="Experience")
        highlights = [_clean_bullet(line) for line in block[1:] if line]
        skills = _extract_skills("\n".join(block), source="resume")
        entries.append(
            ExperienceEntry(
                company=company,
                role=role,
                highlights=highlights,
                skills=skills,
            )
        )
    return entries


def _parse_project_entries(lines: list[str]) -> list[ProjectEntry]:
    entries: list[ProjectEntry] = []
    for block in _split_blocks(lines):
        name = block[0]
        summary = None
        highlights = [_clean_bullet(line) for line in block[1:] if line]
        if highlights:
            summary = highlights[0]
        skills = _extract_skills("\n".join(block), source="project")
        entries.append(ProjectEntry(name=name, summary=summary, highlights=highlights, skills=skills))
    return entries


def _build_evidence_pool(sections: dict[str, list[str]]) -> list[Evidence]:
    evidence_pool: list[Evidence] = []
    counters = {"experience": 0, "project": 0, "education": 0}

    for source_type in ("experience", "projects", "education"):
        section_lines = sections.get(source_type, [])
        if not section_lines:
            continue
        normalized_source = "project" if source_type == "projects" else source_type
        for line in _bullet_candidates(section_lines):
            counters[normalized_source] += 1
            cleaned_line = _clean_bullet(line)
            evidence_pool.append(
                Evidence(
                    id=f"{normalized_source}-{counters[normalized_source]}",
                    text=cleaned_line,
                    source_type=normalized_source,
                    source_ref=f"{normalized_source}:{counters[normalized_source]}",
                    skill_refs=[
                        skill.normalized_name
                        for skill in _extract_skills(cleaned_line, source="resume")
                    ],
                )
            )

    if evidence_pool:
        return evidence_pool

    summary_lines = [line for line in sections.get("summary", []) if line]
    for index, line in enumerate(summary_lines, start=1):
        evidence_pool.append(
            Evidence(
                id=f"summary-{index}",
                text=line,
                source_type="summary",
                source_ref=f"summary:{index}",
                skill_refs=[skill.normalized_name for skill in _extract_skills(line, source="resume")],
            )
        )
    return evidence_pool


def _split_header_pair(header: str, default_role: str) -> tuple[str, str]:
    if " at " in header.lower():
        role, company = re.split(r"\s+at\s+", header, maxsplit=1, flags=re.IGNORECASE)
        return company.strip(), role.strip()
    if " - " in header:
        left, right = header.split(" - ", 1)
        return left.strip(), right.strip()
    return header.strip(), default_role


def _split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _bullet_candidates(lines: list[str]) -> list[str]:
    bullets: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "+")):
            bullets.append(line)
    return bullets


def _clean_bullet(line: str) -> str:
    return re.sub(r"^[\-\*\+\s]+", "", line).strip()


def _join_lines(lines: list[str]) -> str:
    return " ".join(line for line in lines if line)


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _infer_skill_category(skill_name: str) -> str:
    return SKILL_LIBRARY.get(skill_name.lower(), "other")
