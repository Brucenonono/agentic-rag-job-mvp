# Schemas

Normalized domain schema layer for the MVP.

Implemented models:

- `Skill`
- `Requirement`
- `Evidence`
- `JD`
- `EducationEntry`
- `ExperienceEntry`
- `ProjectEntry`
- `Resume`
- `MatchDimension`
- `GapAnalysis`
- `MatchScore`

Design choices:

- Pydantic models stay small and strict with `extra="forbid"`.
- Score-like fields use `0.0` to `1.0` so downstream aggregation remains simple.
- Resume evidence is modeled explicitly to keep scoring and rewriting traceable.
