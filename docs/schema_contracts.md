# Schema Contracts

## Purpose

These contracts define the first shared object model for the whole system.

The goal is not to freeze every field now. The goal is to make sure:

- parsers know what they must output
- retrieval knows what it indexes
- scoring knows what it consumes
- rewrite and interview generation know what evidence is allowed to cite
- evaluation knows what can be measured

## Planned Core Objects

### `Skill`

Minimum fields:

- `name`
- `category`
- `normalized_name`
- `source`
- `confidence`

Why it matters:

- skill normalization is the bridge between raw text and comparable requirements

### `Requirement`

Minimum fields:

- `text`
- `category`
- `importance`
- `is_explicit`
- `skill_refs`

Why it matters:

- a JD contains more than skills; it also contains role expectations, ownership signals, and hidden constraints

### `Evidence`

Minimum fields:

- `id`
- `text`
- `source_type`
- `source_ref`
- `skill_refs`
- `project_ref`
- `confidence`

Why it matters:

- every score and rewrite must be traceable to concrete resume evidence

### `JD`

Minimum fields:

- `title`
- `company`
- `summary`
- `requirements`
- `hard_skills`
- `soft_signals`

Why it matters:

- this is the retrieval query source and the scoring target

### `Resume`

Minimum fields:

- `candidate_name`
- `education`
- `experiences`
- `projects`
- `skills`
- `evidence_pool`

Why it matters:

- this is the evidence corpus from which all matching claims are drawn

### `MatchDimension`

Minimum fields:

- `name`
- `score`
- `reason`
- `supporting_evidence_ids`

Why it matters:

- keeps the scoring system explainable instead of collapsing everything into one opaque number

### `MatchScore`

Minimum fields:

- `overall_score`
- `dimension_scores`
- `strengths`
- `risks`
- `gaps`

Why it matters:

- this is the main downstream artifact consumed by rewrite and interview modules

### `GapAnalysis`

Minimum fields:

- `missing_requirement`
- `gap_type`
- `severity`
- `recoverability`

Why it matters:

- not every missing point should be treated the same way

## First Validation Rule Set

The first implementation should enforce:

1. every scored claim cites one or more `Evidence` objects
2. every `Evidence` object links back to an original source segment
3. every `Requirement` is categorized as explicit or implicit
4. every final `MatchScore` exposes dimension-level reasoning

## Design Principle

Prefer narrower, typed objects over one giant JSON blob.

Reason:

- better retrieval indexing
- better prompt grounding
- easier metric computation
- lower hallucination risk during rewrite and interview generation
