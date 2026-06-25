# MVP Scope

## Minimal Runnable Target

The first runnable version of this project should do exactly these things:

1. accept one JD document and one resume document
2. normalize both into shared schema objects
3. retrieve candidate evidence from the resume and project history
4. score the candidate against the JD using multiple signals
5. output a structured report with:
   - match score
   - top supporting evidence
   - skill gaps
   - rewrite suggestions
   - interview preparation topics
6. run the same sample through multiple retrieval and pipeline variants for evaluation

## Non-Goals For The First Milestone

- batch processing for many candidates
- browser UI
- database-backed persistence
- advanced multi-agent planning
- distributed tool servers

## Module Boundaries

### `schemas`

Inputs:

- raw parsed fields from JD and resume
- normalized skill and evidence objects

Outputs:

- validated domain objects used everywhere else

### `parsers`

Inputs:

- raw JD text
- raw resume text
- optional project descriptions

Outputs:

- structured schema-ready objects

### `retrieval`

Inputs:

- parsed JD requirements
- parsed resume/project evidence corpus

Outputs:

- ranked evidence candidates

### `scoring`

Inputs:

- parsed requirements
- reranked evidence set

Outputs:

- dimension scores
- final match score
- explanation payload

### `tools`

Inputs:

- typed tool requests

Outputs:

- typed tool responses

### `pipelines`

Inputs:

- end-to-end task request

Outputs:

- combined report artifact

### `evaluation`

Inputs:

- datasets
- experiment configs
- pipeline outputs

Outputs:

- metric tables
- experiment comparisons

## Initial Data Contract Targets

The first schema set should include:

- `JD`
- `Resume`
- `Skill`
- `Evidence`
- `Requirement`
- `MatchDimension`
- `MatchScore`
- `GapAnalysis`
- `RewriteInstruction`
- `InterviewTopic`
