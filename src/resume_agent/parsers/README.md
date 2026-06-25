# Parsers

Minimal parser layer for the first runnable version.

Current scope:

- parse `txt` JD files into `JD`
- parse `txt` resume files into `Resume`
- accept `json` inputs when they already resemble schema payloads
- use lightweight rules and keyword extraction instead of LLM parsing

Current entrypoints:

- `parse_jd_file()`
- `parse_resume_file()`
- `parse_jd_text()`
- `parse_resume_text()`

Design note:

- This is intentionally a baseline parser so the project can run before any LLM dependency is introduced.
