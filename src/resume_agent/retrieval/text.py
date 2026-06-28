"""Shared text processing helpers for retrieval backends."""

from __future__ import annotations

import re
from typing import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "with",
}


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for lexical retrieval backends."""

    return [token for token in _iter_tokens(text) if token not in STOPWORDS]


def token_set(text: str) -> set[str]:
    """Return a unique token set for overlap-based methods."""

    return set(tokenize_text(text))


def _iter_tokens(text: str) -> Iterable[str]:
    return (token for token in re.findall(r"[a-z0-9\+]+", text.lower()) if token)
