"""Modular Group 1 tweet preprocessing.

Airline aliases are normalized to the generic token "airline" in handles,
hashtags, and ordinary text. Non-airline handles are removed; non-airline
hashtag words are preserved without the leading '#'. Keep the raw tweet in a
separate column: never overwrite the source text.
"""
from __future__ import annotations

import re
from typing import Collection, Mapping

from config import AIRLINE_ALIASES, P0_CONFIG

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
_HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,!?;:])")
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}"


def _normalized_aliases(aliases: Collection[str]) -> set[str]:
    """Normalize aliases for case-insensitive exact matching."""
    return {re.sub(r"\s+", " ", a.lstrip("@#").strip()).casefold() for a in aliases}


def _alias_pattern(aliases: Collection[str]) -> re.Pattern[str]:
    """Create a case-insensitive whole-word/whole-phrase pattern, longest first."""
    normalized = _normalized_aliases(aliases)
    alternatives = []
    for alias in sorted(normalized, key=len, reverse=True):
        # Match one or more spaces where the configured alias contains spaces.
        alternatives.append(re.escape(alias).replace(r"\ ", r"\s+"))
    if not alternatives:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def remove_urls(text: str) -> str:
    """Remove HTTP(S)/www URLs while retaining trailing sentence punctuation."""
    def replacement(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = url[len(url.rstrip(_URL_TRAILING_PUNCTUATION)):]
        return trailing

    text = _URL_RE.sub(replacement, text)
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)


def replace_airline_mentions(text: str, aliases: Collection[str] = AIRLINE_ALIASES) -> str:
    """Replace known airline @handles with 'airline'."""
    lookup = _normalized_aliases(aliases)
    return _MENTION_RE.sub(
        lambda match: "airline" if match.group(1).casefold() in lookup else match.group(0),
        text,
    )


def remove_mentions(text: str) -> str:
    """Remove remaining, non-airline @mentions."""
    return _MENTION_RE.sub("", text)


def replace_airline_hashtags(text: str, aliases: Collection[str] = AIRLINE_ALIASES) -> str:
    """Replace known airline hashtags with 'airline'; preserve other hashtag words."""
    lookup = _normalized_aliases(aliases)
    return _HASHTAG_RE.sub(
        lambda match: "airline" if match.group(1).casefold() in lookup else match.group(1),
        text,
    )


def replace_airline_names(text: str, aliases: Collection[str] = AIRLINE_ALIASES) -> str:
    """Replace known airline names in ordinary text with 'airline'."""
    return _alias_pattern(aliases).sub("airline", text)


def strip_hashtag_symbols(text: str) -> str:
    """Remove '#' while preserving non-airline hashtag content."""
    return _HASHTAG_RE.sub(lambda match: match.group(1), text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


_STEPS = (
    ("remove_urls", remove_urls),
    ("replace_airline_mentions", replace_airline_mentions),
    ("remove_other_mentions", remove_mentions),
    ("replace_airline_hashtags", replace_airline_hashtags),
    ("replace_airline_names", replace_airline_names),
    ("strip_hashtag_symbols", strip_hashtag_symbols),
    ("normalize_whitespace", normalize_whitespace),
)


def preprocess_text(text: str, config: Mapping[str, bool] | None = None) -> str:
    """Run selected Group 1 operations in dependency-safe order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    settings = P0_CONFIG if config is None else config
    for name, function in _STEPS:
        if settings.get(name, False):
            text = function(text)
    return text
