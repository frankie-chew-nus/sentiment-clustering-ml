"""Modular Group 1 tweet preprocessing.

Research decision:
- Replace configured airline names with the generic token "airline", whether
  they appear as @mentions, #hashtags, or ordinary text.
- Remove other @mentions.
- Preserve non-airline hashtag content without the leading '#'.
- Keep original raw tweets in a separate column; never overwrite source text.

This reduces explicit airline identity in the vectorized text while preserving
surrounding complaint/praise language. Ambiguous aliases (e.g. "American") need
sample review to check for false replacements.
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


def _compact(value: str) -> str:
    """Canonical form for matching handles/hashtags with spaces or underscores."""
    return re.sub(r"[\s_]+", "", value.lstrip("@#").strip()).casefold()


def _normalized_aliases(aliases: Collection[str]) -> set[str]:
    return {_compact(alias) for alias in aliases}


def _alias_pattern(aliases: Collection[str]) -> re.Pattern[str]:
    """Match ordinary airline names as whole words/phrases, longest forms first."""
    normalized = {re.sub(r"\s+", " ", a.strip()).casefold() for a in aliases}
    alternatives = []
    for alias in sorted(normalized, key=len, reverse=True):
        alternatives.append(re.escape(alias).replace(r"\ ", r"\s+"))
    if not alternatives:
        return re.compile(r"(?!x)x")
    # Do not match inside handles/hashtags; those are handled by dedicated functions.
    return re.compile(
        r"(?<![\w@#])(?:" + "|".join(alternatives) + r")(?!\w)",
        re.IGNORECASE,
    )


def remove_urls(text: str) -> str:
    """Remove HTTP(S)/www URLs while retaining trailing sentence punctuation."""
    def replacement(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = url[len(url.rstrip(_URL_TRAILING_PUNCTUATION)):]
        return trailing

    text = _URL_RE.sub(replacement, text)
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)


def replace_airline_mentions(
    text: str, aliases: Collection[str] = AIRLINE_ALIASES
) -> str:
    """Replace recognized airline @mentions with 'airline'."""
    lookup = _normalized_aliases(aliases)
    return _MENTION_RE.sub(
        lambda match: "airline" if _compact(match.group(1)) in lookup else match.group(0),
        text,
    )


def remove_mentions(text: str) -> str:
    """Remove remaining, non-airline @mentions."""
    return _MENTION_RE.sub("", text)


def replace_airline_hashtags(
    text: str, aliases: Collection[str] = AIRLINE_ALIASES
) -> str:
    """Replace recognized airline hashtags with 'airline'; keep other hashtag words."""
    lookup = _normalized_aliases(aliases)
    return _HASHTAG_RE.sub(
        lambda match: "airline" if _compact(match.group(1)) in lookup else match.group(1),
        text,
    )


def replace_airline_names(
    text: str, aliases: Collection[str] = AIRLINE_ALIASES
) -> str:
    """Replace recognized airline names in ordinary text with 'airline'."""
    return _alias_pattern(aliases).sub("airline", text)


def strip_hashtag_symbols(text: str) -> str:
    """Remove '#' while preserving remaining hashtag text."""
    return _HASHTAG_RE.sub(lambda match: match.group(1), text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# Order matters: specialized airline-reference handling runs before generic cleanup.
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
