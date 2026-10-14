""""Modular Group 1 tweet preprocessing.

All functions are deterministic and preserve the input unless their specific
transformation applies. Original tweets should be stored separately.
"""
from __future__ import annotations

import re
from typing import Collection, Mapping

from config import AIRLINE_HASHTAG_ALIASES, P0_CONFIG

# Recognize common URL forms without swallowing adjacent sentence punctuation.
_URL_RE = re.compile(r'(?i)\\b(?:https?://|www\\.)[^\\s<>]+')
_MENTION_RE = re.compile(r'(?<!\\w)@[A-Za-z0-9_]+')
_HASHTAG_RE = re.compile(r'(?<!\\w)#([A-Za-z0-9_]+)')
_WHITESPACE_RE = re.compile(r'\\s+')
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r'\\s+([.,!?;:])')
_URL_TRAILING_PUNCTUATION = '.,!?;:)]}'


def remove_urls(text: str) -> str:
    """Remove HTTP(S)/www URLs while retaining adjacent sentence punctuation."""
    def replacement(match: re.Match[str]) -> str:
        url = match.group(0)
        trailing = url[len(url.rstrip(_URL_TRAILING_PUNCTUATION)):]
        return trailing

    text = _URL_RE.sub(replacement, text)
    # If a URL was followed by a period, avoid leaving "sentence .".
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r'\\1', text)


def remove_mentions(text: str) -> str:
    """Remove Twitter-style @handles, including airline handles."""
    return _MENTION_RE.sub('', text)


def remove_airline_hashtags(
    text: str, aliases: Collection[str] = AIRLINE_HASHTAG_ALIASES
) -> str:
    """Remove only exact, case-insensitive airline hashtag aliases."""
    lookup = {alias.lstrip('#').casefold() for alias in aliases}
    return _HASHTAG_RE.sub(
        lambda match: '' if match.group(1).casefold() in lookup else match.group(0),
        text,
    )


def strip_hashtag_symbols(text: str) -> str:
    """Keep hashtag content, dropping its leading #."""
    return _HASHTAG_RE.sub(lambda match: match.group(1), text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip edges."""
    return _WHITESPACE_RE.sub(' ', text).strip()


# The order is deliberate: remove airline hashtags before stripping #.
_STEPS = (
    ('remove_urls', remove_urls),
    ('remove_mentions', remove_mentions),
    ('remove_airline_hashtags', remove_airline_hashtags),
    ('strip_hashtag_symbols', strip_hashtag_symbols),
    ('normalize_whitespace', normalize_whitespace),
)


def preprocess_text(text: str, config: Mapping[str, bool] | None = None) -> str:
    """Run selected Group 1 operations in dependency-safe order.

    Missing/non-string values are rejected rather than silently converted.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    settings = P0_CONFIG if config is None else config
    for name, function in _STEPS:
        if settings.get(name, False):
            text = function(text)
    return text
"