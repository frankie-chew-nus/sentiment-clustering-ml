"""Modular Group 1 and Group 2 tweet preprocessing.

Group 1:
- Replace configured airline references with the generic token "airline".
- Remove non-airline @mentions.
- Keep non-airline hashtag words and remove '#'.
- Preserve the raw input separately; never overwrite source text.

Group 2:
- Expand common contractions before punctuation handling.
- Support number conversion, lowercasing, configurable punctuation, and emoji
  modes (describe/keep/remove).
- VADER and structural features should use original raw text, not this output.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Collection, Mapping

from config import (
    AIRLINE_ALIASES,
    AIRLINE_HASHTAG_SUFFIX_RULES,
    P0_CONFIG,
    P1_CONFIG,
)

from lexical_normalization import (
    correct_spelling,
    lemmatize_text,
    remove_stopwords,
    split_compound_words,
)

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
_HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,!?;:])")
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}"

# Deliberately expand unambiguous contractions and common informal versions
# found in tweets. Ambiguous forms such as "I'd" and "he's" are omitted.
_CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "cant": "can not",
    "won't": "will not", "wont": "will not",
    "don't": "do not", "dont": "do not",
    "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "haven't": "have not", "havent": "have not",
    "hasn't": "has not", "hasnt": "has not",
    "hadn't": "had not", "hadnt": "had not",
    "wouldn't": "would not", "wouldnt": "would not",
    "shouldn't": "should not", "shouldnt": "should not",
    "couldn't": "could not", "couldnt": "could not",
    "mustn't": "must not", "mustnt": "must not",
    "needn't": "need not", "neednt": "need not",
    "shan't": "shall not", "shant": "shall not",
    "i'm": "i am", "im": "i am",
    "you're": "you are", "youre": "you are",
    "we're": "we are", "were": "we are",
    "they're": "they are", "theyre": "they are",
    "it's": "it is", "its": "its",
    "that's": "that is", "thats": "that is",
    "there's": "there is", "theres": "there is",
    "what's": "what is", "whats": "what is",
    "who's": "who is", "whos": "who is",
    "let's": "let us", "lets": "let us",
    "i've": "i have", "ive": "i have",
    "you've": "you have", "youve": "you have",
    "we've": "we have", "weve": "we have",
    "they've": "they have", "theyve": "they have",
    "i'll": "i will", "ill": "i will",
    "you'll": "you will", "youll": "you will",
    "we'll": "we will", "well": "well",
    "they'll": "they will", "theyll": "they will",
    "would've": "would have", "wouldve": "would have",
    "could've": "could have", "couldve": "could have",
    "should've": "should have", "shouldve": "should have",
}
_CONTRACTION_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(key) for key in sorted(_CONTRACTIONS, key=len, reverse=True)
    ) + r")(?!\w)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w:/-])(?P<number>\d[\d,]*(?:\.\d+)?)(?![\w:/-])")
_EMOJI_START = "__EMOJI_START__"
_EMOJI_END = "__EMOJI_END__"
_EMOJI_MARKER_RE = re.compile(re.escape(_EMOJI_START) + r"(.*?)" + re.escape(_EMOJI_END))


def _compact(value: str) -> str:
    """Canonical form for comparing aliases, handles, and hashtags."""
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
    text: str,
    aliases: Collection[str] = AIRLINE_ALIASES,
    suffix_rules: Collection[str] = AIRLINE_HASHTAG_SUFFIX_RULES,
) -> str:
    """Normalize exact airline hashtags and approved airline-prefixed suffixes."""
    alias_set = _normalized_aliases(aliases)
    suffix_set = {_compact(suffix) for suffix in suffix_rules}
    ordered_aliases = sorted(alias_set, key=len, reverse=True)

    def replacement(match: re.Match[str]) -> str:
        token = match.group(1)
        compact = _compact(token)
        if compact in alias_set:
            return "airline"
        for alias in ordered_aliases:
            if compact.startswith(alias) and len(compact) > len(alias):
                suffix = compact[len(alias):]
                if suffix in suffix_set:
                    return f"airline {suffix}"
        return token

    return _HASHTAG_RE.sub(replacement, text)


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


def expand_contractions(text: str) -> str:
    """Expand common contractions and common no-apostrophe tweet variants."""
    text = text.replace("’", "'")
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0).casefold()], text)


def handle_emojis(text: str, mode: str = "describe") -> str:
    """Handle Unicode emojis as descriptions, unchanged, or removed.

    The emoji package is imported lazily so Group 1 does not require it unless
    an emoji mode is used.
    """
    if mode not in {"describe", "keep", "remove"}:
        raise ValueError("emoji mode must be 'describe', 'keep', or 'remove'")
    if mode == "keep":
        return text

    try:
        import emoji
    except ImportError as exc:
        raise ImportError(
            "Emoji handling requires the 'emoji' package. Install requirements.txt."
        ) from exc

    if mode == "remove":
        # Use offsets from the library so multi-codepoint emojis are removed safely.
        items = emoji.emoji_list(text)
        for item in reversed(items):
            text = text[:item["match_start"]] + text[item["match_end"]:]
        return text

    demojized = emoji.demojize(text, delimiters=(_EMOJI_START, _EMOJI_END))
    return _EMOJI_MARKER_RE.sub(
        lambda match: " " + match.group(1).replace("_", " ") + " ",
        demojized,
    )


def convert_numbers(text: str) -> str:
    """Convert standalone numbers to English words; leave times/ranges/IDs intact."""
    try:
        from num2words import num2words
    except ImportError as exc:
        raise ImportError(
            "Number conversion requires the 'num2words' package. Install requirements.txt."
        ) from exc

    def replacement(match: re.Match[str]) -> str:
        raw = match.group("number").replace(",", "")
        try:
            number = float(raw) if "." in raw else int(raw)
            words = num2words(number, lang="en")
            return words.replace("-", " ")
        except (ValueError, OverflowError):
            return match.group("number")

    return _NUMBER_RE.sub(replacement, text)


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Replace Unicode punctuation characters with spaces; emoji symbols remain."""
    return "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )


# Group 1 cleaning, in dependency-safe order.
_GROUP1_STEPS = (
    ("remove_urls", remove_urls),
    ("replace_airline_mentions", replace_airline_mentions),
    ("remove_other_mentions", remove_mentions),
    ("replace_airline_hashtags", replace_airline_hashtags),
    ("replace_airline_names", replace_airline_names),
    ("strip_hashtag_symbols", strip_hashtag_symbols),
)

# Group 2 functions are applied in a dedicated order below so contraction and
# emoji handling happen before punctuation removal.
def preprocess_text(text: str, config: Mapping[str, Any] | None = None) -> str:
    """Run selected preprocessing operations in a fixed, dependency-safe order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    settings = P0_CONFIG if config is None else config

    for name, function in _GROUP1_STEPS:
        if settings.get(name, False):
            text = function(text)

    if settings.get("expand_contractions", False):
        text = expand_contractions(text)

    emoji_mode = settings.get("emoji_mode", "keep")
    if emoji_mode != "keep" or "emoji_mode" in settings:
        text = handle_emojis(text, emoji_mode)

    if settings.get("convert_numbers", False):
        text = convert_numbers(text)

    if settings.get("lowercase", False):
        text = lowercase_text(text)

    punctuation_mode = settings.get("punctuation_mode", "keep")
    if punctuation_mode not in {"remove", "keep"}:
        raise ValueError("punctuation_mode must be 'remove' or 'keep'")
    if punctuation_mode == "remove":
        text = remove_punctuation(text)

    # Group 3: optional lexical normalization. Each operation is controlled
    # independently so its contribution can be measured in ablation studies.
    text = split_compound_words(
        text,
        method=str(settings.get("segmentation_mode", "none")),
        min_token_length=int(settings.get("segmentation_min_token_length", 7)),
        protected_words=settings.get("protected_words", ()),
        symspell_dictionary_path=settings.get("symspell_dictionary_path"),
    )

    if settings.get("correct_spelling", False):
        text = correct_spelling(
            text,
            dictionary_path=settings.get("symspell_dictionary_path"),
            max_edit_distance=int(settings.get("max_edit_distance", 2)),
            protected_words=settings.get("protected_words", ()),
        )

    if settings.get("lemmatize", False):
        text = lemmatize_text(
            text,
            model=str(settings.get("spacy_model", "en_core_web_sm")),
            preserve_negation=bool(settings.get("preserve_negation", True)),
        )

    if settings.get("remove_stopwords", False):
        text = remove_stopwords(
            text,
            model=str(settings.get("spacy_model", "en_core_web_sm")),
            preserve_negation=bool(settings.get("preserve_negation", True)),
        )

    if settings.get("normalize_whitespace", False):
        text = normalize_whitespace(text)

    return text
