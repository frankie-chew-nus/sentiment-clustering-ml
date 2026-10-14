"""Group 3 optional lexical normalization functions.

Segmentation and spelling correction are experiments, not guaranteed improvements.
Keep them configurable and inspect changes on real tweets before enabling by default.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Collection

from config import (
    AIRLINE_ALIASES,
    NEGATION_WORDS,
    SENTIMENTSENTIMENT_PROTECTED_WORDS,
    STOPWORDSENTIMENT_PROTECTED_WORDS,
)


def _compact(value: str) -> str:
    return re.sub(r"[\s_]+", "", value.lstrip("@#").strip()).casefold()


def split_compound_words(
    text: str,
    method: str = "wordninja",
    min_token_length: int = 7,
    protected_words: Collection[str] | None = None,
    symspell_dictionary_path: str | None = None,
) -> str:
    """Split eligible long alphabetic tokens with a selected method.

    Methods: none, wordninja, symspell, hybrid.
    Hybrid uses Wordninja proposals plus word-frequency checks. This is a
    heuristic, not a guarantee of correct segmentation.
    """
    if method not in {"none", "wordninja", "symspell", "hybrid"}:
        raise ValueError("method must be none, wordninja, symspell, or hybrid")
    if method == "none":
        return text

    protected = {w.casefold() for w in (protected_words or set())}
    protected |= {_compact(alias) for alias in AIRLINE_ALIASES}

    sym_spell = _get_symspell(symspell_dictionary_path) if method == "symspell" else None
    wordninja = None
    wordfreq = None
    if method in {"wordninja", "hybrid"}:
        try:
            import wordninja as wn
            wordninja = wn
        except ImportError as exc:
            raise ImportError("Install wordninja to use this segmentation method.") from exc
    if method == "hybrid":
        try:
            from wordfreq import zipf_frequency
            wordfreq = zipf_frequency
        except ImportError as exc:
            raise ImportError("Install wordfreq to use hybrid segmentation.") from exc

    min_len = max(1, int(min_token_length))

    def split_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) < min_len or token.casefold() in protected:
            return token
        if method == "symspell":
            result = sym_spell.word_segmentation(token, max_edit_distance=0)
            candidate = getattr(result, "corrected_string", token).strip()
            return candidate if " " in candidate else token

        pieces = wordninja.split(token)
        if len(pieces) < 2:
            return token
        if method == "hybrid":
            # Preserve tokens which look like common single words. Reject splits
            # containing rare fragments to reduce (not eliminate) false positives.
            if wordfreq(token, "en") >= 3.0:
                return token
            if any(wordfreq(piece, "en") < 2.0 for piece in pieces):
                return token
        return " ".join(pieces)

    return re.sub(r"\b[A-Za-z]{%d,}\b" % min_len, split_token, text)


@lru_cache(maxsize=8)
def _get_symspell(dictionary_path: str | None, max_edit_distance: int = 2):
    """Load SymSpell from a user-provided term-frequency dictionary."""
    if not dictionary_path:
        raise ValueError(
            "Set symspell_dictionary_path to a SymSpell term-frequency dictionary."
        )
    try:
        from symspellpy import SymSpell
    except ImportError as exc:
        raise ImportError("Install symspellpy to use SymSpell operations.") from exc
    path = Path(dictionary_path)
    if not path.is_file():
        raise FileNotFoundError(f"SymSpell dictionary not found: {path}")
    spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    if not spell.load_dictionary(str(path), term_index=0, count_index=1):
        raise ValueError(f"Could not load SymSpell dictionary: {path}")
    return spell


def correct_spelling(
    text: str,
    dictionary_path: str | None = None,
    max_edit_distance: int = 2,
    protected_words: Collection[str] | None = None,
) -> str:
    """Correct eligible alphabetic tokens with SymSpell, protecting negation/domain terms."""
    spell = _get_symspell(dictionary_path, max_edit_distance)
    try:
        from symspellpy import Verbosity
    except ImportError as exc:
        raise ImportError("Install symspellpy to use spelling correction.") from exc

    protected = set(SENTIMENT_PROTECTED_WORDS) | set(NEGATION_WORDS)
    protected |= {word.casefold() for word in (protected_words or set())}
    protected |= {_compact(alias) for alias in AIRLINE_ALIASES}

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        key = word.casefold()
        if key in protected or len(key) < 4:
            return word
        suggestions = spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
        if not suggestions:
            return word
        candidate = suggestions[0].term
        if word.isupper():
            return candidate.upper()
        if word.istitle():
            return candidate.title()
        return candidate

    return re.sub(r"\b[A-Za-z]+\b", replace, text)


@lru_cache(maxsize=2)
def _load_spacy(model: str):
    try:
        import spacy
    except ImportError as exc:
        raise ImportError("Install spacy to use lemmatization or stopword removal.") from exc
    try:
        return spacy.load(model)
    except OSError as exc:
        raise OSError(
            "spaCy model '" + model + "' is not installed. In Colab run: "
            "python -m spacy download " + model
        ) from exc


def lemmatize_text(
    text: str,
    model: str = "en_core_web_sm",
    preserve_negation: bool = True,
) -> str:
    """Lemmatize with spaCy and preserve negation words verbatim by default."""
    nlp = _load_spacy(model)
    protected = NEGATION_WORDS if preserve_negation else set()
    doc = nlp(text)
    output = []
    for token in doc:
        if token.is_space:
            continue
        if token.is_punct:
            output.append(token.text)
        elif token.text.casefold() in protected:
            output.append(token.text)
        else:
            lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
            output.append(lemma)
    return " ".join(output)


@lru_cache(maxsize=2)
def _load_stopwords(model: str) -> set[str]:
    return {word.casefold() for word in _load_spacy(model).Defaults.stop_words}


def remove_stopwords(
    text: str,
    model: str = "en_core_web_sm",
    preserve_negation: bool = True,
    protected_words: Collection[str] | None = None,
) -> str:
    """Remove spaCy stopwords while preserving configurable sentiment terms.

    By default, configured sentiment/intensifier words are protected. Negation
    terms are additionally protected when preserve_negation=True. Callers can
    add experiment-specific protected words without editing processing logic.
    """
    stopwords = _load_stopwords(model)
    protected = set(STOPWORD_PROTECTED_WORDS)
    if not preserve_negation:
        protected -= set(NEGATION_WORDS)
    protected |= {word.casefold() for word in (protected_words or set())}
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    return " ".join(
        token for token in tokens
        if token.casefold() not in stopwords or token.casefold() in protected
    )
