"""L3 RAW VADER extraction. Ground-truth labels are never loaded."""
from __future__ import annotations
from collections.abc import Sequence
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import hashlib
from feature_engineering_config import VADER_COLUMNS, VADER_VERSION, select_vader_columns

@lru_cache(maxsize=1)
def _analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError("Install vaderSentiment==3.3.2") from exc
    installed = version("vaderSentiment")
    if installed != VADER_VERSION:
        raise RuntimeError(f"Expected vaderSentiment=={VADER_VERSION}; found {installed}")
    return SentimentIntensityAnalyzer()

def extract_vader_scores(raw_text: str) -> dict[str, float]:
    """Calculate all four unscaled VADER scores on original RAW text."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string; missing values are not silently filled")
    s = _analyzer().polarity_scores(raw_text)
    return {"V_pos": s["pos"], "V_neg": s["neg"],
            "V_neu": s["neu"], "V_compound": s["compound"]}

def extract_vader_frame(raw_texts: Sequence[str]):
    """Produce row_id and all four scores, preserving original order."""
    import pandas as pd
    if isinstance(raw_texts, (str, bytes)):
        raise TypeError("raw_texts must be a sequence, not a string")
    rows = [{"row_id": i, **extract_vader_scores(t)} for i, t in enumerate(raw_texts)]
    return pd.DataFrame.from_records(rows, columns=["row_id", *VADER_COLUMNS]).astype(
        {"row_id": "int64", **{c: "float64" for c in VADER_COLUMNS}}
    )

def select_vader_frame(master, variant: str):
    """Return V0/V1/V2 features; row_id remains metadata, not model input."""
    columns = select_vader_columns(variant)
    missing = [c for c in columns if c not in master.columns]
    if missing:
        raise ValueError(f"Missing VADER columns: {missing}")
    return master.loc[:, list(columns)].copy()

def raw_csv_fingerprint(path: str | Path) -> dict[str, object]:
    """Hash exact source bytes to guard sequential row alignment."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"source_filename": path.name, "source_sha256": digest.hexdigest(),
            "source_bytes": path.stat().st_size}

def load_raw_text_only(path: str | Path):
    """Whitelist RAW text; never load airline_sentiment or confidence."""
    import pandas as pd
    raw = pd.read_csv(path, usecols=["text"], dtype={"text": "string"})
    if raw["text"].isna().any():
        raise ValueError("RAW text contains null values; resolve before extraction")
    return raw["text"].tolist()

# L3-N: P1 lexical modifier features. Keep preprocessing functions unchanged.
import re as _re
from config import NEGATION_WORDS, INTENSIFIER_WORDS
from feature_engineering_config import (
    CONSERVATIVE_INTENSIFIER_WORDS, N_COLUMNS, select_n_columns,
)

_N_TOKEN_RE = _re.compile(r"[a-z]+(?:'[a-z]+)?", flags=_re.IGNORECASE)

def extract_n_scores(p1_text: str) -> dict[str, int]:
    """Count exact P1 tokens; never infer sentiment polarity or use labels."""
    if not isinstance(p1_text, str):
        raise TypeError("p1_text must be a string; nulls must be resolved explicitly")
    tokens = _N_TOKEN_RE.findall(p1_text.casefold())
    return {
        "N_negation_count": sum(t in NEGATION_WORDS for t in tokens),
        "N_intensifier_original_count": sum(t in INTENSIFIER_WORDS for t in tokens),
        "N_intensifier_conservative_count": sum(
            t in CONSERVATIVE_INTENSIFIER_WORDS for t in tokens
        ),
    }

def extract_n_frame(p1_texts: Sequence[str]):
    """Preserve P1 row order and create a feature-only N block."""
    import pandas as pd
    if isinstance(p1_texts, (str, bytes)):
        raise TypeError("p1_texts must be a sequence, not a string")
    rows = [{"row_id": i, **extract_n_scores(t)} for i, t in enumerate(p1_texts)]
    return pd.DataFrame.from_records(rows, columns=["row_id", *N_COLUMNS]).astype(
        {"row_id": "int64", **{c: "int64" for c in N_COLUMNS}}
    )

def select_n_frame(master, variant: str):
    """Select N0-N3; do not expose row_id as a clustering feature."""
    columns = select_n_columns(variant)
    missing = [c for c in columns if c not in master.columns]
    if missing:
        raise ValueError(f"Missing N columns: {missing}")
    return master.loc[:, list(columns)].copy()

def load_p1_text_only(path: str | Path):
    """Read only P1 processed_text, never ground-truth sentiment columns."""
    import pandas as pd
    df = pd.read_csv(path, usecols=["processed_text"], dtype={"processed_text": "string"})
    if df["processed_text"].isna().any():
        raise ValueError("P1 processed_text contains nulls; resolve before extraction")
    return df["processed_text"].tolist()

# L3-S: boundary-aware URL exclusion for structural features only.
from feature_engineering_config import S_COLUMNS, select_s_columns

_S_URL_RE = _re.compile(
    r"(?i)(?<![\w@])(?:https?://|www\.)[^\s<>\"']+"
    r"|(?<![\w@])(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|co|io|ly|me|info|biz|us|uk|sg)(?:/[^\s<>\"']*)?"
)
_S_URL_TRAILING_PUNCT = ".,!?;:)]}"
_S_URL_STOP_CHARS = frozenset('“”‘’«»‹›<>')

def _s_emoji_list(text: str):
    """Use emoji package sequence matching, as in S0-C audit."""
    import emoji
    return emoji.emoji_list(text)

def strip_s_urls(raw_text: str) -> str:
    """S0-C audit rule: remove URL spans but preserve adjacent emoji/punctuation."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    spans = [(e["match_start"], e["match_end"]) for e in _s_emoji_list(raw_text)]
    edits = []
    for match in _S_URL_RE.finditer(raw_text):
        start, end = match.span()
        for emoji_start, _ in spans:
            if start < emoji_start < end:
                end = min(end, emoji_start)
                break
        for i in range(start, end):
            if raw_text[i] in _S_URL_STOP_CHARS:
                end = i
                break
        while end > start and raw_text[end - 1] in _S_URL_TRAILING_PUNCT:
            end -= 1
        if end > start:
            edits.append((start, end))
    for start, end in reversed(edits):
        raw_text = raw_text[:start] + raw_text[end:]
    return raw_text

def extract_s_scores(raw_text: str) -> dict[str, float | int]:
    """Extract structural expression from temporary URL-excluded RAW text."""
    cleaned = strip_s_urls(raw_text)
    eligible = [
        token for token in _re.findall(r"\b[^\W\d_]+\b", cleaned, flags=_re.UNICODE)
        if len(token) >= 2 and token.isalpha()
    ]
    uppercase_ratio = (
        sum(token.isupper() for token in eligible) / len(eligible) if eligible else 0.0
    )
    return {
        "S_exclamation_count": cleaned.count("!"),
        "S_uppercase_ratio": uppercase_ratio,
        "S_emoji_count": len(_s_emoji_list(cleaned)),
    }

def extract_s_frame(raw_texts: Sequence[str]):
    """Return row-aligned S feature block; no labels or confidence fields."""
    import pandas as pd
    if isinstance(raw_texts, (str, bytes)):
        raise TypeError("raw_texts must be a sequence, not a string")
    rows = [{"row_id": i, **extract_s_scores(t)} for i, t in enumerate(raw_texts)]
    return pd.DataFrame.from_records(rows, columns=["row_id", *S_COLUMNS]).astype(
        {"row_id": "int64", "S_exclamation_count": "int64",
         "S_uppercase_ratio": "float64", "S_emoji_count": "int64"}
    )

def select_s_frame(master, variant: str):
    """Select S0-S3; row_id is alignment metadata, never a model feature."""
    columns = select_s_columns(variant)
    missing = [c for c in columns if c not in master.columns]
    if missing:
        raise ValueError(f"Missing S columns: {missing}")
    return master.loc[:, list(columns)].copy()
