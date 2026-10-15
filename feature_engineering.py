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
