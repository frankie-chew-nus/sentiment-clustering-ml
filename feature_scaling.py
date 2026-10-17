"""L4-M2 transformations of existing engineered features, without labels.

Fit scaling on the designated analysis corpus. Transform other data using the
same fitted references. No clustering, block weighting, or feature extraction.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd

from feature_scaling_config import (
    FEATURE_SETS, COUNT_COLUMNS, BOUNDED_COLUMNS, ScalingSpec
)

FORBIDDEN = ("airline_sentiment", "sentiment_confidence",
             "airline_sentiment_confidence")

def select_columns(*variants):
    """Return approved columns, rejecting duplicate or repeated families."""
    columns, families = [], set()
    for variant in variants:
        if variant not in FEATURE_SETS:
            raise ValueError(f"Unknown feature variant: {variant}")
        family = variant[0]
        if family in families:
            raise ValueError(f"Repeated feature family: {family}")
        families.add(family)
        columns.extend(FEATURE_SETS[variant])
    return tuple(columns)

def validate_master(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if any(c in FORBIDDEN or "sentiment_confidence" in c.lower()
           for c in frame.columns):
        raise ValueError("Sentiment label/confidence columns are forbidden")
    if "row_id" not in frame:
        raise ValueError("Missing row_id")
    ids = frame["row_id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(frame))):
        raise ValueError("row_id must be sequential and aligned")
    return frame

class EngineeredFeatureScaler:
    """Apply log1p to counts, optionally followed by bounded or standard scaling.

    Bounded method uses the configured percentile of nonzero transformed values
    when the overall percentile is zero; values above reference are clipped to 1.
    Native VADER/uppercase-ratio columns remain unchanged in all methods.
    """
    def __init__(self, variants=("V1", "N1", "S2"), spec=None):
        self.variants = tuple(variants)
        self.spec = spec if spec is not None else ScalingSpec()
        self.columns = select_columns(*self.variants)
        self.references_ = None

    def _values(self, frame):
        validate_master(frame)
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing selected columns: {sorted(missing)}")
        data = frame.loc[:, self.columns].apply(pd.to_numeric, errors="raise")
        values = data.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite feature values")
        for j, col in enumerate(self.columns):
            if col in COUNT_COLUMNS:
                if (values[:, j] < 0).any():
                    raise ValueError(f"Negative count in {col}")
                values[:, j] = np.log1p(values[:, j])
            elif col not in BOUNDED_COLUMNS:
                raise ValueError(f"Unrecognized engineered feature: {col}")
        return values

    def fit(self, frame):
        values = self._values(frame)
        references = {}
        for j, col in enumerate(self.columns):
            if col not in COUNT_COLUMNS:
                continue
            x = values[:, j]
            if self.spec.method == "bounded":
                reference = float(np.percentile(x, self.spec.percentile))
                if reference <= 0 and self.spec.positive_fallback:
                    positive = x[x > 0]
                    reference = (float(np.percentile(positive, self.spec.percentile))
                                 if len(positive) else 0.0)
                references[col] = {"reference": reference}
            elif self.spec.method == "standard":
                references[col] = {"mean": float(x.mean()),
                                   "std": float(x.std(ddof=0))}
            else:
                references[col] = {}
        self.references_ = references
        return self

    def transform(self, frame):
        if self.references_ is None:
            raise RuntimeError("Fit the scaler before transform")
        values = self._values(frame)
        for j, col in enumerate(self.columns):
            if col not in COUNT_COLUMNS:
                continue
            reference = self.references_[col]
            if self.spec.method == "bounded":
                scale = reference["reference"]
                values[:, j] = values[:, j] / scale if scale > 0 else 0.0
                if self.spec.clip:
                    values[:, j] = np.clip(values[:, j], 0.0, 1.0)
            elif self.spec.method == "standard":
                std = reference["std"]
                values[:, j] = ((values[:, j] - reference["mean"]) / std
                                if std > 0 else 0.0)
        return values

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def manifest(self):
        if self.references_ is None:
            raise RuntimeError("Scaler not fitted")
        return {"variants": list(self.variants), "columns": list(self.columns),
                "spec": asdict(self.spec), "references": self.references_}
