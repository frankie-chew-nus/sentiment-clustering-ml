"""L4-M2 label-blind engineered-feature scaling tests."""
import numpy as np
import pandas as pd
import pytest
from feature_scaling import EngineeredFeatureScaler, select_columns
from feature_scaling_config import ScalingSpec

@pytest.fixture
def frame():
    return pd.DataFrame({
        "row_id": [0, 1, 2, 3, 4],
        "V_compound": [-0.8, 0.0, 0.6, 0.1, -0.3],
        "N_negation_count": [0, 0, 1, 2, 4],
        "S_exclamation_count": [0, 0, 0, 3, 27],
        "S_uppercase_ratio": [0, 0.2, 0, 0.5, 0.1],
        "S_emoji_count": [0, 0, 0, 0, 40],
    })

def test_columns():
    assert select_columns("V1", "N1", "S2") == (
        "V_compound", "N_negation_count", "S_exclamation_count",
        "S_uppercase_ratio")
    with pytest.raises(ValueError, match="Repeated"):
        select_columns("N1", "N2")

@pytest.mark.parametrize("method", ["bounded", "log_only", "standard"])
def test_fit_transform_finite(frame, method):
    scaler = EngineeredFeatureScaler(("V1", "N1", "S3"),
                                    ScalingSpec(method=method))
    x = scaler.fit_transform(frame)
    assert x.shape == (5, 5)
    assert np.isfinite(x).all()
    assert np.array_equal(x, scaler.transform(frame))
    assert np.array_equal(x[:, 0], frame["V_compound"])
    assert np.array_equal(x[:, 3], frame["S_uppercase_ratio"])

def test_bounded_and_clipped(frame):
    scaler = EngineeredFeatureScaler(("N1", "S3"))
    x = scaler.fit_transform(frame)
    assert np.min(x) >= 0
    assert np.max(x) <= 1
    assert scaler.manifest()["references"]["S_emoji_count"]["reference"] > 0

def test_zero_constant_count(frame):
    frame["N_negation_count"] = 0
    x = EngineeredFeatureScaler(("N1",)).fit_transform(frame)
    assert np.all(x == 0)

def test_label_leakage(frame):
    frame["airline_sentiment"] = ["negative"] * len(frame)
    with pytest.raises(ValueError, match="Sentiment"):
        EngineeredFeatureScaler().fit(frame)

def test_row_alignment(frame):
    frame.loc[2, "row_id"] = 99
    with pytest.raises(ValueError, match="row_id"):
        EngineeredFeatureScaler().fit(frame)

def test_invalid_counts(frame):
    frame.loc[0, "N_negation_count"] = -1
    with pytest.raises(ValueError, match="Negative count"):
        EngineeredFeatureScaler(("N1",)).fit(frame)

def test_fit_required(frame):
    with pytest.raises(RuntimeError, match="Fit"):
        EngineeredFeatureScaler().transform(frame)

def test_percentile_validation():
    with pytest.raises(ValueError, match="percentile"):
        ScalingSpec(percentile=0)
