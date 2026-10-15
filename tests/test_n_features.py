"""L3-N tests: P1 modifier counts and explicit variant selection."""
import pandas as pd
import pytest

from feature_engineering_config import (
    CONSERVATIVE_INTENSIFIER_WORDS, select_n_columns,
)
from config import INTENSIFIER_WORDS
from feature_engineering import (
    extract_n_scores, extract_n_frame, select_n_frame, load_p1_text_only,
)

@pytest.mark.parametrize("variant,expected", [
    ("N0", ()),
    ("N1", ("N_negation_count",)),
    ("N2", ("N_negation_count", "N_intensifier_original_count")),
    ("N3", ("N_negation_count", "N_intensifier_conservative_count")),
])
def test_variant_columns(variant, expected):
    assert select_n_columns(variant) == expected

def test_unknown_variant():
    with pytest.raises(ValueError):
        select_n_columns("N4")

def test_vocabularies_are_consistent():
    assert len(INTENSIFIER_WORDS) == 13
    assert len(CONSERVATIVE_INTENSIFIER_WORDS) == 8
    assert CONSERVATIVE_INTENSIFIER_WORDS.issubset(INTENSIFIER_WORDS)

@pytest.mark.parametrize("text,expected", [
    ("i can not go", (1, 0, 0)),
    ("not never no", (3, 0, 0)),
    ("really very good", (0, 2, 2)),
    ("so bad", (0, 1, 0)),
    ("me too", (0, 1, 0)),
    ("pretty frustrating quite bad", (0, 2, 0)),
    ("not really very good", (1, 2, 2)),
    ("nothingness nobody", (1, 0, 0)),
    ("", (0, 0, 0)),
])
def test_counts(text, expected):
    actual = extract_n_scores(text)
    assert tuple(actual.values()) == expected

@pytest.mark.parametrize("bad", [None, float("nan"), 2, ["not"]])
def test_invalid_text(bad):
    with pytest.raises(TypeError):
        extract_n_scores(bad)

def test_frame_and_selection():
    df = extract_n_frame(["not really good", "so bad", ""])
    assert df.columns.tolist() == ["row_id", "N_negation_count",
                                  "N_intensifier_original_count",
                                  "N_intensifier_conservative_count"]
    assert df.row_id.tolist() == [0, 1, 2]
    assert select_n_frame(df, "N0").shape == (3, 0)
    assert select_n_frame(df, "N1").shape == (3, 1)
    assert select_n_frame(df, "N2").shape == (3, 2)
    assert select_n_frame(df, "N3").shape == (3, 2)

def test_empty_frame():
    df = extract_n_frame([])
    assert df.shape == (0, 4)

def test_label_columns_never_loaded(tmp_path):
    path = tmp_path / "p1.csv"
    pd.DataFrame({"processed_text": ["not good", "really great"],
                  "airline_sentiment": ["negative", "positive"],
                  "airline_sentiment_confidence": [0.99, 0.9]}).to_csv(path, index=False)
    assert load_p1_text_only(path) == ["not good", "really great"]

def test_null_p1_rejected(tmp_path):
    path = tmp_path / "p1.csv"
    pd.DataFrame({"processed_text": ["valid", None]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_p1_text_only(path)
