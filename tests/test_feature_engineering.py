"""L3 VADER extraction tests; run with pytest and vaderSentiment==3.3.2."""
import pandas as pd
import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from feature_engineering_config import select_vader_columns
from feature_engineering import (
    extract_vader_scores, extract_vader_frame, select_vader_frame,
    load_raw_text_only, raw_csv_fingerprint,
)

@pytest.mark.parametrize("variant,expected", [
    ("V0", ()), ("V1", ("V_compound",)),
    ("V2", ("V_pos", "V_neg", "V_compound")),
])
def test_variant_whitelist(variant, expected):
    assert select_vader_columns(variant) == expected

def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        select_vader_columns("V3")

@pytest.mark.parametrize("text", ["", "THANK YOU!!!", "Terrible service 😡", "not bad", "10/10 service"])
def test_exact_library_scores(text):
    expected = SentimentIntensityAnalyzer().polarity_scores(text)
    actual = extract_vader_scores(text)
    assert actual == {"V_pos": expected["pos"], "V_neg": expected["neg"],
                      "V_neu": expected["neu"], "V_compound": expected["compound"]}
    assert -1 <= actual["V_compound"] <= 1
    assert all(0 <= actual[k] <= 1 for k in ("V_pos", "V_neg", "V_neu"))
    assert abs(sum(actual[k] for k in ("V_pos", "V_neg", "V_neu")) - 1) <= .0011

@pytest.mark.parametrize("invalid", [None, float("nan"), 3, ["hello"]])
def test_invalid_text_rejected(invalid):
    with pytest.raises(TypeError):
        extract_vader_scores(invalid)

def test_schema_order_and_variants():
    frame = extract_vader_frame(["Amazing!", "awful", ""])
    assert frame.columns.tolist() == ["row_id", "V_pos", "V_neg", "V_neu", "V_compound"]
    assert frame.row_id.tolist() == [0, 1, 2]
    assert select_vader_frame(frame, "V0").shape == (3, 0)
    assert select_vader_frame(frame, "V1").columns.tolist() == ["V_compound"]
    assert select_vader_frame(frame, "V2").columns.tolist() == ["V_pos", "V_neg", "V_compound"]

def test_label_columns_not_loaded(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"text": ["GOOD", "bad"], "airline_sentiment": ["positive", "negative"],
                  "airline_sentiment_confidence": [.99, .98]}).to_csv(path, index=False)
    assert load_raw_text_only(path) == ["GOOD", "bad"]
    before = raw_csv_fingerprint(path)["source_sha256"]
    path.write_text(path.read_text() + "\n")
    assert before != raw_csv_fingerprint(path)["source_sha256"]

def test_missing_text_rejected(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"text": ["valid", None]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_raw_text_only(path)
