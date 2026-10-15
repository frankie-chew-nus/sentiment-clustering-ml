"""L3-S: URL-excluded RAW structural-expression feature tests."""
import pandas as pd
import pytest
from feature_engineering_config import select_s_columns
from feature_engineering import (
    strip_s_urls, extract_s_scores, extract_s_frame, select_s_frame,
    load_raw_text_only,
)

@pytest.mark.parametrize("variant,expected", [
    ("S0", ()),
    ("S1", ("S_exclamation_count",)),
    ("S2", ("S_exclamation_count", "S_uppercase_ratio")),
    ("S3", ("S_exclamation_count", "S_uppercase_ratio", "S_emoji_count")),
])
def test_variants(variant, expected):
    assert select_s_columns(variant) == expected

def test_unknown_variant():
    with pytest.raises(ValueError):
        select_s_columns("S4")

def test_exclamation_and_uppercase():
    s = extract_s_scores("I LOVE this AIRLINE!!!")
    assert s["S_exclamation_count"] == 3
    assert s["S_uppercase_ratio"] == pytest.approx(2 / 3)
    assert s["S_emoji_count"] == 0

def test_no_eligible_tokens():
    assert extract_s_scores("I !")["S_uppercase_ratio"] == 0.0

def test_urls_do_not_inflate_uppercase():
    s = extract_s_scores("TERRIBLE service!!! https://EXAMPLE.com/HELP")
    assert s["S_exclamation_count"] == 3
    assert s["S_uppercase_ratio"] == pytest.approx(0.5)

def test_emoji_adjacent_to_url():
    raw = "http://t.co/cRFrwpc1Sx”😂😂😂"
    assert strip_s_urls(raw) == "”😂😂😂"
    assert extract_s_scores(raw)["S_emoji_count"] == 3

def test_emoji_after_angle_bracket():
    raw = "http://t.co/NfjAuW16vZ<--....😕"
    assert strip_s_urls(raw) == "<--....😕"
    assert extract_s_scores(raw)["S_emoji_count"] == 1

def test_terminal_punctuation_retained():
    assert strip_s_urls("Great! https://example.com!") == "Great! !"

def test_no_url_preserves_expression():
    raw = "WOW!!! 😊"
    assert strip_s_urls(raw) == raw
    assert extract_s_scores(raw) == {"S_exclamation_count": 3,
                                     "S_uppercase_ratio": 1.0,
                                     "S_emoji_count": 1}

@pytest.mark.parametrize("bad", [None, 3, float("nan"), ["hello"]])
def test_invalid_raw(bad):
    with pytest.raises(TypeError):
        extract_s_scores(bad)

def test_frame_schema_and_variants():
    df = extract_s_frame(["WOW!", "okay 😊", ""])
    assert df.columns.tolist() == ["row_id", "S_exclamation_count",
                                    "S_uppercase_ratio", "S_emoji_count"]
    assert df.row_id.tolist() == [0, 1, 2]
    assert select_s_frame(df, "S0").shape == (3, 0)
    assert select_s_frame(df, "S1").shape == (3, 1)
    assert select_s_frame(df, "S2").shape == (3, 2)
    assert select_s_frame(df, "S3").shape == (3, 3)

def test_empty_frame():
    assert extract_s_frame([]).shape == (0, 4)

def test_only_raw_text_loaded(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({
        "text": ["THANK YOU!!!", "bad 😡"],
        "airline_sentiment": ["positive", "negative"],
        "airline_sentiment_confidence": [0.99, 0.99],
    }).to_csv(path, index=False)
    assert load_raw_text_only(path) == ["THANK YOU!!!", "bad 😡"]
