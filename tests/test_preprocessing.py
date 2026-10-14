"""Unit tests for Group 1 preprocessing."""

import pytest

from config import P0_CONFIG
from preprocessing import (
    normalize_whitespace,
    preprocess_text,
    remove_airline_hashtags,
    remove_mentions,
    remove_urls,
    strip_hashtag_symbols,
)


def test_remove_urls_preserves_sentence_punctuation():
    assert remove_urls("Look here https://example.com/path.") == "Look here."


def test_remove_mentions():
    assert remove_mentions("Hi @united and @someone!") == "Hi  and !"


def test_remove_only_exact_airline_hashtags():
    text = "#United #WorstFlightEver #DeltaAirlines"
    assert remove_airline_hashtags(text) == " #WorstFlightEver "


def test_ordinary_airline_names_are_preserved():
    assert preprocess_text("United delayed my flight") == "United delayed my flight"


def test_other_hashtag_words_are_preserved_without_hash():
    assert preprocess_text("#WorstFlightEver #Happy") == "WorstFlightEver Happy"


def test_group1_example():
    text = "@united United delayed my flight again! #United #WorstFlightEver"
    assert preprocess_text(text) == "United delayed my flight again! WorstFlightEver"


def test_normalize_whitespace():
    assert normalize_whitespace("  too   many\n spaces  ") == "too many spaces"


def test_invalid_input_raises_type_error():
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]


def test_config_can_disable_mentions_and_urls():
    config = dict(P0_CONFIG)
    config["remove_mentions"] = False
    config["remove_urls"] = False
    assert preprocess_text("@user https://example.com", config) == "@user https://example.com"


def test_strip_hashtag_symbols_standalone():
    assert strip_hashtag_symbols("#good #bad-service") == "good bad-service"
