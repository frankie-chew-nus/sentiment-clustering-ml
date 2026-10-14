"""Unit tests for Group 1 preprocessing."""

import pytest

from config import P0_CONFIG
from preprocessing import (
    normalize_whitespace,
    preprocess_text,
    remove_mentions,
    remove_urls,
    replace_airline_hashtags,
    replace_airline_mentions,
    replace_airline_names,
    strip_hashtag_symbols,
)


def test_remove_urls_preserves_sentence_punctuation():
    assert remove_urls("Look here https://example.com/path.") == "Look here."


def test_airline_mentions_become_generic_token():
    assert replace_airline_mentions("@united delayed my flight") == "airline delayed my flight"


def test_non_airline_mentions_are_removed():
    assert remove_mentions("Hi @someone!") == "Hi !"


def test_airline_hashtags_become_generic_token():
    assert replace_airline_hashtags("#United #WorstFlightEver #DeltaAirlines") == "airline WorstFlightEver airline"


def test_ordinary_airline_names_become_generic_token():
    assert replace_airline_names("United delayed my flight") == "airline delayed my flight"


def test_multiword_airline_names_become_generic_token():
    assert replace_airline_names("Delta Air Lines and Virgin America") == "airline and airline"


def test_other_airline_names_in_sentence_are_replaced():
    assert replace_airline_names("I flew with JetBlue and Southwest Airlines") == "I flew with airline and airline"


def test_non_airline_hashtags_keep_content():
    assert preprocess_text("#WorstFlightEver #Happy") == "WorstFlightEver Happy"


def test_full_group1_example():
    text = "@united United delayed my flight again! #United #WorstFlightEver"
    assert preprocess_text(text) == "airline airline delayed my flight again! airline WorstFlightEver"


def test_normalize_whitespace():
    assert normalize_whitespace("  too   many\n spaces  ") == "too many spaces"


def test_invalid_input_raises_type_error():
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]


def test_config_can_disable_mention_and_url_processing():
    config = dict(P0_CONFIG)
    config["replace_airline_mentions"] = False
    config["remove_other_mentions"] = False
    config["remove_urls"] = False
    assert preprocess_text("@united https://example.com", config) == "@united https://example.com"


def test_strip_hashtag_symbols_standalone():
    assert strip_hashtag_symbols("#good #bad-service") == "good bad-service"
