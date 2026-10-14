"""Unit tests for Group 1 and Group 2 preprocessing."""

import pytest

from config import P0_CONFIG, P1_CONFIG, P2_CONFIG
from preprocessing import (
    convert_numbers,
    expand_contractions,
    handle_emojis,
    lowercase_text,
    normalize_whitespace,
    preprocess_text,
    remove_mentions,
    remove_punctuation,
    remove_urls,
    replace_airline_hashtags,
    replace_airline_mentions,
    replace_airline_names,
    strip_hashtag_symbols,
    split_compound_words,
    correct_spelling,
    lemmatize_text,
    remove_stopwords,
)


def test_remove_urls_preserves_sentence_punctuation():
    assert remove_urls("Look here https://example.com/path.") == "Look here."


def test_airline_mentions_become_generic_token():
    assert replace_airline_mentions("@united delayed my flight") == "airline delayed my flight"


def test_american_air_handle_is_recognized():
    assert replace_airline_mentions("@AmericanAir delayed my flight") == "airline delayed my flight"


def test_virgin_atlantic_handle_is_recognized():
    assert replace_airline_mentions("@VirginAtlantic please help") == "airline please help"


def test_non_airline_mentions_are_removed():
    assert remove_mentions("Hi @someone!") == "Hi !"


def test_exact_airline_hashtags_become_generic_token():
    assert replace_airline_hashtags("#United #WorstFlightEver #DeltaAirlines") == "airline WorstFlightEver airline"


def test_airline_sentiment_hashtag_suffix_is_preserved():
    assert replace_airline_hashtags("#usairwaysfail #unitedsucks") == "airline fail airline sucks"


def test_ordinary_airline_names_become_generic_token():
    assert replace_airline_names("United delayed my flight") == "airline delayed my flight"


def test_explicit_american_airlines_name_becomes_generic_token():
    assert replace_airline_names("American Airlines cancelled my flight") == "airline cancelled my flight"


def test_multiword_airline_names_become_generic_token():
    assert replace_airline_names("Delta Air Lines and Virgin America") == "airline and airline"


def test_other_airline_names_in_sentence_are_replaced():
    assert replace_airline_names("I flew with JetBlue and Southwest Airlines") == "I flew with airline and airline"


def test_non_airline_hashtags_keep_content():
    assert preprocess_text("#WorstFlightEver #Happy") == "WorstFlightEver Happy"


def test_full_group1_example():
    text = "@united United delayed my flight again! #United #WorstFlightEver"
    assert preprocess_text(text) == "airline airline delayed my flight again! airline WorstFlightEver"


def test_full_group1_preserves_sentiment_suffix():
    assert preprocess_text("@USAirways #usairwaysfail") == "airline airline fail"


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


# Group 2 tests
def test_expand_negation_contractions():
    assert expand_contractions("I can't fly because they don't help") == "I can not fly because they do not help"


def test_expand_common_tweet_spellings_without_apostrophes():
    assert expand_contractions("i cant fly, we dont know") == "i can not fly, we do not know"


def test_expand_contractions_before_punctuation_removal():
    config = {**P1_CONFIG, "emoji_mode": "keep", "convert_numbers": False}
    assert preprocess_text("I can't fly!!!", config) == "i can not fly"


def test_lowercase_text():
    assert lowercase_text("TERRIBLE Service") == "terrible service"


def test_remove_punctuation_preserves_emoji_characters():
    assert remove_punctuation("bad!!! 😡") == "bad    😡"


def test_emoji_keep_mode():
    assert handle_emojis("great 😊", mode="keep") == "great 😊"


def test_emoji_remove_mode():
    assert handle_emojis("great 😊", mode="remove") == "great "


def test_emoji_describe_mode():
    result = handle_emojis("bad 😡", mode="describe").lower()
    # Emoji labels can vary by emoji-package version (e.g. "enraged face").
    assert "😡" not in result
    assert any(label in result for label in ("angry face", "enraged face"))


def test_number_conversion():
    assert convert_numbers("3 hours") == "three hours"


def test_number_conversion_does_not_change_times_or_ranges():
    assert convert_numbers("at 1:30, wait 1-4am, flight N351JB") == "at 1:30, wait 1-4am, flight N351JB"


def test_p1_configuration_expands_numbers_and_normalizes_text():
    config = {**P1_CONFIG, "emoji_mode": "keep"}
    result = preprocess_text("I can't wait 3 hours!!!", config)
    assert result == "i can not wait three hours"


def test_punctuation_mode_validation():
    with pytest.raises(ValueError):
        preprocess_text("test", {**P1_CONFIG, "punctuation_mode": "invalid"})


def test_emoji_mode_validation():
    with pytest.raises(ValueError):
        handle_emojis("test", mode="invalid")



def test_segmentation_none_is_identity():
    assert split_compound_words("greatjob worstflightever", method="none") == "greatjob worstflightever"


def test_segmentation_rejects_unknown_method():
    with pytest.raises(ValueError):
        split_compound_words("greatjob", method="unknown")


def test_wordninja_segmentation():
    result = split_compound_words("greatjob worstflightever", method="wordninja", min_token_length=7)
    assert "great job" in result
    assert "worst flight ever" in result


def test_symspell_segmentation_requires_dictionary():
    with pytest.raises(ValueError, match="symspell_dictionary_path"):
        split_compound_words("greatjob", method="symspell", symspell_dictionary_path=None)


def test_spelling_correction_requires_dictionary():
    with pytest.raises(ValueError, match="symspell_dictionary_path"):
        correct_spelling("terribl", dictionary_path=None)


def test_spacy_missing_model_has_actionable_error():
    with pytest.raises(OSError, match="spaCy model"):
        lemmatize_text("delayed flights", model="model_that_does_not_exist")


def test_stopword_removal_requires_spacy_model():
    with pytest.raises(OSError, match="spaCy model"):
        remove_stopwords("this is not good", model="model_that_does_not_exist", preserve_negation=True)


def test_p2_is_conservative_by_default():
    assert P2_CONFIG["segmentation_mode"] == "none"
    assert P2_CONFIG["correct_spelling"] is False
    assert P2_CONFIG["lemmatize"] is False
    assert P2_CONFIG["remove_stopwords"] is False


def test_strip_hashtag_symbols_standalone():
    assert strip_hashtag_symbols("#good #bad-service") == "good bad-service"
