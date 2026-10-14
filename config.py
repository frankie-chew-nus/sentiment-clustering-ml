"""Configuration for airline tweet preprocessing.

P0: Group 1 basic cleaning.
P1: P0 + Group 2 text normalization.
P2: P1 + configurable Group 3 lexical normalization.

Group 3 operations remain configurable because segmentation and spelling
correction can introduce errors. P2 starts conservatively while these options
are benchmarked on this dataset.
"""

AIRLINE_ALIASES = frozenset({
    "united", "united airlines", "unitedairlines", "united_airlines",
    "delta", "delta air lines", "delta airlines", "deltaairlines", "delta_air_lines",
    "american", "american air", "americanair", "american airlines", "americanairlines", "american_airlines",
    "us airways", "usairways", "usair", "us_airways",
    "southwest", "southwest airlines", "southwestair", "southwestairlines", "southwest_airlines",
    "jetblue", "jet blue",
    "virgin america", "virginamerica", "virgin_america",
    "virgin atlantic", "virginatlantic", "virgin_atlantic", "virgin",
})

AIRLINE_HASHTAG_ALIASES = AIRLINE_ALIASES

AIRLINE_HASHTAG_SUFFIX_RULES = {
    "fail", "fails", "failure", "sucks", "suck", "sux",
    "hate", "hates", "terrible", "awful", "bad", "worst",
    "delay", "delays", "delayed", "cancel", "cancelled", "canceled",
    "disappointed", "unacceptable", "poorservice", "badservice",
}

P0_CONFIG = {
    "remove_urls": True,
    "replace_airline_mentions": True,
    "remove_other_mentions": True,
    "replace_airline_hashtags": True,
    "replace_airline_names": True,
    "strip_hashtag_symbols": True,
    "normalize_whitespace": True,
}

P1_CONFIG = {
    **P0_CONFIG,
    "expand_contractions": True,
    "convert_numbers": True,
    "lowercase": True,
    "punctuation_mode": "remove",
    "emoji_mode": "describe",
}

# Group 3 controls remain independent. Segmentation and spelling correction
# are OFF until their method/dictionary is validated; lemmatization and
# stopword-removal ON/OFF effects will be measured rather than assumed.
P2_CONFIG = {
    **P1_CONFIG,
    "segmentation_mode": "none",  # none, wordninja, symspell, hybrid
    "segmentation_min_token_length": 7,
    "correct_spelling": False,
    "symspell_dictionary_path": None,
    "max_edit_distance": 2,
    "lemmatize": False,
    "spacy_model": "en_core_web_sm",
    "remove_stopwords": False,
    "preserve_negation": True,
}
