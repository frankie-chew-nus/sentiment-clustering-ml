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


# Configurable lexical protection lists.
# These are research choices rather than processing logic, so they live here.
NEGATION_WORDS = frozenset({
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "can't", "cant", "don't", "dont", "doesn't",
    "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "haven't", "havent",
    "hasn't", "hasnt", "hadn't", "hadnt", "won't", "wont", "wouldn't",
    "wouldnt", "shouldn't", "shouldnt", "couldn't", "couldnt",
})

SENTIMENT_PROTECTED_WORDS = frozenset({
    "airline", "flight", "flights", "delayed", "delay", "cancelled", "canceled",
    "cancel", "refund", "baggage", "luggage", "service", "terrible", "great",
    "good", "bad", "worst", "best", "happy", "unhappy", "angry", "annoyed",
    "disappointed", "awesome", "awful", "love", "hate",
})

# Candidate sentiment modifiers to protect if stopword removal is enabled.
# This list is intentionally configurable and should be audited against raw text.
INTENSIFIER_WORDS = frozenset({
    "very", "really", "so", "too", "extremely", "absolutely", "totally",
    "completely", "quite", "pretty", "highly", "incredibly", "seriously",
})

STOPWORD_PROTECTED_WORDS = (
    NEGATION_WORDS | SENTIMENT_PROTECTED_WORDS | INTENSIFIER_WORDS
)

P0_CONFIG = {
    "remove_urls": True,
    "replace_airline_mentions": True,
    "remove_other_mentions": True,
    "replace_airline_hashtags": True,
    "replace_airline_names": True,
    "strip_hashtag_symbols": True,
    "hashtag_segmentation_mode": "none",
    "hashtag_segmentation_min_token_length": 7,
    "normalize_whitespace": True,
}

P1_CONFIG = {
    **P0_CONFIG,
    "expand_contractions": True,
    "convert_numbers": False,
    "lowercase": True,
    "punctuation_mode": "keep",
    "emoji_mode": "describe",
}

# P2 defaults reflect the completed Group 3 validation: selective hashtag
# segmentation, lemmatization, and protected stopword removal are ON; global
# segmentation and generic spelling correction remain OFF.
P2_CONFIG = {
    **P1_CONFIG,
    # Global ordinary-text segmentation remains OFF. P2 selectively segments
    # non-airline hashtag bodies while hashtag provenance is still available.
    "segmentation_mode": "none",
    "segmentation_min_token_length": 7,
    "hashtag_segmentation_mode": "wordninja",
    "hashtag_segmentation_min_token_length": 7,
    "correct_spelling": False,
    "symspell_dictionary_path": None,
    "max_edit_distance": 2,
    "lemmatize": True,
    "spacy_model": "en_core_web_sm",
    "remove_stopwords": True,
    "preserve_negation": True,
}
