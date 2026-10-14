"""Configuration for airline tweet preprocessing.

P0: Group 1 basic cleaning only.
P1: P0 plus Group 2 text normalization.
P2 will be defined when Group 3 lexical normalization is implemented.
"""

AIRLINE_ALIASES = frozenset({
    # United Airlines
    "united", "united airlines", "unitedairlines", "united_airlines",
    # Delta Air Lines
    "delta", "delta air lines", "delta airlines", "deltaairlines", "delta_air_lines",
    # American Airlines
    "american", "american air", "americanair", "american airlines",
    "americanairlines", "american_airlines",
    # US Airways
    "us airways", "usairways", "usair", "us_airways",
    # Southwest Airlines
    "southwest", "southwest airlines", "southwestair", "southwestairlines", "southwest_airlines",
    # JetBlue
    "jetblue", "jet blue",
    # Virgin America and Virgin Atlantic
    "virgin america", "virginamerica", "virgin_america",
    "virgin atlantic", "virginatlantic", "virgin_atlantic",
    "virgin",
})

# Backwards compatibility for earlier versions of the Group 1 implementation.
AIRLINE_HASHTAG_ALIASES = AIRLINE_ALIASES

# Approved endings to preserve when an airline name is fused with a sentiment
# hashtag. Expand only after reviewing common forms from the dataset.
AIRLINE_HASHTAG_SUFFIX_RULES = {
    "fail", "fails", "failure", "sucks", "suck", "sux",
    "hate", "hates", "terrible", "awful", "bad", "worst",
    "delay", "delays", "delayed", "cancel", "cancelled", "canceled",
    "disappointed", "unacceptable", "poorservice", "badservice",
}

# Group 1: basic cleaning.
P0_CONFIG = {
    "remove_urls": True,
    "replace_airline_mentions": True,
    "remove_other_mentions": True,
    "replace_airline_hashtags": True,
    "replace_airline_names": True,
    "strip_hashtag_symbols": True,
    "normalize_whitespace": True,
}

# Group 2: each operation remains configurable. These are starting defaults
# for the standard pipeline; later ablations can switch modes on/off.
P1_CONFIG = {
    **P0_CONFIG,
    "expand_contractions": True,
    "convert_numbers": True,
    "lowercase": True,
    "punctuation_mode": "remove",  # "remove" or "keep"
    "emoji_mode": "describe",     # "describe", "keep", or "remove"
}
