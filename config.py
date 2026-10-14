"""Configuration for Group 1 tweet cleaning.

Airline aliases are replaced with the generic token "airline" whether they
appear as @mentions, #hashtags, or ordinary text. Review aliases against the
dataset because some names (such as "American") can also be ordinary words.
"""

AIRLINE_ALIASES = frozenset({
    "united", "united airlines", "unitedairlines",
    "delta", "delta air lines", "delta airlines", "deltaairlines",
    "american", "american airlines", "americanairlines",
    "us airways", "usairways", "usair",
    "southwest", "southwest airlines", "southwestair", "southwestairlines",
    "jetblue", "jet blue",
    "virgin america", "virginamerica", "virgin",
})

# Backwards-compatible name while older code is being updated.
AIRLINE_HASHTAG_ALIASES = AIRLINE_ALIASES

P0_CONFIG = {
    "remove_urls": True,
    "replace_airline_mentions": True,
    "remove_other_mentions": True,
    "replace_airline_hashtags": True,
    "replace_airline_names": True,
    "strip_hashtag_symbols": True,
    "normalize_whitespace": True,
}
