"""Configuration for Group 1 tweet cleaning.

Research decision:
Normalize airline identity to the generic token "airline" wherever a configured
airline alias appears: as an @mention, a #hashtag, or ordinary text. This aims
to reduce airline-identity signals so clusters focus more on sentiment than
which airline is named.

Aliases are based on the six target airlines plus variants observed in the
dataset's @mention audit. Some aliases, especially "American" and "Virgin",
are ambiguous in ordinary English and should be checked with real examples.
"""

AIRLINE_ALIASES = frozenset({
    # United Airlines
    "united", "united airlines", "unitedairlines", "united_airlines",
    # Delta Air Lines
    "delta", "delta air lines", "delta airlines", "deltaairlines", "delta_air_lines",
    # American Airlines; "americanair" was frequent in the dataset audit.
    "american", "american air", "americanair", "american airlines",
    "americanairlines", "american_airlines",
    # US Airways
    "us airways", "usairways", "usair", "us_airways",
    # Southwest Airlines
    "southwest", "southwest airlines", "southwestair", "southwestairlines", "southwest_airlines",
    # JetBlue
    "jetblue", "jet blue",
    # Virgin America and Virgin Atlantic; both observed in the dataset.
    "virgin america", "virginamerica", "virgin_america",
    "virgin atlantic", "virginatlantic", "virgin_atlantic",
    "virgin",
})

# Retain the earlier constant name for backwards compatibility.
AIRLINE_HASHTAG_ALIASES = AIRLINE_ALIASES

# For fused airline+sentiment hashtags, preserve the suffix after a known
# airline prefix. Examples: #usairwaysfail -> "airline fail"; #unitedsucks
# -> "airline sucks". Kept separate so it can be audited and expanded.
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
