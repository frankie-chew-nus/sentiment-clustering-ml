"""Configuration for Group 1 tweet cleaning.

Research decision:
Normalize airline identity to the generic token "airline" wherever a configured
airline alias appears: as an @mention, a #hashtag, or ordinary text. This aims
to reduce airline-identity signals so clusters are more about tweet sentiment
than which airline is named.

The list covers the six airline brands represented in the dataset. Some aliases,
especially "American" and "Virgin", are ambiguous in ordinary English; inspect
examples after processing and revise the list if false replacements occur.
"""

AIRLINE_ALIASES = frozenset({
    # United Airlines
    "united", "united airlines", "unitedairlines", "united_airlines",
    # Delta Air Lines
    "delta", "delta air lines", "delta airlines", "deltaairlines", "delta_air_lines",
    # American Airlines
    "american", "american airlines", "americanairlines", "american_airlines",
    # US Airways
    "us airways", "usairways", "usair", "us_airways",
    # Southwest Airlines
    "southwest", "southwest airlines", "southwestair", "southwestairlines", "southwest_airlines",
    # JetBlue
    "jetblue", "jet blue",
    # Virgin America
    "virgin america", "virginamerica", "virgin_america", "virgin",
})

# Keep the previous constant name available while related code is updated.
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
