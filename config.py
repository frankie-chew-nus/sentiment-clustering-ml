"""Configuration for Group 1 tweet cleaning."""

# Match airline-specific hashtags only; normal airline words are preserved.
# Extend aliases after reviewing the actual dataset's hashtag vocabulary.
AIRLINE_HASHTAG_ALIASES = frozenset({
    'united', 'unitedairlines', 'delta', 'deltaairlines',
    'american', 'americanairlines', 'usairways', 'usair',
    'southwest', 'southwestair', 'southwestairlines',
    'jetblue', 'virginamerica', 'virgin',
})

P0_CONFIG = {
    'remove_urls': True,
    'remove_mentions': True,
    'remove_airline_hashtags': True,
    'strip_hashtag_symbols': True,
    'normalize_whitespace': True,
}
