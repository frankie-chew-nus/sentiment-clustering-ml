# Preprocessing Decision Record: Airline Identity Normalization

**Status:** Agreed design; validate against dataset examples before treating alias coverage as final.  
**Module:** Group 1 basic cleaning (`preprocessing.py`, `config.py`)  
**Date recorded:** 2026-10-14

## Context

The project aims to discover sentiment-aligned groups of airline tweets (negative, neutral, positive), not to group tweets by which airline they discuss. If airline names remain as distinctive terms, clustering may reflect airline identity or airline-specific vocabulary as well as sentiment.

The earlier Group 1 implementation removed an airline only when it appeared as an `@mention` or exact `#hashtag`. This left ordinary references such as "United delayed my flight" unchanged. The team clarified that this was too narrow.

## Decision

Replace configured airline references with the generic token `airline` in all three forms:

- **Handle:** `@united` -> `airline`
- **Hashtag:** `#United` -> `airline`
- **Ordinary text:** `United delayed my flight` -> `airline delayed my flight`

Other rules:

- Remove non-airline `@mentions`.
- Preserve non-airline hashtag text but remove its leading `#`: `#WorstFlightEver` -> `WorstFlightEver`.
- Remove URLs and normalize whitespace.
- Keep the original tweet unchanged in a separate raw-text column.
- Do not use the dataset sentiment label or confidence during preprocessing, vectorization, or clustering. They are for post-clustering evaluation only.

## Why use a placeholder instead of deleting the airline name?

The goal is to suppress airline identity while retaining the sentence's structure. For example:

- Deletion: "United delayed my flight" -> "delayed my flight"
- Placeholder: "United delayed my flight" -> "airline delayed my flight"

The placeholder preserves a signal that an airline is the actor/topic while avoiding a different token for every airline. This is a design hypothesis, not a guaranteed clustering improvement; we can test it later against alternatives if needed.

## Alias configuration

The current aliases are stored centrally in `AIRLINE_ALIASES` in `config.py`. They cover configured forms for United, Delta, American, US Airways, Southwest, JetBlue, and Virgin America, including common spaced and concatenated spellings.

The list should be reviewed against the actual dataset. There is a known ambiguity: "American" and "Virgin" can be ordinary English words as well as parts of airline names. Whole-word matching prevents partial matches inside longer words, but it cannot determine the intended meaning in every sentence. Inspect a sample of replacements for false positives before full-dataset use.

## Alternatives considered

1. **Remove only @mentions and #hashtags.** Rejected because ordinary mentions such as "United delayed my flight" would still preserve airline identity.
2. **Delete airline names wherever they occur.** Rejected as the default because deletion can make sentences less informative or grammatically awkward.
3. **Replace airline references with a generic token.** Selected because it suppresses brand identity but retains a placeholder for the entity.

## Implementation order

1. Remove URLs.
2. Replace configured airline @mentions with `airline`.
3. Remove remaining non-airline @mentions.
4. Replace configured airline hashtags with `airline`.
5. Replace configured airline names in ordinary text with `airline`.
6. Remove `#` from remaining hashtags while preserving their words.
7. Normalize whitespace.

Specialized hashtag/mention handling runs before ordinary text replacement so prefixed forms receive the intended behavior.

## Validation plan

- Unit-test handles, hashtags, ordinary airline names, multiword names, non-airline mentions, non-airline hashtags, punctuation, whitespace, and invalid input.
- Review a random sample of actual before/after tweets and specifically inspect ambiguous names.
- Keep raw text and processed text in separate columns.
- Record this configuration with each experiment so results remain reproducible.
- Compare clustering performance only after running the unsupervised pipeline; use labels only in the permitted evaluation stage.

## Current limitation

The implementation uses a curated alias list, not a complete airline entity recognizer. New spellings, misspellings, unusual hashtags, or aliases absent from the list will not necessarily be normalized. Alias expansion should be driven by data inspection and documented changes, not by looking at sentiment labels.
