# Airline Sentiment Clustering

Modular Python code for unsupervised clustering of airline tweets by sentiment.

## Current status

- **Group 1 preprocessing:** URL removal, generic airline-placeholder replacement, removal of unrelated mentions, hashtag handling, and whitespace normalization.
- **Configuration:** P0_CONFIG and alias lists in config.py.
- **Tests:** tests/test_preprocessing.py.
- **Decision history:** See docs/preprocessing_decisions.md for rationale, dataset-derived alias findings, suffix handling, alternatives considered, limitations, and validation plans.

## Group 1 rules

- Remove URLs.
- Replace configured airline names with the generic token airline, whether they occur as an @mention, a #hashtag, or ordinary text.
- Remove non-airline @mentions.
- Exact airline hashtags become airline. Selected sentiment hashtags preserve their suffix, e.g. #usairwaysfail -> airline fail and #unitedsucks -> airline sucks.
- Preserve other hashtag content while removing #, e.g. #WorstFlightEver -> WorstFlightEver.
- Keep original raw tweet text in a separate column.
- Review ambiguous names (especially American and Virgin) for false replacements.

## Run tests

From the repository root:

    python -m pip install pytest
    python -m pytest -q

## Use in Google Colab

    !git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git

    import sys
    sys.path.insert(0, "/content/sentiment-clustering-ml")
    from preprocessing import preprocess_text
    from config import P0_CONFIG

    result = preprocess_text("@USAirways #usairwaysfail", P0_CONFIG)
    print(result)  # expected: airline airline fail

## Important methodological rule

Do not pass sentiment labels or sentiment confidence to preprocessing, vectorization, or clustering. Use labels only for post-clustering evaluation.
