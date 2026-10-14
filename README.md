# Airline Sentiment Clustering

Modular Python code for the unsupervised airline-tweet sentiment clustering project.

## Current status

- **Group 1 preprocessing:** URL removal, mention removal, airline-hashtag removal, hashtag-symbol stripping, and whitespace normalization.
- **Configuration:** `P0_CONFIG` in `config.py` enables the Group 1 operations by default.
- **Tests:** `tests/test_preprocessing.py` covers core expected behavior.

## Group 1 rules

- Remove URLs and `@mentions`.
- Remove hashtags that exactly match a configured airline alias, such as `#United`.
- Preserve ordinary airline names in the text (for example, `United delayed my flight`).
- For other hashtags, remove only `#` and retain the content (for example, `#WorstFlightEver` becomes `WorstFlightEver`).
- Keep the original raw tweet in a separate column; do not overwrite it.

## Run tests

From the repository root, install pytest if needed and run:

    python -m pip install pytest
    python -m pytest -q

## Use in Google Colab

Clone the repository:

    !git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git

Import from the cloned directory:

    import sys
    sys.path.insert(0, "/content/sentiment-clustering-ml")
    from preprocessing import preprocess_text
    from config import P0_CONFIG

    result = preprocess_text(
        "@united United delayed my flight again! #United #WorstFlightEver",
        P0_CONFIG,
    )
    print(result)

Expected output:

    United delayed my flight again! WorstFlightEver

## Development notes

- The repository currently keeps `preprocessing.py` and `config.py` at its root, so imports reflect that layout.
- The module currently implements Group 1 only. Contractions, emoji handling, number conversion, segmentation, spelling correction, lemmatization, vectorization, clustering and evaluation are planned separately.
- Sentiment labels must not be passed to preprocessing, vectorization, or clustering. They are reserved for post-clustering evaluation.
