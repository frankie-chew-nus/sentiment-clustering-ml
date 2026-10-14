# Airline Sentiment Clustering

Modular Python code for the unsupervised airline-tweet sentiment clustering project.

## Current status

- **Group 1 preprocessing:** URL removal, airline-reference normalization, removal of unrelated mentions, hashtag handling, and whitespace normalization.
- **Configuration:** `P0_CONFIG` in `config.py` enables Group 1 operations by default.
- **Tests:** `tests/test_preprocessing.py` covers the expected behavior for the current Group 1 rules.

## Group 1 rules

- Remove URLs.
- Replace configured airline references with the generic token `airline`, whether they appear as an `@mention`, a `#hashtag`, or ordinary text. Examples: `@united` -> `airline`, `#United` -> `airline`, `United delayed my flight` -> `airline delayed my flight`.
- Remove non-airline `@mentions`.
- Preserve non-airline hashtag content while removing the `#` marker; for example, `#WorstFlightEver` -> `WorstFlightEver`.
- Keep the original raw tweet in a separate column; do not overwrite it.
- The alias list includes potentially ambiguous names such as `American` and `Virgin`; inspect replacements on real tweets before treating the pipeline as final.

## Run tests

From the repository root, run:

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

    airline airline delayed my flight again! airline WorstFlightEver

## Development notes

- The repository currently keeps `preprocessing.py` and `config.py` at its root, so imports reflect that layout.
- The module currently implements Group 1 only. Contractions, emoji handling, number conversion, segmentation, spelling correction, lemmatization, vectorization, clustering and evaluation are planned separately.
- Sentiment labels must not be passed to preprocessing, vectorization, or clustering. They are reserved for post-clustering evaluation.
