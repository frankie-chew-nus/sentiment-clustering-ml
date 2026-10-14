# Airline Sentiment Clustering

Modular Python code for unsupervised clustering of airline tweets by sentiment.

## Current status

- **Group 1:** URL removal, generic airline placeholder, other mention removal, hashtag handling, and whitespace normalization.
- **Group 2:** configurable contraction expansion, number conversion, lowercasing, punctuation mode, and emoji mode.
- **Configuration:** `P0_CONFIG` is basic cleaning; `P1_CONFIG` enables the standard normalization settings.
- **Decision history:** See `docs/preprocessing_decisions.md` for Group 1 rationale and dataset-derived alias findings.

## Group 2 defaults

P1 defaults to contraction expansion ON, number conversion ON, lowercasing ON, punctuation removal, and emoji description mode. The emoji mode is configurable as `describe`, `keep`, or `remove`; punctuation mode is `remove` or `keep`.

These are starting settings, not experimentally established best choices. P0 remains available as Group 1-only cleaning for controlled comparison. P2 will be configured when Group 3 is implemented.

## Install dependencies

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt

## Run tests

    python -m pytest -q

## Use in Google Colab

    !git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git

    import sys
    sys.path.insert(0, "/content/sentiment-clustering-ml")
    from preprocessing import preprocess_text
    from config import P0_CONFIG, P1_CONFIG

    raw = "@united I can't wait 3 hours!!! 😡 #unitedsucks"
    print(preprocess_text(raw, P0_CONFIG))
    print(preprocess_text(raw, P1_CONFIG))

## Methodological rule

Keep the original tweet in a separate column. VADER, uppercase ratio, emoji counts, and punctuation counts must be extracted from the original text rather than normalized output. Sentiment labels/confidence must not be used for preprocessing or clustering; they are reserved for post-clustering evaluation.
