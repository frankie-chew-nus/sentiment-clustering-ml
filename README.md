# Airline Sentiment Clustering

Modular Python code for unsupervised clustering of airline tweets by sentiment.

## Current status

- Group 1: basic cleaning and generic airline-placeholder normalization.
- Group 2: configurable contraction expansion, number conversion, lowercasing, punctuation handling, and emoji modes.
- Group 3: optional segmentation, spelling correction, spaCy lemmatization, and negation-protected stopword removal.
- Configurations: P0 (Group 1), P1 (Groups 1-2), and P2 (Groups 1-3 options).
- Decision history: docs/preprocessing_decisions.md.

## Group 3 defaults

Group 3 operations are exposed but conservative by default:
- segmentation: none while methods are being benchmarked;
- spelling correction: OFF until a frequency dictionary is selected and reviewed;
- lemmatization: OFF pending ON/OFF ablation;
- stopword removal: OFF, with negation preservation available.

Segmentation can be set to wordninja, symspell, or hybrid. These are candidate methods, not presumed improvements.

## Install dependencies

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m spacy download en_core_web_sm

The spaCy model is required only if lemmatization or stopword removal is enabled. SymSpell also needs a compatible frequency dictionary passed as symspell_dictionary_path.

## Run tests

    python -m pytest -q

## Use in Google Colab

    !git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git

    import sys
    sys.path.insert(0, "/content/sentiment-clustering-ml")
    from preprocessing import preprocess_text
    from config import P0_CONFIG, P1_CONFIG, P2_CONFIG

    raw = "@united I can't wait 3 hours!!! 😡 #unitedsucks"
    print(preprocess_text(raw, P0_CONFIG))
    print(preprocess_text(raw, P1_CONFIG))

    # Enable a segmentation candidate only for its ablation:
    test_config = {**P1_CONFIG, "segmentation_mode": "wordninja"}
    print(preprocess_text("greatjob worstflightever", test_config))

## Methodological rule

Keep the raw tweet in a separate column. Structural features and VADER scores must be extracted from original raw text. Sentiment labels/confidence are for post-clustering evaluation only.
