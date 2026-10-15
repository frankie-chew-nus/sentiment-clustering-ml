# Airline Sentiment Clustering

Modular Python code for unsupervised clustering of airline tweets by sentiment.

## Scope and context disclaimer

This project is designed and validated specifically for the supplied **airline-tweet sentiment-clustering context**. Several preprocessing choices are intentionally domain- and corpus-aware, including airline alias normalization, airline hashtag handling, protected airline/sentiment vocabulary, and selective hashtag segmentation.

The pipeline is modular and reusable as software, but the current configurations should **not be assumed to generalize unchanged to other domains or datasets**. If reused elsewhere, re-audit the aliases, protected terms, hashtag behavior, lexical normalization choices, and other corpus-specific assumptions before treating the outputs as appropriate.

The project objective is sentiment-oriented clustering, not perfect grammatical reconstruction of tweets.

## Current status

Preprocessing P0-P2 is complete and validated.

- **P0 — basic cleaning:** HTML entity decoding, URL removal; airline mentions/names/hashtags normalized to the generic token `airline`; non-airline mentions removed; other hashtag content retained.
- **P1 — text normalization:** P0 + contraction expansion, emoji descriptions, lowercasing; numbers and punctuation are preserved by default.
- **P2 — lexical normalization:** P1 + Wordninja segmentation only for non-airline hashtag bodies, spaCy lemmatization, and protected stopword removal.
- **Global Wordninja:** OFF after corpus audit showed damaging ordinary-word splits.
- **Generic SymSpell spelling correction:** OFF after corpus audit showed harmful semantic/negation corrections.
- **Decision history and audit evidence:** `docs/preprocessing_decisions.md`.

The preprocessing suite currently reaches **54 automated tests passing** after the HTML-decoding addition, alongside the earlier manual P2 sentiment-preservation checks and HTML-decoding validation. Functional validation confirms that the code behaves as designed; it does not by itself establish that P2 will produce better clusters than P0/P1. That will be evaluated downstream.

## Final preprocessing defaults

| Operation | P0 | P1 | P2 |
|---|---:|---:|---:|
| HTML entity decoding | ON | ON | ON |
| URL / mention / airline normalization | ON | ON | ON |
| Contraction expansion | OFF | ON | ON |
| Emoji description | OFF | ON | ON |
| Lowercase | OFF | ON | ON |
| Number conversion | OFF | OFF | OFF |
| Punctuation removal | OFF | OFF | OFF |
| Selective non-airline hashtag Wordninja | OFF | OFF | ON |
| Global ordinary-text Wordninja | OFF | OFF | OFF |
| SymSpell spelling correction | OFF | OFF | OFF |
| spaCy lemmatization | OFF | OFF | ON |
| Protected stopword removal | OFF | OFF | ON |
| Negation protection | n/a | n/a | ON |

### HTML entity decoding

P0 decodes HTML character references with Python's built-in `html.unescape()` before URL removal, and P1/P2 inherit this behavior. Examples include `&amp;` -> `&`, `&gt;` -> `>`, `&quot;` -> `"`, and `&#39;` -> `'`.

Decoding runs before URL removal so downstream processing receives restored text. This also allows encoded apostrophes to participate in contraction expansion (for example, `Don&#39;t` -> `Don't` -> `do not`). No extra package is required because `html` is part of Python's standard library.

## Install dependencies

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m spacy download en_core_web_sm

The spaCy model is required when P2 lemmatization or stopword removal is enabled. SymSpell remains available as an experimental function but is OFF in the finalized P2 configuration.

## Run tests

    python -m pytest -q

## Process a single tweet

    from preprocessing import preprocess_text
    from config import P0_CONFIG, P1_CONFIG, P2_CONFIG

    raw = "@united I can't wait 3 hours!!! 😡 #badcustomerservice"

    print(preprocess_text(raw, P0_CONFIG))
    print(preprocess_text(raw, P1_CONFIG))
    print(preprocess_text(raw, P2_CONFIG))

## Define your own processing configuration

The preprocessing function accepts a normal Python mapping. The simplest approach is to start from the closest validated configuration and override only the settings you want to change.

For example, start from P2 but keep stopwords:

    from config import P2_CONFIG
    from preprocessing import preprocess_text

    MY_CONFIG = {
        **P2_CONFIG,
        "remove_stopwords": False,
    }

    processed = preprocess_text("I am not very disappointed with the delayed flights", MY_CONFIG)
    print(processed)

Or start from P1 and enable only lemmatization:

    MY_CONFIG = {
        **P1_CONFIG,
        "lemmatize": True,
        "spacy_model": "en_core_web_sm",
    }

This configuration pattern is intentional: P0/P1/P2 are validated reference configurations, while experiment-specific variants can be created without editing the core preprocessing functions.

## Produce a processed CSV file

For a dataset containing a raw `text` column:

    import pandas as pd

    from preprocessing import preprocess_text
    from config import P2_CONFIG

    INPUT_CSV = "/content/drive/My Drive/MSI5001 project/EDA/Tweets-redacted.csv"
    OUTPUT_CSV = "/content/drive/My Drive/MSI5001 project/EDA/Tweets-processed-p2.csv"

    df = pd.read_csv(INPUT_CSV)

    RUN_CONFIG = {
        **P2_CONFIG,
        # Override any settings for this particular export here.
        # Example:
        # "remove_stopwords": False,
    }

    df["processed_text"] = df["text"].fillna("").apply(
        lambda text: preprocess_text(text, RUN_CONFIG)
    )

    df.to_csv(OUTPUT_CSV, index=False)

    print("Rows:", len(df))
    print("Saved:", OUTPUT_CSV)

Keep the original `text` column. Do not overwrite it. This preserves the raw tweet for auditing and for later structural/VADER feature extraction.

### Reproducibility recommendation

When exporting a processed dataset, record which configuration produced it. For notebook experiments, print the configuration next to the output:

    from pprint import pprint
    pprint(RUN_CONFIG)

For more formal experiment runs, save the configuration (or the P0/P1/P2 name plus overrides) alongside the output filename/experiment metadata. This avoids ambiguity about how a processed file was produced.

## Use in Google Colab

First clone the repository:

    !git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git

Then:

    import sys
    sys.path.insert(0, "/content/sentiment-clustering-ml")

    from preprocessing import preprocess_text
    from config import P0_CONFIG, P1_CONFIG, P2_CONFIG

If the repository is already cloned:

    %cd /content/sentiment-clustering-ml
    !git switch main
    !git pull origin main

After switching branches or pulling code during an active notebook session, reload imported modules or restart the runtime to avoid stale Python imports.

## Methodological rules

- Keep the raw tweet in a separate column.
- Structural/style features and VADER scores should be extracted from original raw text unless an experiment explicitly justifies another representation.
- Dataset sentiment labels and sentiment-confidence fields are reserved for **post-clustering external evaluation only**. They must not be used to choose preprocessing rules, engineered features, vectorization settings, clustering inputs, or unsupervised model tuning.
- Corpus audits used to design preprocessing are label-free.
- P0/P1/P2 are preprocessing hypotheses. Their downstream clustering value must be evaluated rather than assumed.
