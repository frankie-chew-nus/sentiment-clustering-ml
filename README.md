# Airline Sentiment Clustering

Modular Python pipeline for **unsupervised clustering of airline tweets by sentiment**. The objective is sentiment discovery, not perfect linguistic reconstruction or word prediction.

> **Strict leakage rule:** `airline_sentiment` and every sentiment-confidence field are reserved **exclusively for post-clustering external evaluation**. They must not enter preprocessing, feature extraction, vectorization, feature selection, clustering, or tuning.

## Project status

| Stage | Status | Evidence |
|---|---|---|
| L2 — Preprocessing | Complete: P0 / P1 / P2 | [Preprocessing decisions](docs/preprocessing_decisions.md) |
| L3 — Feature engineering | Complete: V / N / S | [Feature-engineering decisions](docs/feature_engineering_decisions.md) |
| L3 — Master feature matrix | Complete: 14,639 tweets × 10 engineered features, plus `row_id` | [PR #22](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/22) |
| L4 — Text representations and feature scaling | Next: methodology and implementation pending | Word/character TF-IDF; TF-IDF-weighted FastText |
| L5 — Clustering and ablations | Pending | Label-blind clustering and controlled comparisons |
| L6 — External evaluation | Pending | Compare discovered clusters with held-out sentiment labels |

**Latest user-verified regression result:** 131 tests passed (L3-M). Tests establish implementation consistency, **not** superior sentiment clustering.

## Scope and generalization

The validated configurations target this airline-tweet corpus. Airline aliases, hashtag rules, protected terms and lexical decisions are corpus-specific; re-audit them before applying this pipeline to another domain. The original RAW tweet must remain available independently of processed variants.

## L2 — Preprocessing

| Operation | P0 | P1 | P2 |
|---|---|---|---|
| HTML entity decoding; URL and airline/mention normalization | On | On | On |
| Contraction expansion; emoji descriptions; lowercase | Off | On | On |
| Preserve numbers and punctuation | Yes | Yes | Yes |
| Wordninja for non-airline hashtag bodies only | Off | Off | On |
| Global Wordninja; generic SymSpell correction | Off | Off | Off |
| spaCy lemmatization; protected stopword removal | Off | Off | On |
| Negation / sentiment / intensifier protection | n/a | n/a | On |

- **P0:** Decode HTML entities; remove URLs; normalize known airline mentions, names and hashtags to `airline`; remove non-airline mentions; preserve useful hashtag content and sentiment-bearing suffixes.
- **P1:** P0 plus contraction expansion, emoji descriptions and lowercasing, while retaining punctuation and numbers.
- **P2:** P1 plus selective non-airline hashtag splitting, lemmatization and protected stopword removal. Global splitting and generic spelling correction remain off after label-blind corpus audits.

P0/P1/P2 are **candidate text inputs** for later vectorization, not an assumption that P2 necessarily clusters sentiment best. See [detailed preprocessing decisions](docs/preprocessing_decisions.md).

## L3 — Engineered feature families

| Family | Input | Master columns | Configurations |
|---|---|---|---|
| **V — VADER** | RAW | `V_pos`, `V_neg`, `V_neu`, `V_compound` | V0 none; V1 compound; V2 pos + neg + compound |
| **N — Negation / intensity** | Validated **P1** | `N_negation_count`, `N_intensifier_original_count`, `N_intensifier_conservative_count` | N0 none; N1 negation; N2 negation + original; N3 negation + conservative |
| **S — Structural expression** | RAW with **temporary URL exclusion** | `S_exclamation_count`, `S_uppercase_ratio`, `S_emoji_count` | S0 none; S1 exclamation; S2 + uppercase; S3 + emoji |

**Why different inputs?** RAW retains capitalization, punctuation and emojis for V/S. P1 expands contractions (e.g., `can't` → `can not`) for consistent modifier counts in N. S uses the audited S0-C URL-boundary rule to preserve adjacent emoji and punctuation. No feature assigns handcrafted positive/negative sentiment classes.

The master feature CSV stores **all 10 unscaled engineered columns** and sequential `row_id`; it does **not** mean all columns are used together in clustering. `row_id` is alignment metadata, never a model feature. `V_neu` is initially audit-only, and N2/N3 are alternative intensifier representations. Selection and block scaling/weighting happen downstream.

**Deferred candidates:** Structured numerical ratings (low coverage and many non-rating matches), airline/domain context indicators (topic-clustering risk), and additional structural measures such as question marks, mixed punctuation and length. These were not all implemented or proven unnecessary.

## Repository guide

| Path | Responsibility |
|---|---|
| `config.py`, `preprocessing.py` | Validated P0/P1/P2 configuration and preprocessing |
| `feature_engineering_config.py` | V0–V2, N0–N3 and S0–S3 selection |
| `feature_engineering.py` | V/N/S feature extraction |
| `master_feature_matrix.py` | Validate and assemble family CSVs; export SHA256 manifest |
| `tests/` | Preprocessing, feature and assembly regression tests |
| `docs/preprocessing_decisions.md` | L2 audit/decision history |
| `docs/feature_engineering_decisions.md` | L3 audits, changes, validation and limitations |
| `notebooks/` | Demonstrations and user-approved notebook artifacts; consult folder for current versions |

## Install and test

```bash
git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git
cd sentiment-clustering-ml
python -m pip install -r requirements-dev.txt
python -m pip install vaderSentiment==3.3.2
python -m spacy download en_core_web_sm
python -m pytest -q
```

The spaCy model is required for P2. VADER is pinned to 3.3.2 for reproducibility. A clean Colab runtime may require reinstalling dependencies.

## Quick examples

**Preprocess one tweet:**

```python
from preprocessing import preprocess_text
from config import P0_CONFIG, P1_CONFIG, P2_CONFIG

raw = "@united I can't wait 3 hours!!! 😡 #badcustomerservice"
for name, cfg in [("P0", P0_CONFIG), ("P1", P1_CONFIG), ("P2", P2_CONFIG)]:
    print(name, preprocess_text(raw, cfg))
```

Custom preprocessing variants can be created with `{**P2_CONFIG, "remove_stopwords": False}`; do not overwrite the validated defaults. HTML decoding precedes URL removal, so encoded apostrophes can participate in contraction expansion.

**Extract the three feature families:**

```python
from feature_engineering import extract_vader_frame, extract_n_frame, extract_s_frame

raw_texts = ["I can't believe this is SO BAD!!! 😡"]
p1_texts = ["i can not believe this is so bad!!! enraged face"]

v = extract_vader_frame(raw_texts)  # RAW
n = extract_n_frame(p1_texts)       # P1
s = extract_s_frame(raw_texts)      # URL-excluded RAW internally
```

**Generate the master CSV from existing family CSVs:**

```python
from master_feature_matrix import assemble_csvs

master, manifest_path = assemble_csvs(
    vader_path="Tweets-L3-vader-features.csv",
    negation_path="Tweets-L3-negation-intensity-features.csv",
    structural_path="Tweets-L3-structural-features.csv",
    output_path="Tweets-L3-master-features.csv",
)
print(master.shape, manifest_path)
```

The assembler enforces strict expected schemas, 14,639 sequential row IDs by default, valid numeric ranges and exact feature-value preservation. It produces a JSON manifest with input/output SHA256 hashes. These checks do not independently prove the entire extraction history of intermediate files.

## Google Colab and Drive

The project's validated Google Drive paths use **`MyDrive`**, not `My Drive`:

```python
from pathlib import Path
BASE = Path("/content/drive/MyDrive/MSI5001 project/EDA")
RAW_PATH = BASE / "Tweets-redacted.csv"
P1_PATH = BASE / "0.Processed-files" / "Tweets-processed-p1.csv"
FEATURE_DIR = BASE / "L3_Feature_Engineering"
```

Mount Drive with `drive.mount("/content/drive")`, then clone/pull `main` (or the relevant review branch). After changing branches in a running notebook, restart the runtime or reload imported Python modules to avoid stale imports.

The executed L3 demonstration notebook is maintained in the repository by the project team. The duplicate, unexecuted `MSI5001_L3_Feature_Generation_Demo.ipynb` was removed as requested; use the current team-maintained executed version in `notebooks/`. Generated CSVs and manifests remain in Google Drive, not GitHub.

## Milestone PR history

| PR | Scope | Latest milestone full-suite validation |
|---|---|---|
| [#19](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/19) | VADER V0–V2 | 70 passed |
| [#20](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/20) | Negation/intensity N0–N3 | 93 passed |
| [#21](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/21) | Structural S0–S3 | 112 passed |
| [#22](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/22) | Master feature matrix | 131 passed |

Detailed audit findings, test corrections and evidence are maintained in `docs/`. The user reviews each PR, adds supporting evidence and merges manually.

## Next steps

L4 will compare **word TF-IDF**, **character TF-IDF**, **combined word + character TF-IDF**, and **TF-IDF-weighted FastText pooling**. We will agree on preprocessing choices, scaling, feature-block weighting and controlled ablations **before** implementation and clustering. Sentiment labels/confidence remain inaccessible until post-clustering external evaluation.
