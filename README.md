# Airline Sentiment Clustering

Modular Python pipeline for **unsupervised clustering of airline tweets by sentiment**. The objective is sentiment discovery, not perfect linguistic reconstruction or word prediction.

> **Strict leakage rule:** `airline_sentiment` and every sentiment-confidence field are reserved **exclusively for post-clustering external evaluation**. They must not enter preprocessing, feature extraction, vectorization, feature selection, clustering, or tuning.

## Project status

| Stage | Status | Evidence |
|---|---|---|
| L2 — Preprocessing | Complete: P0 / P1 / P2 | [Preprocessing decisions](docs/preprocessing_decisions.md) |
| L3 — Feature engineering | Complete: V / N / S | [Feature-engineering decisions](docs/feature_engineering_decisions.md) |
| L3 — Master feature matrix | Complete: 14,639 tweets × 10 engineered features, plus `row_id` | [PR #22](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/22) |
| L4 — Vectorization, scaling and integration | **M1–M4 complete and merged** | [TF-IDF](docs/vectorization_decisions.md), [scaling](docs/feature_scaling_decisions.md), [weighting](docs/feature_integration_decisions.md), [FastText](docs/fasttext_decisions.md) |
| L5 — Clustering and ablations | Pending | Label-blind clustering and controlled comparisons |
| L6 — External evaluation | Pending | Compare discovered clusters with held-out sentiment labels |

**Latest user-verified regression result:** **179/179 tests passed** (L4-M4; Google Colab, 9.69 s). Tests establish implementation consistency, **not** superior sentiment clustering. L5 clustering and L6 label-based external evaluation have not yet been performed.

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

## L4 — Clustering-ready representations

L4 implemented label-blind text vectorization, engineered-feature scaling, block-distance weighting and optional corpus-trained FastText. The matrices are **inputs to L5**, not evidence of optimal sentiment clustering.

| Milestone | Implementation | Validated outcome |
|---|---|---|
| **M1 — TF-IDF** | `vectorization.py`, `vectorization_config.py` | Seven sparse representations from 14,639 tweets; [decision log](docs/vectorization_decisions.md) |
| **M2 — Scaling** | `feature_scaling.py`, `feature_scaling_config.py` | `log1p` + bounded count scaling primary candidate; log-only and standardization alternatives; [decision log](docs/feature_scaling_decisions.md) |
| **M3 — Integration** | `feature_integration.py` | W0/W1/W2 label-blind calibrated sparse matrices; [decision log](docs/feature_integration_decisions.md) |
| **M4 — FastText** | `fasttext_representation.py` | Optional corpus-trained 100-dimensional vectors; [decision log](docs/fasttext_decisions.md) |

### M1 — TF-IDF configurations and decisions

Like L2's P0/P1/P2 variants and L3's V/N/S feature subsets, M1 defines **seven explicit representation configurations**. Every row below was implemented and numerically validated on the same 14,639 tweets. Word uses token 1–2 grams; `char` and `char_wb` use character 3–5 grams. A combined configuration concatenates two independently L2-normalized blocks; it does **not** apply M3 weighting.

| M1 ID | Input and analyzer | Vocabulary dimensions | Nonzero entries | Zero rows | Role / interpretation |
|---|---|---:|---:|---:|---|
| `word_p1` | P1 · word TF-IDF | 29,792 | 371,747 | 16 | **Primary word baseline**; greater lexical coverage than P2 |
| `word_p2` | P2 · word TF-IDF | 17,267 | 173,532 | 71 | Controlled preprocessing alternative; smaller vocabulary, more zero rows |
| `char_p0` | P0 · character TF-IDF | 50,000 | 3,332,264 | 0 | **Primary character baseline**; preserves P0 casing and punctuation |
| `char_p1` | P1 · character TF-IDF | 50,000 | 3,455,072 | 0 | Controlled character-preprocessing alternative |
| `char_wb_p0` | P0 · `char_wb` TF-IDF | 50,000 | 2,252,967 | 1 | Controlled word-boundary character alternative |
| `combined_p1_p1` | P1 word + P1 character | 79,792 | 3,826,819 | 0 | Controlled same-preprocessing combined baseline |
| `combined_p1_p0` | P1 word + P0 character | 79,792 | 3,704,011 | 0 | **Primary combined baseline**; input to M3 integration |

**How to read the names:** `word_p1` means word TF-IDF fitted to P1 text; `char_p0` means character TF-IDF fitted to P0 text; `combined_p1_p0` means P1 word TF-IDF concatenated with P0 character TF-IDF. `char_wb` restricts character n-grams to word boundaries; it is a distinct analyzer, not a separate preprocessing level.

**M1 conclusions:** P2 word vectors have ~42.0% fewer dimensions than P1 word vectors but more zero rows (71 vs 16). Character P0 has no zero rows, although its 50,000-feature vocabulary reaches the configured cap. Both combined configurations eliminate zero rows in this corpus; their mean row norm is approximately **1.4138**, consistent with two separately L2-normalized nonzero blocks. These findings support **P1 word + P0 character as a defensible M3 integration baseline**, not as a demonstrated winner for sentiment clustering.

**M1 versus M3:** M1 generated all seven text representations. M3 used **one selected text combination**—`word_p1` + `char_p0`—then appended **V1 + N1 + S2** (four engineered columns) to produce 79,796-dimensional W0/W1/W2 matrices. W0, W1 and W2 are **three weighting configurations of the same underlying features**, not three new TF-IDF vectorizers. L5 will compare the word-only, character-only, combined-text and weighted-integrated baselines under a prespecified label-blind clustering protocol.

### Validated initial L5 inputs

All rows correspond to **14,639 tweets**. Files shown below are the **read-only master copies** placed by the project owner in Google Drive; the `L4-` prefix was added after the original L4 exports were validated. Use the paired `.json` manifest for each `.npz` file.

| L5 role | Master NPZ filename | Shape | Zero rows |
|---|---|---|---:|
| Word-only baseline | `L4-word_p1.npz` | 14639 × 29792 | 16 |
| Character-only baseline | `L4-char_p0.npz` | 14639 × 50000 | 0 |
| Combined text-only | `L4-combined_p1_p0.npz` | 14639 × 79792 | 0 |
| W0 — direct concatenation | `L4-W0.npz` | 14639 × 79796 | 0 |
| W1 — equal expected block distance | `L4-W1.npz` | 14639 × 79796 | 0 |
| W2 — 70% text / 30% engineered | `L4-W2.npz` | 14639 × 79796 | 0 |
| Optional semantic baseline | `L4-fasttext_p1_100d.npz` | 14639 × 100 | 0 |

Other M1 configurations (word P2, character P1, character-word-boundary P0, and combined P1+P1) are available as controlled alternatives. Do not assume any representation is sentiment-optimal.

**M1 settings:** P1 word TF-IDF uses 1–2 grams (up to 30,000 features); P0 character TF-IDF uses 3–5 grams (up to 50,000). Both use `min_df=2`, `max_df=0.95`, sublinear TF, IDF and per-block L2 normalization. Concatenated text blocks are not automatically renormalized.

### M2 — Engineered-feature scaling configurations

M2 implemented and tested **three scaling methods** for nonnegative count features. All methods retain the native VADER polarity values and uppercase ratio without additional scaling. The same validated L3 master matrix (14,639 rows) was used.

| M2 method | Count transformation | Resulting behavior | Experimental role |
|---|---|---|---|
| **`bounded`** | `log1p(x)` then divide by fitted 95th-percentile reference; positive-value percentile fallback when the overall reference is zero; optionally clip to [0,1] | Preserves count zeros and limits large transformed counts | **Primary scaling candidate; used for M3 W0/W1/W2 exports** |
| `log_only` | `log1p(x)` only | Preserves zeros and compresses outliers, but does not cap large counts | Controlled alternative |
| `standard` | `log1p(x)` then `(x - mean) / std` on transformed counts | Centers counts and gives unit variance; original zeros become negative standardized values | Controlled alternative |

**Why `log1p`?** It keeps absent counts at zero (`log1p(0)=0`), compresses highly right-skewed counts (27 exclamation marks become about 3.3322 before bounded scaling), and preserves count ordering before clipping. This reduces extreme-count influence on Euclidean distances; it does **not** prove optimal clustering.

**Fitted references (bounded, 95th percentile):** `N_negation_count = 1.0986122886681096`; `S_exclamation_count = 1.0986122886681096`; `S_emoji_count = 1.791759469228055` in the V1+N1+S3 audit. The M3 baseline uses **V1+N1+S2**, so emoji is not included in W0/W1/W2.

**Observed M2 distance contributions:** The label-blind audit sampled **49,999 distinct tweet pairs**, using **V1+N1+S3** engineered features only (no text TF-IDF). These are shares of mean squared Euclidean distance, not sentiment-importance scores.

| M2 method | VADER | Negation | Structural |
|---|---:|---:|---:|
| **`bounded`** | 48.51% | 25.20% | 26.28% |
| `log_only` | 38.82% | 25.14% | 36.03% |
| `standard` | 6.91% | 30.86% | 62.23% |

**M2 conclusion:** Standardization greatly increased the structural family's distance contribution, including rare emoji effects. Bounded scaling was retained as the primary *engineering candidate*, with log-only and standardization still available for controlled future comparisons. The current M3 exports use **bounded only**; an M2-method ablation in L5 would require separately generated integrated matrices with other scaling choices.

### M3 — Integrated feature blocks and weighting configurations

M3 combines **one selected M1 text pair**—`word_p1` (29,792 columns) + `char_p0` (50,000 columns)—with **V1** (VADER compound, 1), **N1** (negation count, 1), and **S2** (exclamation count + uppercase ratio, 2). The four engineered columns use M2's bounded configuration where applicable. Total: **79,796 columns** across 14,639 tweets.

**W0/W1/W2 do not denote different vectorizers or scaling methods.** They are three ways to weight these *same five blocks*. The percentages below refer to expected **squared Euclidean distance contributions**, not raw feature coefficients.

| M3 method | Rule | Word | Character | VADER | Negation | Structural | Role |
|---|---|---:|---:|---:|---:|---:|---|
| **W0** | No additional weights (all coefficients = 1) | 40.8265% | 40.4799% | 9.2829% | 4.8141% | 4.5966% | **Default unweighted baseline** |
| **W1** | Equal calibrated contribution per active block | 20% | 20% | 20% | 20% | 20% | Diagnostic equal-distance alternative |
| **W2** | Text-priority allocation (70% text / 30% engineered) | 35% | 35% | 10% | 10% | 10% | Text-priority alternative |

The calibrated coefficient for block `b` is `sqrt(target_share_b / original_mean_squared_distance_b)`. Thus, **equal target contributions do not imply equal numerical coefficients**. W0 retains the original feature magnitudes. The measured W0 text contribution is approximately **81.31%**; W1 gives 40% to text, W2 gives 70%. The W1/W2 targets were achieved on the calibration corpus, but no method is established as sentiment-optimal.

| Block | W0 coefficient | W1 coefficient | W2 coefficient |
|---|---:|---:|---:|
| Word | 1.000000 | 0.318236 | 0.420987 |
| Character | 1.000000 | 0.319596 | 0.422785 |
| VADER | 1.000000 | 0.667387 | 0.471914 |
| Negation | 1.000000 | 0.926749 | 0.655311 |
| Structural | 1.000000 | 0.948425 | 0.670638 |

**M3 output:** `L4-W0.npz`, `L4-W1.npz`, and `L4-W2.npz` are three distinct **14,639 × 79,796** sparse matrices, each with **3,728,386** stored nonzero entries and a paired JSON manifest. They passed the recorded export, SHA256 and numerical checks. **W0 is the baseline; W1 and W2 are controlled L5 alternatives**, not winners. See [M2 decisions](docs/feature_scaling_decisions.md) and [M3 decisions](docs/feature_integration_decisions.md) for complete audits.

**M4 FastText:** Corpus-trained P1 skip-gram, 100 dimensions, window=5, `min_count=2`, 10 epochs, seed=5001, single worker, character n-grams 3–6; IDF-weighted mean pooling followed by L2 normalization. Full-corpus validation: **8,814 explicit vocabulary tokens**, **94.1682% token-occurrence coverage**, **0 zero tweet vectors**, and **29.45 s training plus encoding** in the reported Colab run. Opposing-polarity words can have high cosine similarity (e.g. excellent/terrible **0.8139**); pretrained FastText remains deferred.

### Data access and integrity rules

`/content/drive/MyDrive/MSI5001 project/EDA/0.Processed-files/` is the **strictly READ-ONLY** master directory for validated L1–L4 datasets. Scripts and notebooks may read from it, but **must never create, overwrite, rename, move or delete any file there**. This includes CSV, NPZ and JSON files. Write all regenerated artifacts and L5 experiment outputs to a **separately agreed working directory** outside `0.Processed-files/`; do not assume an output folder name without approval.

Validated source fingerprints: P1 CSV SHA256 `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e`; L3 master CSV SHA256 `abb285f95c28a59bb35cbb797faabdcaac894f1c7d04885b4e7789dd3d88067c`. The L4 NPZ manifests record export hashes and, where implemented, source fingerprints. Renaming a file does not alter its bytes, but **always verify the current file against its paired manifest before use**. Earlier M1 NPZs lack an independent row-level mapping; row alignment relies on validated upstream corpus-order evidence.

The separately provided `L4_Vectorization_Normalization_Demo.ipynb` is **not assumed to be executed or committed**. The project owner will run and upload it manually if desired; do not automatically upload notebooks to GitHub.

## Repository guide

| Path | Responsibility |
|---|---|
| `config.py`, `preprocessing.py` | Validated P0/P1/P2 configuration and preprocessing |
| `feature_engineering_config.py` | V0–V2, N0–N3 and S0–S3 selection |
| `feature_engineering.py` | V/N/S feature extraction |
| `master_feature_matrix.py` | Validate and assemble family CSVs; export SHA256 manifest |
| `vectorization.py`, `vectorization_config.py` | M1 sparse word/character TF-IDF representations |
| `feature_scaling.py`, `feature_scaling_config.py` | M2 engineered-feature scaling |
| `feature_integration.py` | M3 W0/W1/W2 block integration and exact distance calibration |
| `fasttext_representation.py` | M4 optional corpus-trained FastText and pooling |
| `tests/` | Preprocessing, feature and assembly regression tests |
| `docs/preprocessing_decisions.md` | L2 audit/decision history |
| `docs/feature_engineering_decisions.md` | L3 audits, changes, validation and limitations |
| `docs/vectorization_decisions.md` | L4-M1 audit and validation evidence |
| `docs/feature_scaling_decisions.md` | L4-M2 scaling audit and fitted references |
| `docs/feature_integration_decisions.md` | L4-M3 weighting coefficients and artifact verification |
| `docs/fasttext_decisions.md` | L4-M4 FastText feasibility and export verification |
| `notebooks/` | Demonstrations and user-approved notebook artifacts; consult folder for current versions |

## Install and test

```bash
git clone https://github.com/frankie-chew-nus/sentiment-clustering-ml.git
cd sentiment-clustering-ml
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m pytest -q
```

The spaCy model is required for P2. The development requirements include pinned VADER 3.3.2 and Gensim for FastText. A clean Colab runtime may require reinstalling dependencies.

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
MASTER_DIR = BASE / "0.Processed-files"  # READ-ONLY: never write here
P1_PATH = MASTER_DIR / "Tweets-processed-p1.csv"
L3_MASTER_PATH = MASTER_DIR / "Tweets-L3-master-features.csv"
W0_PATH = MASTER_DIR / "L4-W0.npz"  # pair with L4-W0.json
# Set a separate output directory only after the project owner approves it.
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
| [#26](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/26) | L4-M1 TF-IDF vectorization | 142 passed |
| [#27](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/27) | L4-M2 scaling | 153 passed |
| [#28](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/28) | L4-M3 integration and weighting | 169 passed |
| [#30](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/30) | M3 independent export evidence | Documentation only |
| [#31](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/31) | L4-M4 FastText | 179 passed |

Detailed audit findings, test corrections and evidence are maintained in `docs/`. The user reviews each PR, adds supporting evidence and merges manually.

## Next steps

**L4-M1–M4 are complete.** Begin L5 with a label-blind clustering methodology and feasibility review, not automatic K-means execution. Prespecify K range, algorithm (KMeans versus MiniBatchKMeans), random seeds, stability protocol, internal metrics and the seven-representation ablation plan. Compare inertia only within compatible feature spaces; do not use sentiment labels or confidence fields for selection. L6 is reserved for external label-based evaluation. L4-M5 was proposed as a handover task but has **not** been separately implemented or validated. PR #29 was a closed, **unmerged** revert proposal and must not be treated as undoing M3.
