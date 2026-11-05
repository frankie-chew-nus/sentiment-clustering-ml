# Model Exploration — Airline Tweet Sentiment Clustering

This folder contains the exploratory notebooks used to compare text representations, engineered sentiment signals, and clustering algorithms for the MSI5001 airline-tweet project.

The aim is to discover structure in tweet text without training the clustering process on the provided sentiment labels. The project dataset contains 14,639 tweets with `text`, `airline_sentiment`, and `sentiment_confidence`.

## Research safeguards

- Use tweet text and features derived from it as clustering inputs.
- Keep `airline_sentiment` and `sentiment_confidence` out of text preprocessing decisions, feature construction, scaling, clustering, model selection, and tuning.
- Use the reference label and its confidence only after assignments have been produced, for external evaluation and error analysis.
- Treat internal clustering scores, external label agreement, stability, and qualitative interpretability as different kinds of evidence.
- Do not interpret cluster IDs as inherently negative, neutral, or positive. Cluster numbers are arbitrary and must be mapped/evaluated after fitting.

The notebooks contain historical exploratory work and saved outputs. Their outputs should be treated as results reported by those notebook runs, not as independently reproduced or currently approved project findings.

## Notebooks

| Notebook | Scope | How to use it |
|---|---|---|
| [Models exploration.ipynb](./Models%20exploration.ipynb) | A focused 3 × 3 comparison of three representations and three clustering algorithms on the 14,639-tweet corpus. | Start here for the representation/algorithm comparison and its interpretation. |
| [Models exploration1.ipynb](./Models%20exploration1.ipynb) | A broader research-implementation notebook. Its introductory roadmap covers data audit, preprocessing, lexical/semantic representations, clustering, internal/external validation, weighting, sensitivity, human annotation, supplementary Twitter-RoBERTa work, and evaluation-coverage checks. | Use as a staged research record. Review its execution requirements and outputs before relying on any individual stage. |
| [Model exploration 2.ipynb](./Model%20exploration%202.ipynb) | An extensive staged notebook spanning the initial dataset audit, TF-IDF, VADER, FastText, transformer experiments, annotation workbooks, Twitter-RoBERTa, algorithm pilots, sensitivity analyses, and evaluation consistency checks. | Use as the broader development and investigation notebook; it contains multiple stages and should be run selectively with its dependencies and input files available. |

## Main comparison in `Models exploration.ipynb`

### Representations

1. **TF-IDF** — word unigrams and bigrams, representing lexical/topic similarity.
2. **Lexicon sentiment features** — VADER positive and negative scores plus question-mark count. These are text-derived features intended to capture praise, complaint wording, and questions.
3. **MiniLM** — 384-dimensional sentence embeddings from `all-MiniLM-L6-v2`, representing general sentence meaning.

### Clustering algorithms

Each representation is compared with:

- **K-Means**
- **Gaussian Mixture Model (GMM)**
- **Ward agglomerative clustering**

This produces nine representation/algorithm combinations. TF-IDF + K-Means is the stated baseline. The notebook fixes the target at three clusters and uses label-free criteria for choices such as GMM covariance type.

### Notebook-reported results (seed 42)

The saved interpretation in the notebook reports the following macro-F1 / adjusted Rand index (ARI) values after clustering and Hungarian matching to the reference labels:

| Representation | K-Means | GMM | Ward |
|---|---:|---:|---:|
| TF-IDF | 0.359 / 0.075 | 0.431 / 0.083 | 0.347 / 0.041 |
| Lexicon sentiment features | **0.563 / 0.195** | 0.486 / 0.095 | **0.565 / 0.182** |
| MiniLM | 0.391 / 0.079 | 0.437 / 0.110 | 0.345 / 0.031 |

The notebook identifies scaled lexicon features + K-Means as its recommended combination: macro-F1 0.563, ARI 0.195, and NMI 0.165. It reports that this is +0.204 macro-F1 and +0.120 ARI over the TF-IDF + K-Means baseline.

The notebook also reports several important qualifications:

- The lexicon-feature clusters are readable as statements/complaints, questions, and praise, but the question cluster contains many negative questions. Neutral sentiment remains difficult to isolate.
- Scaling matters: without scaling, question-mark count dominates the three-feature representation.
- GMM probabilities are reported as over-confident and weakly related to `sentiment_confidence`.
- VADER can misread context such as the airline name “united”, polite requests, and sarcasm.
- The notebook notes that the compact feature set was selected after comparing feature subsets, so the final feature choice was also informed by evaluation results.
- These are single-notebook run results (seed 42), not a substitute for a frozen, independently reproduced experiment or an out-of-sample validation study.

Use the notebook's own tables and diagnostics when quoting any further metric or per-cluster result. Do not mix these results with results from separate notebook versions or later experiment tracks.

## Data and execution

The notebooks refer to local or uploaded project data such as:

- `Tweets.csv` or `Tweets (1).xlsx` — the 14,639-row tweet dataset.
- Cached MiniLM embeddings and other saved vectors where used.
- Optional model checkpoints, annotation workbooks, frozen assignments, and output directories for later stages.

Paths in the notebooks include machine-specific examples. Before running, update the input path or upload the required file into the notebook environment. Do not assume that all referenced data, caches, or checkpoints are stored in this repository.

### Environment

The notebooks use Python and a combination of:

- pandas and NumPy
- scikit-learn
- vaderSentiment / NLTK VADER
- sentence-transformers and PyTorch
- Gensim (FastText)
- Transformers (including Twitter-RoBERTa experiments)
- SciPy, matplotlib, seaborn, and supporting utilities

Install only the packages required by the notebook or stage you intend to run. Transformer experiments may download model weights and require substantial memory; GPU availability may be useful but is not assumed for every stage.

## Recommended reading order

1. Read this README and the project-level dataset instructions.
2. Start with **Models exploration.ipynb** to understand the nine-model comparison.
3. Consult **Models exploration1.ipynb** and **Model exploration 2.ipynb** for the wider staged investigations and subsequent model families.
4. Before citing results, record the notebook filename, stage/section, execution configuration, input data version, and whether the saved output was freshly reproduced.

## Interpretation and reproducibility notes

- A higher internal silhouette score does not by itself imply better sentiment recovery.
- ARI, NMI, and macro-F1 measure different aspects of agreement with the reference labels; report the exact metric and evaluation procedure.
- Cluster balance and visual separation are descriptive diagnostics, not proof that clusters correspond to the three sentiment classes.
- Saved notebook outputs can become stale when inputs, package versions, preprocessing, or model checkpoints change.
- The exploratory notebooks are not a single canonical pipeline. Prefer the project's frozen, documented experiment specifications when reproducing approved results.

