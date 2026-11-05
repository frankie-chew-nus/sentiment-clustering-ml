# Models Exploration — Airline Tweet Sentiment Clustering

This folder contains three related Jupyter notebooks from the MSI5001 airline-tweet project. They document an early focused model comparison and two broader, overlapping research-workflow notebooks. The uploaded local copies were reviewed to prepare this guide; the GitHub notebook files were not opened.

The notebooks use the supplied sentiment labels as evaluation references, not as inputs to unsupervised feature construction or clustering. However, some later sections explicitly reveal labels after assignments are frozen for external evaluation, human–AI–reference triangulation, and error analysis. The sentiment-specialised Twitter-RoBERTa checkpoint also carries sentiment knowledge learned from external training data; this is distinct from leakage of this project's own labels.

## Notebooks

| Notebook | Scope | How it relates to the others |
|---|---|---|
| [Models exploration.ipynb](./Models%20exploration.ipynb) | Focused comparison of three representations × three clustering algorithms, with EDA, label-free model selection, post-clustering evaluation, cluster interpretation, feature/scaling checks and robustness analysis. | A self-contained seed-42 experiment using 14,639 rows. It is not the same protocol as the later L6 E0–E7 standalone K-means experiments. |
| [Models exploration1.ipynb](./Models%20exploration1.ipynb) | Broad research implementation spanning dataset audit, TF-IDF, VADER/hybrid features, FastText, BERTweet, human annotation work, Twitter-RoBERTa, algorithm comparison, sensitivity and evaluation-coverage auditing. | One edition of the long staged workflow. Includes environment guidance and an integrated research framing. |
| [Model exploration 2.ipynb](./Model%20exploration%202.ipynb) | A closely related staged workflow covering the same broad sequence, with a more stepwise Colab-oriented opening and an explicit text-artifact audit in Stage 1. | An overlapping notebook edition, not an independent replication of Models exploration1. The two long notebooks differ in cell structure and some early audit/setup material; preserve their provenance separately. |

The two long notebooks have 216 and 217 cells respectively and share the same broad stages through Stage 20. They should be treated as related working versions, not as independent runs whose results can be averaged or counted as replications.

### Version and evidence policy

- **Keep both long-form notebooks:** they contain overlapping stages but differ in setup, cell structure and some early auditing material. Neither should be deleted or renamed solely on the basis of this README review.
- **Do not designate a canonical long-form notebook yet:** a definitive successor/predecessor relationship and cell-by-cell equivalence have not been established. Choose a notebook for execution only after checking its required inputs, dependencies and stage-specific changes.
- **Avoid double-counting results:** if the same experiment, artifact or saved metric appears in both long notebooks, cite its underlying run once rather than treating the copies as independent confirmations.
- **Use unambiguous references:** identify every result by notebook filename, stage, row count, representation, clustering algorithm, seed and saved-artifact provenance. Local E0/E7 identifiers alone are not unique across this repository.
- **Separate historical from approved evidence:** saved outputs in these notebooks are exploratory records; the frozen L5 and canonical L6 evidence retain their separately documented status.

## 1. Focused experiment: three representations × three algorithms

[Models exploration.ipynb](./Models%20exploration.ipynb) uses 14,639 tweets and a fixed random seed of 42. It compares:

**Representations**

- **TF-IDF:** word unigrams and bigrams.
- **Lexicon features:** three scaled variables — VADER negative score, VADER positive score, and question-mark count.
- **MiniLM:** sentence embeddings from `all-MiniLM-L6-v2`.

**Clustering algorithms**

- K-Means
- Gaussian Mixture Model (GMM)
- Ward agglomerative clustering

This produces nine representation/algorithm combinations. TF-IDF + K-Means is the stated baseline. The notebook uses label-free diagnostics for choices such as cluster-count exploration and GMM covariance selection, then introduces the project sentiment labels in its evaluation section.

### Important comparability qualification

The nine combinations do not all receive identically transformed inputs. The notebook applies 50-component TruncatedSVD to TF-IDF for GMM and Ward, and 50-component PCA to MiniLM for GMM. K-Means uses the original TF-IDF and MiniLM representations; Ward uses the full MiniLM embedding. The lexicon feature block is scaled before clustering. Consequently, cross-algorithm differences can reflect both the algorithm and the representation's dimensionality/preparation. They are not a perfectly controlled algorithm-only ablation.

### Notebook-reported results (seed 42)

The following values are copied from the saved result table in the notebook. Accuracy is obtained after one-to-one Hungarian cluster-to-sentiment matching; ARI and NMI compare the partition directly with the reference labels.

| Representation | Algorithm | ARI | NMI | Macro-F1 | Silhouette |
|---|---|---:|---:|---:|---:|
| TF-IDF | K-Means | 0.075 | 0.070 | 0.359 | 0.006 |
| TF-IDF | GMM | 0.083 | 0.047 | 0.431 | 0.014 |
| TF-IDF | Ward | 0.041 | 0.059 | 0.347 | 0.031 |
| Lexicon features | K-Means | 0.195 | 0.165 | 0.563 | 0.407 |
| Lexicon features | GMM | 0.095 | 0.127 | 0.486 | 0.368 |
| Lexicon features | Ward | 0.182 | 0.157 | 0.565 | 0.406 |
| MiniLM | K-Means | 0.079 | 0.068 | 0.391 | 0.043 |
| MiniLM | GMM | 0.110 | 0.088 | 0.437 | 0.031 |
| MiniLM | Ward | 0.031 | 0.029 | 0.345 | 0.015 |

The notebook recommends scaled lexicon features + K-Means for this experiment: Macro-F1 0.563, ARI 0.195 and NMI 0.165. Lexicon + Ward has a marginally higher Macro-F1 (0.565), but lower ARI/NMI and less favourable subsample stability in the notebook's analysis. K-Means is also suitable for assigning future observations with `predict()`, unlike the fitted agglomerative model.

The notebook interprets the three lexicon clusters as statements/complaints, questions, and praise. This is a useful qualitative description, not proof that the groups are clean negative/neutral/positive classes. In particular, many negative complaints are phrased as questions.

Other notebook findings include:

- Scaling is essential. Without scaling, question-mark count dominates the lexicon representation; the K-Means and Ward partition largely by the number of question marks.
- Removing VADER positive or question-mark features reduces the reported external scores; VADER positive is especially influential in the feature-ablation table.
- GMM probabilities are over-confident and show little association with the supplied sentiment-confidence values.
- The notebook reports better clustering agreement with reference labels among higher-confidence tweets, but the low-confidence bins are small and this is post-hoc analysis.
- The compact experiment is one saved seed-42 run, not a frozen or independently reproduced benchmark.

## 2. Long-form research workflow: Models exploration1 and Model exploration 2

The two longer notebooks cover a much wider exploratory sequence. Their common research flow includes:

| Stage | Workstream | Main contents |
|---|---|---|
| Stage 1 | Dataset audit and preparation | Load the original source; inspect missingness, duplicates/repeated text, label distribution for audit only, numeric expressions, tweet artefacts and airline identity; create a minimally transformed working copy. |
| Stages 3–7 | Classical and lexical representations | TF-IDF unigram baseline, alternative representations, VADER coverage/lexicon audit and VADER-based clustering. |
| Stage 8 | Hybrid features | Combine TF-IDF with VADER-derived signals and compare feature-weight configurations. |
| Stage 9 | FastText | Train corpus-specific FastText on the unlabeled tweet text, pool word embeddings into tweet vectors, and cluster. |
| Stage 10 | Representation and algorithm comparisons | Compare representations and clustering algorithms; include a BERTweet contextual-embedding benchmark. |
| Stages 11–12 | Assignment and annotation analysis | Freeze assignments, inspect cluster sentiment composition and model disagreements, and generate blinded human annotation workbooks. |
| Stage 14 | Scaling sensitivity | Revisit VADER reproducibility and compare scaled and unscaled lexicon features. |
| Stage 15B | Human–AI–reference triangulation | Compare independent human annotations, reference sentiment/confidence and frozen BERTweet/VADER-based predictions. No model retraining occurs in this audit. |
| Stage 16 | Sentiment-specialised Twitter-RoBERTa | Compare contextual embeddings and sentiment-probability representations; test username/URL normalization; perform label-blind clustering and then external evaluation. A direct classifier comparison is also included. |
| Stage 17 | TF-IDF algorithm feasibility | Pilot K-Means, GMM and Agglomerative clustering on a 3,000-tweet sample, including memory, singleton and assignment audits. |
| Stage 18 | Hybrid and K-Means sensitivity | Examine text/sentiment weighting and K-Means initialization/convergence settings. |
| Stage 19 | RoBERTa × clustering algorithm | Compare K-Means, diagonal-covariance GMM and Ward Agglomerative clustering on the same 14,426 unique-tweet RoBERTa embedding set; freeze assignments before external evaluation and inspect errors. |
| Stage 20 | Evaluation coverage | Audit experiment registers and saved metrics for coverage/consistency; no models are retrained in this final audit. |

### Data audit and row counts

The long-form notebooks begin with 14,639 source rows. Their duplicate audit reports 99 exact duplicate rows and 213 repeated-text rows, with 187 repeated-text groups; 25 groups have conflicting supplied sentiment labels. The notebooks create a modelling dataset with 14,426 unique text values for later stages. This differs from the 14,639-row corpus used in the focused seed-42 notebook and other experiment tracks.

The duplicate-handling choice changes the observation set and therefore affects comparability. Do not silently combine scores from the 14,639-row and 14,426-row workflows.

### Selected Stage 19 result: sentiment-specialised RoBERTa

The saved Stage 19G output compares three clustering algorithms on 14,426 raw-text Twitter-RoBERTa embeddings. The project reference labels were introduced only after assignments were frozen.

| Algorithm | Silhouette | External ARI | NMI | Purity / mapped accuracy | Macro-F1 | Neutral recall |
|---|---:|---:|---:|---:|---:|---:|
| K-Means | 0.28849 | 0.49882 | 0.40535 | 0.78844 | **0.75127** | **0.67910** |
| GMM (diagonal) | 0.28055 | 0.46678 | 0.39849 | 0.77097 | 0.73845 | 0.68858 |
| Ward Agglomerative | **0.29090** | **0.53962** | 0.40042 | **0.80778** | 0.74592 | 0.49951 |

These are notebook-saved results, not independently reproduced estimates. Agglomerative leads on ARI, purity/mapped accuracy and silhouette; K-Means leads on Macro-F1; GMM has slightly higher neutral recall than K-Means. The differences demonstrate metric trade-offs rather than a single universal winner.

**External knowledge qualification:** The Twitter-RoBERTa checkpoint was fine-tuned using external sentiment-labelled data. The clustering sections withhold this project's sentiment labels, but the representation is not devoid of sentiment supervision in its pretraining history. It should not be described as equivalent to a model trained from scratch without sentiment knowledge.

## 3. Relationship to the repository's L5/L6 experiment tracks

This folder's long notebooks use their own stage and experiment labels. Some labels overlap with the later, canonical L6 experiment series, but they do **not** always identify the same configuration:

- The long-form notebooks call TF-IDF unigram clustering **E0** and FastText + K-Means **E7**.
- The canonical [L6 E0–E7 standalone experiments](../data%20exploration/README.md) use E0 for word TF-IDF and E7 for a later sentiment-focused hybrid.

Therefore, an experiment label such as **E7 is ambiguous without its notebook and configuration**. Do not join or compare results by experiment ID alone. Record the notebook filename, stage/section, representation, algorithm, row count, preprocessing, random seed and evaluation protocol.

Likewise, this folder's seed-42 nine-model comparison, its 14,426-row staged RoBERTa work, the frozen label-blind L5 K=3 evidence, and the canonical L6 E0–E7 standalone evaluation are distinct experiments. Do not merge their metric tables.

## 4. Label-use and interpretation safeguards

- The source fields `airline_sentiment` and `sentiment_confidence` are not clustering inputs. In the long-form notebooks, labels/confidence are used in explicitly marked audit-only or post-clustering sections.
- Keep the original project labels out of preprocessing, feature extraction, vectorization, scaling, feature selection, clustering, model selection and tuning.
- Human annotation and confidence-stratified analysis are post-hoc evidence, not unsupervised features.
- Cluster IDs are arbitrary. Any negative/neutral/positive mapping is created only during post-clustering evaluation.
- Internal silhouette, seed stability, external ARI/NMI, mapped accuracy, Macro-F1 and qualitative interpretability answer different questions.
- A sentiment-specialised pretrained model uses externally learned sentiment information even when the project's own labels are withheld.

## 5. Execution and reproducibility

The long-form notebooks contain machine-specific local paths, interactive upload/setup instructions, optional model downloads, cached or saved artifacts and compute-intensive stages. They are not a clean single-command pipeline. Run only the intended sections after checking their prerequisites and paths.

Before citing a result, preserve:

1. The exact notebook filename and stage/cell range.
2. Source-data fingerprint and row alignment (including whether duplicates were removed).
3. Representation, preprocessing and feature configuration.
4. Algorithm, parameters, random seed and dimensionality-reduction steps.
5. Frozen assignments, metrics, manifests and relevant model/checkpoint versions.
6. Whether the result came from saved historical output or a fresh execution.

The saved notebook outputs are useful evidence of the recorded runs, but do not prove that a fresh run with current package versions, model weights and local inputs will reproduce them exactly.

## Suggested reading order

1. Read this guide and the [repository root README](../../README.md).
2. Start with [Models exploration.ipynb](./Models%20exploration.ipynb) for the compact three-by-three comparison.
3. Read [Models exploration1.ipynb](./Models%20exploration1.ipynb) and [Model exploration 2.ipynb](./Model%20exploration%202.ipynb) as overlapping versions of the broader staged investigation. Compare their audit/setup changes before choosing one to execute.
4. For the distinct L6 runs, consult [data exploration](../data%20exploration/README.md), [model ablation](../model%20ablation/README.md) and [general-model investigation](../general-model-investigation/README.md).
