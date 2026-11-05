# Model Exploration — Airline Tweet Sentiment Clustering

This folder contains three Jupyter notebooks documenting the exploratory modelling work for the MSI5001 airline-tweet sentiment-clustering project. Together, they investigate how data preparation, text representation, engineered sentiment signals, and clustering algorithm choices affect the structure and interpretability of clusters.

The task is **unsupervised clustering into three groups** using tweet content. The provided `airline_sentiment` and `sentiment_confidence` fields are reference information for post-clustering evaluation; they must not be used as clustering inputs or to make preprocessing, representation, scaling, model-selection, or tuning decisions.

## Notebooks in this folder

| Notebook | Main purpose | Suggested use |
|---|---|---|
| [Models exploration.ipynb](./Models%20exploration.ipynb) | Focused comparison of three representations (TF-IDF, VADER-derived lexicon features, and MiniLM sentence embeddings) with three clustering algorithms (K-Means, Gaussian Mixture Model, and Ward agglomerative clustering). Includes label-free model-selection diagnostics, a 3 × 3 comparison, interpretation, scaling/feature-ablation checks, and robustness analysis. | Start here for the compact representation-by-algorithm experiment and its interpretation. |
| [Models exploration1.ipynb](./Models%20exploration1.ipynb) | Staged research implementation covering the source-data audit and minimal preprocessing, TF-IDF baselines, VADER and hybrid features, FastText/BERTweet representations, algorithm comparisons, human–AI–reference annotation analysis, Twitter-RoBERTa experiments, sensitivity analyses, and evaluation-coverage checks. | Use as a detailed research record. Run only the stages needed for a particular investigation. |
| [Model exploration 2.ipynb](./Model%20exploration%202.ipynb) | A second staged notebook following a similar broad investigation: initial data audit, TF-IDF and VADER experiments, FastText and transformer embeddings, annotation workbooks, algorithm feasibility/comparison, hybrid-weight and K-Means sensitivity, RoBERTa algorithm comparisons, qualitative error analysis, and evaluation consistency. | Use alongside the staged research notebook when tracing later experiments and supporting diagnostics. |

The two staged notebooks overlap substantially in scope. They are maintained as separate exploratory records, not as two independent replications of every experiment. Check the specific notebook section, input files, configuration, and saved outputs before combining or comparing results.

## Research workflow represented

At a high level, the notebooks cover the following workstreams:

1. **Dataset audit and preparation** — inspect the 14,639-row source, missing values, duplicates/repeated text, tweet length, airline metadata, punctuation, and other text patterns; create working copies while preserving the original source.
2. **Text preprocessing and representation** — compare word-level TF-IDF, lexicon-derived sentiment signals (including VADER), hybrid representations, FastText embeddings, MiniLM/BERTweet embeddings, and sentiment-specialised Twitter-RoBERTa representations.
3. **Clustering** — investigate K-Means, Gaussian Mixture Models, and agglomerative clustering (including Ward linkage where applicable). Not every algorithm/representation combination is run in every notebook stage.
4. **Evaluation and interpretation** — use internal clustering diagnostics and, only after assignments are generated, external comparisons with the reference sentiment labels. Inspect cluster sizes, representative tweets, feature profiles, mismatches, and annotation evidence.
5. **Robustness and sensitivity** — examine random seeds, subsamples/held-out observations where implemented, feature scaling and weighting, hyperparameters, representation choices, and evaluation coverage.

## Data and label-use safeguards

- The project corpus contains 14,639 airline-related tweets. The core source fields are `text`, `airline_sentiment`, and `sentiment_confidence`.
- Use `text` and features derived from the tweet text as the unsupervised modelling inputs.
- Keep `airline_sentiment` and `sentiment_confidence` out of feature construction, vectorizer fitting, scaling, clustering, model selection, and tuning.
- The reference sentiment and confidence may be joined back only after clustering for external evaluation, confidence-stratified analysis, and error interpretation, as described in the notebooks.
- Cluster IDs (0, 1, 2) are arbitrary. They do not inherently mean negative, neutral, and positive; any correspondence must be established during post-hoc evaluation.
- Internal separation, repeatability, cluster balance, external label agreement, and human interpretability are different forms of evidence. None should be presented as a substitute for the others.

## Interpreting results

The focused 3 × 3 notebook reports one exploratory run (seed 42) in which the lexicon-feature representation outperformed its TF-IDF + K-Means baseline on the reported external metrics. Its saved interpretation recommends scaled lexicon features with K-Means for that comparison, while also discussing limitations such as the difficulty of isolating neutral sentiment, the effect of question-mark scaling, and VADER's sensitivity to context and sarcasm.

Treat those metrics and recommendations as **notebook-reported exploratory results**, not as a frozen project-wide conclusion or independently reproduced benchmark. The broader staged notebooks include additional representations, configurations, and analyses; results from separate notebook versions or experiment stages should not be merged without checking that their inputs, preprocessing, evaluation protocol, and configurations match.

In particular:

- A higher silhouette score or more balanced clusters does not by itself establish better recovery of negative, neutral, and positive sentiment.
- ARI, NMI, purity, macro-F1, internal scores, and stability scores answer different questions. State the metric and evaluation procedure when quoting a result.
- Confidence-stratified results use the reference confidence after clustering; they are post-hoc analyses, not unsupervised input signals.
- Human annotation and model-assisted analyses should be interpreted with their documented coverage, agreement, and limitations.
- Saved cell outputs may be stale relative to the current source files or code. Verify provenance before citing them.

## Running the notebooks

The notebooks were developed for interactive Jupyter environments, including VS Code/Jupyter and Google Colab. They contain staged workflows and may depend on local data files, cached embeddings, annotation workbooks, saved assignments, model checkpoints, and external pretrained model downloads.

Before running:

1. Obtain the project dataset and any stage-specific supporting files from the appropriate project location.
2. Update machine-specific paths and configure the expected input files for the notebook environment.
3. Install the dependencies required by the stage being run. The notebooks use Python data/scientific tooling such as pandas, NumPy, scikit-learn, SciPy, and plotting libraries, with additional packages for VADER/NLTK, Gensim/FastText, sentence-transformers/PyTorch, and Hugging Face Transformers in relevant stages.
4. Run cells in order within the selected stage, ensuring that prerequisite variables and artifacts have been created.
5. Record the notebook, stage, input-data version, package/model configuration, random seed, and output artifacts for any result intended for reporting.

Transformer embedding and model stages may download pretrained weights and require substantial storage, memory, and compute. A full clean-kernel run may therefore take materially more resources than the smaller classical-feature experiments. The notebooks do not imply that every referenced data file, cache, or checkpoint is included in this folder.

## Suggested reading order

1. Read the project-level dataset instructions and this folder guide.
2. Work through [Models exploration.ipynb](./Models%20exploration.ipynb) for the focused three-representation, three-algorithm comparison.
3. Use [Models exploration1.ipynb](./Models%20exploration1.ipynb) and [Model exploration 2.ipynb](./Model%20exploration%202.ipynb) to trace the broader staged experiments, annotation work, robustness checks, and later transformer investigations.
4. Before using a result in a report, verify its source notebook and section, data and artifact provenance, and whether the output was freshly reproduced.

## Reproducibility note

These notebooks are an additional exploratory track, separate from the repository's frozen L5 K=3 representation ablation and numbered L6 E0–E26 experiments. They are exploratory research artifacts rather than one canonical end-to-end pipeline. Prefer the project's separately documented and frozen experiment specifications when reproducing approved results. Preserve the original notebooks and source data; make changes in versioned working copies and document the exact configuration and outputs associated with each reported finding.
