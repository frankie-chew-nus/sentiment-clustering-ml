# Model Ablation — Four-Stage Airline Sentiment Study

This folder documents the **model-ablation showcase** for the MSI5001 airline-tweet project. It asks:

> As NLP representations evolve from statistical features to learned embeddings and pretrained language models, do they improve the recovery of negative, neutral and positive airline sentiment? How much does the clustering algorithm matter?

The study uses **14,639 redacted airline tweets**. It compares three **unsupervised clustering** algorithms across three text-representation families, followed by a **separate zero-shot classification** benchmark. The purpose is to compare complete modelling approaches, not to claim that newer models must be better.

## 1. Study design

| Stage | Approach | Text preparation | Representation / task | Models evaluated |
|---|---|---|---|---|
| **1 — Statistical NLP** | Word TF-IDF | P1-equivalent cleaning and normalization | Word unigrams and bigrams | K-means (E0), GMM (E14), Agglomerative Ward (E15) |
| **2 — Neural word embeddings** | FastText | Same P1-equivalent preprocessing | Corpus-trained 100-dimensional word vectors, IDF-weighted tweet pooling, L2 normalization | K-means (E6), GMM (E12), Agglomerative Ward (E13) |
| **3 — Contextual Transformers** | MiniLM | Original redacted tweet text, without additional P0/P1 cleaning | Pretrained `all-MiniLM-L6-v2` 384-dimensional sentence embeddings | K-means (E16), GMM (E17), Agglomerative Ward (E18) |
| **4 — Foundation-model classification** | Qwen2.5-3B-Instruct | Original redacted tweet text | Zero-shot instruction: predict negative, neutral or positive | Direct classification (E19); no clustering |

**Key controls:** Same 14,639 observations and source ordering; clustering experiments target **K=3**, with seed **5001** where applicable. Human sentiment labels are withheld during representation building and clustering and used only after predictions are fixed for external evaluation. E19 receives the *names* of the three sentiment classes in its instruction, but not labelled training examples.

**Important differences:** E14 and E15 apply **50-dimensional TruncatedSVD** after word TF-IDF for computational feasibility; E0 uses the original sparse TF-IDF representation. Ward linkage uses **Euclidean distance**. Therefore, not all cross-model comparisons isolate the clustering algorithm alone.

## 2. Experiment inventory

| Experiment | Representation | Algorithm | Role |
|---|---|---|---|
| E0 | Word TF-IDF | K-means | Reused baseline |
| E14 | Word TF-IDF + 50D SVD | GMM | New model-ablation run |
| E15 | Word TF-IDF + 50D SVD | Agglomerative, **Ward / Euclidean** | Updated linkage run |
| E6 | FastText 100D | K-means | Reused baseline |
| E12 | FastText 100D | GMM | Reused run |
| E13 | FastText 100D | Agglomerative, **Ward / Euclidean** | Updated linkage run |
| E16 | MiniLM 384D | K-means | New model-ablation run |
| E17 | MiniLM 384D | GMM | New model-ablation run |
| E18 | MiniLM 384D | Agglomerative, **Ward / Euclidean** | Updated linkage run |
| E19 | Qwen2.5-3B-Instruct | Zero-shot classification | Separate task-directed benchmark |

Earlier **Average-linkage** Agglomerative results were retained as historical sensitivity evidence. The main table below reports the subsequently verified **Ward-linkage** results for E13, E15 and E18.

## 3. Evaluation protocol

- **ARI (higher is better):** Chance-adjusted agreement between the discovered groups and human sentiment labels.
- **NMI (higher is better):** Shared information between group assignments and sentiment labels.
- **Purity (higher is better):** Fraction of observations belonging to the majority true class in each predicted group; can be inflated by class imbalance.
- **Accuracy (higher is better):** For E0–E18, **optimally mapped cluster accuracy** after one-to-one Hungarian cluster-to-sentiment matching. For E19, **direct classification accuracy** without cluster mapping.
- **Macro F1 (higher is better):** Unweighted average of negative, neutral and positive F1, highlighting uneven class performance.
- **Silhouette (higher is better):** Internal geometric separation, calculated on a fixed 2,000-tweet sample for clustering. It is **not** sentiment accuracy. Not applicable to E19's direct classification.
- **Per-class precision, recall, F1 and contingency/confusion matrices:** Retained as necessary diagnostics, especially for neutral sentiment.

The corpus is imbalanced: **9,178 negative**, **3,099 neutral** and **2,362 positive** tweets (majority negative approximately **62.70%**). Overall accuracy and purity must therefore be interpreted with Macro F1 and class-level results.

## 4. Consolidated results

All values below are rounded to four decimal places from the previously executed notebooks and verified metrics artifacts.

| Stage | Experiment | ARI ↑ | NMI ↑ | Purity ↑ | Accuracy ↑ | Macro F1 ↑ | Silhouette ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | E0 — TF-IDF + K-means | 0.1277 | 0.0970 | 0.6643 | 0.6507 | 0.3825 | 0.0047 |
| 1 | E14 — TF-IDF + GMM | 0.1318 | 0.0950 | 0.6651 | 0.5537 | **0.5087** | 0.0256 |
| 1 | E15 — TF-IDF + Ward | 0.1089 | 0.0900 | 0.6602 | 0.6401 | 0.3562 | 0.0404 |
| 2 | E6 — FastText + K-means | 0.0954 | 0.0968 | 0.6786 | 0.4659 | 0.4555 | 0.0795 |
| 2 | E12 — FastText + GMM | 0.0884 | 0.0700 | 0.6270 | 0.4463 | 0.4025 | 0.0565 |
| 2 | E13 — FastText + Ward | 0.0757 | 0.0938 | 0.6725 | 0.4837 | 0.4297 | 0.0556 |
| 3 | E16 — MiniLM + K-means | **0.1325** | 0.0903 | 0.6486 | 0.4987 | 0.4357 | 0.0456 |
| 3 | E17 — MiniLM + GMM | 0.1120 | 0.0793 | 0.6513 | 0.4834 | 0.4364 | 0.0376 |
| 3 | E18 — MiniLM + Ward | 0.0551 | 0.0221 | 0.6337 | 0.5119 | 0.3659 | 0.0325 |
| 4 | E19 — Qwen zero-shot | **0.5308** | **0.4004** | **0.7927** | **0.7927** | **0.6697** | N/A |

**Comparability:** E19's accuracy is direct classification accuracy, while E0–E18 use label-informed optimal mapping of otherwise unnamed clusters. These are distinct evaluation settings. Silhouette scores span different feature spaces and must not be treated as a single standardized ranking. E19 purity happens to equal its direct accuracy in this run; that equality is not guaranteed generally.

## 5. What the showcases demonstrated

### 5.1 Newer embeddings did not guarantee better unsupervised sentiment recovery

Among the **unsupervised** runs, **E16 MiniLM + K-means** achieved the highest ARI (**0.1325**), only slightly above **E14 TF-IDF + GMM** (**0.1318**). **E14** had the highest Macro F1 (**0.5087**), exceeding the MiniLM runs.

A plausible explanation is **objective mismatch**: general-purpose sentence embeddings encode semantic similarity, which can reflect topic or situation rather than sentiment polarity. For example, complaints and praise about the same flight delay may be semantically related but emotionally opposite. This mechanism is a **hypothesis**, not directly proven by these scores.

### 5.2 The clustering algorithm and its configuration matter

K-means, GMM and Agglomerative Clustering produced different partitions for each representation. The original **Average-linkage** Agglomerative runs often collapsed into a dominant cluster. Switching to **Ward / Euclidean** improved ARI and Macro F1 for TF-IDF, FastText and MiniLM compared with their earlier Average-linkage configurations, but Ward still did not lead the unsupervised comparison.

For FastText and MiniLM, the Ward reruns changed **both linkage and distance** (Average/cosine to Ward/Euclidean). The observed improvement cannot be assigned exclusively to linkage.

### 5.3 Better cluster geometry is not equivalent to sentiment accuracy

Internal silhouette and external label-agreement metrics answer different questions. Earlier Average-linkage runs demonstrated that apparently high silhouette can accompany near-single-cluster collapse. The current Ward results are more interpretable, but silhouette still measures geometry **within each representation**, not sentiment meaning.

### 5.4 Task-directed foundation-model classification performed substantially better

**E19 Qwen zero-shot** achieved **79.27% direct accuracy**, **0.6697 Macro F1**, **0.5308 ARI** and **0.4004 NMI**. It was explicitly instructed to predict sentiment, unlike unsupervised clustering.

This does **not** establish that foundation-model embeddings inherently outperform statistical or neural representations. It shows that **pretrained language understanding combined with an explicit sentiment-classification task** performed better than the tested unsupervised sentiment-discovery approaches.

### 5.5 Neutral sentiment remained difficult

E19's reported recall was approximately **95.45% negative**, **23.14% neutral** and **90.09% positive**. Despite strong overall accuracy, the model missed most neutral tweets. Class-level metrics therefore remain essential.


## 6. Prompt refinement follow-up: E19–E23

After the first E19 evaluation, prompt iterations investigated why the model missed many human-labelled neutral tweets. These are **exploratory, label-informed prompt experiments**: results on this corpus were inspected to motivate later prompts, so they are not independent confirmation of generalization.

| Experiment | Prompt strategy | Direct accuracy ↑ | Macro F1 ↑ | Neutral recall |
|---|---|---:|---:|---:|
| E19 | Original, minimal three-label instruction | 79.27% | 0.6697 | 23.14% |
| E20 | Explicit sentiment definitions and neutral boundaries | 82.30% | 0.7529 | 47.18% |
| E21 | Two-stage neutral-vs-opinionated, then polarity | 76.17% | 0.6829 | 48.85% |
| E22 | Neutral-boundary refinement: operational issues and politeness are not sentiment by themselves | 81.99% | 0.7533 | 50.66% |
| E23 | Context-aware question interpretation; punctuation is not sentiment by itself | 82.30% | **0.7603** | **52.92%** |

These follow-ups use the same Qwen2.5-3B-Instruct model and deterministic zero-shot classification. They differ in prompting/decision strategy, not model size or supervised fine-tuning. Values are from the previously reviewed executed notebooks; retain the corresponding metrics/manifests as the evidence source.

### Main prompt findings

1. **The original prompt under-specified class definitions.** E19 had strong negative and positive recall, but neutral recall of only 23.14%. The model appeared reluctant to predict neutral.
2. **Explicit class definitions helped substantially.** E20 raised Macro F1 from 0.6697 to 0.7529 and neutral recall from 23.14% to 47.18%, without changing the foundation model.
3. **A two-stage design was not the best trade-off.** E21 improved neutral recall compared with E19 but had the lowest overall accuracy and weak neutral precision; the first-stage neutral decision was difficult to correct later.
4. **Additional neutral-boundary rules gave only a small overall change.** E22 slightly exceeded E20 on Macro F1 (0.7533 vs 0.7529), but was a near tie on aggregate performance.
5. **Context-aware question instructions produced the strongest aggregate balance.** E23 achieved the highest recorded Macro F1 (0.7603), ARI (0.5873) and NMI (0.4347), while matching E20's rounded accuracy (82.30%). E23's neutral recall rose to 52.92%, but it still missed 1,459 of the 3,099 human-labelled neutral tweets.

### E23 per-class results

| Sentiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 90.15% | 90.43% | 0.9029 |
| Neutral | 68.30% | 52.92% | 0.5964 |
| Positive | 69.55% | 89.25% | 0.7818 |

E23's remaining neutral misclassifications were split between negative (818) and positive (641) predictions. This suggests the residual problem is not solely a one-directional negative bias.

### Error-analysis hypotheses

Inspection of E20 errors found associations between neutral tweets misclassified as negative and terms/signals such as negation, cancellation and requests for help; neutral tweets misclassified as positive more often contained exclamation marks, gratitude and positive vocabulary. These are **descriptive associations, not causal explanations**. Correctly classified neutral tweets also often contained question marks, so punctuation alone did not explain the errors.

E23 therefore instructed Qwen to judge questions by meaning, intent and full context:
- factual information requests without an expressed attitude should be neutral;
- rhetorical questions that express criticism or frustration should be negative;
- questions that genuinely express praise should be positive;
- a question mark alone should not determine sentiment.

This targeted prompt was associated with improved results, but because it was developed after examining labelled errors on the same corpus, the result remains exploratory. A separate development set and untouched test set—or new independently annotated tweets—are needed for confirmatory claims.

### Current recommendation

**E23 is the leading exploratory prompt configuration so far**, based on Macro F1 and neutral recall. E20 remains a reasonable simpler alternative: it achieves the same rounded accuracy and is marginally below E23 on Macro F1. There is no evidence here that the small E20/E22/E23 aggregate differences are statistically significant. Retain the original E19 baseline and all prompt versions for transparency.


## 7. Research conclusion

> **There is no demonstrated one-size-fits-all text preparation, feature representation or clustering configuration for this corpus.** Increasing representation sophistication alone did not consistently improve unsupervised sentiment recovery. Results depended on the interaction among preprocessing, representation, dimensionality reduction, clustering method and evaluation objective.
>
> **Direct zero-shot classification was much more effective at identifying the specified sentiment labels**, but it addresses a different, task-directed problem from unsupervised cluster discovery.

The showcase traces representative NLP approaches from statistical features through word embeddings and contextual Transformers to an instruction-following foundation model. It is **not** a controlled historical study proving that advances over time caused particular score changes.

## 8. Limitations and reproducibility

1. **Single corpus:** Findings are specific to the 14,639 airline tweets and tested configurations; they do not establish general performance on other domains.
2. **Exploratory comparisons:** Prior E0–E13 results informed subsequent choices. The original annotations are not an untouched independent test set for every later design decision.
3. **Confounded comparisons:** Stage 1 changes representation dimensionality (SVD) as well as clustering method. Stages 1–2 use P1-equivalent cleaning, whereas Stages 3–4 use raw redacted text.
4. **No uncertainty analysis:** Differences such as E16 ARI 0.1325 versus E14 ARI 0.1318 have not been shown statistically significant.
5. **Different task types:** E19 is zero-shot classification, not a fourth clustering algorithm. It is presented separately.
6. **Reproducibility:** Preserve source CSV fingerprints, row order, model versions, configurations, cluster assignments or frozen predictions, metrics JSON, manifests and figures. Qwen's saved E19 predictions were originally generated under Transformers **4.51.3** and later reused for evaluation in an environment with Transformers **4.57.6**; this does not prove fresh inference would be identical.
7. **Data protection:** Do not commit original sentiment labels, tweet-level checkpoint outputs or secret credentials to GitHub. The redacted input data and evidence artifacts should remain in their designated storage locations.

## 9. Repository organization

The reusable preprocessing and feature-building modules live in the repository's source code. These notebooks are **experiment assemblies**, not new reusable library functions.

- [Earlier E0–E7 exploration](../data%20exploration/README.md) — representation and weighting ablations.
- [E19 Qwen zero-shot notebook](L6_E19_Qwen_ZeroShot_E0_Style_Install.ipynb) — the latest committed model-ablation notebook, with execution outputs cleared.

**Evidence inventory:** E0, E6, E12, E13 and E14–E18 were executed and reviewed in the project's Colab/Drive workflow. This README summarizes their recorded scores; it does **not** imply all those notebooks or their raw outputs are currently committed to this folder. Add them separately if the repository is intended to contain the complete model-ablation notebook collection.
