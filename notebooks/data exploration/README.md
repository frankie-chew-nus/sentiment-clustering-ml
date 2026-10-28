# MSI5001 — L6 Standalone Clustering Experiments (E0–E7)

**Purpose:** Document the independent L6 experiment notebooks, their design choices, measured results, and interpretation. This README belongs **beside the notebooks** in `notebooks/data exploration/`, not in the repository's root README. It documents experiments rather than reusable library functions.

**Status:** E0–E7 executed; E4 was rerun after correcting E3-prefixed output filenames. The figures below are taken from the supplied executed notebooks and should be checked against the saved evidence artifacts before formal publication.

## 1. What we were trying to learn

Can unsupervised K-means clustering discover three groups of airline tweets that correspond to the original **negative, neutral and positive** human sentiment annotations? More specifically, how do text cleaning, numerical text representations, sentiment-oriented features and their weights affect the results?

We used **14,639 airline tweets**, **K=3**, `k-means++`, `n_init=10`, `max_iter=300`, `tol=0.0001`, and **random seed 5001**. Every experiment generated its own inputs afresh from the redacted tweet text, using reusable GitHub implementation modules; **none loaded the previously generated L1–L5 feature matrices or cluster assignments**. The original labelled CSV was accessed only in the evaluation stage after cluster generation and row-order/text alignment checks.

Cluster numbers 0, 1 and 2 have **no fixed sentiment meanings**. We never assumed cluster 0 was negative. For mapped accuracy and per-class precision/recall/F1, clusters were matched one-to-one to sentiment classes with the Hungarian algorithm. ARI and NMI do not need such mapping.

## 2. What each notebook tested

| Notebook | Experiment in plain language | Key design choice |
|---|---|---|
| **E0** | Word-based text only | Clean/normalize tweets, describe emojis, lowercase, expand contractions; word TF-IDF with single words and adjacent word pairs. |
| **E1** | Character patterns only | Basic airline/URL/mention cleanup while retaining case and punctuation; character TF-IDF using 3–5-character sequences. |
| **E2** | Combined word and character patterns | Concatenate E0-style word and E1-style character TF-IDF; no engineered sentiment features. |
| **E3** | Text plus sentiment/expression signals | E2 text plus VADER compound polarity, negation count, exclamation count and uppercase-word ratio; direct concatenation without block calibration. |
| **E4** | Give all five blocks equal influence | Same five blocks as E3; calibrate each block to **20% expected squared-distance contribution** (word, character, VADER, negation, structural). |
| **E5** | Prioritize text over engineered signals | Same blocks; **70%** total expected squared-distance contribution to word+character text and **30%** to VADER+negation+structural features. |
| **E6** | Test learned semantic word relationships | Corpus-trained, 100-dimensional FastText embeddings from cleaned/lowercased tweets, with IDF-weighted pooling. |
| **E7** | Exploratory custom sentiment-focused hybrid | Word TF-IDF **70%**, character TF-IDF **10%**, VADER positive/negative/compound **15%**, negation **5%**; exclude structural expression block. |

**Note:** The percentages above are calibrated contributions to *expected squared distance*, not feature counts, model accuracy, or learned importance. E7 is a newly designed, **label-informed exploratory** configuration, not one of the frozen E0–E6 baselines.

## 3. What the scores mean

- **ARI (higher better; 1 perfect, ~0 chance-level):** chance-adjusted agreement between discovered clusters and human sentiment classes.
- **NMI (0–1, higher better):** information shared between cluster assignments and sentiment classes.
- **Purity (higher better):** fraction of tweets belonging to the majority sentiment of their cluster; may flatter imbalanced data.
- **Optimally mapped accuracy (higher better):** proportion correct after the best one-to-one cluster-to-class mapping; not predictive accuracy on unseen tweets.
- **Per-class precision / recall / F1:** performance for each sentiment *after* optimal mapping. **Macro F1** averages the three class F1 scores equally, highlighting minority-class failures.
- **Silhouette (−1 to 1, higher means better geometric separation):** internal cluster compactness/separation in the representation space; **not a sentiment agreement score**. Calculated using the fixed label-blind 2,000-tweet sample protocol.
- **Contingency matrices and within-cluster sentiment percentages:** show which sentiments actually populate each cluster; a clean three-sentiment recovery would have a different dominant sentiment in each cluster.

The original label distribution is imbalanced: **9,178 negative, 3,099 neutral and 2,362 positive** tweets (negative approximately **62.7%**). This makes mapped accuracy and purity alone insufficient for declaring a winner.

## 4. Consolidated measured results

| Experiment | ARI ↑ | NMI ↑ | Purity ↑ | Mapped accuracy ↑ | Macro F1 ↑ | Silhouette ↑ |
|---|---:|---:|---:|---:|---:|---:|
| E0 | **0.127673** | 0.097009 | 0.664253 | **0.650728** | 0.382478 | 0.004727 |
| E1 | 0.065613 | 0.108383 | 0.668966 | 0.615684 | 0.409713 | 0.003365 |
| E2 | 0.011965 | 0.055120 | 0.644853 | 0.591639 | 0.333360 | 0.002403 |
| E3 | 0.043477 | **0.126191** | 0.626955 | 0.458706 | 0.444792 | 0.050672 |
| E4 | 0.052099 | 0.095692 | 0.626955 | 0.491973 | 0.479529 | **0.227258** |
| E5 | 0.052237 | 0.095790 | 0.626955 | 0.492110 | **0.479657** | 0.109319 |
| E6 | 0.095433 | 0.096847 | **0.678598** | 0.465947 | 0.455532 | 0.079546 |
| E7 (exploratory) | 0.030345 | 0.120703 | 0.626955 | 0.447503 | 0.434275 | 0.0694 |

*Numbers are descriptive results from the executed notebooks, not uncertainty-adjusted estimates or evidence of out-of-sample generalization. E7 silhouette is displayed rounded as reported in the executed analysis.*

## 5. Observations from E0–E6

1. **E0 led on ARI and mapped accuracy.** Single-word and two-word text patterns retained useful sentiment information. But its Macro F1 was low (0.3825), reflecting uneven class recovery, especially for neutral tweets.
2. **E2 underperformed.** Adding character n-grams to word n-grams without engineered features did not automatically improve sentiment grouping; its ARI was only 0.0120.
3. **E3 had the highest NMI (0.1262).** Adding sentiment/expression features increased information association, but did not produce accurate one-to-one recovery of all three sentiments.
4. **E4 had the highest silhouette (0.2273).** Equal-weight engineered blocks produced clear numerical separation, but this did **not** imply strong sentiment alignment (ARI 0.0521).
5. **E4 and E5 were practically tied on external metrics.** E5 had a slightly higher Macro F1 (0.479657 versus 0.479529) even though E4 had much higher silhouette. Their differences are too small to support a meaningful external winner.
6. **E6 had the highest purity (0.6786)** and the second-highest ARI, but purity is sensitive to class imbalance and cannot establish three-class recovery.
7. **Neutral sentiment remained difficult.** Across representations, better negative or positive identification did not consistently translate to reliable neutral identification. The contingency matrices and per-class scores are essential to this interpretation.

**Core finding:** Stronger internal cluster separation does not necessarily mean closer agreement with human sentiment annotations.

## 6. Why we designed E7

After observing E0–E6, we hypothesized that a more sentiment-focused blend could combine E0's lexical alignment with the better class balance of engineered representations, while avoiding the strong emphasis on negation and punctuation that characterized E4/E5.

- **70% word TF-IDF:** prioritize the word and two-word context associated with E0's leading ARI.
- **10% character TF-IDF:** retain some robustness to informal spelling without letting character patterns dominate.
- **15% VADER positive, negative and compound:** test more explicit polarity information than the single compound score in E3–E5.
- **5% negation:** retain the important “not/no” signal, but limit its ability to dictate cluster boundaries.
- **Exclude exclamation and uppercase features:** reduce the risk of separating expressive writing styles instead of sentiment polarity.
- **No FastText:** keep the first custom experiment interpretable and limited to the modular sparse-feature pipeline.

These were **hypotheses**, not findings established before E7 was run. Because E7's design was informed by observed external scores, its subsequent evaluation on the same labels is exploratory.

## 7. E7 results and trade-offs

| Measure | E4 | E7 | Interpretation |
|---|---:|---:|---|
| ARI | **0.052099** | 0.030345 | Worse overall partition agreement for E7. |
| NMI | 0.095692 | **0.120703** | E7 has greater measured information association. |
| Macro F1 | **0.479529** | 0.434275 | Worse balanced class recovery for E7. |
| Mapped accuracy | **0.491973** | 0.447503 | More one-to-one sentiment mismatches for E7. |
| Purity | 0.626955 | 0.626955 | No difference. |
| Silhouette | **0.227258** | 0.0694 | Weaker internal geometric separation for E7. |

E7 recovered more positive tweets (**positive recall ~89.46%**, versus **~50.68%** for E4), but with low positive precision (**~32.62%**). Its neutral F1 (**~0.3227**) fell below E4's (**~0.4282**). Thus higher positive recall came with more false positives and weaker balanced recovery.

**Conclusion:** E7 was **not an overall improvement**. It improved NMI and positive recall relative to E4, but worsened ARI, Macro F1, mapped accuracy, neutral recovery and silhouette. Optimizing one sentiment objective can trade off against others.

## 8. What we can and cannot conclude

**Supported:** E0 best ARI/mapped accuracy among the tested configurations; E3 best NMI among E0–E6; E4 best silhouette; E5 marginally best Macro F1; E6 best purity; E7's trade-offs are real descriptive observations in these runs.

**Not established:** a universal best representation, statistically significant performance differences, generalization to new tweets, or that K=3 is optimal. The experiments used one corpus and one clustering seed, and some methods may vary with software versions. Cluster mapping uses the same annotations being evaluated, so mapped metrics describe the current corpus rather than a deployed classifier.

If future work uses the observed sentiment labels to select or tune new representations, those labels cannot then serve as an untouched independent evaluation set. A genuinely new labelled dataset or an appropriately separated evaluation design would be needed for confirmatory claims.

## 9. Notebook-folder organization and evidence

The experiment documentation and eight notebooks were merged into GitHub through [PR #36](https://github.com/frankie-chew-nus/sentiment-clustering-ml/pull/36):

```text
notebooks/data exploration/
  README.md
  L6_E0_Word_TFIDF_Standalone.ipynb
  L6_E1_Character_TFIDF_Standalone.ipynb
  L6_E2_Combined_TFIDF_Standalone.ipynb
  L6_E3_W0_Standalone.ipynb
  L6_E4_W1_Standalone.ipynb
  L6_E5_W2_Standalone.ipynb
  L6_E6_FastText_Standalone.ipynb
  L6_E7_Exploratory_Hybrid_Standalone.ipynb
```

The consolidated E0–E7 analysis notebook was prepared separately but **was not included in PR #36**. The committed E0–E7 notebooks have their execution outputs cleared; the user-run notebooks and Google Drive experiment output directories retain the executed results, JSON metrics, assignments, figures and provenance evidence. Preserve those external artifacts as the numerical evidence: this README is an interpretation, not a replacement for saved experiment outputs.

**Repository scope:** L6 uses the existing modular Python code as a library, but does not introduce new reusable pipeline modules. The root README links to this experiment record and gives only a brief project-level findings summary.
