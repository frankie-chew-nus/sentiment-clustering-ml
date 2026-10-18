# L5 K=3 representation-ablation decision log

## Status

**Frozen before external evaluation.** This record documents the label-blind L5 conclusions before any sentiment-label alignment scores are inspected.

## Scope

L5 evaluated how text representation, sentiment-oriented engineered features, feature weighting, and FastText affect K-means clustering at the prespecified primary configuration **K=3**.

Sentiment labels and sentiment-confidence fields were not used for preprocessing, feature construction, weighting, K-means fitting, configuration selection, internal evaluation, or the label-blind interpretation audit.

## Experimental design

Seven validated L4 representations were compared:

| ID | Representation |
|---|---|
| E0 | Word TF-IDF P1 |
| E1 | Character TF-IDF P0 |
| E2 | Combined word P1 + character P0 TF-IDF |
| E3 | W0 direct concatenation with engineered features |
| E4 | W1 equal expected squared-distance contribution |
| E5 | W2 text-priority weighting |
| E6 | FastText P1 100d |

K-means used K=3, k-means++ initialization, n_init=10, max_iter=300, tol=1e-4, and seeds 5001-5005. Silhouette used a fixed 2,000-row label-blind sample; DBI, CH, inertia, cluster sizes, runtime, iterations, and seed stability were also captured.

## Consolidated evidence

| Rep | Silhouette mean | DBI mean | CH mean | Mean seed ARI | Min seed ARI |
|---|---:|---:|---:|---:|---:|
| E0 | 0.004676 | 4.721389 | 69.893565 | 0.739890 | 0.539160 |
| E1 | 0.003145 | 6.227104 | 78.481191 | 0.831430 | 0.717381 |
| E2 | 0.003167 | 6.414162 | 66.861272 | 0.698508 | 0.281528 |
| E3 | 0.050672 | 3.894211 | 814.607082 | 1.000000 | 1.000000 |
| E4 | 0.227258 | 1.611038 | 3821.196730 | 1.000000 | 1.000000 |
| E5 | 0.109319 | 2.596665 | 1523.883225 | 1.000000 | 1.000000 |
| E6 | 0.079629 | 2.849459 | 1270.937320 | 0.993186 | 0.986674 |

Raw inertia is not used for cross-representation ranking because the feature spaces have different geometry/scaling.

## Weighting ablation

Cross-representation comparison at seed 5001:

| Comparison | ARI | Aligned agreement | Different tweets |
|---|---:|---:|---:|
| E3 W0 vs E4 W1 | 0.324336 | 64.4170% | 5,209 |
| E3 W0 vs E5 W2 | 0.324548 | 64.4306% | 5,207 |
| E4 W1 vs E5 W2 | 0.999714 | 99.9863% | 2 |

Therefore calibrated weighting materially changes the W0 solution. W1 and W2, however, converge on essentially the same K=3 partition despite substantially different internal metric values. Feature-space geometry and partition membership must therefore be interpreted separately.

## Label-blind E4/W1 interpretation

E4 seed 5001 cluster sizes were 4,317 / 7,409 / 2,913.

| Cluster | Mean VADER compound | Negation present | Exclamation present | Uppercase present | Label-blind description |
|---|---:|---:|---:|---:|---|
| 0 | -0.0389 | 100.0% | 7.25% | 25.62% | Negation-dominant |
| 1 | 0.1562 | 0.0% | 0.0% | 25.91% | Low engineered-signal / unmarked |
| 2 | 0.3396 | 20.56% | 100.0% | 30.59% | Exclamation-dominant / expressive |

Centroid-nearest examples and a fixed random sample (seed 5001) supported these feature descriptions. The random sample also showed mixed favourable and unfavourable language within the expressive cluster. Consequently, strong internal separation is **not** treated as proof of sentiment-class recovery.

## Conclusions

1. Text-only TF-IDF produced weakly separated, often highly imbalanced K=3 partitions.
2. Adding engineered sentiment-oriented features materially changed cluster geometry, balance, and initialization stability.
3. Calibrated weighting further changed the partition relative to W0.
4. W1 had the strongest observed internal K=3 geometry and perfect tested seed stability.
5. W1 and W2 produced virtually identical assignments (14,637 of 14,639 aligned) despite different internal scores.
6. W1's clean geometry is strongly structured by engineered negation/exclamation signals; internal clustering quality must not be equated with sentiment recovery.

## Carry-forward recommendation

For subsequent clustering-model exploration:

- **Primary:** E4 / W1.
- **Weighting sensitivity:** E5 / W2.
- **Text-only control:** E2.
- **Optional semantic control:** E6 / FastText.
- E3 remains the direct-concatenation ablation reference.

Exploration of K=2-8 is intentionally deferred to subsequent clustering-model exploration and does not alter the frozen L5 K=3 representation conclusions.

## Evidence and reproducibility

The Colab evidence archive records matrix fingerprints, configuration, five seeds, internal metrics, cluster assignments, stability outputs, cross-representation comparisons, interpretation tables, representative/random tweet samples, figures, and an `L5_K3_FREEZE_MANIFEST.json` generated before external evaluation.

External sentiment evaluation is explicitly outside this frozen L5 record.
