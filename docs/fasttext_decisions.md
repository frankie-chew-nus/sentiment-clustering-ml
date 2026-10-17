# L4-M4 — Optional corpus-trained FastText representation

Status: user-run Colab regression and full-corpus validation passed; exported artifacts independently verified. Pending PR review. No clustering or sentiment labels.

## Scope and audit rationale
The L4 feasibility audit trained corpus FastText on P1 and found it computationally
feasible. However, cosine similarity of opposing-polarity terms (e.g. excellent
and terrible, approximately 0.815 in the audit) raised concern that embeddings
may capture shared airline contexts rather than sentiment polarity. This is
not evidence that FastText performs poorly in clustering. It remains optional.

## Initial training configuration
- Input: P1 processed_text; whitespace tokenization.
- Gensim FastText skip-gram (sg=1), vector_size=100, window=5,
  min_count=2, epochs=10, seed=5001, workers=1, character ngrams 3-6.
- Train FastText and unigram TF-IDF on the same P1 analysis corpus.
- Pool each tweet using IDF-weighted mean of token vectors; L2-normalize
  each nonzero pooled tweet vector.
- Preserve all-zero rows explicitly; record row indices in export manifest.
- Out-of-explicit-vocabulary FastText words may have subword vectors, but
  unseen TF-IDF tokens are excluded by design.
- Fitted model and IDF weights must be reused for later held-out transforms;
  no refitting on held-out data.

## Provenance and validation
- Export 14,639 x 100 matrix as compressed NPZ, plus JSON containing
  configuration, zero-row indices, SHA256 and P1 source fingerprint.
- Use Drive path: EDA/L4-Vec_norm/M4 FastText representation/.
- Report corpus token coverage, vocabulary size, zero rows, pooled-vector
  norm distribution, training time, and selected polarity-opposition
  similarity examples in the full-dataset Colab audit.
- Train deterministically where practical with single worker and fixed seed.
  Numerical reproducibility across different library/BLAS environments
  is not guaranteed; record library versions.
- Unit tests use synthetic tweets and do not establish sentiment quality.

## Deferred and excluded
- Pretrained FastText is deferred: no model weights have been supplied or audited.
- No SVD/PCA or other dimensionality reduction.
- No sentiment-label tuning, K-means, or external evaluation in M4.
- No claim that FastText is better than word/character TF-IDF.


## L4-M4-VAL-01 — User-run Colab validation (2026-10-17)

- Targeted FastText unit tests: **10/10 passed**.
- Full regression suite: **179/179 passed (9.69 seconds)**, Python 3.13.16, pytest 8.4.2.
- Full P1 corpus: **14,639 tweets**, **262,713** whitespace token occurrences, **24,135** unique whitespace tokens.
- Gensim FastText explicit vocabulary: **8,814**; explicit token-occurrence coverage **94.1682%** (subword inference is not captured by this statistic).
- TF-IDF unigram vocabulary: **24,135**.
- Training plus encoding: **29.45 seconds** in the reported Colab runtime.
- Output: **14,639 x 100** dense tweet vectors; **0 zero vectors**; mean norm **1.0**; min **0.9999999999999996**, max **1.0000000000000002**.
- Word cosine similarities (diagnostic examples, not label-based evaluation): excellent/terrible **0.8139**, good/bad **0.6190**, great/awful **0.6617**, happy/angry **0.6430**.
- Compared with the earlier feasibility audit's one zero vector, the implemented pooling procedure produced zero zero-vectors. This discrepancy may reflect differences in tokenization, TF-IDF coverage, or pooling; the two procedures have not been proven identical.
- These values establish computational feasibility and representation geometry, **not superiority in sentiment clustering**.

## L4-M4-VAL-02 — Independently verified uploaded export (2026-10-17)

The user uploaded `M4 FastText representation-20261017T062159Z-1-001.zip`, containing `fasttext_p1_100d.npz` and `fasttext_p1_100d.json`. Independent inspection of the uploaded files confirmed:

| Check | Result |
|---|---|
| NPZ opens; key `embeddings` present | PASS |
| Shape | 14639 x 100 |
| All entries finite | PASS |
| Zero rows | 0 |
| Row L2 norms | min 0.9999999999999996; mean 1.0; max 1.0000000000000002 |
| SHA256 matches JSON manifest | PASS |
| Matrix SHA256 | `566def64727e2630ea8699589557b28f20795e6d4be15ffeec55c74af344d574` |
| P1 CSV SHA256 in manifest | `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e` |
| Git revision recorded in manifest | `2a4a88e49e735eb0a0cee706d98a8a1e8021806f` |

The manifest records corpus-trained skip-gram FastText, 100 dimensions, window=5, min_count=2, 10 epochs, seed=5001, workers=1, subword ngrams 3–6, P1 whitespace tokenization and IDF-weighted mean pooling followed by L2 normalization.

**Limitations:** The uploaded NPZ and JSON independently verify exported array integrity and manifest consistency, but do not independently reproduce training or establish sentiment-label alignment. FastText training can vary across environments; record dependency versions for future reruns. No K-means or sentiment labels used. Pretrained FastText remains deferred.
