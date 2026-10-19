# L4 text representation decisions — implementation record

Status: L4-M1 user-validated in Colab (142/142 regression tests) and exported matrices independently verified from uploaded ZIP. **Not merged.**

## Approved experimental inputs
- Word TF-IDF: P1 primary, P2 controlled alternative.
- Character TF-IDF: P0 primary, P1 controlled alternative.
- Combined: P1 word + P1 character baseline; P1 word + P0 character mixed-input alternative.
- `char_wb`: controlled alternative to `char`.
- No additional stopword removal, no dimensionality reduction, no sentiment labels.
- All configurations are experimental; none is a demonstrated clustering winner.

## Initial parameters
- Word: ngrams (1,2), min_df=2, max_df=0.95, max_features=30000.
- Character: ngrams (3,5), min_df=2, max_df=0.95, max_features=50000.
- Both: sublinear_tf=True, use_idf=True, norm='l2'.
- Lowercase=False in vectorizer because input variants already implement their
  respective casing decisions; P0 character case must be preserved.
- No combined-vector renormalization or relative weighting in M1. These are M3 responsibilities.

## Why L2 normalization for sparse TF-IDF?

Word and character TF-IDF encode many term-pattern weights in sparse vectors. Per-row L2 normalization makes each **nonzero** text vector unit-length, limiting variation in magnitude associated with tweet length while retaining the relative pattern weights within that vector. Zero vectors remain zero. This is a methodological rationale for the configured `norm='l2'`, not evidence that it outperforms alternative normalization methods on sentiment recovery.

Feature-wise mean-centering and z-scoring would answer a different question: they would alter term-level weighting and mean-centering would generally destroy sparse storage efficiency. M1 therefore independently L2-normalizes the word and character blocks, concatenates them without another row normalization, and leaves cross-block distance allocation to M3. See [M2 scaling](feature_scaling_decisions.md) and [M3 weighting](feature_integration_decisions.md).

## Invariants and limitations
- Strict positional alignment of P0/P1/P2 by row_id must be verified at ingestion.
- Label and confidence columns are forbidden inputs; row_id is metadata only.
- Output is sparse CSR. Persist matrices and manifests in Drive, not GitHub.
- Vocabulary cap is an initial engineering setting, not an optimum.
- Fit IDF on the designated analysis corpus only; use transform for new data.
- Corpus statistics, feature geometry and numerical stability are label-blind.
- No K-means, sentiment-label evaluation, or configuration ranking in M1.

## Deferred decisions
- M2: log1p, bounded scaling (positive-value percentile fallback), StandardScaler alternative.
- M3: engineered-family blocks, W0/W1/W2 weighting and distance audits.
- M4: optional corpus-trained FastText; pretrained model remains unaudited/deferred.
- M5: end-to-end Colab validation, provenance, regression and final decision review.

## Validation gate
Run targeted and full regression suites in Colab, then validate full-dataset
shapes, sparsity, row alignment, feature counts, and fingerprints against audits.
User reviews results before a pull request or merge.

## L4-M1-FIX-01 — TF-IDF roundtrip test (2026-10-17)
- Initial targeted test: 8 passed, 3 failed due to exact floating-point equality assertions in character configurations.
- Updated test compares maximum sparse-matrix absolute difference against 1e-12; production vectorization code unchanged.
- User-reported subsequent full Colab regression suite: **142 passed in 6.18 s** (Python 3.13.16).
- Fresh Colab full-suite collection initially required installing vaderSentiment==3.3.2; declared this and sklearn/scipy in requirements-dev.txt.

## L4-M1-VAL-01 — Full dataset and export audit (2026-10-17)
- 14,639 aligned rows in P0, P1 and P2; identical original text columns and zero empty processed texts.
- Input CSV SHA256: P0 `d349148dce04344180bf7969f12b6ccd2180e387927738cd911da07dc30c00a4`; P1 `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e`; P2 `70600d91b2f1327e36b1dc20d3f079328623b9ccb4546a74228046051b989755`.
- P1 matches previously validated fingerprint. P0/P2 fingerprints are user-reported Colab values.
- Outputs stored outside GitHub under `EDA/L4-Vec_norm/M1 full-dataset vectorization/`.

| Representation | Shape | NNZ | Zero rows | Mean L2 row norm | NPZ SHA256 |
|---|---|---:|---:|---:|---|
| word_p1 | 14639 x 29792 | 371747 | 16 | 0.99891 | fa4fdd28368f0c2f4f9cd234127ab1699b3c64935db4c08ed003fe821b1af672 |
| word_p2 | 14639 x 17267 | 173532 | 71 | 0.99515 | 207bb6f737fea26bb2cc6e89d6cc40c4245091947170b9fd31a5a9b52d1530a9 |
| char_p0 | 14639 x 50000 | 3332264 | 0 | 1.00000 | f94b8e38dd3b0904905fef09e33607cb3c591f49391725d0f9a1753613818d20 |
| char_p1 | 14639 x 50000 | 3455072 | 0 | 1.00000 | 95710a39e7384158890f8a096236a6c00a1a43dba0d17e96ee842759e26e9ed9 |
| char_wb_p0 | 14639 x 50000 | 2252967 | 1 | 0.99993 | 2f75d861dfc9a2d71883a0e0acf5657b1fb2f676afefb5ece0797eeccf54b48a |
| combined_p1_p1 | 14639 x 79792 | 3826819 | 0 | 1.41376 | 1cb7e85d3a5ecbaac91ce79cbda89eab6a6e0b70677916ed6ea49b3c5dbac74b |
| combined_p1_p0 | 14639 x 79792 | 3704011 | 0 | 1.41376 | 4e2577047204795c78fb9aa18c2b1996f918368c9a14ef98b08a654f82a58f41 |

Independent verification of user-uploaded ZIP: all 7 NPZ matrices load as sparse matrices; all 7 SHA256 hashes match corresponding JSON manifests; shapes, NNZ, zero rows and row norms match Colab summaries; all stored values finite.

**Limitations:** Existing export manifests include matrix hashes, vectorizer settings, shape and NNZ, but not source CSV fingerprints or Git commit. These input fingerprints are recorded here, not cryptographically linked to individual NPZ exports. Combined blocks are independently L2-normalized and have mean norm ~sqrt(2); weighting is deferred to M3. No clustering or external evaluation performed. Updated dependency declaration requires a fresh Colab test run before PR approval.
