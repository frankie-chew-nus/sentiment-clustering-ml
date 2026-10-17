# L4 text representation decisions — implementation record

Status: L4-M1 implementation candidate on review branch. **Not yet Colab-validated or merged.**

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
