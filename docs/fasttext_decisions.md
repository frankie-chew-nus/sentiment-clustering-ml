# L4-M4 — Optional corpus-trained FastText representation

Status: implementation candidate, Colab validation pending. No clustering or sentiment labels.

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
