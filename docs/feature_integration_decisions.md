# L4-M3 — Feature integration and block-distance weighting

Status: implementation candidate, Colab validation pending. No clustering or sentiment-label use.

## Scope
Combine independently constructed word/character TF-IDF matrices and scaled
V/N/S engineered features in explicit named blocks. Preserve row ordering,
finite numerical values and sparse CSR representation. Do not alter M1/M2 outputs.

## Approved alternatives
- W0: direct concatenation; every block coefficient = 1 (default baseline).
- W1: equal expected squared-distance contribution across active blocks.
- W2: 70% text / 30% engineered expected squared-distance allocation when
  both groups are present, split equally within each group.
- W2 with only text or only engineered blocks: redistribute equally across
  present blocks. When a group is missing, its allocation is not discarded.
- Any zero-distance block is rejected for W1/W2; no silent dropping.
- 60/40 and 80/20 sensitivity analysis is deferred to L5, not implemented
  as separate named M3 configurations.

## Exact calibration
For n rows, D_b = (2n/(n-1)) * [mean_i ||x_ib||^2 - ||mean_i x_ib||^2].
This equals the mean squared Euclidean distance over all ordered distinct
tweet pairs, computed without O(n^2) enumeration or stochastic sampling.
For W1/W2, coefficient w_b = sqrt(p_b / D_b). Therefore w_b^2 D_b = p_b
and the sum of calibrated block distances is 1 on the fit corpus.
A common multiplier would not affect relative shares.
W0 preserves original M1/M2 magnitudes and has no imposed targets.

## Alignment and leakage
- Block names are explicitly whitelisted: word, char, V, N, S.
- Labels/confidence/row_id are not accepted as named blocks.
- Equal row counts alone do NOT establish correct source alignment.
- Before M3 full-dataset integration, validate P0/P1 original-text alignment,
  L3 master fingerprint, and input NPZ provenance; document this externally.
- Fit weights once on the designated analysis corpus and reuse for transforms.
- Never tune weights using sentiment labels, K-means metrics, or external
  evaluation. Do not renormalize the entire combined vector after weighting,
  since that would change the calibrated geometry.

## Outputs
Sparse CSR matrices and SHA256 manifests containing block ordering,
dimensions, calibration distances, target shares, actual coefficients,
source SHA256 fingerprints and Git revision. Store generated artifacts in
Google Drive under EDA/L4-Vec_norm/M3 feature integration/, not GitHub.

## Limitations and review gate
W1 and W2 are mathematical allocations, not proven sentiment-optimal weights.
Correlated word and character features can duplicate information. W1 can
amplify structural expression unrelated to polarity. Block matrices must be
loaded and aligned by a validated external ingestion workflow. Run targeted
and full regression tests, full 14,639-row numerical audit, verify achieved
distance contributions and exported fingerprints before PR review.
