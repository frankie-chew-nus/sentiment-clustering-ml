# L4-M3 — Feature integration and block-distance weighting

Status: user-run Colab regression, full-dataset calibration and export checks passed; pending PR review. No clustering or sentiment-label use.

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


## L4-M3-VAL-01 — User-run full-dataset validation (2026-10-17)

**Status:** Colab regression and numerical export checks passed, as reported by user. Full regression: **169/169 passed (5.43 s)**. Targeted test run originally failed 2/16; FIX-01 corrected floating-point exact equality and nonfinite error wording. Subsequent full regression passed.

**Inputs:** 14,639 aligned rows. L3 master fingerprint matched the approved L3 hash `abb285f95c28a59bb35cbb797faabdcaac894f1c7d04885b4e7789dd3d88067c`. The M1 `word_p1` (14,639 x 29,792) and `char_p0` (14,639 x 50,000) exports passed manifest SHA256, shape, NNZ and finite-value checks. Master feature selection: V1+N1+S2; bounded scaling with percentile 95; count reference for N_negation_count and S_exclamation_count = `1.0986122886681096` each. M1 manifests lack row-level identifiers, so upstream row-order alignment relies on earlier validated corpus checks.

**Code revision used in Colab:** `04e2674432b92eb8a0afab9db6b4bc300ec80fcb`. This is the tested implementation revision, not necessarily the subsequent documentation commit.

### Exact calibration audit

Distances below are exact all-distinct-pairs mean squared Euclidean distances calculated by the algebraic identity, not estimates from random sampled pairs. Values and coefficients are rounded to six decimals.

| Block | Original mean squared distance | W0 coefficient / share | W1 coefficient / share | W2 coefficient / share |
|---|---:|---|---|---|
| word | 1.974836 | 1.000000 / 40.826494% | 0.318236 / 20.000000% | 0.420987 / 35.000000% |
| char | 1.958069 | 1.000000 / 40.479868% | 0.319596 / 20.000000% | 0.422785 / 35.000000% |
| V | 0.449029 | 1.000000 / 9.282941% | 0.667387 / 20.000000% | 0.471914 / 10.000000% |
| N | 0.232866 | 1.000000 / 4.814117% | 0.926749 / 20.000000% | 0.655311 / 10.000000% |
| S | 0.222343 | 1.000000 / 4.596580% | 0.948425 / 20.000000% | 0.670638 / 10.000000% |

W0 text share = 81.306362%; W1 text share = 40%; W2 text share = 70%. W1 and W2 percentages are **achieved calibration targets**, not evidence of better sentiment alignment. W0 coefficients remain exactly one.

### Export and provenance verification

Output folder: `/content/drive/MyDrive/MSI5001 project/EDA/L4-Vec_norm/M3 feature integration/`.

| Matrix | Shape | NNZ | Export SHA256 |
|---|---|---:|---|
| W0 | 14639 x 79796 | 3728386 | `e2c1ab5f4dd463d4b7280ff13a93ef02638090a06406672c88296a4fbcdb3abd` |
| W1 | 14639 x 79796 | 3728386 | `90cdac5f0d5b1cb094a083840bc39feec10bbb7187a76b2cb60a252b16d352fe` |
| W2 | 14639 x 79796 | 3728386 | `4bc8084385ebfdf95f3f5422661895a1081a97fe561f93ef654198001811759e` |

All three exported NPZ matrices were reloaded in Colab and passed shape, NNZ, finite-value, exact reload equality and SHA256 checks against their JSON manifests. This was initially **user-run verification**; the exports were subsequently uploaded and independently inspected, as recorded in L4-M3-VAL-02 below.

**Limitations:** No clustering, labels, or outcome-based tuning. M3 does not determine optimal weights. The manifests link the M1 files and master dataset by hashes but cannot independently prove M1-to-master row mapping; preserve the L2/L3/M1 alignment evidence. Combined word and character information can be redundant. Review PR before merge.


## L4-M3-VAL-02 — Independently reviewed exported artifacts (2026-10-17)

The user uploaded `M3 feature integration-20261017T060551Z-1-001.zip` containing W0/W1/W2 NPZ matrices and their JSON manifests. The files were independently inspected outside Colab, following the user-run full-dataset validation recorded in L4-M3-VAL-01.

| Artifact | Matrix shape | Stored nonzero entries | SHA256 |
|---|---|---:|---|
| W0.npz | 14639 x 79796 | 3728386 | `e2c1ab5f4dd463d4b7280ff13a93ef02638090a06406672c88296a4fbcdb3abd` |
| W1.npz | 14639 x 79796 | 3728386 | `90cdac5f0d5b1cb094a083840bc39feec10bbb7187a76b2cb60a252b16d352fe` |
| W2.npz | 14639 x 79796 | 3728386 | `4bc8084385ebfdf95f3f5422661895a1081a97fe561f93ef654198001811759e` |

**Independent checks:** 3/3 matrices loaded as SciPy sparse matrices; 3/3 file SHA256 hashes matched the corresponding JSON manifests; 3/3 shapes and nonzero counts matched the Colab report; all stored numerical values were finite. Each manifest records the expected block coefficients, calibration method, source hashes and code revision.

**Recorded provenance:** L3 master SHA256 `abb285f95c28a59bb35cbb797faabdcaac894f1c7d04885b4e7789dd3d88067c`; M1 `word_p1` SHA256 `fa4fdd28368f0c2f4f9cd234127ab1699b3c64935db4c08ed003fe821b1af672`; M1 `char_p0` SHA256 `f94b8e38dd3b0904905fef09e33607cb3c591f49391725d0f9a1753613818d20`. The recorded M3 implementation revision is `04e2674432b92eb8a0afab9db6b4bc300ec80fcb`.

**Interpretation:** W0 observed 81.306362% text distance contribution; W1 achieved 20% per block; W2 achieved 70% text and 30% engineered contributions. These are distance-calibration results, not clustering-performance measurements. Row-level alignment relies on earlier L2/L3/M1 evidence rather than a row mapping stored in the NPZ files. No labels, K-means or outcome-based tuning were used.

**GitHub workflow clarification:** PR #28 was merged. PR #29 was an unmerged revert proposal and was closed without merging. This documentation-only evidence update retains the M3 implementation on `main`.
