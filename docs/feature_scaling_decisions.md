# L4-M2 engineered-feature scaling — implementation candidate

Status: user-reported Colab regression and full-dataset numerical validation passed; master SHA256 and exact scaling references verified. Not merged.

## Scope
Transform selected columns from the unscaled L3 master matrix. Never modify
the original master CSV. Preserve row_id only for alignment, not as a feature.
No sentiment labels, K-means, block weighting or clustering outcomes.

## Configurations
- Default variants: V1 + N1 + S2; V2, N2/N3 and S3 are controlled alternatives.
- Native VADER and uppercase-ratio columns remain unchanged.
- Count columns receive log1p. Three alternatives:
  - bounded (primary candidate): divide by a fitted percentile reference,
    optionally using positive-value fallback when overall percentile is zero;
    clip at 1.
  - log_only: log1p with no further scaling.
  - standard: log1p then population-standardization of transformed counts.
- Default bounded percentile: 95 (audited candidate, configurable).
- All fitted references are saved as metadata; downstream transforms reuse
  the same references without refitting.

## Interpretation and limitations
- Bounded scaling is not proven superior to alternatives for sentiment clustering.
- The positive-value percentile fallback avoids zero references for zero-inflated
  features, but can magnify rare observations; clipping limits their maximum.
- Feature-wise scaling does not equalize block norms or distance contributions.
  W0/W1/W2 weighting remains an L4-M3 responsibility.
- L4-M2 does not implement text-feature integration.

## Validation gate
Run targeted and full regression suites in Colab. Validate on the actual
14,639-row L3 master CSV, record feature distributions and fitted references,
and compare label-blind squared-distance contributions with L4 scaling audits.
Preserve dataset fingerprint and Git revision in validation evidence.


## L4-M2-VAL-01 — Colab regression and full-corpus scaling (2026-10-17)

**Validation evidence:** User-run Colab, Python 3.13.16, pytest 8.4.2.
Targeted `tests/test_feature_scaling.py`: **11/11 passed (0.56 s)**.
Full regression suite: **153/153 passed (5.55 s)**. No failures or errors.
The following full-dataset figures were reported for the 14,639-row master matrix, using `V1 + N1 + S3` (five columns), rather than the default `V1 + N1 + S2`.

### Distribution statistics (user-reported, rounded)

| Method | Feature | Mean | Std | Median | P95 | Max | Zero % |
|---|---|---:|---:|---:|---:|---:|---:|
| bounded | V_compound | 0.1352 | 0.4738 | 0.1144 | 0.8313 | 0.9778 | 15.8686 |
| bounded | N_negation_count | 0.2352 | 0.3412 | 0 | 1 | 1 | 66.4185 |
| bounded | S_exclamation_count | 0.1668 | 0.3244 | 0 | 1 | 1 | 77.9630 |
| bounded | S_uppercase_ratio | 0.0288 | 0.0770 | 0 | 0.1429 | 0.9565 | 73.2427 |
| bounded | S_emoji_count | 0.0176 | 0.1013 | 0 | 0 | 1 | 96.6391 |
| log_only | V_compound | 0.1352 | 0.4738 | 0.1144 | 0.8313 | 0.9778 | 15.8686 |
| log_only | N_negation_count | 0.2607 | 0.3809 | 0 | 1.0986 | 1.6094 | 66.4185 |
| log_only | S_exclamation_count | 0.1989 | 0.4064 | 0 | 1.0986 | 3.3322 | 77.9630 |
| log_only | S_uppercase_ratio | 0.0288 | 0.0770 | 0 | 0.1429 | 0.9565 | 73.2427 |
| log_only | S_emoji_count | 0.0324 | 0.1912 | 0 | 0 | 3.7136 | 96.6391 |
| standard | V_compound | 0.1352 | 0.4738 | 0.1144 | 0.8313 | 0.9778 | 15.8686 |
| standard | N_negation_count | 0 | 1 | -0.6844 | 2.1997 | 3.5407 | 0 |
| standard | S_exclamation_count | 0 | 1 | -0.4894 | 2.2141 | 7.7105 | 0 |
| standard | S_uppercase_ratio | 0.0288 | 0.0770 | 0 | 0.1429 | 0.9565 | 73.2427 |
| standard | S_emoji_count | 0 | 1 | -0.1694 | -0.1694 | 19.2542 | 0 |

The `Zero %` column measures exactly zero *after transformation*. Under standardization, original count zeros become negative standardized values; the underlying presence/absence of emojis or punctuation has not changed.

### L4-M2-VAL-02 — Label-blind distance contribution audit

User-run Colab generated **49,999 distinct tweet pairs**, using NumPy RNG seed 5001, with 50,000 candidate index pairs and equal-index pairs excluded. Contributions are the family share of **mean squared Euclidean distance**, calculated from `V1 + N1 + S3` engineered features only; TF-IDF is excluded.

| Method | VADER distance / share | Negation distance / share | Structural distance / share |
|---|---|---|---|
| bounded | 0.4511 / 48.5138% | 0.2343 / 25.2026% | 0.2444 / 26.2836% |
| log_only | 0.4511 / 38.8235% | 0.2921 / 25.1420% | 0.4187 / 36.0345% |
| standard | 0.4511 / 6.9138% | 2.0133 / 30.8564% | 4.0603 / 62.2298% |

**Interpretation:** VADER compound is unchanged across methods. Standardization substantially increases the distance contribution of sparse structural counts, while bounded scaling limits count maxima. The 95th percentile of transformed emoji counts is zero, motivating the audited positive-value fallback. None of these measurements establishes better sentiment clustering.

**Decision:** Retain `bounded` with log1p and a configurable 95th-percentile reference as the *primary experimental candidate*. Retain `log_only` and `standard` as controlled alternatives. No final family weights are selected in M2; block weighting is deferred to M3.

**Provenance limitations / review gate:** Master CSV SHA256, fitted references and checked-out Git revision were subsequently confirmed in L4-M2-PROV-01 below. The distance audit uses V1+N1+S3 and is not directly interchangeable with V1+N1+S2 results. No sentiment labels, K-means, or external evaluation were used.


## L4-M2-PROV-01 — Source fingerprint and fitted references (2026-10-17)

User-run Google Colab verification:
- L3 master CSV SHA256: `abb285f95c28a59bb35cbb797faabdcaac894f1c7d04885b4e7789dd3d88067c` — **matches validated L3 handover fingerprint**.
- Colab checked-out Git revision: `ac999ebfde6b3ab4b932c0fd082659d70705080f`. This is the implementation revision observed during validation, **not** the later documentation commit or necessarily the final PR head.
- Scaling specification: `EngineeredFeatureScaler(variants=("V1","N1","S3"), spec=ScalingSpec(method="bounded", percentile=95))`.
- Fitted transformed-count references:
  - `N_negation_count`: `1.0986122886681096` = log1p(2).
  - `S_exclamation_count`: `1.0986122886681096` = log1p(2).
  - `S_emoji_count`: `1.791759469228055` = log1p(5).
- These references are fit from the corpus; the emoji positive-value fallback is relevant because its overall 95th percentile is zero.
- The values confirm the scaling parameters used in the 14,639-row validation; they are **not evidence of superior sentiment clustering**.
- The user-reported regression suite passed **153/153 tests** before this documentation-only change. No code modifications in PROV-01.
