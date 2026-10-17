# L4-M2 engineered-feature scaling — implementation candidate

Status: methodology approved for audit and implementation; Colab validation pending.

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
