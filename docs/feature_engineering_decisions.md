# L3 feature-engineering decisions — VADER branch

Status: **implementation candidate; Colab validation pending; not approved for main**.

## Methodology
- Objective: unsupervised sentiment clustering, with airline_sentiment and all sentiment-confidence columns reserved strictly for external evaluation.
- Input: original RAW text only. No changes to P0/P1/P2.
- Use vaderSentiment==3.3.2; compute all four scores unscaled once.
- Master CSV: sequential row_id, V_pos, V_neg, V_neu, V_compound. V0 selects no VADER columns; V1 selects compound; V2 selects pos, neg and compound.
- Explicit feature whitelist. row_id is metadata, never a clustering feature.
- Save SHA256, row count and source filename alongside exports; positional IDs require exact row-order consistency.

## V0 audit evidence (executed label-blind notebook)
- RAW tweets: 14,639; SHA256 a41aeefcff9740c090153f608691d345c84f59084e40f03169c2f667df66578e.
- Compound exactly zero: 2,323 (15.87%); pos and neg both zero: 2,269.
- Unique compound scores: 2,915. Mixed component signals and score redundancy motivate testing both V1 and V2, not choosing a winner.
- V_neu retained for audit, excluded from initial feature subsets because pos + neg + neu approximately equals one.
- No claims about improved sentiment clustering before post-clustering external evaluation.

## Earlier structured-rating S-R audit
- 197 fraction-like expressions in 180/14,639 tweets (1.23%).
- Many matches were dates, quantities, identifiers; S-R deferred, P2 unchanged.

## Verification gate
- Run new VADER unit tests and full existing preprocessing regression suite in Colab.
- Validate full dataset schema, row order, ranges, and source SHA256 against V0 audit.
- Record outputs and limitations before merge to main. No ground-truth labels used.

## Limitations
- RAW names/hashtags can affect VADER; sarcasm and mixed opinions remain challenging.
- Sequential row IDs are valid only for the same source file and row order.

## L3-VADER-FIX-01 — Empty-string component-sum test correction (2026-10-15)

**Evidence:** First Colab run of `tests/test_feature_engineering.py` reported **15 passed, 1 failed**. The failing case was `test_exact_library_scores[""]`: the test asserted `V_pos + V_neg + V_neu ≈ 1` for an empty string, but the sum was 0.0. The preceding direct-library equality assertion passed, supporting that the extractor matched `vaderSentiment==3.3.2` and the test assumption was incorrect.

**Decision and rationale:** Keep the empty-string case in the test suite. For exactly empty text, explicitly assert that all four VADER scores are zero. For nonempty test strings, retain the existing approximate sum-to-one assertion (tolerance 0.0011). No handcrafted sentiment thresholds or modifications to the VADER library.

**Changed file:** `tests/test_feature_engineering.py` only. `feature_engineering.py`, `feature_engineering_config.py`, preprocessing P0/P1/P2 and dataset are unchanged.

**Verification status:** Correction committed to feature branch; **Colab rerun pending**. Expected 16 passing tests is a hypothesis, not yet an observed result. After rerun, capture test output and full regression results before merge.

**Scope:** This is a test-correctness fix, not evidence of improved clustering or a feature-methodology change. Sentiment labels/confidence remain excluded.

## L3-VADER-ENV-01 — Full regression suite dependency setup (2026-10-15)

**Observed in user-run Google Colab:** After FIX-01, the targeted VADER suite passed **16/16**. The first full regression run reported **56 passed, 14 failed**. Failure traces were ImportErrors for preprocessing dependencies `emoji`, `num2words`, and `wordninja`, already declared in `requirements.txt`. This indicates an incomplete Colab environment rather than demonstrated preprocessing regression.

**Corrective action:** Install repository dependencies using `%pip install -r requirements-dev.txt` and the P2 spaCy model using `!python -m spacy download en_core_web_sm`, then rerun `!python -m pytest -q`.

**User-reported verification:** Full regression suite **70 passed** after setup. This is user-reported console output, not independently reproduced by this document update. The VADER implementation and preprocessing source were not modified for ENV-01.

**Decision:** Treat ENV-01 as an environment/setup issue; retain existing P0/P1/P2 code. Full-dataset VADER extraction and V0 audit parity checks remain the next validation gate before merge.

## L3-VADER-VAL-01 — Full-dataset feature CSV review (2026-10-15)

**Evidence source:** User-provided `Tweets-L3-vader-features.csv`, reviewed against the previously executed label-blind V0 audit. This CSV is an output artifact retained outside GitHub, not a repository input or a source of ground-truth sentiment.

**Reported CSV validation results:**

| Check | Observed | Assessment |
|---|---:|---|
| Rows | 14,639 | Matches V0 audit |
| Sequential `row_id` | 0–14,638 | Expected order/shape |
| Engineered columns | `V_pos`, `V_neg`, `V_neu`, `V_compound` | Expected feature-only schema |
| Exact zero compound | 2,323 | Matches V0 audit |
| Both positive and negative components zero | 2,269 | Matches V0 audit |
| Unique compound scores | 2,915 | Matches V0 audit |
| Invalid/non-finite score values | 0 | Passed |
| Sentiment-label or confidence columns | None | Passed |

**Score summary (rounded to four decimals):**

| Feature | Mean | Min | Max |
|---|---:|---:|---:|
| `V_pos` | 0.1569 | 0.0000 | 1.0000 |
| `V_neg` | 0.0817 | 0.0000 | 0.7840 |
| `V_neu` | 0.7615 | 0.0000 | 1.0000 |
| `V_compound` | 0.1352 | -0.9574 | 0.9778 |

**Interpretation:** Structural and numerical parity supports the VADER extraction implementation and retaining the configurable V0/V1/V2 variants. These are extraction-validity checks, not evidence that one variant produces better sentiment clusters.

**Important qualification:** A standalone output CSV does not authenticate which RAW file or row order produced it. The V0 RAW source SHA256 was `a41aeefcff9740c090153f608691d345c84f59084e40f03169c2f667df66578e`; source-to-output provenance must be confirmed from the executed validation notebook or recorded export metadata before final sign-off. The VADER component-sum check should also allow all-zero scores for empty RAW text.

**Testing evidence:** User reported targeted VADER tests **16/16 passed** and full regression suite **70/70 passed** after dependency installation. No preprocessing changes were required.

**Decision / status:** Feature CSV numerical review accepted; **merge remains subject to source-provenance confirmation and user review**. Keep `V_neu` in the master feature CSV but exclude it from initial V1/V2 clustering selections. Do not commit notebooks or CSV datasets to GitHub without explicit user approval.

## L3-VADER-PROV-01 — RAW source fingerprint verification (2026-10-15)

**User-run Google Colab verification:** `raw_csv_fingerprint()` returned source filename `Tweets-redacted.csv` and SHA256 `a41aeefcff9740c090153f608691d345c84f59084e40f03169c2f667df66578e`. Comparison with the original V0 audit fingerprint returned **True**; the assertion passed.

**Combined validation evidence:** Targeted VADER suite 16/16 passed; full regression suite 70/70 passed (user-reported Colab outputs). Uploaded master VADER CSV passed numerical/schema checks: 14,639 rows, row IDs 0–14,638, four VADER columns, 2,323 exact-zero compound values, 2,269 with both pos/neg zero, 2,915 unique compound values, and no sentiment label/confidence columns.

**Scope of confirmation:** The current RAW file matches the audited source byte-for-byte. Matching CSV statistics support consistency but do not independently prove the export's generation history or every row-level correspondence. For future exports, save the input SHA256 and code revision in an external run manifest or executed notebook.

**Disposition:** Ready for user merge review of `feature/l3-vader-engineering`. No merge performed. V1/V2 remain alternative unsupervised clustering configurations; no ground-truth-label-based selection has occurred. No notebooks or datasets committed to GitHub.

## L3-N0 / N0-B — Negation and intensity audits (2026-10-15)

**Status:** Label-blind audit reviewed; N0–N3 definitions approved as experimental alternatives. Implementation and Colab validation pending.

**Input:** Validated P1 `processed_text` (not RAW/P2); 14,639 tweets. Reported P1 SHA256: `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e`. P1 contraction expansion makes e.g. `can't` into `can not`.

**N0 prevalence:** Negation in 4,916 tweets (33.58%); original-vocabulary intensifier in 1,823 (12.45%); both in 680 (4.65%); neither in 8,580 (58.61%). Dominant negation matches: `not` 3,906, `no` 1,515, `never` 321, `nothing` 157. Frequent original intensifiers: `so` 901, `really` 302, `very` 268, `too` 240.

**N0-B comparison:** Original 13-word vocabulary: 1,823 tweets (12.45%), 1,993 token matches. Conservative 8-word candidate: 693 tweets (4.73%), 722 token matches. Reduction: 1,130 tweets lose all matches and 1,271 token matches (63.8%). Removed words: `so` 901, `too` 240, `seriously` 80, `pretty` 43, `quite` 7.

**Qualitative audit:** 55 word-stratified sampled contexts reviewed without sentiment labels: 14 clear sentiment intensifiers, 8 other degree modifiers, 31 discourse/other, 2 uncertain. This is **not** a corpus-wide error estimate. The removed words include genuine sentiment uses (e.g. `pretty frustrating`) and unrelated uses (e.g. `so I called`, `me too`).

**Agreed feature candidates:** N0 none; N1 negation count; N2 negation + original intensifier count; N3 negation + conservative intensifier count. N2 and N3 are mutually exclusive selection variants. Compute both count columns once for master features; do not include ground-truth labels or confidence columns. Do not treat negation count as negative polarity. No sentiment thresholds or large grammar rules.

**Implementation constraints:** Reuse existing `config.NEGATION_WORDS` and `config.INTENSIFIER_WORDS` as canonical lists; retain a frozen conservative subset in `feature_engineering_config.py`. Match whole tokens on P1. Preserve row order and sequential row_id; record input SHA256. Scaling and clustering performance comparisons are downstream.

**Evidence location:** N0 and N0-B executed Colab notebooks, review CSVs, and team slide deck are retained outside GitHub by project policy. This decision record summarizes observed findings and agreed methodology; no notebook/CSV uploaded.

**Verification pending:** Unit tests, P1 corpus parity, full regression suite, and user review before PR/merge. No performance superiority claimed.

## L3-N-VAL-01 — P1 full-dataset parity and regression verification (2026-10-15)

**Evidence source:** User-run Google Colab, using the agreed P1 CSV and feature branch implementation. Results supplied in the conversation; this GitHub record does not claim independent rerun.

| Check | Observed |
|---|---|
| P1 source filename | `Tweets-processed-p1.csv` |
| P1 SHA256 | `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e` |
| Source size | 3,012,428 bytes |
| Rows extracted | 14,639 |
| Tweets with `N_negation_count > 0` | 4,916 |
| Tweets with `N_intensifier_original_count > 0` | 1,823 |
| Tweets with `N_intensifier_conservative_count > 0` | 693 |
| N-feature automated tests | 23 passed (user reported) |
| Full project regression tests | 93 passed (user reported) |

The executed Colab assertion comparing row count and three coverage counts with the prior N0/N0-B audit returned `N0/N0-B audit parity: PASSED`. The P1 SHA256 matches the previously recorded P1 audit fingerprint.

**Interpretation:** The implemented N features reproduce audited corpus-level counts and pass the reported unit/regression checks. This validates extraction consistency, **not** sentiment-clustering improvement, nor the superiority of N2 versus N3.

**Remaining validation before PR:** Export/review the feature-only N CSV schema, sequential row identifiers, value ranges, and absence of sentiment labels/confidence; document any anomalies. Keep notebooks and datasets outside GitHub. User reviews PR and supplies any additional evidence before merging.

## L3-N-VAL-02 — Final N feature CSV validation (2026-10-15)

**Evidence:** User-provided `Tweets-L3-negation-intensity-features.csv`, reviewed after user-run N0/N0-B corpus parity and 23/23 N tests, 93/93 full regression tests.

| Check | Observed |
|---|---|
| Rows | 14,639 |
| Sequential `row_id` | 0–14,638 |
| Feature columns | `N_negation_count`, `N_intensifier_original_count`, `N_intensifier_conservative_count` |
| Tweets with negation count > 0 | 4,916 |
| Tweets with original intensifier count > 0 | 1,823 |
| Tweets with conservative intensifier count > 0 | 693 |
| Total negation matches / max per tweet | 5,961 / 4 |
| Total original intensifier matches / max | 1,993 / 4 |
| Total conservative intensifier matches / max | 722 / 3 |
| All three counts zero | 8,580 |
| Invalid, missing, or negative feature counts | 0 |
| Sentiment labels/confidence columns | None |

**Interpretation:** Schema, row sequence, nonnegative integer counts, subset relation (conservative <= original), and corpus-level audit counts are consistent with the approved extractor. The previously reported P1 SHA256 was `fdeab9faa57b7ffb7ac33335aebafc5f153849d94647e0118d02e809a86c632e`.

**Limitations:** The CSV alone does not authenticate the full extraction history or prove sentiment-clustering benefit. N1, N2, and N3 remain controlled alternatives; neither intensifier vocabulary has been selected using ground-truth labels.

**Disposition:** Candidate implementation and evidence ready for user PR review. User supplies/reviews supporting evidence and performs the merge. No notebooks, slide decks, or CSV datasets committed to GitHub.

## L3-S0 / S0-B / S0-C — Structural expression audit and implementation (2026-10-15)

**Status:** Candidate S0–S3 implementation committed to feature branch; Colab unit tests, full regression and corpus parity are **pending**. No PR or merge yet.

**Objective and leakage control:** Capture sentiment expression (not sentiment polarity or topic) using original RAW tweets, with a temporary URL-excluded view. Do not use airline_sentiment or confidence columns. RAW, P0/P1/P2 and existing V/N feature definitions are unchanged.

**S0 label-blind audit:** 14,639 RAW tweets; exclamation marks in 3,226 (22.04%), uppercase emphasis in 4,256 (29.07%), emojis in 492 (3.36%). Exclamation mean 0.363, median 0, p95 2, max 27; emoji mean 0.0676, median 0, p95 0, max 40. Spearman correlations: exclamation–uppercase 0.0514, exclamation–emoji 0.0497, uppercase–emoji 0.0065. These are expression indicators, not positive/negative labels.

**S0-B URL sensitivity:** 1,175 tweets (8.03%) contained matched URLs; 388 tweets (2.65%) changed at least one S feature. Exclamation changes 0; uppercase-ratio changes 386; emoji changes 3. S0-B sometimes consumed emojis directly adjacent to URLs.

**S0-C boundary correction:** Revised URL matching preserves adjacent emoji sequences and terminal punctuation. All 14,639 tweets retained their original emoji sequences (0 losses), exclamation changes remained 0; compared with S0-B, uppercase changed in 2 tweets and emoji count in 3. Means (RAW / S0-B / S0-C): exclamation 0.362866 / 0.362866 / 0.362866; uppercase 0.028206 / 0.028815 / 0.028830; emoji 0.067559 / 0.067218 / 0.067559. The S0 initial audit and later audits used slightly different uppercase tokenization rules; **do not interpret cross-audit uppercase means as pure URL effects**.

**Agreed candidate feature definitions:**
- `S_exclamation_count`: number of literal `!` characters in temporary URL-excluded RAW text.
- `S_uppercase_ratio`: count of eligible alphabetic tokens (at least 2 letters) that are entirely uppercase, divided by eligible-token count; 0 if none. Uses the S0-C Unicode-aware regex definition.
- `S_emoji_count`: count of emoji sequences via `emoji.emoji_list` after temporary URL exclusion.
- S0 none; S1 exclamation; S2 exclamation + uppercase; S3 all three. Raw counts/ratios retained unscaled in feature CSV; scaling is deferred to the clustering pipeline.

**Limitations:** URL boundaries are heuristic; uppercase acronyms, airport codes, handles and hashtags may still contribute noise. Emoji count is sparse. VADER also reacts to punctuation, case and emoji; marginal clustering value must be tested later. No handcrafted polarity thresholds.

**Evidence storage:** S0, S0-B, S0-C audit ZIPs and permanent team PowerPoint are held outside GitHub; no notebooks, datasets, slides or large outputs are committed.

**Verification gate:** Run `tests/test_s_features.py`, then `python -m pytest -q` for full regression; verify RAW SHA256 `a41aeefcff9740c090153f608691d345c84f59084e40f03169c2f667df66578e`, 14,639 rows, S0-C corpus means and emoji preservation. Export and review feature-only CSV before opening PR. User adds evidence and merges only after review.

## L3-S-VAL-01 — Colab test and full-corpus parity evidence (2026-10-15)

**Evidence source:** User-reported Colab outputs from the S feature branch; results were not independently rerun by this documentation update.

- S-specific automated tests: **19 passed**.
- Full project regression suite: **112 passed**.
- RAW source: `Tweets-redacted.csv`, SHA256 `a41aeefcff9740c090153f608691d345c84f59084e40f03169c2f667df66578e`, 1,561,768 bytes.
- Rows: **14,639**; `row_id` 0–14,638.
- Output columns: `row_id`, `S_exclamation_count`, `S_uppercase_ratio`, `S_emoji_count`.

| Statistic | Exclamation count | Uppercase ratio | Emoji count |
|---|---:|---:|---:|
| Mean | 0.3628663160051916 | 0.028830265685066452 | 0.06755925951226177 |
| Standard deviation | 0.919735 | 0.076977 | 0.617202 |
| Minimum | 0 | 0 | 0 |
| 25th percentile | 0 | 0 | 0 |
| Median | 0 | 0 | 0 |
| 75th percentile | 0 | 0.041667 | 0 |
| Maximum | 27 | 0.956522 | 40 |

Tweets with `S_emoji_count > 0`: **492**.

**Assessment:** Feature means and emoji prevalence match S0-C audit values to displayed precision. The RAW source fingerprint matches the earlier audit. The reported 19/19 and 112/112 tests passed. These establish extraction and regression consistency, not a sentiment-clustering benefit.

**Next verification:** Export and review feature-only CSV for schema, nonnegative counts, valid ratios, sequential row IDs, and absence of sentiment labels/confidence; capture any anomalies before PR. No notebook or CSV committed.
