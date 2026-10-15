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
