# Out-of-fold target encoding: correct, and slightly worse here

## What changed

`category_risk`, `state_risk` and `merchant_risk` were computed as `AVG(is_fraud)`
per level over the whole training split, then applied back to those same training
rows. Every row contributed to the encoding it was scored on.

They are now computed **out-of-fold with empirical-Bayes smoothing**
(`macros/oof_target_encoding.sql`):

- A training row is encoded from the other four of five folds, so its own label never
  reaches its own feature value. Folds are assigned by hashing `trans_num`, so they
  are stable across runs and independent of row order.
- `risk = (fraud_sum + m·global_rate) / (level_count + m)` shrinks thinly-observed
  levels toward the prior.
- Test rows use full training statistics, which is correct — no test label is
  involved.
- Unseen levels fall back to the global rate. Previously they fell back to `0.0`,
  which made a novel merchant look maximally *safe*.

## Measured effect

| Held-out test | In-sample encoding | OOF, m=100 | OOF, m=20 |
|---|---|---|---|
| Meta PR AUC | **0.8010** | 0.7837 | 0.7823 |
| Meta ROC AUC | **0.9941** | 0.9914 | 0.9912 |
| Recall | **69.79%** | 69.23% | 67.93% |
| Precision | 84.82% | 84.71% | **85.91%** |
| Frauds caught | **1,497** | 1,485 | 1,457 |

The methodologically correct construction is **~2.3% worse on PR AUC**.

## Why the "fix" costs performance

This is worth being precise about, because the instinct is that removing leakage
should improve generalisation.

**There was very little leakage to remove.** Every encoded level is densely observed
— the least-frequent merchant has 727 training rows, categories and states far more.
A single row contributes about 0.1% of its level's mean. The in-sample encoding was
therefore almost identical to a leave-one-out encoding already.

**Out-of-fold adds variance.** Splitting into five folds means the same merchant
receives five different encodings across training rows — 2,868 distinct values for
693 merchants. That variance is noise the model has to fit around, and here it costs
more than the leakage it removes.

Smoothing is not the cause: m=100 shrinks a 727-row level by 12% and m=20 by 2.7%,
and both land within 0.002 PR AUC of each other. `m` was set from level support, not
searched against the test set.

## What was shipped, and why

**OOF is kept**, despite the lower number.

The in-sample encoding is only safe *because* this dataset happens to have dense
levels. That is a property of the data, not of the code. On a real portfolio with a
long tail of merchants — most seen a handful of times — in-sample encoding leaks
heavily and fails exactly where fraud concentrates. Choosing the construction that
is robust when level support is thin, and paying 2% on a dataset where support is
uniformly thick, is the right trade.

The comparison is recorded here rather than buried because the number moved the wrong
way, and a reader comparing this README against an earlier one deserves the
explanation.

## Expected side effect on drift monitoring

`scripts/drift.py` now reports **MODERATE** drift on `category_risk` (PSI 0.158) and
`state_risk` (PSI 0.199). This is intended, not a regression.

Training rows carry per-fold encodings while test rows carry a single full-training
encoding, so the two distributions have genuinely different shapes even though their
means agree to four decimals (0.006 vs 0.006). PSI measures distributional shift and
correctly sees it.

The distinction that matters: this is drift *by construction* between two encoding
schemes, not drift in the underlying population. It does not indicate a broken
pipeline, and it stays below the SIGNIFICANT threshold. If a serving-time monitor is
ever pointed at live data, the reference distribution should be the training rows'
*serving-equivalent* encodings, not their out-of-fold ones.

## Related

The graph features in [graph-features.md](graph-features.md) failed for the
neighbouring reason: a label-derived feature that looked strong in training because
label history was informative there, and was not once it had to generalise. High
level support is what separates the two outcomes — dense levels make in-sample
encoding nearly harmless, while a card-level constant derived from labels is not
dense in any useful sense.

## Not addressed

`card_txn_cnt`, `card_mean_amt` and `card_std_amt` are still full-window training
aggregates applied to training rows. They are **not** label-derived, so they cannot
leak the target, but they do let a training row see its own card's future spend. A
strictly causal version would compute them over a trailing window per row, as the
velocity features already do.
