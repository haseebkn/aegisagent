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

| Development holdout | In-sample encoding | OOF, m=100 | OOF, m=20 |
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

## Drift monitoring: measured on serving-equivalent columns

Comparing the out-of-fold column on training rows against the full-training column on
scoring rows measures the encoding *construction*, not the data. It read as MODERATE
drift on the full dataset (PSI 0.16-0.20) and SIGNIFICANT on the CI fixture (PSI above
5), in both cases while the means agreed to three decimals.

The first attempt at a fix excluded these three features from the drift gate. That was
wrong: `merchant_risk` and `category_risk` are two of the model's strongest inputs
(ROC AUC 0.72 each, behind only `amt`), so excluding them meant a broken encoding join
could never fail the gate. A monitor that cannot fail on its most important features
invites trust it has not earned.

The mart therefore emits a **serving-equivalent** column per encoding —
`category_risk_serving`, `state_risk_serving`, `merchant_risk_serving` — holding the
value each row would receive at scoring time. `scripts/drift.py` measures on those.
The comparison is apples-to-apples, all 19 monitored features stay under the gate, and
no exclusion list is needed.

| | Before | After |
|---|---|---|
| Full dataset | 0 significant, **2 moderate** | **0 significant, 0 moderate, 19 stable** |
| CI fixture | **3 significant** (excluded from gate) | **0 significant, 0 moderate, 19 stable** |
| Features under the gate | 16 of 19 | **19 of 19** |

Models still train on the out-of-fold column; the serving columns are never a model
input, and `test_serving_equivalent_encodings_are_not_model_inputs` enforces that. On a
training row the serving value is computed from statistics that include that row, so
feeding it to a model would reintroduce exactly the leakage out-of-fold encoding exists
to remove.

**Verified to still catch a real break.** Simulating a failed encoding join — every
scoring row collapsing to the global fraud rate — leaves the mean *unchanged* at
0.005791, so any mean-ratio check passes it. PSI flags it at 12.43, SIGNIFICANT.

### A note on the reference window

Train-versus-test is an artifact of this project having two static splits. In a real
deployment the reference is a rolling window of recent scored traffic compared against
the current one; both sides are serving-equivalent by definition and the mismatch never
arises. The serving columns make the static-split comparison behave the way a rolling
one naturally would.

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
