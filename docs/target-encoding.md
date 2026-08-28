# Causal target encoding and prior-only card history

## Why out-of-fold was not enough

The Phase 0 pipeline replaced in-sample target encoding with deterministic random
folds. That prevented a row's own label from entering its feature, but it was not
causal: an early transaction still used labels from later dates in the other four
folds. This made internal training and calibration evidence inconsistent with the
chronological model split.

Phase 1 replaces random folds with a **prequential encoding**. For each training row,
category, state, and merchant histories end at the immediately preceding event:

```text
risk = (prior_level_fraud + m × prior_global_rate)
       / (prior_level_count + m)
```

Both the level statistics and the global prior use earlier labels only. `trans_num`
is a deterministic tie-breaker. Before any label history exists, the configurable
cold-start prior is 0.005. Scoring rows use maps frozen from the complete training
period; scoring labels never update them.

Implementation: [macros/temporal_target_encoding.sql](../macros/temporal_target_encoding.sql).

## Card-history features

`card_txn_cnt`, `card_mean_amt`, and `card_std_amt` now use transactions from the
strictly preceding 90 days. The bounded window matters for two reasons:

1. It excludes the current transaction and all future activity.
2. It prevents a lifetime count from growing mechanically with calendar time.

An initial cumulative implementation was causal but failed the drift gate:
`card_txn_cnt` moved from a mean of 908.9 in training to 2,110.2 in development
(PSI 4.98). The 90-day definition reduced that to 285.7 versus 330.2 (PSI 0.1705,
moderate); card mean and standard deviation are stable.

Amount z-scores remain neutral until five prior transactions exist. Two observations
are mathematically sufficient for a sample standard deviation but too unstable to
present as meaningful investigation evidence.

The 24-hour and 7-day velocity windows also end one microsecond before the current
timestamp, so their minimum is zero rather than one.

## Serving-equivalent drift references

Training rows receive their historical prequential encoding while scoring rows
receive a frozen full-training map. Directly comparing those columns would measure
construction differences. The mart therefore retains monitoring-only
`*_risk_serving` columns: the value a training row would receive from the frozen map.
All model feature lists explicitly exclude these columns.

On the Phase 1 mart, the drift gate reports zero significant, one moderate, and 18
stable features. The machine-readable result is in
[reports/drift.json](../reports/drift.json).

## Verification

The implementation is protected at three levels:

- `assert_target_encodings_are_temporal.sql` independently reconstructs all three
  training encodings from earlier labels and compares every value.
- `assert_card_history_is_prior_only.sql` independently reconstructs the 90-day
  prior count and verifies first-event velocity is zero.
- `test_temporal_feature_contract.py` rejects random-fold hashes, current-row velocity,
  or removal of the explicit temporal window contracts.

The full mart passes all 15 dbt tests over 1,852,394 rows.

## Evaluation consequence

After causal reconstruction and bounded card history, the final ensemble achieves
PR AUC 0.7766 on the development window and 0.6711 on the prospectively locked tail.
The gap is disclosed rather than averaged away. The locked tail is not historically
pristine because older project revisions reported aggregate metrics over the entire
public test file; a genuinely external dataset is still required.
