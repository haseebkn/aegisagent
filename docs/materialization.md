# Materialization strategy: measured, not assumed

Incremental materialization was on the roadmap as a performance improvement. It was
measured and **rejected**. So was materializing the intermediate models as tables.
This records the numbers, because "we didn't bother" and "we measured and it was
worse" are different statements.

## Measurements

Full rebuild of 1,852,394 rows, DuckDB, warm cache:

| Configuration | Total | Notes |
|---|---|---|
| **Intermediates as views** (shipped) | **25.8 – 27.6s** | mart materialization is ~24s of it |
| Intermediates as tables | 36.8s | **33% slower** |

Per-model, with intermediates as tables:

```
int_feature_engineering    8.98s
int_velocity_features      7.79s
int_all_features           1.81s
int_graph_features        16.92s
fct_fraud_features         0.92s
```

Materializing the chain forces DuckDB to write and re-read four intermediate results
totalling ~7M rows. Left as views, the whole pipeline collapses into a single fused
query the optimiser can plan end to end — projection pushdown means columns no
downstream model selects are never computed at all. The engine is better at
optimising the whole than the parts.

## Why incremental was rejected

**The saving does not exist.** The staging models are views over `read_csv_auto`, so
any run re-reads the source CSVs regardless of how the mart is materialized. Making
only `fct_fraud_features` incremental would skip a 24-second write while still paying
the full upstream scan. To make incremental genuinely save work, staging and
intermediate would have to become tables — which the measurement above shows costs
more than it saves at this volume.

**The correctness risk is real.** `txns_24h` and `txns_7d` are trailing-window
aggregates. An incremental run that processes only new rows computes them over a
truncated history and silently produces wrong values for every row near the batch
boundary. Correct incremental velocity needs a 7-day lookback re-read and a merge on
`trans_num`.

That is exactly the shape of the bug this project already shipped once: velocity
features that were wrong in a way no null check or range check could see
(`assert_velocity_no_train_serve_skew` exists because of it). Taking on that risk to
save 24 seconds is a bad trade.

## When this decision should be revisited

Revisit when any of these becomes true:

- **Volume.** Full refresh exceeds a few minutes, or source data no longer fits
  comfortably in memory. The crossover is a property of scale, not of principle.
- **Arrival pattern.** Data arrives as periodic batches appended to history rather
  than as two static files. Incremental pays off when the new fraction is small.
- **Engine.** On a warehouse billing per byte scanned — BigQuery, Snowflake,
  Redshift — the economics invert immediately and incremental with partition pruning
  becomes the obvious default. DuckDB reading a local file has no such meter.

If incremental is implemented then, the velocity lookback must be verified by
building the same window both ways and asserting the outputs are identical, not by
inspection.
