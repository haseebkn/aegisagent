-- A frozen full-training label map is safe only when every scoring event is later
-- than the training period. Reject accidentally swapped or overlapping CSVs.
WITH bounds AS (
    SELECT
        MAX(CASE WHEN dataset_split = 'train' THEN trans_date_trans_time END) AS train_end,
        MIN(CASE WHEN dataset_split = 'test' THEN trans_date_trans_time END) AS test_start
    FROM {{ ref('fct_fraud_features') }}
)
SELECT * FROM bounds
WHERE train_end IS NULL OR test_start IS NULL OR train_end >= test_start
