-- Regression guard for the train/serve skew that zeroed the velocity features on
-- the development holdout. A distribution that differs by more than 2x, or
-- a split where the feature is almost always zero while the other is not, means
-- the feature is not being computed the same way at training and scoring time.
--
-- Returns one row per offending feature; dbt fails the test if any row comes back.
WITH stats AS (
    SELECT
        dataset_split,
        AVG(txns_24h)                                          AS avg_24h,
        AVG(txns_7d)                                           AS avg_7d,
        AVG(CASE WHEN txns_24h = 0 THEN 1.0 ELSE 0.0 END)      AS zero_frac_24h,
        AVG(CASE WHEN txns_7d  = 0 THEN 1.0 ELSE 0.0 END)      AS zero_frac_7d
    FROM {{ ref('fct_fraud_features') }}
    GROUP BY dataset_split
),

pivoted AS (
    SELECT
        MAX(CASE WHEN dataset_split = 'train' THEN avg_24h       END) AS train_avg_24h,
        MAX(CASE WHEN dataset_split = 'test'  THEN avg_24h       END) AS test_avg_24h,
        MAX(CASE WHEN dataset_split = 'train' THEN avg_7d        END) AS train_avg_7d,
        MAX(CASE WHEN dataset_split = 'test'  THEN avg_7d        END) AS test_avg_7d,
        MAX(CASE WHEN dataset_split = 'train' THEN zero_frac_24h END) AS train_zero_24h,
        MAX(CASE WHEN dataset_split = 'test'  THEN zero_frac_24h END) AS test_zero_24h,
        MAX(CASE WHEN dataset_split = 'train' THEN zero_frac_7d  END) AS train_zero_7d,
        MAX(CASE WHEN dataset_split = 'test'  THEN zero_frac_7d  END) AS test_zero_7d
    FROM stats
),

violations AS (
    SELECT 'txns_24h' AS feature, 'mean ratio outside [0.5, 2.0]' AS reason,
           train_avg_24h AS train_value, test_avg_24h AS test_value
    FROM pivoted
    WHERE test_avg_24h / NULLIF(train_avg_24h, 0) NOT BETWEEN 0.5 AND 2.0

    UNION ALL
    SELECT 'txns_7d', 'mean ratio outside [0.5, 2.0]', train_avg_7d, test_avg_7d
    FROM pivoted
    WHERE test_avg_7d / NULLIF(train_avg_7d, 0) NOT BETWEEN 0.5 AND 2.0

    UNION ALL
    SELECT 'txns_24h', 'zero-fraction gap exceeds 0.25', train_zero_24h, test_zero_24h
    FROM pivoted
    WHERE ABS(test_zero_24h - train_zero_24h) > 0.25

    UNION ALL
    SELECT 'txns_7d', 'zero-fraction gap exceeds 0.25', train_zero_7d, test_zero_7d
    FROM pivoted
    WHERE ABS(test_zero_7d - train_zero_7d) > 0.25
)

SELECT * FROM violations
