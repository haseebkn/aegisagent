SELECT
    *,
    CAST(SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '1 DAY' PRECEDING AND CURRENT ROW
    ) AS INTEGER) AS txns_24h,
    CAST(SUM(CASE WHEN dataset_split = 'train' THEN 1 ELSE 0 END) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND CURRENT ROW
    ) AS INTEGER) AS txns_7d
FROM {{ ref('int_feature_engineering') }}
