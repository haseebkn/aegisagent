-- Card transaction velocity over trailing windows.
--
-- NOTE: the window intentionally counts EVERY preceding transaction for the card,
-- not just the ones carrying dataset_split = 'train'. Filtering on the split here
-- silently zeroed both features for 99.5% of test rows (train ends 2020-06-21,
-- test begins the same day, so a test row's trailing window contains only test
-- rows), which meant the velocity model was trained on real counts and scored on
-- zeros. See the avg_txns_24h_by_split singular test for the regression guard.
SELECT
    *,
    CAST(COUNT(*) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '1 DAY' PRECEDING AND CURRENT ROW
    ) AS INTEGER) AS txns_24h,
    CAST(COUNT(*) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND CURRENT ROW
    ) AS INTEGER) AS txns_7d
FROM {{ ref('int_feature_engineering') }}
