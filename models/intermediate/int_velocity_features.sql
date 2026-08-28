-- Card transaction velocity over trailing windows.
--
-- The window counts every strictly earlier transaction for the card,
-- not just the ones carrying dataset_split = 'train'. Filtering on the split here
-- silently zeroed both features for 99.5% of test rows (train ends 2020-06-21,
-- test begins the same day, so a test row's trailing window contains only test
-- rows), which meant the velocity model was trained on real counts and scored on
-- zeros. Ending one microsecond before the current timestamp excludes the current
-- transaction and all same-timestamp peers: these features mean PRIOR activity.
SELECT
    *,
    CAST(COUNT(*) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '1 DAY' PRECEDING AND INTERVAL '1 MICROSECOND' PRECEDING
    ) AS INTEGER) AS txns_24h,
    CAST(COUNT(*) OVER (
        PARTITION BY cc_num
        ORDER BY trans_date_trans_time ASC
        RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND INTERVAL '1 MICROSECOND' PRECEDING
    ) AS INTEGER) AS txns_7d
FROM {{ ref('int_feature_engineering') }}
