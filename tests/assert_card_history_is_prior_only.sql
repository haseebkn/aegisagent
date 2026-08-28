-- Every card-history feature must describe rows strictly before the current event
-- and remain bounded to the same 90-day window at training and scoring time.

WITH ordered AS (
    SELECT
        trans_num,
        card_txn_cnt,
        txns_24h,
        txns_7d,
        COUNT(*) OVER (
            PARTITION BY cc_num
            ORDER BY trans_date_trans_time
            RANGE BETWEEN INTERVAL '90 DAYS' PRECEDING
                  AND INTERVAL '1 MICROSECOND' PRECEDING
        ) AS expected_prior_count
    FROM {{ ref('fct_fraud_features') }}
)

SELECT *
FROM ordered
WHERE card_txn_cnt <> expected_prior_count
   OR (expected_prior_count = 0 AND (txns_24h <> 0 OR txns_7d <> 0))
