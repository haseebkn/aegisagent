-- Guards unstable early-history normalization and divide-by-near-zero failures.
-- Fewer than five prior observations must produce the neutral z-score; established
-- histories must remain within a deliberately generous sanity bound.
SELECT
    trans_num,
    card_txn_cnt,
    card_std_amt,
    amt_z_card,
    amt_over_mean_card
FROM {{ ref('fct_fraud_features') }}
WHERE (card_txn_cnt < 5 AND amt_z_card <> 0)
   OR ABS(amt_z_card) > 1000
   OR ABS(amt_over_mean_card) > 1000
