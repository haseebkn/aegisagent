-- Guards the divide-by-near-zero in amt_z_card / amt_over_mean_card. Cards with no
-- prior history get card_std_amt = 0, and the 1e-6 epsilon then produces z-scores
-- in the billions, which flow straight into the STR prompt as "fact".
SELECT
    trans_num,
    card_txn_cnt,
    card_std_amt,
    amt_z_card,
    amt_over_mean_card
FROM {{ ref('fct_fraud_features') }}
WHERE ABS(amt_z_card) > 1000
   OR ABS(amt_over_mean_card) > 1000
