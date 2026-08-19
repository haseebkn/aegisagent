{{ config(
    materialized='table'
) }}

SELECT
    row_id,
    trans_num,
    cc_num,
    trans_date_trans_time,
    dataset_split,
    is_fraud,
    -- Core features
    amt,
    log_amt,
    distance_km,
    night,
    hour,
    day_of_week,
    hour_sin,
    hour_cos,
    is_online,
    -- Risk encodings
    category_risk,
    state_risk,
    merchant_risk,
    -- Card stats
    card_txn_cnt,
    card_mean_amt,
    card_std_amt,
    -- Relative card features
    amt_z_card,
    amt_over_mean_card,
    -- Velocities
    txns_24h,
    txns_7d,
    -- Interactions
    amt_x_catRisk,
    dist_x_online,
    amt_x_night,
    -- Graph / entity features (see int_graph_features.sql)
    merchant_card_degree,
    card_merchant_degree,
    merchant_fraud_card_cnt,
    merchant_fraud_card_ratio,
    card_2hop_fraud_cards
FROM {{ ref('int_graph_features') }}
