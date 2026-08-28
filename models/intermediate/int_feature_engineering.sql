{#
    Label-derived encodings and card statistics are strictly prior-only for every
    training row. Scoring rows use frozen training-label maps. See
    macros/temporal_target_encoding.sql and docs/target-encoding.md.
#}
{% set encodings = [
    ('category_risk', 'category'),
    ('state_risk',    'state'),
    ('merchant_risk', 'merchant')
] %}

WITH train_only AS (
    SELECT * FROM {{ ref('stg_transactions_train') }}
),

global_stats AS (
    SELECT AVG(is_fraud) AS global_rate FROM train_only
),

settings AS (
    SELECT CAST({{ var('target_encoding_cold_start_prior', 0.005) }} AS DOUBLE)
        AS cold_start_prior
),

train_temporal AS (
    SELECT
        *,
        SUM(is_fraud) OVER (
            ORDER BY trans_date_trans_time, trans_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS global_fraud_prior,
        COUNT(*) OVER (
            ORDER BY trans_date_trans_time, trans_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS global_count_prior
    FROM train_only
),

{% for name, key_col in encodings %}
{{ temporal_encoding_cte(name, key_col) }},
{% endfor %}

unioned AS (
    SELECT *, 'train' AS dataset_split FROM {{ ref('stg_transactions_train') }}
    UNION ALL
    SELECT *, 'test' AS dataset_split FROM {{ ref('stg_transactions_test') }}
),

with_card_history AS (
    SELECT
        *,
        -- A bounded 90-day history is causal and avoids the mechanical calendar-time
        -- drift of lifetime counts. Same-timestamp peers are excluded.
        COUNT(*) OVER (
            PARTITION BY cc_num
            ORDER BY trans_date_trans_time
            RANGE BETWEEN INTERVAL '90 DAYS' PRECEDING
                  AND INTERVAL '1 MICROSECOND' PRECEDING
        ) AS card_txn_cnt,
        AVG(amt) OVER (
            PARTITION BY cc_num
            ORDER BY trans_date_trans_time
            RANGE BETWEEN INTERVAL '90 DAYS' PRECEDING
                  AND INTERVAL '1 MICROSECOND' PRECEDING
        ) AS card_mean_amt,
        STDDEV_SAMP(amt) OVER (
            PARTITION BY cc_num
            ORDER BY trans_date_trans_time
            RANGE BETWEEN INTERVAL '90 DAYS' PRECEDING
                  AND INTERVAL '1 MICROSECOND' PRECEDING
        ) AS card_std_amt
    FROM unioned
),

joined AS (
    SELECT
        u.* EXCLUDE (card_txn_cnt, card_mean_amt, card_std_amt),
        COALESCE(cat.category_risk, g.global_rate) AS category_risk,
        COALESCE(st.state_risk,     g.global_rate) AS state_risk,
        COALESCE(mer.merchant_risk, g.global_rate) AS merchant_risk,
        -- Serving-equivalent references, for drift monitoring only.
        COALESCE(cat.category_risk_serving, g.global_rate) AS category_risk_serving,
        COALESCE(st.state_risk_serving,     g.global_rate) AS state_risk_serving,
        COALESCE(mer.merchant_risk_serving, g.global_rate) AS merchant_risk_serving,
        CAST(u.card_txn_cnt AS INTEGER) AS card_txn_cnt,
        COALESCE(u.card_mean_amt, 0.0) AS card_mean_amt,
        COALESCE(u.card_std_amt, 0.0) AS card_std_amt
    FROM with_card_history u
    CROSS JOIN global_stats g
    LEFT JOIN category_risk_map cat ON u.trans_num = cat.trans_num
    LEFT JOIN state_risk_map    st  ON u.trans_num = st.trans_num
    LEFT JOIN merchant_risk_map mer ON u.trans_num = mer.trans_num
)

SELECT
    row_id,
    trans_date_trans_time,
    cc_num,
    merchant,
    category,
    amt,
    first,
    last,
    gender,
    street,
    city,
    state,
    zip,
    lat,
    long,
    city_pop,
    job,
    dob,
    trans_num,
    unix_time,
    merch_lat,
    merch_long,
    is_fraud,
    dataset_split,
    hour,
    day_of_week,
    night,
    -- Cyclical hour
    SIN(2.0 * 3.141592653589793 * hour / 24.0) AS hour_sin,
    COS(2.0 * 3.141592653589793 * hour / 24.0) AS hour_cos,
    -- Distance
    (2.0 * 6371.0 * ASIN(SQRT(
        SIN(RADIANS(merch_lat - lat)/2.0)^2 + 
        COS(RADIANS(lat)) * COS(RADIANS(merch_lat)) * SIN(RADIANS(merch_long - long)/2.0)^2
    ))) AS distance_km,
    -- Log amt
    LN(1.0 + amt) AS log_amt,
    -- Online flag
    CASE WHEN category LIKE '%_net' THEN 1 ELSE 0 END AS is_online,
    -- Risk features
    category_risk,
    state_risk,
    merchant_risk,
    category_risk_serving,
    state_risk_serving,
    merchant_risk_serving,
    -- Card stats
    card_txn_cnt,
    card_mean_amt,
    card_std_amt,
    -- Relative amt features.
    -- A sample standard deviation is technically defined with two observations but
    -- is too unstable to present as evidence at that support. Emit the neutral value
    -- until five prior transactions exist, and floor the denominator at $1 for
    -- genuinely low-variance cards.
    CASE
        WHEN card_txn_cnt < 5 OR card_std_amt = 0 THEN 0.0
        ELSE (amt - card_mean_amt) / GREATEST(card_std_amt, 1.0)
    END AS amt_z_card,
    CASE
        WHEN card_txn_cnt = 0 THEN 1.0
        ELSE amt / GREATEST(card_mean_amt, 1.0)
    END AS amt_over_mean_card
FROM joined
