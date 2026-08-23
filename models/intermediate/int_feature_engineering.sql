{#
    Target encodings are computed out-of-fold and smoothed; see
    macros/oof_target_encoding.sql for why.
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

-- Deterministic fold assignment, stable across runs because it hashes the
-- transaction id rather than relying on row order.
train_folded AS (
    SELECT *, ABS(HASH(trans_num)) % 5 AS enc_fold FROM train_only
),

{% for name, key_col in encodings %}
{{ oof_encoding_cte(name, key_col) }},
{% endfor %}

card_rates AS (
    SELECT
        cc_num,
        COUNT(*) AS card_txn_cnt,
        AVG(amt) AS card_mean_amt,
        COALESCE(STDDEV_SAMP(amt), 0.0) AS card_std_amt
    FROM train_only
    GROUP BY cc_num
),

unioned AS (
    SELECT *, 'train' AS dataset_split FROM {{ ref('stg_transactions_train') }}
    UNION ALL
    SELECT *, 'test' AS dataset_split FROM {{ ref('stg_transactions_test') }}
),

joined AS (
    SELECT
        u.*,
        COALESCE(cat.category_risk, g.global_rate) AS category_risk,
        COALESCE(st.state_risk,     g.global_rate) AS state_risk,
        COALESCE(mer.merchant_risk, g.global_rate) AS merchant_risk,
        -- Serving-equivalent references, for drift monitoring only.
        COALESCE(cat.category_risk_serving, g.global_rate) AS category_risk_serving,
        COALESCE(st.state_risk_serving,     g.global_rate) AS state_risk_serving,
        COALESCE(mer.merchant_risk_serving, g.global_rate) AS merchant_risk_serving,
        COALESCE(cr.card_txn_cnt, 0) AS card_txn_cnt,
        COALESCE(cr.card_mean_amt, 0.0) AS card_mean_amt,
        COALESCE(cr.card_std_amt, 0.0) AS card_std_amt
    FROM unioned u
    CROSS JOIN global_stats g
    LEFT JOIN category_risk_map cat ON u.trans_num = cat.trans_num
    LEFT JOIN state_risk_map    st  ON u.trans_num = st.trans_num
    LEFT JOIN merchant_risk_map mer ON u.trans_num = mer.trans_num
    LEFT JOIN card_rates        cr  ON u.cc_num    = cr.cc_num
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
    -- Cards with no prior history have card_mean_amt = card_std_amt = 0. Dividing by
    -- an epsilon there produced z-scores in the billions, which were rendered into
    -- the STR prompt as established fact. Emit the neutral value instead, and floor
    -- the denominators at $1 so a genuinely low-variance card cannot blow up either.
    CASE
        WHEN card_txn_cnt = 0 THEN 0.0
        ELSE (amt - card_mean_amt) / GREATEST(card_std_amt, 1.0)
    END AS amt_z_card,
    CASE
        WHEN card_txn_cnt = 0 THEN 1.0
        ELSE amt / GREATEST(card_mean_amt, 1.0)
    END AS amt_over_mean_card
FROM joined
