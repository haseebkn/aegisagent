WITH train_only AS (
    SELECT * FROM {{ ref('stg_transactions_train') }}
),

category_rates AS (
    SELECT 
        category, 
        AVG(is_fraud) AS category_risk
    FROM train_only
    GROUP BY category
),

state_rates AS (
    SELECT 
        state, 
        AVG(is_fraud) AS state_risk
    FROM train_only
    GROUP BY state
),

merchant_rates AS (
    SELECT 
        merchant, 
        AVG(is_fraud) AS merchant_risk
    FROM train_only
    GROUP BY merchant
),

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
        COALESCE(c.category_risk, 0.0) AS category_risk,
        COALESCE(s.state_risk, 0.0) AS state_risk,
        COALESCE(m.merchant_risk, 0.0) AS merchant_risk,
        COALESCE(cr.card_txn_cnt, 0) AS card_txn_cnt,
        COALESCE(cr.card_mean_amt, 0.0) AS card_mean_amt,
        COALESCE(cr.card_std_amt, 0.0) AS card_std_amt
    FROM unioned u
    LEFT JOIN category_rates c ON u.category = c.category
    LEFT JOIN state_rates s ON u.state = s.state
    LEFT JOIN merchant_rates m ON u.merchant = m.merchant
    LEFT JOIN card_rates cr ON u.cc_num = cr.cc_num
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
    -- Card stats
    card_txn_cnt,
    card_mean_amt,
    card_std_amt,
    -- Relative amt features
    (amt - card_mean_amt) / (card_std_amt + 1e-6) AS amt_z_card,
    amt / (card_mean_amt + 1e-6) AS amt_over_mean_card
FROM joined
