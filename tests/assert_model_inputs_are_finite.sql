-- Null/NaN/Infinity must fail before model fitting or serving; SQL comparisons
-- alone do not detect NULL and drift must never silently drop invalid rows.
{% set features = [
    'amt', 'log_amt', 'distance_km', 'night', 'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'is_online', 'category_risk', 'state_risk',
    'merchant_risk', 'category_risk_serving', 'state_risk_serving',
    'merchant_risk_serving', 'card_txn_cnt', 'card_mean_amt', 'card_std_amt',
    'amt_z_card', 'amt_over_mean_card', 'txns_24h', 'txns_7d',
    'amt_x_catRisk', 'dist_x_online', 'amt_x_night'
] %}

SELECT trans_num
FROM {{ ref('fct_fraud_features') }}
WHERE trans_num IS NULL OR cc_num IS NULL OR trans_date_trans_time IS NULL
   OR is_fraud IS NULL OR is_fraud NOT IN (0, 1)
   OR dataset_split IS NULL OR dataset_split NOT IN ('train', 'test')
   OR evaluation_role IS NULL
   OR evaluation_role NOT IN ('model_development', 'development_holdout', 'locked_evaluation')
   OR amt < 0 OR distance_km < 0 OR card_txn_cnt < 0 OR card_std_amt < 0
   OR txns_24h < 0 OR txns_7d < txns_24h
   OR night NOT IN (0, 1) OR is_online NOT IN (0, 1)
   OR hour NOT BETWEEN 0 AND 23 OR day_of_week NOT BETWEEN 0 AND 6
   OR category_risk NOT BETWEEN 0 AND 1 OR state_risk NOT BETWEEN 0 AND 1
   OR merchant_risk NOT BETWEEN 0 AND 1
   {% for feature in features %}
   OR {{ feature }} IS NULL OR NOT isfinite({{ feature }})
   {% endfor %}
