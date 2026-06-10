SELECT 1
FROM (
    SELECT COUNT(*) AS row_count FROM {{ ref('fct_fraud_features') }}
)
WHERE row_count != 1852394
