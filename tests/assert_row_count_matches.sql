-- Guards against silent row loss in the union/join chain.
--
-- The expected count is a var so the same test runs against the full Kaggle dataset
-- and against the small CI fixture:
--   dbt test --vars '{expected_row_count: 14500}'
SELECT 1
FROM (
    SELECT COUNT(*) AS row_count FROM {{ ref('fct_fraud_features') }}
)
WHERE row_count != {{ var('expected_row_count') }}
