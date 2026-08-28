-- The evaluation tail must be later than every development-holdout transaction,
-- and ordinary training rows must never be assigned to either test role.

WITH bounds AS (
    SELECT
        MAX(CASE WHEN evaluation_role = 'development_holdout'
                 THEN trans_date_trans_time END) AS development_end,
        MIN(CASE WHEN evaluation_role = 'locked_evaluation'
                 THEN trans_date_trans_time END) AS evaluation_start,
        SUM(CASE WHEN dataset_split = 'train'
                  AND evaluation_role <> 'model_development' THEN 1 ELSE 0 END) AS bad_train,
        SUM(CASE WHEN dataset_split = 'test'
                  AND evaluation_role = 'development_holdout' THEN 1 ELSE 0 END) AS dev_rows,
        SUM(CASE WHEN dataset_split = 'test'
                  AND evaluation_role = 'locked_evaluation' THEN 1 ELSE 0 END) AS eval_rows
    FROM {{ ref('fct_fraud_features') }}
)

SELECT * FROM bounds
WHERE bad_train <> 0
   OR dev_rows = 0
   OR eval_rows = 0
   OR development_end > evaluation_start
