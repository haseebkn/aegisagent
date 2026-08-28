-- Reconstruct each training-row encoding using prior labels only. Any future-label
-- aggregate, random fold, or accidental inclusion of the current label will differ.

{% set encodings = [
    ('category_risk', 'category'),
    ('state_risk', 'state'),
    ('merchant_risk', 'merchant')
] %}
{% set smoothing = 20 %}

WITH source AS (
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
    FROM {{ ref('stg_transactions_train') }}
),

{% for name, key_col in encodings %}
{{ name }}_expected AS (
    SELECT
        trans_num,
        (
            COALESCE(SUM(is_fraud) OVER (
                PARTITION BY {{ key_col }}
                ORDER BY trans_date_trans_time, trans_num
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0.0)
            + {{ smoothing }} * COALESCE(
                global_fraud_prior / NULLIF(global_count_prior, 0),
                CAST({{ var('target_encoding_cold_start_prior', 0.005) }} AS DOUBLE)
            )
        ) / (
            COUNT(*) OVER (
                PARTITION BY {{ key_col }}
                ORDER BY trans_date_trans_time, trans_num
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) + {{ smoothing }}
        ) AS expected_value
    FROM source
){% if not loop.last %},{% endif %}
{% endfor %},

violations AS (
    {% for name, key_col in encodings %}
    SELECT
        '{{ name }}' AS feature,
        f.trans_num,
        f.{{ name }} AS actual_value,
        e.expected_value
    FROM {{ ref('fct_fraud_features') }} f
    JOIN {{ name }}_expected e USING (trans_num)
    WHERE f.dataset_split = 'train'
      AND ABS(f.{{ name }} - e.expected_value) > 1e-10
    {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
)

SELECT * FROM violations
