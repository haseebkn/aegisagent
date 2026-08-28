{#
    Prequential, smoothed target encoding.

    For a training row, only labels from rows ordered strictly before it may enter
    the feature. The prior also evolves using earlier rows only. This is causal in
    event time (with trans_num as a deterministic tie-breaker) and invariant to any
    mutation of later labels.

        risk = (prior_level_fraud + m * prior_global_rate)
               / (prior_level_count + m)

    Before any labels exist, target_encoding_cold_start_prior supplies the prior.
    Scoring rows use a map frozen from the complete training period; scoring labels
    never update it. The serving-equivalent column is retained only for drift checks.
#}

{% macro temporal_encoding_cte(name, key_col, smoothing=20) %}

{{ name }}_history AS (
    SELECT
        trans_num,
        SUM(is_fraud) OVER (
            PARTITION BY {{ key_col }}
            ORDER BY trans_date_trans_time, trans_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS level_fraud_prior,
        COUNT(*) OVER (
            PARTITION BY {{ key_col }}
            ORDER BY trans_date_trans_time, trans_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS level_count_prior
    FROM train_temporal
),

{{ name }}_totals AS (
    SELECT
        {{ key_col }} AS enc_key,
        SUM(is_fraud) AS fraud_sum,
        COUNT(*) AS level_count
    FROM train_only
    GROUP BY {{ key_col }}
),

{{ name }}_map AS (
    SELECT
        tt.trans_num,
        (
            COALESCE(h.level_fraud_prior, 0.0)
            + {{ smoothing }} * COALESCE(
                tt.global_fraud_prior / NULLIF(tt.global_count_prior, 0),
                s.cold_start_prior
            )
        ) / (COALESCE(h.level_count_prior, 0) + {{ smoothing }}) AS {{ name }},
        COALESCE(
            (tot.fraud_sum + {{ smoothing }} * g.global_rate)
            / NULLIF(tot.level_count + {{ smoothing }}, 0),
            g.global_rate
        ) AS {{ name }}_serving
    FROM train_temporal tt
    CROSS JOIN settings s
    CROSS JOIN global_stats g
    LEFT JOIN {{ name }}_history h ON tt.trans_num = h.trans_num
    LEFT JOIN {{ name }}_totals tot ON tt.{{ key_col }} = tot.enc_key

    UNION ALL

    SELECT
        te.trans_num,
        COALESCE(
            (tot.fraud_sum + {{ smoothing }} * g.global_rate)
            / NULLIF(tot.level_count + {{ smoothing }}, 0),
            g.global_rate
        ) AS {{ name }},
        COALESCE(
            (tot.fraud_sum + {{ smoothing }} * g.global_rate)
            / NULLIF(tot.level_count + {{ smoothing }}, 0),
            g.global_rate
        ) AS {{ name }}_serving
    FROM {{ ref('stg_transactions_test') }} te
    CROSS JOIN global_stats g
    LEFT JOIN {{ name }}_totals tot ON te.{{ key_col }} = tot.enc_key
)

{% endmacro %}
