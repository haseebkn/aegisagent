{#
    Out-of-fold, smoothed target encoding.

    The previous implementation computed AVG(is_fraud) per level over the whole
    training split and then applied it back to those same training rows. Each row
    therefore contributed to the encoding it was scored on. With 693 merchants at
    727+ rows each the practical leakage was small, but it is still the same failure
    mode that made the graph features collapse (docs/graph-features.md): a feature
    built from labels looks better in training than it can possibly be at serving.

    Two corrections:

      Out-of-fold  -- a training row is encoded using every OTHER fold, so its own
                      label never reaches its own feature value.
      Smoothing    -- empirical-Bayes shrinkage toward the global fraud rate, so a
                      thinly-observed level is pulled to the prior instead of
                      reporting a noisy extreme.

          risk = (fraud_sum + m * global_rate) / (level_count + m)

    m is set from level support, not searched against the test set. Every level here
    has at least 727 training rows, so m = 20 shrinks a well-observed level by under
    3% while still pulling a thinly-observed one toward the prior. m = 100 was the
    first value tried and shrank every level by ~12%, discarding real signal.

    Test rows use the full training statistics, which is correct: no test label is
    involved. Unseen levels fall back to the global rate rather than to 0.0, which
    previously made a novel merchant look maximally safe.
#}

{% macro oof_encoding_cte(name, key_col, folds=5, smoothing=20) %}

{{ name }}_totals AS (
    SELECT
        {{ key_col }} AS enc_key,
        SUM(is_fraud) AS fraud_sum,
        COUNT(*)      AS level_count
    FROM train_folded
    GROUP BY {{ key_col }}
),

{{ name }}_by_fold AS (
    SELECT
        {{ key_col }} AS enc_key,
        enc_fold,
        SUM(is_fraud) AS fraud_sum,
        COUNT(*)      AS level_count
    FROM train_folded
    GROUP BY {{ key_col }}, enc_fold
),

{{ name }}_map AS (
    -- Training rows: subtract the row's own fold from the level statistics.
    SELECT
        tf.trans_num,
        COALESCE(
            (t.fraud_sum - COALESCE(f.fraud_sum, 0) + {{ smoothing }} * g.global_rate)
            / NULLIF(t.level_count - COALESCE(f.level_count, 0) + {{ smoothing }}, 0),
            g.global_rate
        ) AS {{ name }}
    FROM train_folded tf
    CROSS JOIN global_stats g
    LEFT JOIN {{ name }}_totals  t ON tf.{{ key_col }} = t.enc_key
    LEFT JOIN {{ name }}_by_fold f ON tf.{{ key_col }} = f.enc_key
                                  AND tf.enc_fold      = f.enc_fold

    UNION ALL

    -- Test rows: full training statistics, smoothed.
    SELECT
        te.trans_num,
        COALESCE(
            (t.fraud_sum + {{ smoothing }} * g.global_rate)
            / NULLIF(t.level_count + {{ smoothing }}, 0),
            g.global_rate
        ) AS {{ name }}
    FROM {{ ref('stg_transactions_test') }} te
    CROSS JOIN global_stats g
    LEFT JOIN {{ name }}_totals t ON te.{{ key_col }} = t.enc_key
)

{% endmacro %}
