SELECT
    *,
    (amt * category_risk) AS amt_x_catRisk,
    (distance_km * is_online) AS dist_x_online,
    (amt * night) AS amt_x_night
FROM {{ ref('int_velocity_features') }}
