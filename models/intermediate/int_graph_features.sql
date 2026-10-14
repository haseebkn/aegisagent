-- Entity/network features over the card <-> merchant bipartite graph.
--
-- This is the class of signal that per-transaction supervised models cannot see:
-- whether an entity sits in a neighbourhood that has already produced fraud. It is
-- also the closest analogue in this dataset to consortium-style analytics.
--
-- Leakage discipline: the graph is built from the TRAIN split only, and every
-- fraud-derived count excludes the current card's own contribution, so a row can
-- never be scored on its own label.
--
-- See docs/graph-features.md for the measured outcome on this dataset.

WITH train AS (
    SELECT cc_num, merchant, is_fraud
    FROM {{ ref('stg_transactions_train') }}
),

-- Cards with at least one confirmed fraud in the training window.
fraud_cards AS (
    SELECT DISTINCT cc_num
    FROM train
    WHERE is_fraud = 1
),

edges AS (
    SELECT DISTINCT cc_num, merchant
    FROM train
),

-- Merchant node: how many distinct cards, and how many of those are fraud cards.
merchant_nodes AS (
    SELECT
        e.merchant,
        COUNT(DISTINCT e.cc_num) AS merchant_card_degree,
        COUNT(DISTINCT CASE WHEN f.cc_num IS NOT NULL THEN e.cc_num END)
            AS merchant_fraud_card_cnt
    FROM edges e
    LEFT JOIN fraud_cards f ON e.cc_num = f.cc_num
    GROUP BY e.merchant
),

-- Card node: how many distinct merchants, and whether the card is itself a fraud card.
card_nodes AS (
    SELECT
        e.cc_num,
        COUNT(DISTINCT e.merchant) AS card_merchant_degree,
        MAX(CASE WHEN f.cc_num IS NOT NULL THEN 1 ELSE 0 END) AS card_is_fraud_card
    FROM edges e
    LEFT JOIN fraud_cards f ON e.cc_num = f.cc_num
    GROUP BY e.cc_num
),

-- Two-hop reach: distinct fraud cards a card shares at least one merchant with,
-- excluding the card itself.
two_hop AS (
    SELECT
        a.cc_num,
        COUNT(DISTINCT b.cc_num) AS card_2hop_fraud_cards
    FROM edges a
    JOIN edges b
      ON a.merchant = b.merchant
     AND a.cc_num <> b.cc_num
    JOIN fraud_cards f ON b.cc_num = f.cc_num
    GROUP BY a.cc_num
)

SELECT
    u.*,
    COALESCE(mn.merchant_card_degree, 0) AS merchant_card_degree,
    COALESCE(cn.card_merchant_degree, 0) AS card_merchant_degree,

    -- Subtract the current card only if the training graph contains this exact
    -- card/merchant edge. A known card visiting a new merchant contributed nothing
    -- to that merchant's historical numerator or denominator.
    GREATEST(
        COALESCE(mn.merchant_fraud_card_cnt, 0)
        - CASE WHEN own_edge.cc_num IS NOT NULL THEN COALESCE(cn.card_is_fraud_card, 0) ELSE 0 END,
        0
    ) AS merchant_fraud_card_cnt,

    -- Share of the merchant's other cards that have known fraud.
    CASE
        WHEN COALESCE(mn.merchant_card_degree, 0)
             - CASE WHEN own_edge.cc_num IS NOT NULL THEN 1 ELSE 0 END > 0
        THEN GREATEST(COALESCE(mn.merchant_fraud_card_cnt, 0)
                      - CASE WHEN own_edge.cc_num IS NOT NULL
                             THEN COALESCE(cn.card_is_fraud_card, 0) ELSE 0 END, 0)::DOUBLE
             / (mn.merchant_card_degree
                - CASE WHEN own_edge.cc_num IS NOT NULL THEN 1 ELSE 0 END)
        ELSE 0.0
    END AS merchant_fraud_card_ratio,

    COALESCE(th.card_2hop_fraud_cards, 0) AS card_2hop_fraud_cards

FROM {{ ref('int_all_features') }} u
LEFT JOIN merchant_nodes mn ON u.merchant = mn.merchant
LEFT JOIN card_nodes     cn ON u.cc_num   = cn.cc_num
LEFT JOIN two_hop        th ON u.cc_num   = th.cc_num
LEFT JOIN edges own_edge ON u.cc_num = own_edge.cc_num AND u.merchant = own_edge.merchant
