# Entity/graph features: what the card–merchant network actually carries

## Why build this

Per-transaction supervised models see one row at a time. They cannot see that an
entity sits in a neighbourhood that has already produced fraud — which is the signal
consortium-style financial-crime platforms are built around. This layer tests how
much of that signal exists in this dataset.

Implemented in `models/intermediate/int_graph_features.sql`; measured by
`scripts/graph_signal.py`.

## The graph

Built from the training split only, over the bipartite card ↔ merchant graph.

| | |
|---|---|
| Cards | 983 |
| Merchants | 693 |
| Distinct edges | 479,072 |
| **Density** | **0.703** |

Density 0.703 means the average card transacted with 70% of all merchants. For
comparison, a real card portfolio is extremely sparse — a cardholder touches a
handful of merchants out of millions.

Consequences measured directly:

- **All 983 cards** touch at least one merchant with fraud history.
- 762 of 983 cards (77.5%) have at least one fraud in training.
- Every merchant has between 319 and 654 fraud-associated cards, out of ~691 cards.

A binary "is this entity connected to known fraud?" feature is therefore constant at
1 for every row. Any community-detection or ring-finding approach has nothing to
find: the graph is close to complete, so every entity is a neighbour of every other.

## Features implemented

All fraud-derived counts exclude the current card's own contribution, so a row can
never be scored on its own label.

| Feature | Definition |
|---|---|
| `merchant_card_degree` | distinct cards seen at the merchant |
| `card_merchant_degree` | distinct merchants used by the card |
| `merchant_fraud_card_cnt` | distinct fraud cards at the merchant, self-excluded |
| `merchant_fraud_card_ratio` | that count over the merchant's other cards |
| `card_2hop_fraud_cards` | distinct fraud cards sharing ≥1 merchant with this card |

## Univariate signal (development holdout)

Base rate — the PR AUC of a random ranker — is 0.0039.

| Feature | ROC AUC | PR AUC | Lift |
|---|---|---|---|
| `card_2hop_fraud_cards` | **0.7918** | 0.0127 | 3.3× |
| `merchant_fraud_card_ratio` | 0.6057 | 0.0056 | 1.5× |
| `merchant_fraud_card_cnt` | 0.5364 | 0.0052 | 1.3× |
| `merchant_card_degree` | 0.5233 | 0.0047 | 1.2× |
| `card_merchant_degree` | 0.3350 | 0.0028 | 0.7× (inverted) |
| *reference:* `amt` | 0.8332 | 0.1369 | 35.5× |
| *reference:* `category_risk` | 0.7246 | 0.0092 | 2.4× |
| *reference:* `merchant_risk` | 0.7150 | 0.0091 | 2.3× |

This was more signal than the density figure predicted. `card_2hop_fraud_cards`
ranks better than either existing risk encoding on ROC AUC — and, as the next
section shows, it still made the model substantially worse.

`card_merchant_degree` carries inverted signal (AUC 0.335): cards using *fewer*
distinct merchants are more likely to be defrauded. Inverted is still informative —
a tree model uses it fine.

## Marginal contribution: strongly negative

Univariate AUC measures whether a feature *can* separate the classes. It does not
measure what the feature adds on top of the features already present. The three
graph features with signal were added to Model 4 and the ensemble retrained.

| Development holdout | Without graph features | With graph features |
|---|---|---|
| Model 4 PR AUC | **0.7974** | 0.1797 |
| Meta PR AUC | **0.8010** | 0.6063 |
| Meta ROC AUC | **0.9941** | 0.9911 |
| Recall | **69.79%** | 50.26% |
| Precision | 84.82% | 79.97% |
| Frauds caught | **1,497** | 1,078 |

Model 4's PR AUC fell by 77%. The ensemble lost 419 caught frauds. The graph
features were reverted; they remain available in the mart but are deliberately not
fed to any model, and `scripts/config.py` records why.

### Why a 0.79-AUC feature made the model worse

The graph features absorbed only 5.3% of Model 4's total importance, so the damage
is not simply that they crowded out better features.

`card_2hop_fraud_cards` and `card_merchant_degree` are **card-level constants derived
from training labels**. Every transaction on a given card carries the same value, and
that value encodes how much fraud that card's neighbourhood produced during training.
The forest therefore learns to order transactions partly by *card identity* rather
than by evidence about the transaction itself. Within the training window that
ordering is close to free accuracy. On the test window it is mostly noise — 77.5% of
cards have fraud history, so the partition separates almost nothing — and the
resulting score ordering is worse than the one built on transaction-level evidence
alone. Ranking quality (PR AUC) collapses even though the feature only appears in a
minority of splits.

This is the same failure mode as in-sample target encoding, one level up: a feature
built from labels looks excellent until it has to generalise to entities whose label
history is no longer informative.

## Caveat on the two-hop feature

`card_2hop_fraud_cards` is a card-level aggregate computed over the training window.
Its predictive power partly reflects that cards compromised during training are
likely to be compromised again in test — the simulator reuses victim cards. That is
legitimate in production (a previously compromised card genuinely is higher risk)
but it is closer to a card-history feature than to a network-structure feature. It
should not be read as evidence that ring detection works on this data. It does not;
there are no rings to detect.

## What this does not establish

The absence of network structure here is a property of the **Sparkov generator**,
which assigns merchants to customers close to uniformly at random. It says nothing
about whether graph features work on real card data, where merchant affinity is
strong, cardholder–merchant graphs are sparse, and mule networks and funnel accounts
produce genuine community structure. On real data this is where the signal a
per-transaction model cannot reach tends to live.

The correct reading: **this dataset cannot be used to evaluate graph-based fraud
detection.** Demonstrating that requires data with realistic entity relationships.

## What was kept

- `models/intermediate/int_graph_features.sql` — the graph layer, with self-exclusion
  discipline, materialised into the mart.
- `scripts/graph_signal.py` — the measurement harness.
- This document.

The features are not wired into any model. Retaining the machinery while rejecting
the features is deliberate: the capability is real and portable to data with genuine
network structure, and the negative result is worth more than a feature that looked
plausible and quietly degraded the ensemble.
