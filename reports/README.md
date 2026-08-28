# Evaluation reports

Machine-readable reports in this directory are generated from the causal Phase 1
feature mart. They are evidence for reproducibility, not claims of external validity.

- `rolling_validation.json`: expanding-window validation over `fraudTrain.csv`.
- `development_evaluation.json`: full decision-focused report on the development window.
- `locked_evaluation.json`: one-time Phase 1 result on the prospectively locked tail.

The locked tail is not historically pristine because pre-Phase-1 versions reported
aggregate results over all of `fraudTest.csv`. Every report retains that caveat.
