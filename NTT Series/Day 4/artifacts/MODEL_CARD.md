# Model Card: FraudDemo v1.0
**Intended use:** Imbalanced binary classification demo for fraud-like detection.
**Data:** Synthetic (sklearn.make_classification); class imbalance ~5% positives.
**Metrics (test):** ROC-AUC 0.883, PR-AUC 0.624 (95% CI 0.554–0.693).
**Threshold policy:** recall_target with threshold=0.000.
**Fairness:** Not applicable (synthetic, no sensitive attributes included).
**Limitations:** Synthetic distribution; performance may not transfer to real data.
**Monitoring suggestion:** Track PR-AUC, recall, calibration error, drift in feature distributions; trigger retraining on ≥5% PR-AUC drop.
