import json
import os
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
import pickle

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# Objective: maximize PR-AUC (1.0 is the best) for an imbalanced classification (fraud-like) problem.
# Decision: binary decision aligned to cost-sensitive thresholding.

X, y = make_classification(
    n_samples=20000, n_features=30, n_informative=10, n_redundant=5,
    weights=[0.95, 0.05], class_sep=1.2, flip_y=0.01, random_state=42
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"
    ))
])

# Use probability calibration to improve thresholding behavior
calibrated = CalibratedClassifierCV(estimator=baseline, method="sigmoid", cv=3)
calibrated.fit(X_train, y_train)

def eval_probs(y_true, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

y_prob_val = calibrated.predict_proba(X_val)[:, 1]
y_prob_test = calibrated.predict_proba(X_test)[:, 1]

metrics_val = eval_probs(y_val, y_prob_val)
metrics_test = eval_probs(y_test, y_prob_test)
print("Validation metrics:", metrics_val)
print("Test metrics:", metrics_test)

# Cost-sensitive threshold selection 
def expected_cost(y_true, y_prob, threshold, c_fn=10.0, c_fp=1.0):
    y_pred = (y_prob >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return c_fn * fn + c_fp * fp

def pick_threshold_cost(y_true, y_prob, c_fn=10.0, c_fp=1.0):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, y_prob, t, c_fn, c_fp) for t in thresholds]
    t_opt = thresholds[int(np.argmin(costs))]
    return float(t_opt), float(np.min(costs))

t_cost, min_cost = pick_threshold_cost(y_val, y_prob_val, c_fn=10.0, c_fp=1.0)
print("Cost-optimal threshold:", t_cost, "Min expected cost:", min_cost)

# Alternative: pick threshold to achieve target recall (e.g., fraud ops requirement)
def pick_threshold_by_target_recall(y_true, y_prob, target_recall=0.90):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Find earliest threshold meeting target recall
    idx = np.where(rec >= target_recall)[0]
    if len(idx) == 0:
        return 0.5
    # thr is length-1 smaller than prec/rec
    thr_idx = min(idx[0], len(thr) - 1)
    return float(thr[thr_idx])

t_recall90 = pick_threshold_by_target_recall(y_val, y_prob_val, 0.90)
print("Recall@90% threshold:", t_recall90)

# Cross-validation (Stratified K-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
for tr_idx, va_idx in skf.split(X_train, y_train):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    m = CalibratedClassifierCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"
            ))
        ]),
        method="sigmoid", cv=3
    )
    m.fit(X_tr, y_tr)
    probs = m.predict_proba(X_va)[:, 1]
    cv_aucs.append(roc_auc_score(y_va, probs))
print("CV ROC-AUC:", float(np.mean(cv_aucs)))

# Bootstrap CI (Confidence Interval) for PR-AUC
def bootstrap_ci(metric_fn, y_true, y_prob, B=1000, alpha=0.05):
    n = len(y_true); stats = []
    rng = np.random.default_rng(42)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    stats.sort()
    lo = stats[int((alpha/2)*B)]
    hi = stats[int((1-alpha/2)*B)]
    return float(np.mean(stats)), (float(lo), float(hi))

mean_prauc, (prauc_lo, prauc_hi) = bootstrap_ci(average_precision_score, y_test, y_prob_test)
print(f"Test PR-AUC mean={mean_prauc:.4f}, 95% CI=({prauc_lo:.4f}, {prauc_hi:.4f})")

# Final threshold selection policy
# Choose between cost-optimal and recall target; demo picks cost-optimal if recall>=80%, else recall-90
def apply_threshold_policy(y_true, y_prob, t_cost, t_recall, min_recall=0.80):
    # Evaluate recall at cost threshold
    y_pred_cost = (y_prob >= t_cost).astype(int)
    rec_cost = recall_score(y_true, y_pred_cost)
    if rec_cost >= min_recall:
        return t_cost, "cost_optimal"
    else:
        return t_recall, "recall_target"

THRESHOLD, policy = apply_threshold_policy(y_val, y_prob_val, t_cost, t_recall90, min_recall=0.80)
print("Selected threshold:", THRESHOLD, "Policy:", policy)

# Final test metrics at selected threshold
def classification_report_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

final_report = classification_report_at_threshold(y_test, y_prob_test, THRESHOLD)
print("Final test metrics @threshold:", final_report)

# Persist artifacts
# Save model, threshold, scaler inside pipeline via pickle
with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(calibrated, f)

with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
    json.dump({
        "threshold": THRESHOLD,
        "policy": policy,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "cv_roc_auc_mean": float(np.mean(cv_aucs)),
        "test_pr_auc_ci": [prauc_lo, prauc_hi]
    }, f, indent=2)

# Create a simple model card
with open(os.path.join(ARTIFACTS_DIR, "MODEL_CARD.md"), "w") as f:
    f.write(f"""# Model Card: FraudDemo v1.0
**Intended use:** Imbalanced binary classification demo for fraud-like detection.
**Data:** Synthetic (sklearn.make_classification); class imbalance ~5% positives.
**Metrics (test):** ROC-AUC {metrics_test['roc_auc']:.3f}, PR-AUC {metrics_test['pr_auc']:.3f} (95% CI {prauc_lo:.3f}–{prauc_hi:.3f}).
**Threshold policy:** {policy} with threshold={THRESHOLD:.3f}.
**Fairness:** Not applicable (synthetic, no sensitive attributes included).
**Limitations:** Synthetic distribution; performance may not transfer to real data.
**Monitoring suggestion:** Track PR-AUC, recall, calibration error, drift in feature distributions; trigger retraining on ≥5% PR-AUC drop.
""")

print(f"Artifacts saved to: {ARTIFACTS_DIR}")
