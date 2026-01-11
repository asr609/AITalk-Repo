# ML workflow for Kaggle tabular datasets: ingestion, modeling, ensembles, evaluation, and business value.
# Author: Amandeep Singh Reen. Reproducible, production-minded pipeline.

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

# Base models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Label encoding
from sklearn.preprocessing import LabelEncoder

# Plotting
import matplotlib.pyplot as plt

import logging
import sys

# --------------------------
# Logging configuration
# --------------------------
LOG_FILE = "Final_Day/pipeline.log"

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),   # write to log file
        logging.StreamHandler(sys.stdout)          # also logger.info to console
    ]
)

logger = logging.getLogger(__name__)


CONFIG = {
    # Path to your Kaggle CSV (download from https://www.kaggle.com/datasets and set path)
    "dataset_path": "Final_Day/ecommerce_customer_behavior.csv",  # <-- change to your file
    # Target column name in your dataset
    "target_col": "Customer_Type",
    # Optional: list of columns to drop (IDs, leakage)
    "drop_cols": [],
    # Optional: force column types (use dict: {"col": "categorical"/"numeric"})
    "type_overrides": {},
    # Train/val/test split ratios
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    # Class imbalance handling
    "use_class_weight": True,
    # Cross-validation folds
    "cv_folds": 5,
    # Business costs (per decision): set realistic values for your problem
    "cost_fn": 10.0,   # cost of false negative (missed conversion/fraud/etc.)
    "cost_fp": 2.0     # cost of false positive (unnecessary action/alert/etc.)
}


# --------------------------
# Data loading and splitting
# --------------------------

def load_dataset(path, target_col, drop_cols):
    assert os.path.exists(path), f"File not found: {path}"
    df = pd.read_csv(path)
    assert target_col in df.columns, f"Target '{target_col}' not in columns."
    for c in drop_cols:
        assert c in df.columns, f"Drop column '{c}' not in columns."
    df = df.drop(columns=drop_cols)
    return df

def split_dataset(df, target_col, test_size, val_size, random_state):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    stratify = y if y.nunique() <= 20 else None  # stratify for classification
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    val_ratio_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_relative,
        random_state=random_state, stratify=(y_trainval if stratify is not None else None)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# --------------------------
# Feature typing and preprocessing
# --------------------------

def detect_feature_types(df, type_overrides, target_col):
    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols, num_cols = [], []
    for c in feature_cols:
        if c in type_overrides:
            if type_overrides[c] == "categorical":
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
    return cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor

# --------------------------
# Models and parameter grids
# --------------------------

def get_models(class_weight_enabled):
    class_weight = "balanced" if class_weight_enabled else None
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight=class_weight, solver="lbfgs"),
        "rf": RandomForestClassifier(class_weight=class_weight, random_state=CONFIG["random_state"]),
        "gb": GradientBoostingClassifier(random_state=CONFIG["random_state"]),
        "ada": AdaBoostClassifier(random_state=CONFIG["random_state"])
    }
    param_grids = {
        "logreg": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"]
        },
        "rf": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 5],
            "model__max_features": ["sqrt", 0.5]
        },
        "gb": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0]
        },
        "ada": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.5, 1.0]
        }
    }
    return models, param_grids

# --------------------------
# Training and evaluation
# --------------------------

def evaluate_classifier(y_true, y_proba, threshold=None):
    if threshold is None:
        # default threshold = 0.5; later we’ll sweep for business value
        threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba))
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return metrics, cm, report

def expected_business_cost(y_true, y_proba, threshold, cost_fp, cost_fn):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    # Expected cost per decision
    cost = (fp * cost_fp + fn * cost_fn) / max(total, 1)
    return cost, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def sweep_threshold_for_business(y_true, y_proba, cost_fp, cost_fn):
    thresholds = np.linspace(0.0, 1.0, 51)
    costs = []
    for t in thresholds:
        c, _ = expected_business_cost(y_true, y_proba, t, cost_fp, cost_fn)
        costs.append(c)
    best_idx = int(np.argmin(costs))
    return thresholds[best_idx], costs[best_idx], pd.DataFrame({"threshold": thresholds, "cost": costs})

def fit_and_tune(X_train, y_train, preprocessor, model_key, model, param_grid, cv_folds):
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=CONFIG["random_state"])
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="average_precision",  # robust under imbalance
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid

def plot_curves(y_true, y_proba, title_prefix, out_dir):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC curve")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_roc.png"), bbox_inches="tight")
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_pr.png"), bbox_inches="tight")
    plt.close()

def plot_business_curve(df_thr, title_prefix, out_dir):
    plt.figure(figsize=(6,4))
    plt.plot(df_thr["threshold"], df_thr["cost"], color="purple")
    plt.xlabel("Threshold")
    plt.ylabel("Expected cost per decision")
    plt.title(f"{title_prefix} Business cost vs threshold")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_business.png"), bbox_inches="tight")
    plt.close()

# --------------------------
# Main workflow
# --------------------------

def main():
    # Load
    df = load_dataset(CONFIG["dataset_path"], CONFIG["target_col"], CONFIG["drop_cols"])
    logger.info(f"Loaded dataset shape: {df.shape}")
    logger.info("Target distribution:")
    logger.info(df[CONFIG["target_col"]].value_counts(normalize=True))

    #Label encoding for binary classification if neede

    # Encode target
    le = LabelEncoder()
    df[CONFIG["target_col"]] = le.fit_transform(df[CONFIG["target_col"]])
    logger.info("Classes:", le.classes_)  # ['New', 'Returning']

    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, CONFIG["target_col"], CONFIG["test_size"], CONFIG["val_size"], CONFIG["random_state"]
    )

    # Feature types and preprocessor
    cat_cols, num_cols = detect_feature_types(df, CONFIG["type_overrides"], CONFIG["target_col"])
    logger.info(f"Detected categorical: {cat_cols}")
    logger.info(f"Detected numeric: {num_cols}")
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # Models
    models, param_grids = get_models(CONFIG["use_class_weight"])

    results = []
    best_model_key, best_grid, best_val_score = None, None, -np.inf

    for key, model in models.items():
        logger.info(f"\nTraining and tuning: {key}")
        grid = fit_and_tune(X_train, y_train, preprocessor, key, model, param_grids[key], CONFIG["cv_folds"])

        # Evaluate on validation
        val_proba = grid.predict_proba(X_val)[:, 1]
        metrics, cm, report = evaluate_classifier(y_val, val_proba, threshold=0.5)
        logger.info(f"Validation metrics ({key}): {json.dumps(metrics, indent=2)}")
        logger.info(f"Confusion matrix ({key}):\n{cm}")

        # Track best by PR-AUC
        if metrics["pr_auc"] > best_val_score:
            best_val_score = metrics["pr_auc"]
            best_model_key = key
            best_grid = grid

        results.append({
            "model": key,
            "best_params": grid.best_params_,
            "val_metrics": metrics
        })

    logger.info("\nModel comparison (validation PR-AUC):")
    for r in sorted(results, key=lambda x: x["val_metrics"]["pr_auc"], reverse=True):
        logger.info(f"- {r['model']}: PR-AUC={r['val_metrics']['pr_auc']:.4f}  ROC-AUC={r['val_metrics']['roc_auc']:.4f}")

    logger.info(f"\nSelected model: {best_model_key} with params: {best_grid.best_params_}")

    # Test evaluation with business threshold optimization
    test_proba = best_grid.predict_proba(X_test)[:, 1]
    best_thr, best_cost, df_thr = sweep_threshold_for_business(
        y_test.values, test_proba, CONFIG["cost_fp"], CONFIG["cost_fn"]
    )
    metrics_05, cm_05, _ = evaluate_classifier(y_test.values, test_proba, threshold=0.5)
    metrics_bt, cm_bt, _ = evaluate_classifier(y_test.values, test_proba, threshold=best_thr)

    logger.info("\nTest metrics at threshold=0.5:")
    logger.info(json.dumps(metrics_05, indent=2))
    logger.info(f"Confusion matrix:\n{cm_05}")

    logger.info(f"\nBusiness-optimized threshold: {best_thr:.3f} (expected cost per decision={best_cost:.4f})")
    logger.info("Test metrics at business-optimized threshold:")
    logger.info(json.dumps(metrics_bt, indent=2))
    logger.info(f"Confusion matrix:\n{cm_bt}")

    # Plots
    out_dir = "artifacts"
    plot_curves(y_test.values, test_proba, f"{best_model_key}_test", out_dir)
    plot_business_curve(df_thr, f"{best_model_key}_test", out_dir)

    # Permutation importance (on validation)
    # Fit a single pipeline with best params on train
    final_pipe = best_grid.best_estimator_
    final_pipe.fit(X_train, y_train)

    # Note: permutation importance requires a callable 'predict' or 'predict_proba'; we use predict_proba for classification
    result = permutation_importance(
        final_pipe, X_val, y_val, n_repeats=5, random_state=CONFIG["random_state"], scoring="average_precision"
    )
    # Get feature names from preprocessor
    prep = final_pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    n = min(len(feature_names), len(result.importances_mean))
    importances = pd.DataFrame({
        "feature": feature_names[n],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    logger.info("Feature names length:", len(feature_names))
    logger.info("Importance length:", len(result.importances_mean))
    
    logger.info("\nTop 15 features by permutation importance (validation PR-AUC drop):")
    logger.info(importances.head(15))

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    importances.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
    pd.DataFrame(results).to_json(os.path.join(out_dir, "model_results.json"), orient="records", indent=2)

    # Optional: serialize model with joblib for deployment
    try:
        import joblib
        joblib.dump(final_pipe, os.path.join(out_dir, f"{best_model_key}_model.joblib"))
        logger.info(f"\nSaved model to {os.path.join(out_dir, f'{best_model_key}_model.joblib')}")
    except ImportError:
        logger.info("\njoblib not installed; skipping model serialization.")

if __name__ == "__main__":
    main()
