#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS 402 – Lab: Interpretable ML with InterpretML
Author: (Your Name)
Instructor/TA: Shikha Soneji
Course: Trends in Data Science
---------------------------------------------------------------------
This lab mirrors the previous SHAP-based lab but uses Microsoft's
InterpretML library. You will reuse the SAME dataset and train/test split
format from the SHAP lab. Fill in the TODOs where indicated.

Helpful docs:
- InterpretML: https://interpret.ml/
- Guide: https://github.com/interpretml/interpret
---------------------------------------------------------------------

LEARNING GOALS
1) Train a glassbox model (Explainable Boosting Machine - EBM) and interpret
   global & local behavior (feature importances, per-feature contributions).
2) Compare a blackbox model (e.g., RandomForest/XGBoost) to EBM performance.
3) Use InterpretML dashboards for interactive exploration.
4) Produce PDP/ICE-style insights and "what-if" reasoning using EBM scores.
5) Communicate trade-offs: accuracy vs interpretability.

WHAT TO SUBMIT
- A short report (PDF or Markdown) answering the questions in TASKS.
- Plots/screenshots exported from the dashboard (png/jpg) as asked.
- Your final code file (.py or .ipynb).

ENVIRONMENT
- Python >= 3.9
- pip install interpret scikit-learn pandas numpy matplotlib
  (If you want XGBoost, also: pip install xgboost)
"""

# ==============================
# 0) Imports & Setup
# ==============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# InterpretML - glassbox and blackbox
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression
from interpret import show
from interpret.blackbox import LimeTabular

# Sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost (uncomment if available)
# from xgboost import XGBClassifier


# ==============================
# 1) Load the SAME dataset as the SHAP lab
# ==============================

# TODO: Set these to match your SHAP lab exactly
DATA_PATH = Path("data")  # folder or absolute path where your CSV lives
CSV_FILE  = "your_dataset.csv"   # e.g., "telco_churn.csv" or "bank_marketing.csv"
TARGET_COL = "target"            # e.g., "churn", "y", etc.

# If your SHAP lab used a pre-split (X_train.npy etc.), you can load those instead.
# Below assumes a single CSV; adapt as needed to mirror the SHAP lab.

def load_data() -> pd.DataFrame:
    csv_path = DATA_PATH / CSV_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}.\n"
                                f"Please set DATA_PATH/CSV_FILE to match your SHAP lab.")
    df = pd.read_csv(csv_path)
    return df


# ==============================
# 2) Train/Validation Split + Preprocess
# ==============================

def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"TARGET_COL='{target_col}' not in columns. Available: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Detect numeric vs categorical columns (simple heuristic)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False))  # with_mean=False supports sparse matrices
    ])

    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor


# ==============================
# 3) Models: EBM (glassbox) and RandomForest (blackbox)
# ==============================

def make_ebm_classifier() -> ExplainableBoostingClassifier:
    # Feel free to tune (e.g., interactions=10, outer_bags=8, learning_rate=0.05)
    return ExplainableBoostingClassifier(random_state=42)

def make_blackbox_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    # Or try XGBoost:
    # return XGBClassifier(
    #     n_estimators=400,
    #     max_depth=5,
    #     learning_rate=0.05,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     reg_lambda=1.0,
    #     random_state=42,
    #     tree_method="hist",
    # )


# ==============================
# 4) Fit Pipelines
# ==============================

def fit_pipelines(X_train, y_train, X_val, y_val, preprocessor):
    # Glassbox (EBM) pipeline
    ebm = make_ebm_classifier()
    pipe_ebm = Pipeline(steps=[("pre", preprocessor), ("clf", ebm)])
    pipe_ebm.fit(X_train, y_train)

    # Blackbox (RF) pipeline
    rf = make_blackbox_classifier()
    pipe_rf = Pipeline(steps=[("pre", preprocessor), ("clf", rf)])
    pipe_rf.fit(X_train, y_train)

    # Optional: Logistic Regression (glassbox linear) for comparison
    logreg = LogisticRegression(random_state=42)
    pipe_lr = Pipeline(steps=[("pre", preprocessor), ("clf", logreg)])
    pipe_lr.fit(X_train, y_train)

    return pipe_ebm, pipe_rf, pipe_lr


# ==============================
# 5) Evaluate
# ==============================

def evaluate_model(pipe, X_val, y_val, model_name="Model"):
    y_pred = pipe.predict(X_val)
    try:
        y_prob = pipe.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, y_prob)
    except Exception:
        y_prob = None
        roc = np.nan

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="binary" if len(np.unique(y_val)) == 2 else "macro")
    print(f"\n[{model_name}]")
    print("Accuracy:", round(acc, 4), "| F1:", round(f1, 4), "| ROC-AUC:", (round(roc, 4) if not np.isnan(roc) else "NA"))
    print(classification_report(y_val, y_pred))

    return {"acc": acc, "f1": f1, "roc_auc": roc}


# ==============================
# 6) Interpretability – Global & Local
# ==============================

def ebm_global_local_explanations(pipe_ebm, X_train, X_val, feature_names):
    """Create global and local explanations for the EBM model."""
    ebm_model = pipe_ebm.named_steps["clf"]
    pre = pipe_ebm.named_steps["pre"]

    # Interpret's show() returns a dashboard you can open in notebook or save as HTML.
    # Global explanation
    ebm_global = ebm_model.explain_global(name="EBM Global Explanation")
    # Local explanation (first 5 validation rows for demo)
    ebm_local = ebm_model.explain_local(pre.transform(X_val[:5]), y=None, name="EBM Local Explanation")

    # Show in the default browser / notebook
    try:
        show(ebm_global)
        show(ebm_local)
    except Exception as e:
        print("Dashboard display failed (this is OK on headless machines):", e)

    # Save figures for report (feature importance bar plot)
    try:
        # InterpretML doesn't expose a direct matplotlib plot for global importance,
        # but we can access scores and make our own bar chart.
        scores = ebm_global.data()["scores"]
        names = feature_names  # Already encoded names if preprocessor is sparse; see note below.
        # NOTE: If you want exact post-OHE feature names, you can get them via:
        # names = pipe_ebm.named_steps["pre"].get_feature_names_out()
        # and then aggregate by original feature if desired.
        idx = np.argsort(scores)[::-1][:15]
        plt.figure()
        plt.bar(range(len(idx)), np.array(scores)[idx])
        plt.xticks(range(len(idx)), np.array(names)[idx], rotation=60, ha="right")
        plt.title("EBM Top Feature Scores (Global)")
        plt.tight_layout()
        plt.savefig("ebm_global_feature_scores.png", dpi=160)
        print("Saved: ebm_global_feature_scores.png")
    except Exception as e:
        print("Could not save EBM global bar plot:", e)

    return ebm_global, ebm_local


def rf_lime_local(pipe_rf, X_train, X_val, feature_names):
    """Local explanations for the blackbox RF using LIME (tabular)."""
    pre = pipe_rf.named_steps["pre"]
    rf = pipe_rf.named_steps["clf"]

    # Build a predict_proba wrapper on the raw (untransformed) features,
    # letting the pipeline inside handle the transform:
    def predict_proba_raw(X_raw):
        return pipe_rf.predict_proba(X_raw)

    # LIME expects raw X and a list of feature names
    lime = LimeTabular(predict_fn=predict_proba_raw, data=X_train, feature_names=feature_names, class_names=["0", "1"])
    explanations = []
    for i in range(min(5, len(X_val))):
        exp = lime.explain_instance(X_val.iloc[i], num_features=10, top_labels=1)
        explanations.append(exp)

    # Display LIME (prints/text). For a notebook, exp.visualize() could be used.
    for i, exp in enumerate(explanations):
        print(f"\n[LIME] Instance {i}")
        try:
            print(exp.local_exp)  # feature weights per class
        except Exception as e:
            print("LIME print failed:", e)

    return explanations


# ==============================
# 7) PDP/ICE-style reasoning & What-if with EBM
# ==============================

def ebm_partial_dependence_like(ebm_model: ExplainableBoostingClassifier, X_raw: pd.DataFrame, feature: str, num_points: int = 20):
    """
    Plot EBM's learned shape function for a single feature as a PDP-like curve.
    For categorical features: shows per-category score.
    For numeric features: grid over the observed range.
    """
    # InterpretML provides per-feature plots via show(global); here is a simple programmatic approach.

    # Get transformed feature names (post-OHE). We'll try to align with original feature.
    # For teaching purposes, we'll do a simple numeric-only curve by quantiles.
    if feature not in X_raw.columns:
        raise KeyError(f"Feature '{feature}' not found in original columns.")

    x = X_raw[feature]
    if np.issubdtype(x.dtype, np.number):
        grid = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), num_points)
        contribs = []
        base = ebm_model.intercept_[0] if hasattr(ebm_model, "intercept_") else 0.0
        for v in grid:
            # Create a synthetic row using the median/mode of other features, and override 'feature' with v.
            row = X_raw.median(numeric_only=True).to_dict()
            # Fallback for categoricals—use the first category (rough but OK for a didactic lab)
            for c in X_raw.columns:
                if c not in row:
                    try:
                        row[c] = X_raw[c].mode().iloc[0]
                    except Exception:
                        row[c] = X_raw[c].iloc[0]
            row[feature] = v
            df_one = pd.DataFrame([row])
            # Use model to get per-feature contributions via predict_and_contrib
            try:
                pred, contribs_one = ebm_model.predict_and_contrib(df_one)
                # Sum only the contribution of 'feature' by matching by name (simple approach)
                # NOTE: For OHE features, this is approximate. OK for teaching; see bonus task.
                # contribs_one returns a list of dicts per row: [{"scores": [...], "names": [...]}]
                names = contribs_one[0]["names"]
                scores = contribs_one[0]["scores"]
                feat_score = 0.0
                for n, s in zip(names, scores):
                    if n.startswith(feature):
                        feat_score += s
                contribs.append(feat_score + base)
            except Exception:
                # Fallback: just use predicted prob
                p = ebm_model.predict_proba(df_one)[:, 1][0]
                contribs.append(p)

        plt.figure()
        plt.plot(grid, contribs, marker="o")
        plt.title(f"EBM PDP-like curve: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Score (approx) or P(y=1)")
        plt.tight_layout()
        out = f"ebm_pdp_{feature}.png"
        plt.savefig(out, dpi=160)
        print(f"Saved: {out}")
    else:
        # Categorical: bar scores per category
        cats = X_raw[feature].astype("category").cat.categories.tolist()
        vals = []
        base = ebm_model.intercept_[0] if hasattr(ebm_model, "intercept_") else 0.0
        for v in cats:
            row = {}
            for c in X_raw.columns:
                col = X_raw[c]
                if np.issubdtype(col.dtype, np.number):
                    row[c] = np.nanmedian(col)
                else:
                    try:
                        row[c] = col.mode().iloc[0]
                    except Exception:
                        row[c] = col.iloc[0]
            row[feature] = v
            df_one = pd.DataFrame([row])
            try:
                pred, contribs_one = ebm_model.predict_and_contrib(df_one)
                names = contribs_one[0]["names"]
                scores = contribs_one[0]["scores"]
                feat_score = 0.0
                for n, s in zip(names, scores):
                    if n.startswith(feature):
                        feat_score += s
                vals.append(feat_score + base)
            except Exception:
                p = ebm_model.predict_proba(df_one)[:, 1][0]
                vals.append(p)

        plt.figure()
        plt.bar(cats, vals)
        plt.title(f"EBM Category Contributions: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Score (approx) or P(y=1)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = f"ebm_cat_{feature}.png"
        plt.savefig(out, dpi=160)
        print(f"Saved: {out}")


# ==============================
# 8) Main driver
# ==============================

def main():
    print("Loading data...")
    df = load_data()

    print("Splitting X/y...")
    X, y = split_X_y(df, TARGET_COL)

    # Ensure y is binary int if needed
    if y.dtype.kind not in "biu":
        # Try to convert common string labels to binary 0/1 by mapping the most frequent to 0 and the positive label to 1
        if y.nunique() == 2:
            mapping = {lab: i for i, lab in enumerate(sorted(y.unique()))}
            y = y.map(mapping).astype(int)
            print(f"INFO: Converted y to binary ints with mapping: {mapping}")
        else:
            raise ValueError("This template expects a binary target. Please adapt for multiclass if needed.")

    # Match your SHAP lab's split (set test_size and random_state to same values)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Building preprocessor...")
    pre = build_preprocessor(X_train)

    print("Fitting models (EBM, RF, LR)...")
    pipe_ebm, pipe_rf, pipe_lr = fit_pipelines(X_train, y_train, X_val, y_val, pre)

    # Evaluate
    metrics_ebm = evaluate_model(pipe_ebm, X_val, y_val, "EBM (Glassbox)")
    metrics_rf  = evaluate_model(pipe_rf,  X_val, y_val, "RandomForest (Blackbox)")
    metrics_lr  = evaluate_model(pipe_lr,  X_val, y_val, "LogisticRegression (Glassbox Linear)")

    # Feature names for LIME (raw column names)
    feature_names = X_train.columns.tolist()

    print("\nCreating EBM explanations...")
    ebm_global, ebm_local = ebm_global_local_explanations(pipe_ebm, X_train, X_val, feature_names)

    print("\nCreating RF LIME local explanations...")
    lime_exps = rf_lime_local(pipe_rf, X_train, X_val, feature_names)

    print("\nGenerating PDP/ICE-like curves for 1–2 key features...")
    ebm_model = pipe_ebm.named_steps["clf"]
    # TODO: Set FEATURES_OF_INTEREST to 1–2 columns you analyzed in the SHAP lab
    FEATURES_OF_INTEREST = []  # e.g., ["tenure", "MonthlyCharges"]
    for feat in FEATURES_OF_INTEREST:
        try:
            ebm_partial_dependence_like(ebm_model, X_train, feat, num_points=20)
        except Exception as e:
            print(f"Skipping PDP for {feat}: {e}")

    print("\nDone. Review generated images (png) and the InterpretML dashboards (if they opened).")


# ==============================
# 9) TASKS (include in your report)
# ==============================
TASKS = r"""
TASKS – Include concise answers and figures in your submission.

A. Data & Setup (5 pts)
   1. Briefly describe the dataset (rows, columns, target balance).
   2. List preprocessing steps (scaling, OHE, handling missing values if any).

B. Modeling (10 pts)
   1. Report metrics on the validation set for EBM, RandomForest, and Logistic Regression:
      - Accuracy, F1, ROC-AUC
   2. Which model performs best? Briefly justify.

C. Global Interpretability (10 pts)
   1. Show the EBM global explanation top features (screenshot + your own bar chart).
   2. Compare EBM's top features to what you observed in the SHAP lab. What's similar/different?
   3. For 1 numeric feature, show the PDP-like curve (saved png) and describe the shape (monotonic? threshold?).

D. Local Interpretability (10 pts)
   1. Pick two validation instances where the model predicts differently than you expected.
   2. Provide EBM local explanation (screenshot or textual) and one RF/LIME explanation.
   3. Explain what drove each prediction (key features & contributions).

E. What-if Analysis (10 pts)
   1. For one feature, vary it across a plausible range and observe the EBM score/probability change.
   2. Propose an actionable change (if any) that could flip an adverse prediction.

F. Reflection (5 pts)
   1. Discuss trade-offs between accuracy and interpretability in your results.
   2. If RF slightly outperforms EBM, is the interpretability gain worth it? Why/why not?
"""

# ==============================
# 10) Entry point
# ==============================

if __name__ == "__main__":
    print(TASKS)
    main()
