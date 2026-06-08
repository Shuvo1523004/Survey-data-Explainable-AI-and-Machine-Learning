#!/usr/bin/env python3
"""
Comprehensive experiment runner for WYDOT Exit Survey revision.
Runs all experiments needed to address reviewer comments:

1. Forward-coded vs Reverse-coded (F1-F32 only, no leakage)
2. With SMOTE vs Without SMOTE
3. Sensitivity analysis excluding retirees
4. Harman's single-factor test for common method bias
5. EFA: 2-factor vs 3-factor structure
6. Spearman rank correlation between RII ranking and SHAP ranking (corrected)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

warnings.filterwarnings("ignore")
np.random.seed(42)
RANDOM_STATE = 42

# ── Paths ──
BASE = Path("/Users/uw-user/Desktop/datamining_pyspark")
FWD_PATH = BASE / "clean data 307.xlsx"
REV_PATH = BASE / "clean data 307_reverse.xlsx"
OUT_DIR = BASE / "experiment_outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Feature columns ──
ORG_ITEMS = [f"F{i}" for i in range(1, 11)]
PERCEPT_ITEMS = [f"F{i}" for i in range(11, 24)]
SUPERV_ITEMS = [f"F{i}" for i in range(24, 33)]
ALL_F = ORG_ITEMS + PERCEPT_ITEMS + SUPERV_ITEMS  # F1-F32
TARGET = "Turnover_Type_Final"
REASON_COL = "Whatareyourreasonsforleavingwydot"


def load_data(path, exclude_retirees=False):
    """Load dataset, encode target, optionally drop retirees."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    if exclude_retirees and REASON_COL in df.columns:
        n_before = len(df)
        df = df[df[REASON_COL] != "Retirement"].copy()
        n_after = len(df)
        print(f"  Excluded {n_before - n_after} retirees → {n_after} remaining")

    # Encode target
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"Voluntary": 1, "Involuntary": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Extract F1-F32 only (no multi-select)
    X = df[ALL_F].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mean())
    y = df[TARGET].copy()

    return X, y, df


def run_models(X_train, X_test, y_train, y_test, use_smote=False, label=""):
    """Train LR, RF, XGBoost. Return results dict."""
    cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if use_smote:
        PipeClass = ImbPipeline
        smote_step = [("smote", SMOTE(random_state=RANDOM_STATE))]
    else:
        PipeClass = SKPipeline
        smote_step = []

    models = {
        "Logistic Regression": {
            "steps": [("scaler", StandardScaler())] + smote_step + [
                ("clf", LogisticRegression(solver="saga", max_iter=5000,
                                           random_state=RANDOM_STATE))
            ],
            "grid": {"clf__C": [0.1, 0.5, 1.0, 2.0], "clf__penalty": ["l2"]}
        },
        "Random Forest": {
            "steps": [("scaler", StandardScaler())] + smote_step + [
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
            ],
            "grid": {"clf__n_estimators": [400, 700],
                     "clf__max_depth": [None, 10, 16],
                     "clf__min_samples_leaf": [1, 2, 4]}
        },
        "XGBoost": {
            "steps": [("scaler", StandardScaler())] + smote_step + [
                ("clf", XGBClassifier(random_state=RANDOM_STATE,
                                      eval_metric="logloss",
                                      tree_method="hist",
                                      n_estimators=600))
            ],
            "grid": {"clf__max_depth": [3, 5, 7],
                     "clf__learning_rate": [0.05, 0.1, 0.2],
                     "clf__subsample": [0.8, 1.0],
                     "clf__colsample_bytree": [0.8, 1.0]}
        }
    }

    results = {}
    for name, spec in models.items():
        pipe = PipeClass(steps=spec["steps"])
        gs = GridSearchCV(pipe, spec["grid"], scoring="f1", cv=cvk,
                          n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "best_params": gs.best_params_,
            "cv_best_f1": round(gs.best_score_, 4)
        }
        results[name] = metrics
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        results[name]["confusion_matrix"] = cm.tolist()

        print(f"  {name}: Acc={metrics['accuracy']:.3f}  "
              f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}")

    return results


def experiment_forward_vs_reverse():
    """Experiment 1: Forward vs Reverse coded, F1-F32 only, no SMOTE."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Forward-Coded vs Reverse-Coded (F1-F32 only, no SMOTE)")
    print("="*70)

    all_results = {}

    for coding, path in [("Forward", FWD_PATH), ("Reverse", REV_PATH)]:
        print(f"\n--- {coding} Coding ({path.name}) ---")
        X, y, _ = load_data(path)
        print(f"  N={len(X)}, Features={X.shape[1]}, "
              f"Voluntary={int((y==1).sum())}, Involuntary={int((y==0).sum())}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
        print(f"  Train={len(X_train)}, Test={len(X_test)}")

        results = run_models(X_train, X_test, y_train, y_test,
                             use_smote=False, label=coding)
        all_results[coding] = results

    return all_results


def experiment_smote_comparison():
    """Experiment 2: With SMOTE vs Without SMOTE on reverse-coded data."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SMOTE vs No-SMOTE (Reverse-Coded, F1-F32 only)")
    print("="*70)

    X, y, _ = load_data(REV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    all_results = {}

    for smote_label, use_smote in [("No SMOTE", False), ("With SMOTE", True)]:
        print(f"\n--- {smote_label} ---")
        results = run_models(X_train, X_test, y_train, y_test,
                             use_smote=use_smote, label=smote_label)
        all_results[smote_label] = results

    # Also run on forward-coded with SMOTE
    print(f"\n--- Forward-Coded With SMOTE ---")
    X_f, y_f, _ = load_data(FWD_PATH)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_f, y_f, test_size=0.20, random_state=RANDOM_STATE, stratify=y_f)
    results = run_models(Xf_train, Xf_test, yf_train, yf_test,
                         use_smote=True, label="Forward+SMOTE")
    all_results["Forward + SMOTE"] = results

    print(f"\n--- Forward-Coded No SMOTE ---")
    results = run_models(Xf_train, Xf_test, yf_train, yf_test,
                         use_smote=False, label="Forward+NoSMOTE")
    all_results["Forward + No SMOTE"] = results

    return all_results


def experiment_exclude_retirees():
    """Experiment 3: Sensitivity analysis excluding retirees."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Sensitivity Analysis — Excluding Retirees")
    print("="*70)

    all_results = {}

    for coding, path in [("Forward (no retirees)", FWD_PATH),
                          ("Reverse (no retirees)", REV_PATH)]:
        print(f"\n--- {coding} ---")
        X, y, df = load_data(path, exclude_retirees=True)
        print(f"  N={len(X)}, Features={X.shape[1]}, "
              f"Voluntary={int((y==1).sum())}, Involuntary={int((y==0).sum())}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
        print(f"  Train={len(X_train)}, Test={len(X_test)}")

        # Without SMOTE
        print(f"\n  [No SMOTE]")
        results_no = run_models(X_train, X_test, y_train, y_test,
                                use_smote=False, label=coding)
        all_results[f"{coding} - No SMOTE"] = results_no

        # With SMOTE
        print(f"\n  [With SMOTE]")
        results_sm = run_models(X_train, X_test, y_train, y_test,
                                use_smote=True, label=coding+" SMOTE")
        all_results[f"{coding} - SMOTE"] = results_sm

    return all_results


def experiment_harmans_test():
    """Experiment 4: Harman's single-factor test for common method bias."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Harman's Single-Factor Test (Common Method Bias)")
    print("="*70)

    X, _, _ = load_data(REV_PATH)

    # Bartlett's test of sphericity
    chi2, p_bartlett = calculate_bartlett_sphericity(X)
    print(f"\n  Bartlett's test: chi2={chi2:.2f}, p={p_bartlett:.6f}")

    # KMO test
    kmo_all, kmo_model = calculate_kmo(X)
    print(f"  KMO measure: {kmo_model:.4f}")

    # Unrotated single-factor EFA
    fa_single = FactorAnalyzer(n_factors=1, rotation=None, method="principal")
    fa_single.fit(X)
    ev, _ = fa_single.get_eigenvalues()
    total_var = ev.sum()
    first_factor_var = ev[0]
    pct_single = (first_factor_var / total_var) * 100

    print(f"\n  Eigenvalue of first factor: {first_factor_var:.4f}")
    print(f"  Total variance: {total_var:.4f}")
    print(f"  % variance explained by single factor: {pct_single:.2f}%")
    print(f"  CMB present (>50%)? {'YES' if pct_single > 50 else 'NO'}")

    # Also get variance explained by each factor
    print(f"\n  Top 10 eigenvalues:")
    for i, e in enumerate(ev[:10]):
        pct = (e / total_var) * 100
        print(f"    Factor {i+1}: eigenvalue={e:.4f}  ({pct:.1f}%)")

    return {
        "bartlett_chi2": round(float(chi2), 2),
        "bartlett_p": float(p_bartlett),
        "kmo": round(float(kmo_model), 4),
        "single_factor_pct_variance": round(float(pct_single), 2),
        "cmb_present": pct_single > 50,
        "eigenvalues": [round(float(e), 4) for e in ev[:10]]
    }


def experiment_efa():
    """Experiment 5: EFA to determine factor structure."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Exploratory Factor Analysis (Factor Structure)")
    print("="*70)

    X, _, _ = load_data(REV_PATH)

    # Scree test — get eigenvalues
    fa_temp = FactorAnalyzer(n_factors=X.shape[1], rotation=None, method="principal")
    fa_temp.fit(X)
    ev, _ = fa_temp.get_eigenvalues()

    print("\n  Eigenvalues (Kaiser criterion: keep factors with eigenvalue > 1):")
    n_kaiser = 0
    for i, e in enumerate(ev):
        marker = " <-- retain" if e > 1 else ""
        if e > 1:
            n_kaiser = i + 1
        print(f"    Factor {i+1}: {e:.4f}{marker}")
        if e < 0.5 and i > 5:
            break
    print(f"\n  Kaiser criterion suggests: {n_kaiser} factors")

    # Parallel analysis (Monte Carlo)
    n_iter = 100
    n_obs, n_vars = X.shape
    random_ev = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        rand_data = np.random.normal(size=(n_obs, n_vars))
        rand_corr = np.corrcoef(rand_data, rowvar=False)
        random_ev[i] = np.sort(np.linalg.eigvalsh(rand_corr))[::-1]
    pa_threshold = np.percentile(random_ev, 95, axis=0)

    n_parallel = 0
    print(f"\n  Parallel analysis (95th percentile):")
    for i in range(min(10, len(ev))):
        retain = ev[i] > pa_threshold[i]
        if retain:
            n_parallel = i + 1
        print(f"    Factor {i+1}: actual={ev[i]:.4f}  "
              f"threshold={pa_threshold[i]:.4f}  "
              f"{'RETAIN' if retain else 'drop'}")
    print(f"\n  Parallel analysis suggests: {n_parallel} factors")

    # Fit 2-factor and 3-factor solutions
    results = {}
    for n_fac in [2, 3]:
        print(f"\n  --- {n_fac}-Factor Solution (Varimax rotation) ---")
        fa = FactorAnalyzer(n_factors=n_fac, rotation="varimax", method="principal")
        fa.fit(X)
        loadings = pd.DataFrame(
            fa.loadings_,
            index=ALL_F,
            columns=[f"Factor_{i+1}" for i in range(n_fac)]
        )
        var_explained = fa.get_factor_variance()

        print(f"  Variance explained:")
        for i in range(n_fac):
            print(f"    Factor {i+1}: {var_explained[1][i]*100:.1f}% "
                  f"(cumulative: {var_explained[2][i]*100:.1f}%)")

        print(f"\n  Factor loadings (>0.4):")
        for f_col in loadings.columns:
            high = loadings[loadings[f_col].abs() > 0.4].index.tolist()
            print(f"    {f_col}: {high}")

        results[f"{n_fac}_factor"] = {
            "loadings": loadings.round(4).to_dict(),
            "variance_explained_pct": [round(v * 100, 2) for v in var_explained[1]],
            "cumulative_variance_pct": [round(v * 100, 2) for v in var_explained[2]],
        }

    results["kaiser_n_factors"] = n_kaiser
    results["parallel_n_factors"] = n_parallel
    return results


def run_shap_analysis():
    """Run SHAP on best XGBoost model for forward-coded data to get rankings."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: SHAP Feature Rankings (Forward-Coded for Comparison)")
    print("="*70)

    try:
        import shap
    except ImportError:
        print("  SHAP not installed, skipping.")
        return None

    X, y, _ = load_data(FWD_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # Train XGBoost
    pipe = SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                              tree_method="hist", n_estimators=600,
                              max_depth=5, learning_rate=0.1))
    ])
    pipe.fit(X_train, y_train)

    # SHAP
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    Xte_scaled = scaler.transform(X_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(Xte_scaled)

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=ALL_F
    ).sort_values(ascending=False)

    print("\n  SHAP Feature Rankings (Forward-Coded XGBoost):")
    for i, (feat, val) in enumerate(mean_abs_shap.items()):
        print(f"    {i+1}. {feat}: {val:.4f}")

    mean_abs_shap.to_csv(OUT_DIR / "shap_rankings_forward.csv", header=["mean_abs_shap"])

    return mean_abs_shap


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_outputs = {}

    # Experiment 1: Forward vs Reverse
    r1 = experiment_forward_vs_reverse()
    all_outputs["exp1_forward_vs_reverse"] = r1

    # Experiment 2: SMOTE comparison
    r2 = experiment_smote_comparison()
    all_outputs["exp2_smote_comparison"] = r2

    # Experiment 3: Exclude retirees
    r3 = experiment_exclude_retirees()
    all_outputs["exp3_exclude_retirees"] = r3

    # Experiment 4: Harman's test
    r4 = experiment_harmans_test()
    all_outputs["exp4_harmans_test"] = r4

    # Experiment 5: EFA
    r5 = experiment_efa()
    all_outputs["exp5_efa"] = r5

    # Experiment 6: SHAP rankings
    r6 = run_shap_analysis()
    if r6 is not None:
        all_outputs["exp6_shap_rankings"] = r6.to_dict()

    # Save all results as JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return super().default(obj)

    with open(OUT_DIR / "all_experiment_results.json", "w") as f:
        json.dump(all_outputs, f, indent=2, cls=NumpyEncoder)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUT_DIR}")
    print("="*70)
