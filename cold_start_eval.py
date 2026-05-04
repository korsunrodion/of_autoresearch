"""
cold_start_eval.py — evaluate cold-start VH detection using training data.

Simulates the seed_verify_db.py cold-start scenario entirely in-memory:
- Takes all models from main DB
- Picks N cold-start models (VH-containing, reset to 'no risk')
- Runs full feature computation + prediction
- Reports precision/recall for each risk level on cold-start models

Usage:
  python cold_start_eval.py [--n-cold 5] [--seed 42]
"""
import sys, os, argparse, warnings, json, pickle, random
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from classifier_v2_cpu_predict import (
    compute_per_user, make_cold_X, FEATURES, COLD_OVERRIDE
)
from base import fetch_df, clean
from db import db as _db

MODEL_DIR = 'models_cpu'

def predict_with_models(per_user, feats, models, thresholds):
    X_base = per_user[feats].values

    e_proba   = models['extreme'].predict_proba(X_base)[:, 1]
    h_proba   = models['high'].predict_proba(X_base)[:, 1]
    l_proba   = models['low'].predict_proba(X_base)[:, 1]
    ord_proba = models['ordinal'].predict_proba(X_base)[:, 1]

    X_aug    = np.column_stack([X_base, e_proba, h_proba, l_proba, ord_proba])
    vh_proba = models['vh'].predict_proba(X_aug)[:, 1]

    X_base_cold  = make_cold_X(X_base, feats)
    cold_vh_proba = (models['cold_vh'].predict_proba(X_base_cold)[:, 1]
                     if 'cold_vh' in models else np.zeros(len(per_user)))
    warm_rescue_proba = (models['warm_vh_rescue'].predict_proba(X_aug)[:, 1]
                         if 'warm_vh_rescue' in models else np.zeros(len(per_user)))

    t_e           = thresholds.get('extreme', 0.5)
    t_vh          = thresholds.get('vh', 0.5)
    t_h           = thresholds.get('high', 0.5)
    t_l           = thresholds.get('low', 0.5)
    COLD_T_E      = thresholds.get('extreme_cold', 0.50)
    t_vh_cold     = thresholds.get('vh_cold',   t_vh)
    t_h_cold      = thresholds.get('high_cold', t_h)
    t_l_cold      = thresholds.get('low_cold',  t_l)
    t_cold_vh_e   = thresholds.get('vh_extreme_cold', 0.5)
    t_warm_rescue = thresholds.get('warm_vh_rescue', 0.47)

    cold_mask = (per_user['model_any_risk_frac'].fillna(0) == 0).values

    predicted = []
    for i in range(len(per_user)):
        t_e_i = COLD_T_E if cold_mask[i] else t_e
        if e_proba[i] >= t_e_i:
            if cold_mask[i]:
                rescue = cold_vh_proba[i] >= t_cold_vh_e
            else:
                rescue = warm_rescue_proba[i] >= t_warm_rescue
            if rescue:
                predicted.append('Very High')
            else:
                predicted.append('Extreme')
        elif vh_proba[i] >= t_vh:
            predicted.append('Very High')
        elif cold_mask[i] and vh_proba[i] >= t_vh_cold:
            predicted.append('Very High')
        elif h_proba[i] >= t_h:
            predicted.append('High')
        elif cold_mask[i] and h_proba[i] >= t_h_cold:
            predicted.append('High')
        elif l_proba[i] >= t_l:
            predicted.append('Low')
        elif cold_mask[i] and l_proba[i] >= t_l_cold:
            predicted.append('Low')
        else:
            predicted.append('No risk')

    return predicted, vh_proba, e_proba, cold_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-cold', type=int, default=5,
                        help='Number of cold-start models to test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading training data...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    try:
        _db.close()
    except Exception:
        pass
    print(f"  {len(df)} rows, {df['tracking_model_name'].nunique()} models", flush=True)

    # ── Pick cold-start candidate models (VH-containing, ≥20 users) ──────────
    model_vh = df.groupby('tracking_model_name').apply(
        lambda g: ((g['risk_level'] == 'Very High').any() and len(g) >= 20)
    )
    candidates = model_vh[model_vh].index.tolist()
    print(f"  VH-containing models with ≥20 users: {len(candidates)}", flush=True)

    n_cold = min(args.n_cold, len(candidates))
    cold_models = random.sample(candidates, n_cold)
    print(f"  Selected {n_cold} cold models: {cold_models}", flush=True)

    # ── Build test dataframe ──────────────────────────────────────────────────
    # Warm portion: all rows NOT from cold models (with real labels)
    # Cold portion: cold model rows with labels reset to 'No risk'
    df_warm = df[~df['tracking_model_name'].isin(cold_models)].copy()
    df_cold_true = df[df['tracking_model_name'].isin(cold_models)].copy()
    df_cold_sim  = df_cold_true.copy()
    df_cold_sim['risk_level'] = 'No risk'   # simulate cold start

    df_test = pd.concat([df_warm, df_cold_sim], ignore_index=True)
    print(f"\nTest set: {len(df_warm)} warm + {len(df_cold_sim)} cold = {len(df_test)} total",
          flush=True)

    # ── Feature computation ───────────────────────────────────────────────────
    print("\nComputing features...", flush=True)
    per_user = compute_per_user(df_test)
    feats = [f for f in FEATURES if f in per_user.columns]

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...", flush=True)
    models = {}
    for name in ['extreme', 'high', 'low', 'ordinal', 'vh', 'cold_vh', 'warm_vh_rescue']:
        path = os.path.join(MODEL_DIR, f'{name}_model.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'thresholds.json')) as f:
        thresholds = json.load(f)
    print(f"  Thresholds: {thresholds}", flush=True)

    # ── Run prediction ────────────────────────────────────────────────────────
    print("\nRunning prediction...", flush=True)
    predicted, vh_proba, e_proba, cold_mask = predict_with_models(
        per_user, feats, models, thresholds)

    # ── Evaluate on cold-start models only ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COLD-START MODEL EVALUATION  (n={n_cold} models)")
    print(f"{'='*60}")

    # Align predictions with true labels for cold model users
    pred_df = pd.DataFrame({
        'user_name': per_user['user_name'].values,
        'predicted': predicted,
        'vh_proba':  np.round(vh_proba, 4),
        'e_proba':   np.round(e_proba, 4),
        'is_cold':   cold_mask,
    })

    # True labels from df_cold_true
    true_labels = df_cold_true.groupby('user_name')['risk_level'].first().reset_index()
    true_labels.columns = ['user_name', 'true_risk']

    cold_results = pred_df[pred_df['is_cold']].merge(true_labels, on='user_name', how='inner')
    print(f"Cold-start users evaluated: {len(cold_results)}", flush=True)
    print(f"\nTrue distribution (cold models):")
    print(cold_results['true_risk'].value_counts().to_string())
    print(f"\nPredicted distribution (cold models):")
    print(cold_results['predicted'].value_counts().to_string())

    # Per-level metrics
    print(f"\nPer-level metrics on cold-start users:")
    for level in ['Very High', 'Extreme', 'High', 'Low']:
        true_bin = (cold_results['true_risk'] == level).astype(int).values
        pred_bin = (cold_results['predicted'] == level).astype(int).values
        n_true = true_bin.sum()
        if n_true == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            true_bin, pred_bin, pos_label=1, average='binary', zero_division=0)
        print(f"  {level:<12}: P={p:.2%} R={r:.2%} F1={f:.2%}  (n_true={n_true})")

    # VH detail: show false negatives
    vh_true = cold_results[cold_results['true_risk'] == 'Very High']
    if not vh_true.empty:
        print(f"\nVH details (n={len(vh_true)}):")
        vh_fn = vh_true[vh_true['predicted'] != 'Very High']
        vh_tp = vh_true[vh_true['predicted'] == 'Very High']
        print(f"  TP={len(vh_tp)}  FN={len(vh_fn)}")
        if not vh_fn.empty:
            print("  False negatives (predicted as):")
            for _, row in vh_fn.iterrows():
                print(f"    {row['user_name']:<30} → {row['predicted']:<12}"
                      f"  vh_proba={row['vh_proba']:.4f}  e_proba={row['e_proba']:.4f}")

    # Also show warm-model VH accuracy for comparison
    warm_results = pred_df[~pred_df['is_cold']].merge(
        df_warm.groupby('user_name')['risk_level'].first().reset_index().rename(
            columns={'risk_level': 'true_risk'}),
        on='user_name', how='inner')
    warm_vh_true = (warm_results['true_risk'] == 'Very High').astype(int).values
    warm_vh_pred = (warm_results['predicted'] == 'Very High').astype(int).values
    if warm_vh_true.sum() > 0:
        p, r, f, _ = precision_recall_fscore_support(
            warm_vh_true, warm_vh_pred, pos_label=1, average='binary', zero_division=0)
        print(f"\nWarm VH (reference): P={p:.2%} R={r:.2%} F1={f:.2%}"
              f"  (n={warm_vh_true.sum()})")


if __name__ == '__main__':
    main()
