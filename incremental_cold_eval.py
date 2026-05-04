"""
incremental_cold_eval.py — simulate gradual warm-up of cold-start tracking models.

Splits each cold model's subscribers into N temporal batches, then adds one batch
per round and checks how prediction accuracy improves as the model accumulates data.

Usage:
  python incremental_cold_eval.py [--n-cold 15] [--n-rounds 4] [--seed 42]
                                   [--min-rows 50]
"""
import sys, os, argparse, warnings, json, pickle, random
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from classifier_v2_cpu_predict import (
    compute_per_user, make_cold_X, FEATURES, COLD_OVERRIDE
)
from base import fetch_df, clean
from db import db as _db

MODEL_DIR = 'models_cpu'


def load_models_and_thresholds(model_dir):
    models = {}
    for name in ['extreme', 'high', 'low', 'ordinal', 'vh', 'cold_vh', 'warm_vh_rescue']:
        path = os.path.join(model_dir, f'{name}_model.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    with open(os.path.join(model_dir, 'thresholds.json')) as f:
        thresholds = json.load(f)
    with open(os.path.join(model_dir, 'features.json')) as f:
        feats = json.load(f)['base_features']
    return models, thresholds, feats


def predict_df(df, models, thresholds, feats):
    """
    Run full prediction cascade on df.
    df must have is_internal_data column so the feedback-loop fix can fire.
    Returns per_user DataFrame with 'predicted' column appended.
    """
    per_user = compute_per_user(df)

    # Feedback-loop fix: recompute model-context features from warm rows only.
    if 'is_internal_data' in df.columns:
        df_warm = df[df['is_internal_data'] == True]
        warm_risk_fracs = {}
        warm_norisk_rates = {}
        for model_name, mdf in df_warm.groupby('tracking_model_name'):
            n = len(mdf)
            warm_risk_fracs[model_name] = {
                'vh':      (mdf['risk_level'] == 'Very High').sum() / n,
                'extreme': (mdf['risk_level'] == 'Extreme').sum() / n,
                'any':     (~mdf['risk_level'].isin(['No risk'])).sum() / n,
            }
            warm_norisk_rates[model_name] = (mdf['risk_level'] == 'No risk').sum() / n

        user_to_model = df.drop_duplicates('user_name').set_index('user_name')['tracking_model_name']
        pu_models = per_user['user_name'].map(user_to_model)

        per_user['model_vh_frac']      = pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('vh', 0.0))
        per_user['model_extreme_frac'] = pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('extreme', 0.0))
        per_user['model_any_risk_frac']= pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('any', 0.0))
        per_user['model_norisk_rate']  = pu_models.map(lambda m: warm_norisk_rates.get(m, 1.0))

    X_b_full = per_user[feats].values

    e_proba   = models['extreme'].predict_proba(X_b_full)[:, 1]
    h_proba   = models['high'].predict_proba(X_b_full)[:, 1]
    l_proba   = models['low'].predict_proba(X_b_full)[:, 1]
    ord_proba = models['ordinal'].predict_proba(X_b_full)[:, 1]
    X_aug     = np.column_stack([X_b_full, e_proba, h_proba, l_proba, ord_proba])
    vh_proba  = models['vh'].predict_proba(X_aug)[:, 1]

    X_base_cold   = make_cold_X(X_b_full, feats)
    if 'cold_vh' in models:
        cold_vh_obj = models['cold_vh']
        if isinstance(cold_vh_obj, list):
            cold_vh_proba = np.mean([clf.predict_proba(X_base_cold)[:, 1]
                                     for clf in cold_vh_obj], axis=0)
        else:
            cold_vh_proba = cold_vh_obj.predict_proba(X_base_cold)[:, 1]
    else:
        cold_vh_proba = np.zeros(len(per_user))
    warm_rescue_proba = (models['warm_vh_rescue'].predict_proba(X_aug)[:, 1]
                         if 'warm_vh_rescue' in models else np.zeros(len(per_user)))

    t_e  = thresholds.get('extreme', 0.5)
    t_vh = thresholds.get('vh', 0.5)
    t_h  = thresholds.get('high', 0.5)
    t_l  = thresholds.get('low', 0.5)

    COLD_T_E  = thresholds.get('extreme_cold', 0.50)
    t_vh_cold = thresholds.get('vh_cold',   t_vh)
    t_h_cold  = thresholds.get('high_cold', t_h)
    t_l_cold  = thresholds.get('low_cold',  t_l)

    cold_mask = (per_user['model_any_risk_frac'].fillna(0) == 0).values

    t_cold_vh_e    = thresholds.get('vh_extreme_cold', 0.5)
    t_warm_rescue  = thresholds.get('warm_vh_rescue', 0.47)
    t_cold_vh_soft = thresholds.get('cold_vh_soft_gate', 0.01)
    t_cold_ord     = thresholds.get('cold_ord_gate', 0.10)

    predicted = []
    for i in range(len(per_user)):
        if cold_mask[i]:
            if e_proba[i] >= COLD_T_E:
                if cold_vh_proba[i] >= t_cold_vh_e:
                    predicted.append('Very High')
                else:
                    predicted.append('Extreme')
            elif cold_vh_proba[i] >= t_cold_vh_e and (
                    e_proba[i] >= t_cold_vh_soft or ord_proba[i] >= t_cold_ord):
                predicted.append('Very High')
            elif h_proba[i] >= t_h_cold:
                predicted.append('High')
            elif l_proba[i] >= t_l_cold:
                predicted.append('Low')
            else:
                predicted.append('No risk')
        else:
            if e_proba[i] >= t_e:
                rescue = warm_rescue_proba[i] >= t_warm_rescue
                predicted.append('Very High' if rescue else 'Extreme')
            elif vh_proba[i] >= t_vh:
                predicted.append('Very High')
            elif h_proba[i] >= t_h:
                predicted.append('High')
            elif l_proba[i] >= t_l:
                predicted.append('Low')
            else:
                predicted.append('No risk')

    per_user['predicted']  = predicted
    per_user['vh_proba']   = vh_proba
    per_user['e_proba']    = e_proba
    per_user['is_cold']    = cold_mask
    return per_user


def metrics_for(true_series, pred_series, level):
    t = (true_series == level).astype(int).values
    p = (pred_series == level).astype(int).values
    n_true = int(t.sum())
    if n_true == 0:
        return None
    pr, rc, f1, _ = precision_recall_fscore_support(
        t, p, pos_label=1, average='binary', zero_division=0)
    return pr, rc, f1, n_true


def print_round_metrics(round_label, cold_results):
    levels = ['Very High', 'Extreme', 'High', 'Low']
    print(f"\n  {'Level':<12} {'P':>7} {'R':>7} {'F1':>7}  n_true")
    for lvl in levels:
        m = metrics_for(cold_results['true_risk'], cold_results['predicted'], lvl)
        if m:
            pr, rc, f1, n = m
            print(f"  {lvl:<12} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}")
    # distribution
    pred_dist = cold_results['predicted'].value_counts()
    true_dist = cold_results['true_risk'].value_counts()
    print(f"  True:  {dict(true_dist)}")
    print(f"  Pred:  {dict(pred_dist)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-cold',   type=int, default=15,
                        help='Number of cold-start models')
    parser.add_argument('--n-rounds', type=int, default=4,
                        help='Number of incremental batches per model')
    parser.add_argument('--min-rows',     type=int, default=50,
                        help='Minimum subscribers per cold model')
    parser.add_argument('--target-level', type=str, default=None,
                        choices=['Very High', 'Extreme', 'High', 'Low'],
                        help='Bias model selection toward models containing this risk level')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    target = args.target_level

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data...", flush=True)
    df_all = fetch_df(selected=False)
    df_all = clean(df_all)
    try:
        _db.close()
    except Exception:
        pass
    print(f"  {len(df_all)} rows, {df_all['tracking_model_name'].nunique()} models",
          flush=True)

    # ── Select cold models ────────────────────────────────────────────────────
    model_stats = df_all.groupby('tracking_model_name').agg(
        n_rows=('user_name', 'count'),
        has_target=('risk_level', lambda s: (s == target).any() if target else True),
        n_target=('risk_level',  lambda s: (s == target).sum() if target else 0),
    )

    # Primary candidates: meet min_rows and contain the target level
    candidates = model_stats[
        (model_stats['n_rows'] >= args.min_rows) & model_stats['has_target']
    ].sort_values('n_target', ascending=False).index.tolist()

    if len(candidates) < args.n_cold:
        # Relax: drop target-level requirement, keep min_rows
        candidates = model_stats[
            model_stats['n_rows'] >= args.min_rows
        ].sort_values('n_rows', ascending=False).index.tolist()

    n_cold = min(args.n_cold, len(candidates))
    cold_models = random.sample(candidates[:max(n_cold * 3, 30)], n_cold)
    cold_set = set(cold_models)

    label = f"target={target}" if target else "no filter"
    print(f"\nSelected {n_cold} cold models ({label}, >= {args.min_rows} rows):",
          flush=True)
    for cm in cold_models:
        row = model_stats.loc[cm]
        dist = df_all[df_all['tracking_model_name'] == cm]['risk_level'].value_counts()
        dist_str = '  '.join(f"{lvl}={cnt}" for lvl, cnt in dist.items() if cnt > 0)
        print(f"  {cm[:55]:<55} {int(row['n_rows']):>4} rows  {dist_str}")

    # ── Split warm / cold data ─────────────────────────────────────────────────
    df_warm = df_all[~df_all['tracking_model_name'].isin(cold_set)].copy()
    df_warm['is_internal_data'] = True

    # True labels for cold models (kept separate for evaluation only)
    df_cold_true = df_all[df_all['tracking_model_name'].isin(cold_set)].copy()

    # Sort each cold model's rows by subscription time for temporal batching
    df_cold_true = df_cold_true.sort_values(
        ['tracking_model_name', 'subscribed_at']).reset_index(drop=True)

    # Assign batch indices per model
    def assign_batches(g, n_rounds):
        idx = np.arange(len(g))
        return pd.Series(
            (idx * n_rounds // len(g)).astype(int), index=g.index)

    df_cold_true['_batch'] = df_cold_true.groupby('tracking_model_name',
                                                    group_keys=False).apply(
        lambda g: assign_batches(g, args.n_rounds))

    print(f"\nBatch sizes per model (n_rounds={args.n_rounds}):", flush=True)
    for cm in cold_models:
        bc = df_cold_true[df_cold_true['tracking_model_name'] == cm]['_batch'].value_counts().sort_index()
        print(f"  {cm[:50]:<50} {dict(bc)}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...", flush=True)
    models, thresholds, feats = load_models_and_thresholds(MODEL_DIR)
    print(f"  Models: {list(models.keys())}", flush=True)
    print(f"  Thresholds: {thresholds}", flush=True)

    # True label lookup for cold users (first-occurrence label per user_name)
    true_labels = (df_cold_true.drop_duplicates('user_name')
                   .set_index('user_name')['risk_level']
                   .rename('true_risk'))

    # Summary table across rounds
    summary = []  # list of (round, level, P, R, F1, n_true, n_cold_users)

    # ── Incremental rounds ────────────────────────────────────────────────────
    for rnd in range(args.n_rounds):
        # Batches 0..rnd of cold models, reset to 'No risk', is_internal_data=False
        df_cold_batches = df_cold_true[df_cold_true['_batch'] <= rnd].copy()
        df_cold_batches['risk_level']      = 'No risk'
        df_cold_batches['risk_score']      = 1
        df_cold_batches['is_internal_data'] = False
        df_cold_batches = df_cold_batches.drop(columns=['_batch'])

        df_sim = pd.concat([df_warm, df_cold_batches], ignore_index=True)

        n_cold_users = df_cold_batches['user_name'].nunique()
        n_cold_rows  = len(df_cold_batches)
        print(f"\n{'='*65}", flush=True)
        print(f"Round {rnd+1}/{args.n_rounds}  "
              f"({n_cold_rows} cold rows, {n_cold_users} cold users)", flush=True)
        print(f"  Total rows: {len(df_sim)} "
              f"(warm={len(df_warm)}, cold={n_cold_rows})", flush=True)

        per_user = predict_df(df_sim, models, thresholds, feats)

        # Evaluate only on cold-model users
        cold_mask_series = per_user['is_cold']
        cold_pu = per_user[cold_mask_series].copy()
        cold_pu = cold_pu.merge(
            true_labels.reset_index(), on='user_name', how='inner')

        print(f"  Cold users in prediction: {len(cold_pu)}", flush=True)
        if cold_pu.empty:
            print("  (no cold users found — check model selection)", flush=True)
            continue

        print_round_metrics(f"Round {rnd+1}", cold_pu)

        for lvl in ['Very High', 'Extreme', 'High', 'Low']:
            m = metrics_for(cold_pu['true_risk'], cold_pu['predicted'], lvl)
            if m:
                pr, rc, f1, n = m
                summary.append((rnd + 1, lvl, pr, rc, f1, n, len(cold_pu)))

    # ── Summary table ─────────────────────────────────────────────────────────
    focus = target if target else 'Very High'
    print(f"\n{'='*65}", flush=True)
    print(f"SUMMARY — {focus} accuracy across rounds", flush=True)
    print(f"{'Round':<7} {'P':>7} {'R':>7} {'F1':>7}  n_true", flush=True)
    for row in summary:
        rnd, lvl, pr, rc, f1, n, _ = row
        if lvl == focus:
            print(f"  {rnd:<5} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}", flush=True)

    print(f"\nSUMMARY — All levels across rounds", flush=True)
    print(f"{'Round':<7} {'Level':<12} {'P':>7} {'R':>7} {'F1':>7}  n_true", flush=True)
    for row in summary:
        rnd, lvl, pr, rc, f1, n, _ = row
        print(f"  {rnd:<5} {lvl:<12} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}",
              flush=True)


if __name__ == '__main__':
    main()
