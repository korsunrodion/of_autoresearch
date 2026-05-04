"""
pure_cold_test.py — Cold-start test on randomly selected models.

Unlike cold_start_test.py (top 20 by n_extreme+n_vh), this picks models
randomly so the test covers typical models, not just the densest risk populations.

The existing trained model is used as-is. Cold cascade fires whenever
model_any_risk_frac==0 (all subscribers injected as 'No risk').

Pass criteria (at final round, all cold data present):
  - Extreme recall >= 0.95  (only asserted if n_extreme >= 10)
  - VH recall    >= 0.90   (only asserted if n_vh >= 5)

Usage:
  python pure_cold_test.py [--n-cold 20] [--n-rounds 4] [--min-rows 50]
                            [--seed 42] [--model-dir models_cpu]
"""
import sys
import os
import argparse
import json
import pickle
import random
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from classifier_v2_cpu_predict import (
    compute_per_user, make_cold_X, FEATURES, COLD_OVERRIDE,
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
    per_user = compute_per_user(df)

    if 'is_internal_data' in df.columns:
        df_warm = df[df['is_internal_data'] == True]
        warm_risk_fracs, warm_norisk_rates = {}, {}
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
        per_user['model_vh_frac']       = pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('vh', 0.0))
        per_user['model_extreme_frac']  = pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('extreme', 0.0))
        per_user['model_any_risk_frac'] = pu_models.map(lambda m: warm_risk_fracs.get(m, {}).get('any', 0.0))
        per_user['model_norisk_rate']   = pu_models.map(lambda m: warm_norisk_rates.get(m, 1.0))

    X_b = per_user[feats].values
    e_proba   = models['extreme'].predict_proba(X_b)[:, 1]
    h_proba   = models['high'].predict_proba(X_b)[:, 1]
    l_proba   = models['low'].predict_proba(X_b)[:, 1]
    ord_proba = models['ordinal'].predict_proba(X_b)[:, 1]
    X_aug     = np.column_stack([X_b, e_proba, h_proba, l_proba, ord_proba])
    vh_proba  = models['vh'].predict_proba(X_aug)[:, 1]

    X_base_cold = make_cold_X(X_b, feats)
    if 'cold_vh' in models:
        cold_vh_obj = models['cold_vh']
        if isinstance(cold_vh_obj, list):
            cold_vh_proba = np.mean([c.predict_proba(X_base_cold)[:, 1]
                                     for c in cold_vh_obj], axis=0)
        else:
            cold_vh_proba = cold_vh_obj.predict_proba(X_base_cold)[:, 1]
    else:
        cold_vh_proba = np.zeros(len(per_user))

    warm_rescue_proba = (models['warm_vh_rescue'].predict_proba(X_aug)[:, 1]
                         if 'warm_vh_rescue' in models else np.zeros(len(per_user)))

    t_e           = thresholds.get('extreme', 0.5)
    t_vh          = thresholds.get('vh', 0.5)
    t_h           = thresholds.get('high', 0.5)
    t_l           = thresholds.get('low', 0.5)
    COLD_T_E      = thresholds.get('extreme_cold', 0.50)
    t_h_cold      = thresholds.get('high_cold', t_h)
    t_l_cold      = thresholds.get('low_cold', t_l)
    t_cold_vh_e    = thresholds.get('vh_extreme_cold', 0.5)
    t_warm_rescue  = thresholds.get('warm_vh_rescue', 0.47)
    t_cold_vh_soft = thresholds.get('cold_vh_soft_gate', 0.01)
    t_cold_ord     = thresholds.get('cold_ord_gate', 0.10)
    cold_mask = (per_user['model_any_risk_frac'].fillna(0) == 0).values

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

    per_user['predicted'] = predicted
    per_user['e_proba']   = e_proba
    per_user['vh_proba']  = vh_proba
    per_user['is_cold']   = cold_mask
    return per_user


def metrics_for(true_series, pred_series, level):
    t = (true_series == level).astype(int).values
    p = (pred_series == level).astype(int).values
    n_true = int(t.sum())
    if n_true == 0:
        return None
    pr, rc, f1, _ = precision_recall_fscore_support(
        t, p, pos_label=1, average='binary', zero_division=0)
    return float(pr), float(rc), float(f1), n_true


def print_metrics_table(cold_results):
    levels = ['Extreme', 'Very High', 'High', 'Low']
    print(f"  {'Level':<12} {'P':>7} {'R':>7} {'F1':>7}  n_true")
    for lvl in levels:
        m = metrics_for(cold_results['true_risk'], cold_results['predicted'], lvl)
        if m:
            pr, rc, f1, n = m
            print(f"  {lvl:<12} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}")
        else:
            print(f"  {lvl:<12} {'—':>7} {'—':>7} {'—':>7}  0")
    print(f"  True:  {dict(cold_results['true_risk'].value_counts())}")
    print(f"  Pred:  {dict(cold_results['predicted'].value_counts())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-cold',    type=int, default=20,
                        help='Number of cold models to test (default 20)')
    parser.add_argument('--n-rounds',  type=int, default=4,
                        help='Incremental rounds (default 4)')
    parser.add_argument('--min-rows',  type=int, default=50,
                        help='Minimum subscribers per model (default 50)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for model selection (default 42)')
    parser.add_argument('--model-dir', default=MODEL_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    df_all = fetch_df(selected=False)
    df_all = clean(df_all)
    try:
        _db.close()
    except Exception:
        pass
    print(f"  {len(df_all)} rows, {df_all['tracking_model_name'].nunique()} models",
          flush=True)

    # ── Select cold models randomly (no risk-level bias) ───────────────────────
    model_stats = df_all.groupby('tracking_model_name').agg(
        n_rows=('user_name', 'count'),
        n_extreme=('risk_level', lambda s: (s == 'Extreme').sum()),
        n_vh=('risk_level', lambda s: (s == 'Very High').sum()),
    )
    eligible = model_stats[model_stats['n_rows'] >= args.min_rows].index.tolist()
    n_cold = min(args.n_cold, len(eligible))
    cold_models = random.sample(eligible, n_cold)
    cold_set = set(cold_models)

    n_ext_total = int(model_stats.loc[list(cold_set), 'n_extreme'].sum())
    n_vh_total  = int(model_stats.loc[list(cold_set), 'n_vh'].sum())

    print(f"\nSelected {n_cold} cold models "
          f"(seed={args.seed}, random, >= {args.min_rows} rows):", flush=True)
    for cm in cold_models:
        row = model_stats.loc[cm]
        dist = df_all[df_all['tracking_model_name'] == cm]['risk_level'].value_counts()
        dist_str = '  '.join(f"{lvl}={cnt}" for lvl, cnt in dist.items() if cnt > 0)
        print(f"  {cm[:55]:<55} {int(row['n_rows']):>4} rows  {dist_str}")
    print(f"\n  Totals across cold set: Extreme={n_ext_total}  VH={n_vh_total}",
          flush=True)

    # ── Warm baseline and true labels ──────────────────────────────────────────
    df_warm = df_all[~df_all['tracking_model_name'].isin(cold_set)].copy()
    df_warm['is_internal_data'] = True

    df_cold_true = df_all[df_all['tracking_model_name'].isin(cold_set)].copy()
    df_cold_true = df_cold_true.sort_values(
        ['tracking_model_name', 'subscribed_at']).reset_index(drop=True)

    true_labels = (df_cold_true.drop_duplicates('user_name')
                   .set_index('user_name')['risk_level']
                   .rename('true_risk'))

    def assign_batches(g):
        idx = np.arange(len(g))
        return pd.Series((idx * args.n_rounds // len(g)).astype(int), index=g.index)

    df_cold_true['_batch'] = df_cold_true.groupby(
        'tracking_model_name', group_keys=False).apply(assign_batches)

    # ── Load models ────────────────────────────────────────────────────────────
    print(f"\nLoading models from {args.model_dir}/...", flush=True)
    models, thresholds, feats = load_models_and_thresholds(args.model_dir)
    print(f"  Thresholds: {thresholds}", flush=True)

    # ── Incremental rounds ─────────────────────────────────────────────────────
    summary = []
    round_final_metrics = {}

    for rnd in range(args.n_rounds):
        frac_pct = int(round((rnd + 1) / args.n_rounds * 100))
        df_cold_batches = df_cold_true[df_cold_true['_batch'] <= rnd].copy()
        df_cold_batches['risk_level']       = 'No risk'
        df_cold_batches['risk_score']       = 1
        df_cold_batches['is_internal_data'] = False
        df_cold_batches = df_cold_batches.drop(columns=['_batch'])

        df_sim = pd.concat([df_warm, df_cold_batches], ignore_index=True)

        n_cold_users = df_cold_batches['user_name'].nunique()
        n_cold_rows  = len(df_cold_batches)
        print(f"\n{'='*65}", flush=True)
        print(f"Round {rnd+1}/{args.n_rounds}  ({frac_pct}% of cold rows)"
              f"  {n_cold_rows} cold rows, {n_cold_users} cold users", flush=True)
        print(f"  Total rows: {len(df_sim)} "
              f"(warm={len(df_warm)}, cold={n_cold_rows})", flush=True)

        per_user = predict_df(df_sim, models, thresholds, feats)

        cold_pu = per_user[per_user['is_cold']].copy()
        cold_pu = cold_pu.merge(true_labels.reset_index(), on='user_name', how='inner')

        print(f"  Cold users predicted: {len(cold_pu)}", flush=True)
        if cold_pu.empty:
            print("  (no cold users — check model selection)", flush=True)
            continue

        print_metrics_table(cold_pu)

        for lvl in ['Extreme', 'Very High', 'High', 'Low']:
            m = metrics_for(cold_pu['true_risk'], cold_pu['predicted'], lvl)
            if m:
                pr, rc, f1, n = m
                summary.append((rnd + 1, lvl, pr, rc, f1, n))
                if rnd == args.n_rounds - 1:
                    round_final_metrics[lvl] = (pr, rc, f1, n)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print("SUMMARY — All levels across rounds", flush=True)
    print(f"{'Round':<7} {'%Data':>6} {'Level':<12} {'P':>7} {'R':>7} {'F1':>7}  n_true",
          flush=True)
    for row in summary:
        rnd, lvl, pr, rc, f1, n = row
        frac_pct = int(round(rnd / args.n_rounds * 100))
        print(f"  {rnd:<5} {frac_pct:>5}%  {lvl:<12} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}",
              flush=True)

    # ── Assertions ─────────────────────────────────────────────────────────────
    MIN_N_EXTREME            = 10
    MIN_N_VH                 = 5
    EXTREME_RECALL_THRESHOLD = 0.95
    VH_RECALL_THRESHOLD      = 0.90

    print(f"\n{'='*65}", flush=True)
    print("ASSERTIONS (final round — all cold data present):", flush=True)

    assertion_results = []

    for level, threshold, min_n, label in [
        ('Extreme',   EXTREME_RECALL_THRESHOLD, MIN_N_EXTREME, 'Extreme recall'),
        ('Very High', VH_RECALL_THRESHOLD,      MIN_N_VH,      'VH recall'),
    ]:
        if level not in round_final_metrics:
            print(f"  [FAIL] {label} — no {level} users in cold set", flush=True)
            assertion_results.append((label, False, 0.0, threshold))
            continue
        _, rc, _, n = round_final_metrics[level]
        if n < min_n:
            print(f"  [SKIP] {label} {rc:.1%}  (n={n} < {min_n} — too few to assert)",
                  flush=True)
            continue
        passed = rc >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label} {rc:.1%} >= {threshold:.0%}  (n={n})", flush=True)
        assertion_results.append((label, passed, rc, threshold))

    failed = [x for x in assertion_results if not x[1]]

    print(f"\n{'='*65}", flush=True)
    if not assertion_results:
        print("RESULT: SKIP — all levels below minimum n for assertion.", flush=True)
        sys.exit(0)
    elif not failed:
        print("RESULT: PASS — all assertions met.", flush=True)
        sys.exit(0)
    else:
        print("RESULT: FAIL:", flush=True)
        for label, _, actual, thr in failed:
            print(f"  - {label}: got {actual:.1%}, required >= {thr:.0%}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
