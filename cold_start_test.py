"""
cold_start_test.py — Deterministic cold-start classifier test suite.

Pass criteria (checked at round 4, all cold-model data present):
  - Extreme recall >= 0.95
  - VH (Very High) recall >= 0.90

Usage:
  python cold_start_test.py [--model-dir models_cpu] [--n-rounds 4] [--min-rows 50] [--n-cold 20]

Model selection is deterministic (no random seed):
  - All models with >= min_rows subscribers.
  - Sorted by (n_extreme + n_vh) descending, stable sort.
  - Top n_cold models selected.

Exits with code 0 on PASS, 1 on FAIL.
"""
import sys
import os
import argparse
import json
import pickle
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


# ── Model loading ──────────────────────────────────────────────────────────────

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


# ── Prediction (mirrors incremental_cold_eval.predict_df exactly) ─────────────

def predict_df(df, models, thresholds, feats):
    """
    Run full prediction cascade on df.
    df must have is_internal_data column so the feedback-loop fix can fire.
    Returns per_user DataFrame with 'predicted' and 'is_cold' columns.
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
    cold_vh_proba = (models['cold_vh'].predict_proba(X_base_cold)[:, 1]
                     if 'cold_vh' in models else np.zeros(len(per_user)))
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

    t_cold_vh_e   = thresholds.get('vh_extreme_cold', 0.5)
    t_warm_rescue = thresholds.get('warm_vh_rescue', 0.47)

    predicted = []
    for i in range(len(per_user)):
        if cold_mask[i]:
            if cold_vh_proba[i] >= t_cold_vh_e:
                predicted.append('Very High')
            elif e_proba[i] >= COLD_T_E:
                predicted.append('Extreme')
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
    per_user['vh_proba']  = vh_proba
    per_user['e_proba']   = e_proba
    per_user['is_cold']   = cold_mask
    return per_user


# ── Metrics helpers ────────────────────────────────────────────────────────────

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
    pred_dist = cold_results['predicted'].value_counts()
    true_dist = cold_results['true_risk'].value_counts()
    print(f"  True:  {dict(true_dist)}")
    print(f"  Pred:  {dict(pred_dist)}")


# ── Main test runner ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', default=MODEL_DIR)
    parser.add_argument('--n-rounds',  type=int, default=4)
    parser.add_argument('--min-rows',  type=int, default=50)
    parser.add_argument('--n-cold',    type=int, default=20)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading training data...", flush=True)
    df_all = fetch_df(selected=False)
    df_all = clean(df_all)
    try:
        _db.close()
    except Exception:
        pass
    print(f"  {len(df_all)} rows, {df_all['tracking_model_name'].nunique()} models",
          flush=True)

    # ── Deterministic cold model selection ─────────────────────────────────────
    # No random seed — selection is a stable sort, fully reproducible.
    model_stats = df_all.groupby('tracking_model_name').agg(
        n_rows=('user_name', 'count'),
        n_extreme=('risk_level', lambda s: (s == 'Extreme').sum()),
        n_vh=('risk_level', lambda s: (s == 'Very High').sum()),
    )
    model_stats['n_top2'] = model_stats['n_extreme'] + model_stats['n_vh']

    eligible = model_stats[model_stats['n_rows'] >= args.min_rows].copy()
    # Stable sort: primary key n_top2 desc, secondary key n_rows desc (for tie-breaking).
    eligible = eligible.sort_values(['n_top2', 'n_rows'], ascending=[False, False])

    cold_models = eligible.index.tolist()[:args.n_cold]
    cold_set = set(cold_models)

    print(f"\nSelected {len(cold_models)} cold models "
          f"(>= {args.min_rows} rows, top {args.n_cold} by n_extreme+n_vh):",
          flush=True)
    for cm in cold_models:
        row = model_stats.loc[cm]
        dist = df_all[df_all['tracking_model_name'] == cm]['risk_level'].value_counts()
        dist_str = '  '.join(f"{lvl}={cnt}" for lvl, cnt in dist.items() if cnt > 0)
        print(f"  {cm[:55]:<55} {int(row['n_rows']):>4} rows  {dist_str}")

    # ── Split warm / cold data ─────────────────────────────────────────────────
    df_warm = df_all[~df_all['tracking_model_name'].isin(cold_set)].copy()
    df_warm['is_internal_data'] = True

    # True labels for cold models (first-occurrence label per user_name)
    df_cold_true = df_all[df_all['tracking_model_name'].isin(cold_set)].copy()
    df_cold_true = df_cold_true.sort_values(
        ['tracking_model_name', 'subscribed_at']).reset_index(drop=True)

    true_labels = (df_cold_true.drop_duplicates('user_name')
                   .set_index('user_name')['risk_level']
                   .rename('true_risk'))

    # Assign batch indices per model (temporal order, n_rounds quantiles)
    def assign_batches(g, n_rounds):
        idx = np.arange(len(g))
        return pd.Series((idx * n_rounds // len(g)).astype(int), index=g.index)

    df_cold_true['_batch'] = df_cold_true.groupby(
        'tracking_model_name', group_keys=False).apply(
        lambda g: assign_batches(g, args.n_rounds))

    print(f"\nBatch sizes per model (n_rounds={args.n_rounds}):", flush=True)
    for cm in cold_models:
        bc = (df_cold_true[df_cold_true['tracking_model_name'] == cm]
              ['_batch'].value_counts().sort_index())
        print(f"  {cm[:50]:<50} {dict(bc)}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading models...", flush=True)
    models, thresholds, feats = load_models_and_thresholds(args.model_dir)
    print(f"  Models: {list(models.keys())}", flush=True)
    print(f"  Thresholds: {thresholds}", flush=True)

    # ── Incremental rounds ─────────────────────────────────────────────────────
    summary = []  # (round, level, P, R, F1, n_true)
    round4_metrics = {}  # level -> (P, R, F1, n_true) at final round

    for rnd in range(args.n_rounds):
        frac_pct = int(round((rnd + 1) / args.n_rounds * 100))
        # Batches 0..rnd: cold users injected with risk_level='No risk'
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

        # Evaluate only on cold-model users
        cold_pu = per_user[per_user['is_cold']].copy()
        cold_pu = cold_pu.merge(true_labels.reset_index(), on='user_name', how='inner')

        print(f"  Cold users in prediction: {len(cold_pu)}", flush=True)
        if cold_pu.empty:
            print("  (no cold users found — check model selection)", flush=True)
            continue

        print_metrics_table(cold_pu)

        for lvl in ['Extreme', 'Very High', 'High', 'Low']:
            m = metrics_for(cold_pu['true_risk'], cold_pu['predicted'], lvl)
            if m:
                pr, rc, f1, n = m
                summary.append((rnd + 1, lvl, pr, rc, f1, n))
                if rnd == args.n_rounds - 1:
                    round4_metrics[lvl] = (pr, rc, f1, n)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print("SUMMARY — All levels across rounds", flush=True)
    print(f"{'Round':<7} {'%Data':>6} {'Level':<12} {'P':>7} {'R':>7} {'F1':>7}  n_true",
          flush=True)
    for row in summary:
        rnd, lvl, pr, rc, f1, n = row
        frac_pct = int(round(rnd / args.n_rounds * 100))
        print(f"  {rnd:<5} {frac_pct:>5}%  {lvl:<12} {pr:>7.1%} {rc:>7.1%} {f1:>7.1%}  {n}",
              flush=True)

    # ── Assertions at round 4 ──────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print("ASSERTIONS (round 4 — all cold data present):", flush=True)

    EXTREME_RECALL_THRESHOLD = 0.95
    VH_RECALL_THRESHOLD      = 0.90

    assertion_results = []

    # Extreme recall
    if 'Extreme' in round4_metrics:
        _, rc_e, _, n_e = round4_metrics['Extreme']
        passed = rc_e >= EXTREME_RECALL_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Extreme recall {rc_e:.1%} >= {EXTREME_RECALL_THRESHOLD:.0%}"
              f"  (n={n_e})", flush=True)
        assertion_results.append(('Extreme recall', passed, rc_e, EXTREME_RECALL_THRESHOLD))
    else:
        print(f"  [FAIL] Extreme recall — no Extreme users found in cold models", flush=True)
        assertion_results.append(('Extreme recall', False, 0.0, EXTREME_RECALL_THRESHOLD))

    # VH recall
    if 'Very High' in round4_metrics:
        _, rc_vh, _, n_vh = round4_metrics['Very High']
        passed = rc_vh >= VH_RECALL_THRESHOLD
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] VH recall {rc_vh:.1%} >= {VH_RECALL_THRESHOLD:.0%}"
              f"  (n={n_vh})", flush=True)
        assertion_results.append(('VH recall', passed, rc_vh, VH_RECALL_THRESHOLD))
    else:
        print(f"  [FAIL] VH recall — no Very High users found in cold models", flush=True)
        assertion_results.append(('VH recall', False, 0.0, VH_RECALL_THRESHOLD))

    # Final summary
    failed = [(name, actual, threshold)
              for name, passed, actual, threshold in assertion_results
              if not passed]

    print(f"\n{'='*65}", flush=True)
    if not failed:
        print("RESULT: PASS — all assertions met.", flush=True)
        sys.exit(0)
    else:
        print("RESULT: FAIL — the following assertions did not pass:", flush=True)
        for name, actual, threshold in failed:
            print(f"  - {name}: got {actual:.1%}, required >= {threshold:.0%}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
