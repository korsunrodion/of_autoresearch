"""
classifier_v10.py — v5 features + RandomForest/ExtraTrees ensemble members

v5/v8/v9 all plateau at 93.23-93.28%. The feature space is exhausted (only
user_name, subscribed_at, user_id_num, tracking_model_name columns available).
Remaining lever: ensemble diversity. Add RF and ExtraTrees — these use
bagging + feature subsetting (not boosting), giving genuinely different
predictions for borderline users.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier, ExtraTreesClassifier)
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

ID_RANGES  = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['1k', '5k', '10k', '20k', '50k', '100k']

WINDOWS_MINUTES = [60, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]

KEY_WINDOWS_H = {6, 8, 10, 12, 24}

FEATURES = (
    [f'log_min_id_diff_{w}h' for w in WINDOW_HOURS] +
    [f'partner_count_{w}h'   for w in WINDOW_HOURS] +
    [f'cic_{tag}_{w}h'      for tag in RANGE_TAGS for w in WINDOW_HOURS] +
    [f'cic_frac_{tag}_{w}h' for tag in RANGE_TAGS for w in WINDOW_HOURS] +
    [f'log_id_span_{w}h'        for w in sorted(KEY_WINDOWS_H)] +
    [f'log_id_std_{w}h'         for w in sorted(KEY_WINDOWS_H)] +
    [f'log_min_consec_gap_{w}h' for w in sorted(KEY_WINDOWS_H)] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num',
     'cic_ratio_6h_24h', 'cic10k_ratio_6h_12h', 'pc_ratio_6h_24h']
)


def _precompute_model_stats(df):
    max_window = pd.Timedelta(minutes=max(WINDOWS_MINUTES))
    times = df['subscribed_at']
    records = []
    for _, user in df[df['risk_level'] == 'No risk'].iterrows():
        t = user['subscribed_at']
        mask = (
            (times >= t - max_window) & (times <= t + max_window) &
            (df['tracking_model_name'] == user['tracking_model_name']) &
            (df['user_name'] != user['user_name'])
        )
        partners = df[mask]
        if partners.empty:
            continue
        records.append({
            'tracking_model_name': user['tracking_model_name'],
            'min_id_diff': (partners['user_id_num'] - user['user_id_num']).abs().min()
        })
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).groupby('tracking_model_name')['min_id_diff'].median()


def _precompute_cross_model(df):
    user_models   = df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count')
    user_max_risk = df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk')
    return pd.concat([user_models, user_max_risk], axis=1)


def compute_per_user(df):
    print("  Precomputing model stats...", flush=True)
    model_norisk_median = _precompute_model_stats(df)
    print("  Precomputing cross-model stats...", flush=True)
    cross = _precompute_cross_model(df)

    max_window = pd.Timedelta(minutes=max(WINDOWS_MINUTES))
    times = df['subscribed_at']
    rows = []

    for _, user in df.iterrows():
        t     = user['subscribed_at']
        model = user['tracking_model_name']
        uid   = user['user_id_num']
        base_mask = (
            (times >= t - max_window) & (times <= t + max_window) &
            (df['tracking_model_name'] == model) &
            (df['user_name'] != user['user_name'])
        )
        all_partners = df[base_mask]

        row = {'user_name': user['user_name'], 'risk_level': user['risk_level']}
        any_empty = False

        for w_min, w_h in zip(WINDOWS_MINUTES, WINDOW_HOURS):
            w_td = pd.Timedelta(minutes=w_min)
            partners = all_partners[
                (all_partners['subscribed_at'] >= t - w_td) &
                (all_partners['subscribed_at'] <= t + w_td)
            ]
            if partners.empty:
                row[f'log_min_id_diff_{w_h}h'] = np.nan
                row[f'partner_count_{w_h}h']   = np.nan
                for tag in RANGE_TAGS:
                    row[f'cic_{tag}_{w_h}h']      = np.nan
                    row[f'cic_frac_{tag}_{w_h}h'] = np.nan
                if w_h in KEY_WINDOWS_H:
                    row[f'log_id_span_{w_h}h']        = np.nan
                    row[f'log_id_std_{w_h}h']         = np.nan
                    row[f'log_min_consec_gap_{w_h}h'] = np.nan
                any_empty = True
            else:
                id_diffs    = (partners['user_id_num'] - uid).abs().values
                partner_ids = partners['user_id_num'].values
                pc = len(partners)
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                row[f'partner_count_{w_h}h']   = float(pc)
                for tag, rng in zip(RANGE_TAGS, ID_RANGES):
                    cic = float((id_diffs <= rng).sum())
                    row[f'cic_{tag}_{w_h}h']      = cic
                    row[f'cic_frac_{tag}_{w_h}h'] = cic / (pc + 1)
                if w_h in KEY_WINDOWS_H:
                    span = float(partner_ids.max() - partner_ids.min()) if pc > 0 else 0.0
                    std  = float(np.std(partner_ids)) if pc > 1 else 0.0
                    row[f'log_id_span_{w_h}h'] = np.log1p(span)
                    row[f'log_id_std_{w_h}h']  = np.log1p(std)
                    if pc >= 2:
                        mgap = float(np.diff(np.sort(partner_ids)).min())
                        row[f'log_min_consec_gap_{w_h}h'] = np.log1p(mgap)
                    else:
                        row[f'log_min_consec_gap_{w_h}h'] = np.nan

        if any_empty:
            for i in range(len(WINDOWS_MINUTES) - 1):
                wh, nwh = WINDOW_HOURS[i], WINDOW_HOURS[i + 1]
                for key in ([f'log_min_id_diff_{wh}h', f'partner_count_{wh}h'] +
                            [f'cic_{tag}_{wh}h'      for tag in RANGE_TAGS] +
                            [f'cic_frac_{tag}_{wh}h' for tag in RANGE_TAGS]):
                    if np.isnan(row.get(key, np.nan)):
                        row[key] = row.get(f'{key[:key.rfind("_")]}_{nwh}h', np.nan)

        if not all_partners.empty:
            row['log_min_time_diff'] = np.log1p(
                (all_partners['subscribed_at'] - t).abs().dt.total_seconds().min() / 60
            )
        else:
            row['log_min_time_diff'] = np.nan

        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0:
            row['rel_min_id_diff_24h'] = raw_24h / norisk_med
        else:
            row['rel_min_id_diff_24h'] = np.nan

        uname = user['user_name']
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
        row['log_user_id_num'] = np.log1p(uid) if not np.isnan(uid) else np.nan
        rows.append(row)

    per_user = pd.DataFrame(rows)
    per_user['cic_ratio_6h_24h']    = per_user['cic_50k_6h']  / (per_user['cic_50k_24h']  + 1)
    per_user['cic10k_ratio_6h_12h'] = per_user['cic_10k_6h']  / (per_user['cic_10k_12h']  + 1)
    per_user['pc_ratio_6h_24h']     = per_user['partner_count_6h'] / (per_user['partner_count_24h'] + 1)
    return per_user


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def nan_fill(X_train, X_val):
    col_med = np.nanmedian(X_train, axis=0)
    return (
        np.where(np.isnan(X_train), col_med, X_train),
        np.where(np.isnan(X_val),   col_med, X_val),
    )


def accumulate_oof(X, y, make_fn, seeds, n_folds, needs_nan_fill=False):
    pool = np.zeros(len(y))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(len(y))
        for ti, vi in cv.split(X, y):
            X_tr, X_va = (nan_fill(X[ti], X[vi]) if needs_nan_fill else (X[ti], X[vi]))
            m = make_fn(seed)
            m.fit(X_tr, y[ti])
            proba[vi] = m.predict_proba(X_va)[:, 1]
        pool += proba
    return pool / len(seeds)


if __name__ == '__main__':
    df = fetch_df(selected=True)
    df = clean(df)

    per_user = compute_per_user(df)
    print(f"Total rows: {len(per_user)}", flush=True)

    levels = ['Very High']
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)

    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values
    y = subset['is_risky'].astype(int).values
    print(f"Dataset: {y.sum()} positives, {len(y)-y.sum()} negatives, {len(feats)} features", flush=True)

    SEEDS   = [42, 7, 123, 999, 2025, 1337, 2024, 101]
    N_FOLDS = 15

    configs = [
        # Gradient boosting (7 HistGB + 3 XGB) — v5 core ensemble
        ("HGB-d4",    False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_depth=4, learning_rate=0.02, min_samples_leaf=8,  l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-d6",    False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_depth=6, learning_rate=0.03, min_samples_leaf=5,  l2_regularization=0.1, class_weight='balanced', random_state=s)),
        ("HGB-d3",    False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=3, learning_rate=0.01, min_samples_leaf=10, l2_regularization=1.0, class_weight='balanced', random_state=s)),
        ("HGB-d5a",   False, lambda s: HistGradientBoostingClassifier(max_iter=500,  max_depth=5, learning_rate=0.05, min_samples_leaf=5,  l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("HGB-d5b",   False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=5, learning_rate=0.02, min_samples_leaf=3,  l2_regularization=0.3, class_weight='balanced', random_state=s)),
        ("HGB-mln31", False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_leaf_nodes=31, learning_rate=0.02, min_samples_leaf=8, l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-mln15", False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_leaf_nodes=15, learning_rate=0.03, min_samples_leaf=5, l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("XGB-spw3",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=3.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw5",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=5.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw10", True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=10.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        # Random Forest — bagging + feature subsetting, different inductive bias
        ("RF-d8",     True,  lambda s: RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=s)),
        ("RF-d6",     True,  lambda s: RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=s)),
        # Extra Trees — maximally random splits, even more diversity
        ("ET-d8",     True,  lambda s: ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=s)),
        ("ET-d6",     True,  lambda s: ExtraTreesClassifier(n_estimators=500, max_depth=6, min_samples_leaf=5, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=s)),
    ]

    print(f"\n=== v10 Ensemble: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
    all_probas = []
    for name, needs_fill, cfg_fn in configs:
        p = accumulate_oof(X, y, cfg_fn, SEEDS, N_FOLDS, needs_nan_fill=needs_fill)
        all_probas.append(p)
        print(f"  Done: {name}", flush=True)

    y_proba = np.mean(all_probas, axis=0)
    best_thresh = find_best_threshold(y, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    pv, rv, fv, _ = precision_recall_fscore_support(y, y_pred, pos_label=1, average='binary', zero_division=0)

    print(f"\n{'='*60}")
    print(f"v10 Ensemble ({len(all_probas)} configs, opt threshold={best_thresh:.3f}):")
    print(f"  Precision: {pv:.2%}  Recall: {rv:.2%}  F1: {fv:.2%}")

    # Show what HGB-only and RF-only give for comparison
    hgb_xgb_proba = np.mean(all_probas[:10], axis=0)
    thresh_hgb = find_best_threshold(y, hgb_xgb_proba)
    y_pred_hgb = (hgb_xgb_proba >= thresh_hgb).astype(int)
    ph, rh, fh, _ = precision_recall_fscore_support(y, y_pred_hgb, pos_label=1, average='binary', zero_division=0)
    print(f"\n  HGB+XGB only (v5 core): P={ph:.2%} R={rh:.2%} F1={fh:.2%} @ {thresh_hgb:.3f}")

    rf_et_proba = np.mean(all_probas[10:], axis=0)
    thresh_re = find_best_threshold(y, rf_et_proba)
    y_pred_re = (rf_et_proba >= thresh_re).astype(int)
    pre, rre, fre, _ = precision_recall_fscore_support(y, y_pred_re, pos_label=1, average='binary', zero_division=0)
    print(f"  RF+ET only:             P={pre:.2%} R={rre:.2%} F1={fre:.2%} @ {thresh_re:.3f}")

    print(f"\nThreshold sweep:")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for thresh in np.arange(0.3, 0.9, 0.05):
        yt = (y_proba >= thresh).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y, yt, pos_label=1, average='binary', zero_division=0)
        print(f"  {thresh:>7.2f}  {pt:>9.2%}  {rt:>6.2%}  {ft:>6.2%}")

    Xf, _ = nan_fill(X, X)
    xgb_imp = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                             eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb_imp.fit(Xf, y)
    importances = sorted(zip(feats, xgb_imp.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop-15 feature importances (XGB):")
    for feat, imp in importances[:15]:
        print(f"  {feat:<40} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  v5 best:   93.28%")
    print(f"  v10 F1:    {fv:.2%}")
    print(f"  vs v5: {fv - 0.9328:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.93:
        print(f"\n  Above 93%! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%}")
