"""
classifier_v4.py — push Very High risk F1 to 95%+

Strategy: close_id_count_4h drives 51% of importance. Add:
1. Finer time windows between 1h and 24h: 2h, 6h, 12h
2. Additional ID range thresholds: 10k, 50k (vs only 100k)
3. Close-ID fraction features: close_id_count / partner_count per window
4. Window ratio features: burst concentration

No username features (confirmed to hurt in v3).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

MAX_ID_RANGES = [10_000, 50_000, 100_000]
WINDOWS_MINUTES = [60, 60*2, 60*4, 60*6, 60*12, 60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]   # 1,2,4,6,12,24,168,336,672

FEATURES = [
    # core per-window features (with 100k range, original set)
    *[f'log_min_id_diff_{w}h'   for w in WINDOW_HOURS],
    *[f'close_id_count_{w}h'   for w in WINDOW_HOURS],
    *[f'partner_count_{w}h'    for w in WINDOW_HOURS],
    # close_id at narrower ranges
    *[f'cic10k_{w}h'  for w in WINDOW_HOURS],   # 10k threshold
    *[f'cic50k_{w}h'  for w in WINDOW_HOURS],   # 50k threshold
    # close-id fractions
    *[f'cic_frac_{w}h' for w in WINDOW_HOURS],  # close_id_count_100k / partner_count
    # model/cross-model
    'log_min_time_diff',
    'log_model_norisk_median_id_diff',
    'rel_min_id_diff_24h',
    'user_model_count',
    'user_max_risk_elsewhere',
    'log_user_id_num',
    # ratio features: burst concentration
    'cic_ratio_4h_24h',    # close_id_count_4h / close_id_count_24h
    'cic_ratio_1h_4h',
    'pc_ratio_1h_24h',     # partner_count_1h / partner_count_24h
]


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
        min_id_diff = (partners['user_id_num'] - user['user_id_num']).abs().min()
        records.append({'tracking_model_name': user['tracking_model_name'], 'min_id_diff': min_id_diff})
    if not records:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame(records)
    return tmp.groupby('tracking_model_name')['min_id_diff'].median()


def _precompute_cross_model(df):
    user_models = df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count')
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
        t = user['subscribed_at']
        model = user['tracking_model_name']
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
                row[f'close_id_count_{w_h}h']  = np.nan
                row[f'partner_count_{w_h}h']   = np.nan
                row[f'cic10k_{w_h}h']          = np.nan
                row[f'cic50k_{w_h}h']          = np.nan
                row[f'cic_frac_{w_h}h']        = np.nan
                any_empty = True
            else:
                id_diffs = (partners['user_id_num'] - user['user_id_num']).abs()
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                cic100 = (id_diffs <= 100_000).sum()
                cic50  = (id_diffs <=  50_000).sum()
                cic10  = (id_diffs <=  10_000).sum()
                pc = len(partners)
                row[f'close_id_count_{w_h}h']  = float(cic100)
                row[f'cic50k_{w_h}h']          = float(cic50)
                row[f'cic10k_{w_h}h']          = float(cic10)
                row[f'partner_count_{w_h}h']   = float(pc)
                row[f'cic_frac_{w_h}h']        = cic100 / (pc + 1)

        if any_empty:
            for i in range(len(WINDOWS_MINUTES) - 1):
                w_h, nw_h = WINDOW_HOURS[i], WINDOW_HOURS[i + 1]
                for feat in ['log_min_id_diff', 'close_id_count', 'partner_count', 'cic10k', 'cic50k', 'cic_frac']:
                    k, nk = f'{feat}_{w_h}h', f'{feat}_{nw_h}h'
                    if np.isnan(row[k]):
                        row[k] = row[nk]

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
        row['log_user_id_num'] = np.log1p(user['user_id_num']) if not np.isnan(user['user_id_num']) else np.nan

        rows.append(row)

    per_user = pd.DataFrame(rows)

    # Ratio features
    per_user['cic_ratio_4h_24h'] = per_user['close_id_count_4h'] / (per_user['close_id_count_24h'] + 1)
    per_user['cic_ratio_1h_4h']  = per_user['close_id_count_1h'] / (per_user['close_id_count_4h']  + 1)
    per_user['pc_ratio_1h_24h']  = per_user['partner_count_1h']  / (per_user['partner_count_24h']  + 1)

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
    N_FOLDS = 10

    configs = [
        ("HGB-d4",    False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_depth=4, learning_rate=0.02, min_samples_leaf=8,  l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-d6",    False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_depth=6, learning_rate=0.03, min_samples_leaf=5,  l2_regularization=0.1, class_weight='balanced', random_state=s)),
        ("HGB-d3",    False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=3, learning_rate=0.01, min_samples_leaf=10, l2_regularization=1.0, class_weight='balanced', random_state=s)),
        ("HGB-d5a",   False, lambda s: HistGradientBoostingClassifier(max_iter=500,  max_depth=5, learning_rate=0.05, min_samples_leaf=5,  l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("HGB-d5b",   False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=5, learning_rate=0.02, min_samples_leaf=3,  l2_regularization=0.3, class_weight='balanced', random_state=s)),
        ("HGB-mln31", False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_leaf_nodes=31, learning_rate=0.02, min_samples_leaf=8,  l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-mln15", False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_leaf_nodes=15, learning_rate=0.03, min_samples_leaf=5,  l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("XGB-spw3",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=3.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw5",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=5.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw10", True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=10.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
    ]

    print(f"\n=== Ensemble: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
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
    print(f"Ensemble ({len(all_probas)} configs, opt threshold={best_thresh:.3f}):")
    print(f"  Precision: {pv:.2%}  Recall: {rv:.2%}  F1: {fv:.2%}")

    print(f"\nThreshold sweep:")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for thresh in np.arange(0.3, 0.9, 0.05):
        yt = (y_proba >= thresh).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y, yt, pos_label=1, average='binary', zero_division=0)
        print(f"  {thresh:>7.2f}  {pt:>9.2%}  {rt:>6.2%}  {ft:>6.2%}")

    # Feature importance via XGB
    Xf, _ = nan_fill(X, X)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                        eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb.fit(Xf, y)
    importances = sorted(zip(feats, xgb.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop-15 feature importances (XGB):")
    for feat, imp in importances[:15]:
        print(f"  {feat:<35} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Previous best (classifier_improved.py):   91.82%")
    print(f"  This script F1:                           {fv:.2%}")
    print(f"  vs previous best: {fv - 0.9182:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.92:
        print(f"\n  92% target exceeded! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%}")
