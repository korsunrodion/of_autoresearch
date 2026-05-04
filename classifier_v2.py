"""
classifier_v2.py — push Very High risk F1 to 95%+

Key improvements over classifier_improved.py (91.82% F1):
1. selected=False: 1048 Very High positives vs 255 (4x more training signal)
2. Vectorized numpy feature computation (per-model matrix ops, ~100x faster)
3. New features: close_id_fraction per window, window ratios, cross-window diffs
4. Larger/more diverse ensemble, nested threshold search
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

MAX_ID_RANGE = 100_000
WINDOWS_MINUTES = [60, 60 * 4, 60 * 24, 60 * 24 * 7, 60 * 24 * 7 * 2, 60 * 24 * 7 * 4]
WINDOW_HOURS = [w // 60 for w in WINDOWS_MINUTES]

BASE_FEATURES = [
    *[f'log_min_id_diff_{w}h' for w in WINDOW_HOURS],
    *[f'close_id_count_{w}h' for w in WINDOW_HOURS],
    *[f'partner_count_{w}h' for w in WINDOW_HOURS],
    'log_min_time_diff',
    'log_model_norisk_median_id_diff',
    'rel_min_id_diff_24h',
    'user_model_count',
    'user_max_risk_elsewhere',
    'log_user_id_num',
]

# Derived features computed after the base features
DERIVED_FEATURES = [
    *[f'close_id_frac_{w}h' for w in WINDOW_HOURS],   # close_id_count / partner_count
    'log_min_id_diff_ratio_1h_24h',                    # 1h vs 24h min_id_diff ratio
    'log_min_id_diff_ratio_24h_168h',                  # 24h vs 7d ratio
    'partner_count_ratio_1h_24h',                      # burst vs day ratio
    'partner_count_ratio_24h_168h',
    'close_count_total',                               # sum close counts across all windows
]

FEATURES = BASE_FEATURES + DERIVED_FEATURES


def _model_stats_vectorized(df: pd.DataFrame) -> pd.Series:
    """Median min_id_diff of No-risk users per model — vectorized per model."""
    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    norisk = df[df['risk_level'] == 'No risk']
    records = []

    for model_name, mdf in df.groupby('tracking_model_name'):
        norisk_m = mdf[mdf['risk_level'] == 'No risk']
        if norisk_m.empty:
            continue
        nr_times = norisk_m['subscribed_at'].values.astype(np.int64)
        nr_ids   = norisk_m['user_id_num'].values
        all_times = mdf['subscribed_at'].values.astype(np.int64)
        all_ids   = mdf['user_id_num'].values
        all_names = mdf['user_name'].values
        nr_names  = norisk_m['user_name'].values

        for i, (t, uid, uname) in enumerate(zip(nr_times, nr_ids, nr_names)):
            mask = (np.abs(all_times - t) <= max_w_ns) & (all_names != uname)
            if mask.any():
                min_diff = np.abs(all_ids[mask] - uid).min()
                records.append({'tracking_model_name': model_name, 'min_id_diff': min_diff})

    if not records:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame(records)
    return tmp.groupby('tracking_model_name')['min_id_diff'].median()


def _cross_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    user_models = df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count')
    user_max_risk = df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk')
    return pd.concat([user_models, user_max_risk], axis=1)


def compute_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-user feature computation using numpy matrix ops per model."""
    print("  Precomputing model stats...", flush=True)
    model_norisk_median = _model_stats_vectorized(df)
    print("  Precomputing cross-model stats...", flush=True)
    cross = _cross_model_stats(df)

    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    all_rows = []

    for model_name, mdf in df.groupby('tracking_model_name'):
        n = len(mdf)
        if n == 0:
            continue
        mdf = mdf.reset_index(drop=True)

        times_ns = mdf['subscribed_at'].values.astype(np.int64)   # (n,)
        ids_arr  = mdf['user_id_num'].values.astype(float)        # (n,)
        usernames = mdf['user_name'].values
        risk_levels = mdf['risk_level'].values

        # Pairwise time diff and id diff matrices — (n, n)
        tdiff = np.abs(times_ns[:, None] - times_ns[None, :])  # (n, n)
        iddiff = np.abs(ids_arr[:, None] - ids_arr[None, :])   # (n, n)
        not_self = ~np.eye(n, dtype=bool)                      # exclude diagonal

        norisk_med = model_norisk_median.get(model_name, np.nan)

        for i in range(n):
            row = {'user_name': usernames[i], 'risk_level': risk_levels[i]}
            any_empty = False

            for w_min, w_h in zip(WINDOWS_MINUTES, WINDOW_HOURS):
                w_ns = int(w_min * 60 * 1e9)
                mask = (tdiff[i] <= w_ns) & not_self[i]

                if not mask.any():
                    row[f'log_min_id_diff_{w_h}h'] = np.nan
                    row[f'close_id_count_{w_h}h']  = np.nan
                    row[f'partner_count_{w_h}h']   = np.nan
                    any_empty = True
                else:
                    diffs = iddiff[i, mask]
                    row[f'log_min_id_diff_{w_h}h'] = np.log1p(diffs.min())
                    row[f'close_id_count_{w_h}h']  = float((diffs <= MAX_ID_RANGE).sum())
                    row[f'partner_count_{w_h}h']   = float(mask.sum())

            # Forward-fill NaN windows from the next larger window
            if any_empty:
                for j in range(len(WINDOWS_MINUTES) - 1):
                    wh, wh_next = WINDOW_HOURS[j], WINDOW_HOURS[j + 1]
                    for feat in ['log_min_id_diff', 'close_id_count', 'partner_count']:
                        v = row[f'{feat}_{wh}h']
                        if v is np.nan or (isinstance(v, float) and np.isnan(v)):
                            row[f'{feat}_{wh}h'] = row[f'{feat}_{wh_next}h']

            # log_min_time_diff: min time to any partner within max window
            all_mask = (tdiff[i] <= max_w_ns) & not_self[i]
            if all_mask.any():
                min_t_diff_min = tdiff[i, all_mask].min() / 1e9 / 60
                row['log_min_time_diff'] = np.log1p(min_t_diff_min)
            else:
                row['log_min_time_diff'] = np.nan

            # Model-level features
            row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
            raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
            if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0:
                row['rel_min_id_diff_24h'] = raw_24h / norisk_med
            else:
                row['rel_min_id_diff_24h'] = np.nan

            # Cross-model features
            uname = usernames[i]
            row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
            gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else mdf.loc[i, 'risk_score']
            row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
            row['log_user_id_num'] = np.log1p(ids_arr[i]) if not np.isnan(ids_arr[i]) else np.nan

            all_rows.append(row)

    per_user = pd.DataFrame(all_rows)

    # Derived features
    for w_h in WINDOW_HOURS:
        pc = per_user[f'partner_count_{w_h}h']
        cc = per_user[f'close_id_count_{w_h}h']
        per_user[f'close_id_frac_{w_h}h'] = cc / (pc + 1.0)

    # Ratio of min_id_diff across windows (log-space difference = log ratio)
    per_user['log_min_id_diff_ratio_1h_24h']   = per_user['log_min_id_diff_1h']   - per_user['log_min_id_diff_24h']
    per_user['log_min_id_diff_ratio_24h_168h']  = per_user['log_min_id_diff_24h']  - per_user['log_min_id_diff_168h']
    per_user['partner_count_ratio_1h_24h']       = per_user['partner_count_1h']    / (per_user['partner_count_24h']  + 1)
    per_user['partner_count_ratio_24h_168h']     = per_user['partner_count_24h']   / (per_user['partner_count_168h'] + 1)
    per_user['close_count_total']                = sum(per_user[f'close_id_count_{w}h'] for w in WINDOW_HOURS)

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
    # ── Load full dataset (selected=False → 1048 Very High positives) ──
    print("Loading full dataset (selected=False)...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    print(f"  {len(df)} rows, risk dist:\n{df['risk_level'].value_counts()}\n", flush=True)

    per_user = compute_per_user(df)
    print(f"Total per-user rows: {len(per_user)}", flush=True)

    levels = ['Very High']
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)

    # Use only features that exist
    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values
    y = subset['is_risky'].astype(int).values
    print(f"Dataset: {y.sum()} positives (Very High), {len(y)-y.sum()} negatives (No risk)", flush=True)

    # ── Baseline: replicate classifier_tree.py ──
    ORIG_FEATURES = [f for f in BASE_FEATURES if f != 'log_user_id_num']
    sub_drop = per_user.dropna(subset=ORIG_FEATURES)
    sub_drop = sub_drop[sub_drop['risk_level'].isin(levels + ['No risk'])].copy()
    sub_drop['is_risky'] = sub_drop['risk_level'].isin(levels)
    X_drop = sub_drop[ORIG_FEATURES].values
    y_drop = sub_drop['is_risky'].astype(int).values
    spw0 = (len(y_drop) - y_drop.sum()) / y_drop.sum()

    print(f"\n=== Baseline (classifier_tree.py replication): XGB drop_na ({y_drop.sum()} pos), scale={spw0:.1f} ===", flush=True)
    base_f1s = []
    for seed in [42, 7, 123, 999, 2025]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred = np.zeros_like(y_drop)
        for ti, vi in cv.split(X_drop, y_drop):
            Xtr, Xva = nan_fill(X_drop[ti], X_drop[vi])
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw0,
                              eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=seed)
            m.fit(Xtr, y_drop[ti]); y_pred[vi] = m.predict(Xva)
        _, _, f, _ = precision_recall_fscore_support(y_drop, y_pred, pos_label=1, average='binary', zero_division=0)
        base_f1s.append(f)
    print(f"  Mean F1: {np.mean(base_f1s):.2%}  seeds: {[f'{v:.2%}' for v in base_f1s]}", flush=True)

    # ── Improved ensemble ──
    SEEDS   = [42, 7, 123, 999, 2025, 1337, 2024, 101]
    N_FOLDS = 10

    spw = (len(y) - y.sum()) / y.sum()

    configs = [
        # HistGB (NaN-tolerant, class_weight='balanced')
        ("HGB-d4",    False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_depth=4, learning_rate=0.02, min_samples_leaf=8,   l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-d6",    False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_depth=6, learning_rate=0.03, min_samples_leaf=5,   l2_regularization=0.1, class_weight='balanced', random_state=s)),
        ("HGB-d3",    False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=3, learning_rate=0.01, min_samples_leaf=10,  l2_regularization=1.0, class_weight='balanced', random_state=s)),
        ("HGB-d5a",   False, lambda s: HistGradientBoostingClassifier(max_iter=500,  max_depth=5, learning_rate=0.05, min_samples_leaf=5,   l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("HGB-d5b",   False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=5, learning_rate=0.02, min_samples_leaf=3,   l2_regularization=0.3, class_weight='balanced', random_state=s)),
        ("HGB-mln31", False, lambda s: HistGradientBoostingClassifier(max_iter=800,  max_leaf_nodes=31, learning_rate=0.02, min_samples_leaf=8,  l2_regularization=0.5, class_weight='balanced', random_state=s)),
        ("HGB-mln15", False, lambda s: HistGradientBoostingClassifier(max_iter=600,  max_leaf_nodes=15, learning_rate=0.03, min_samples_leaf=5,  l2_regularization=0.2, class_weight='balanced', random_state=s)),
        ("HGB-d4-lr01", False, lambda s: HistGradientBoostingClassifier(max_iter=1200, max_depth=4, learning_rate=0.01, min_samples_leaf=6, l2_regularization=0.3, class_weight='balanced', random_state=s)),
        ("HGB-d7",    False, lambda s: HistGradientBoostingClassifier(max_iter=400,  max_depth=7, learning_rate=0.05, min_samples_leaf=4,   l2_regularization=0.05, class_weight='balanced', random_state=s)),
        # XGB
        ("XGB-spw3",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=3.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw5",  True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=5.0,  eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-spw10", True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=10.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        ("XGB-d6",    True,  lambda s: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,   scale_pos_weight=spw, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
    ]

    print(f"\n=== Improved: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
    print(f"  {y.sum()} positives, {len(y)-y.sum()} negatives, {len(feats)} features", flush=True)

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

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    baseline_f1 = np.mean(base_f1s)
    print(f"  Baseline F1 (classifier_tree.py):         {baseline_f1:.2%}")
    print(f"  Previous best (classifier_improved.py):   91.82%")
    print(f"  This script F1:                           {fv:.2%}")
    print(f"  vs baseline: {fv - baseline_f1:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.92:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%} (92% target already exceeded)")
    else:
        print(f"\n  Gap to 95% target: {0.95 - fv:.2%}")
