"""
classifier_v7.py — push Very High risk F1 to 95%+

Strategy: use selected=False (1048 Very High positives vs 255) with the proven
feature set from v5 (6h/7h/8h/10h windows + 10k/20k/50k/100k ranges) and vectorized
numpy computation per model (max 550 users → sub-second per model).

Dropped: cross-model features (zero importance in v6b), stacking (hurts vs mean).
Added: 7h window (top feature in both v6 and v6b), vectorized compute for speed.
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

WINDOWS_MINUTES = [60, 60*4, 60*6, 60*7, 60*8, 60*10, 60*12, 60*24,
                   60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]  # 1,4,6,7,8,10,12,24,168,336,672

ID_RANGES = [10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['10k', '20k', '50k', '100k']
KEY_WH    = {6, 7, 8, 10, 12}    # windows with extra features

FEATURES = (
    [f'log_min_id_diff_{w}h' for w in WINDOW_HOURS] +
    [f'partner_count_{w}h'   for w in WINDOW_HOURS] +
    [f'cic_{tag}_{w}h' for tag in RANGE_TAGS for w in WINDOW_HOURS] +
    [f'cic_frac_{tag}_{w}h' for tag in RANGE_TAGS for w in KEY_WH] +
    [f'log_id_span_{w}h'        for w in KEY_WH] +
    [f'log_min_consec_gap_{w}h' for w in KEY_WH] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num']
)


def _model_norisk_median_vectorized(df):
    """Median nearest-neighbour ID diff for No-risk users, per model."""
    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    records = []
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if nr.empty:
            continue
        all_times_ns = mdf['subscribed_at'].values.astype(np.int64)
        all_ids      = mdf['user_id_num'].values
        all_names    = mdf['user_name'].values
        for t_ns, uid, uname in zip(nr['subscribed_at'].values.astype(np.int64),
                                     nr['user_id_num'].values,
                                     nr['user_name'].values):
            mask = (np.abs(all_times_ns - t_ns) <= max_w_ns) & (all_names != uname)
            if mask.any():
                records.append({'tracking_model_name': model_name,
                                'min_id_diff': np.abs(all_ids[mask] - uid).min()})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).groupby('tracking_model_name')['min_id_diff'].median()


def _cross_model_stats(df):
    user_models   = df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count')
    user_max_risk = df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk')
    return pd.concat([user_models, user_max_risk], axis=1)


def compute_per_user(df):
    """Vectorized per-model numpy feature computation."""
    print("  Precomputing model stats...", flush=True)
    model_norisk_median = _model_norisk_median_vectorized(df)
    print("  Precomputing cross-model stats...", flush=True)
    cross = _cross_model_stats(df)

    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    all_rows = []

    for model_name, mdf in df.groupby('tracking_model_name'):
        n = len(mdf)
        if n == 0:
            continue
        mdf = mdf.reset_index(drop=True)
        times_ns  = mdf['subscribed_at'].values.astype(np.int64)
        ids_arr   = mdf['user_id_num'].values.astype(float)
        usernames = mdf['user_name'].values
        risk_lvls = mdf['risk_level'].values
        risk_scs  = mdf['risk_score'].values

        # Pairwise matrices (n × n) — fast for n ≤ 550
        tdiff  = np.abs(times_ns[:, None] - times_ns[None, :])   # nanoseconds
        iddiff = np.abs(ids_arr[:, None]  - ids_arr[None, :])
        not_self = ~np.eye(n, dtype=bool)

        norisk_med = model_norisk_median.get(model_name, np.nan)

        for i in range(n):
            row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
            any_empty = False

            for w_min, w_h in zip(WINDOWS_MINUTES, WINDOW_HOURS):
                w_ns = int(w_min * 60 * 1e9)
                mask = (tdiff[i] <= w_ns) & not_self[i]

                if not mask.any():
                    row[f'log_min_id_diff_{w_h}h'] = np.nan
                    row[f'partner_count_{w_h}h']   = np.nan
                    for tag in RANGE_TAGS:
                        row[f'cic_{tag}_{w_h}h'] = np.nan
                    if w_h in KEY_WH:
                        for tag in RANGE_TAGS:
                            row[f'cic_frac_{tag}_{w_h}h'] = np.nan
                        row[f'log_id_span_{w_h}h']        = np.nan
                        row[f'log_min_consec_gap_{w_h}h'] = np.nan
                    any_empty = True
                else:
                    diffs = iddiff[i, mask]
                    pids  = ids_arr[mask]
                    pc    = float(mask.sum())
                    row[f'log_min_id_diff_{w_h}h'] = np.log1p(diffs.min())
                    row[f'partner_count_{w_h}h']   = pc
                    for tag, rng in zip(RANGE_TAGS, ID_RANGES):
                        cic = float((diffs <= rng).sum())
                        row[f'cic_{tag}_{w_h}h'] = cic
                        if w_h in KEY_WH:
                            row[f'cic_frac_{tag}_{w_h}h'] = cic / (pc + 1)
                    if w_h in KEY_WH:
                        row[f'log_id_span_{w_h}h'] = np.log1p(float(pids.max() - pids.min()))
                        row[f'log_min_consec_gap_{w_h}h'] = (
                            np.log1p(float(np.diff(np.sort(pids)).min())) if pc >= 2 else np.nan
                        )

            # Forward-fill NaN from next larger window
            if any_empty:
                for j in range(len(WINDOWS_MINUTES) - 1):
                    wh, nwh = WINDOW_HOURS[j], WINDOW_HOURS[j + 1]
                    for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in RANGE_TAGS]:
                        k, nk = f'{base}_{wh}h', f'{base}_{nwh}h'
                        if np.isnan(row.get(k, np.nan)):
                            row[k] = row.get(nk, np.nan)

            # Min time diff to any partner in max window
            all_mask = (tdiff[i] <= max_w_ns) & not_self[i]
            if all_mask.any():
                row['log_min_time_diff'] = np.log1p(tdiff[i, all_mask].min() / 1e9 / 60)
            else:
                row['log_min_time_diff'] = np.nan

            row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
            raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
            row['rel_min_id_diff_24h'] = (raw_24h / norisk_med
                                          if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0
                                          else np.nan)
            uname = usernames[i]
            row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
            gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else risk_scs[i]
            row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
            row['log_user_id_num'] = np.log1p(ids_arr[i]) if not np.isnan(ids_arr[i]) else np.nan

            all_rows.append(row)

    return pd.DataFrame(all_rows)


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
    print("Loading full dataset (selected=False)...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    print(f"  {len(df)} rows", flush=True)

    per_user = compute_per_user(df)
    print(f"Total per-user rows: {len(per_user)}", flush=True)

    levels = ['Very High']
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)

    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values
    y = subset['is_risky'].astype(int).values
    print(f"Dataset: {y.sum()} positives, {len(y)-y.sum()} negatives, {len(feats)} features", flush=True)

    SEEDS   = [42, 7, 123, 999, 2025, 1337, 2024, 101]
    N_FOLDS = 10   # fewer folds since we have 4x more positives

    spw = (len(y) - y.sum()) / y.sum()

    configs = [
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

    Xf, _ = nan_fill(X, X)
    xgb_imp = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                             eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb_imp.fit(Xf, y)
    importances = sorted(zip(feats, xgb_imp.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop-15 feature importances:")
    for feat, imp in importances[:15]:
        print(f"  {feat:<40} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  selected=True best (v5):        93.28%  (255 positives)")
    print(f"  selected=False this run:         {fv:.2%}  ({y.sum()} positives)")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.92:
        print(f"\n  Above 92%! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%}")
        print(f"  Note: selected=False brings harder negatives — if still improving, try larger ensemble")
