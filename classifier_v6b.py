"""
classifier_v6b.py — v6 with fixed cross-model join (no duplicate rows).

Bug in v6: xm_feats indexed by user_name → join multiplied rows for users with
multiple subscriptions (11326 → 12572 rows, 255 → 355 positives). Fixed by
computing cross-model features inline in the main loop using pre-sorted arrays
(searchsorted for efficient window lookup), so each row in per_user gets exactly
the cross-model count for that specific subscription event's time window.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

INTRA_WINDOWS_H = [1, 4, 6, 7, 8, 9, 10, 12, 24, 168, 336, 672]
CROSS_WINDOWS_H = [6, 8, 10, 12]
INTRA_RANGES    = [10_000, 20_000, 50_000, 100_000]
INTRA_TAGS      = ['10k', '20k', '50k', '100k']
CROSS_RANGES    = [10_000, 50_000]
CROSS_TAGS      = ['10k', '50k']
KEY_WINDOWS_H   = {6, 7, 8, 9, 10, 12}

FEATURES = (
    [f'log_min_id_diff_{w}h' for w in INTRA_WINDOWS_H] +
    [f'partner_count_{w}h'   for w in INTRA_WINDOWS_H] +
    [f'cic_{tag}_{w}h' for tag in INTRA_TAGS for w in INTRA_WINDOWS_H] +
    [f'cic_frac_{tag}_{w}h' for tag in INTRA_TAGS for w in KEY_WINDOWS_H] +
    [f'log_id_span_{w}h'        for w in KEY_WINDOWS_H] +
    [f'log_min_consec_gap_{w}h' for w in KEY_WINDOWS_H] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num'] +
    # cross-model features (no model filter)
    [f'xm_cic_{tag}_{w}h'      for tag in CROSS_TAGS for w in CROSS_WINDOWS_H] +
    [f'xm_pc_{w}h'             for w in CROSS_WINDOWS_H] +
    [f'xm_cic_frac_{tag}_{w}h' for tag in CROSS_TAGS for w in CROSS_WINDOWS_H]
)


def _precompute_model_stats(df):
    max_window = pd.Timedelta(hours=max(INTRA_WINDOWS_H))
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

    # Pre-sort all rows by time for efficient cross-model searchsorted lookup
    df_sorted = df.sort_values('subscribed_at').reset_index(drop=True)
    xm_times_ns = df_sorted['subscribed_at'].values.astype(np.int64)
    xm_ids      = df_sorted['user_id_num'].values
    xm_names    = df_sorted['user_name'].values

    max_window = pd.Timedelta(hours=max(INTRA_WINDOWS_H))
    times = df['subscribed_at']
    rows = []

    for _, user in df.iterrows():
        t      = user['subscribed_at']
        model  = user['tracking_model_name']
        uname  = user['user_name']
        uid    = user['user_id_num']
        t_ns   = np.int64(t.value)

        base_mask = (
            (times >= t - max_window) & (times <= t + max_window) &
            (df['tracking_model_name'] == model) &
            (df['user_name'] != uname)
        )
        all_partners = df[base_mask]

        row = {'user_name': uname, 'risk_level': user['risk_level']}
        any_empty = False

        # ── Intra-model features ──
        for w_h in INTRA_WINDOWS_H:
            w_td = pd.Timedelta(hours=w_h)
            partners = all_partners[
                (all_partners['subscribed_at'] >= t - w_td) &
                (all_partners['subscribed_at'] <= t + w_td)
            ]
            if partners.empty:
                row[f'log_min_id_diff_{w_h}h'] = np.nan
                row[f'partner_count_{w_h}h']   = np.nan
                for tag in INTRA_TAGS:
                    row[f'cic_{tag}_{w_h}h'] = np.nan
                if w_h in KEY_WINDOWS_H:
                    for tag in INTRA_TAGS:
                        row[f'cic_frac_{tag}_{w_h}h'] = np.nan
                    row[f'log_id_span_{w_h}h']        = np.nan
                    row[f'log_min_consec_gap_{w_h}h'] = np.nan
                any_empty = True
            else:
                id_diffs    = (partners['user_id_num'] - uid).abs().values
                partner_ids = partners['user_id_num'].values
                pc = len(partners)
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                row[f'partner_count_{w_h}h']   = float(pc)
                for tag, rng in zip(INTRA_TAGS, INTRA_RANGES):
                    cic = float((id_diffs <= rng).sum())
                    row[f'cic_{tag}_{w_h}h'] = cic
                    if w_h in KEY_WINDOWS_H:
                        row[f'cic_frac_{tag}_{w_h}h'] = cic / (pc + 1)
                if w_h in KEY_WINDOWS_H:
                    row[f'log_id_span_{w_h}h'] = np.log1p(float(partner_ids.max() - partner_ids.min()))
                    row[f'log_min_consec_gap_{w_h}h'] = (
                        np.log1p(float(np.diff(np.sort(partner_ids)).min())) if pc >= 2 else np.nan
                    )

        if any_empty:
            for i in range(len(INTRA_WINDOWS_H) - 1):
                wh, nwh = INTRA_WINDOWS_H[i], INTRA_WINDOWS_H[i + 1]
                for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in INTRA_TAGS]:
                    k, nk = f'{base}_{wh}h', f'{base}_{nwh}h'
                    if np.isnan(row.get(k, np.nan)):
                        row[k] = row.get(nk, np.nan)

        if not all_partners.empty:
            row['log_min_time_diff'] = np.log1p(
                (all_partners['subscribed_at'] - t).abs().dt.total_seconds().min() / 60
            )
        else:
            row['log_min_time_diff'] = np.nan

        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        row['rel_min_id_diff_24h'] = (raw_24h / norisk_med
                                      if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0
                                      else np.nan)
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
        row['log_user_id_num'] = np.log1p(uid) if not np.isnan(uid) else np.nan

        # ── Cross-model features (all models, searchsorted on pre-sorted array) ──
        for w_h in CROSS_WINDOWS_H:
            w_ns = int(w_h * 3600 * 1e9)
            lo = int(np.searchsorted(xm_times_ns, t_ns - w_ns))
            hi = int(np.searchsorted(xm_times_ns, t_ns + w_ns, side='right'))
            mask = xm_names[lo:hi] != uname
            xm_partner_ids = xm_ids[lo:hi][mask]
            pc_xm = len(xm_partner_ids)
            row[f'xm_pc_{w_h}h'] = float(pc_xm)
            if pc_xm > 0:
                xm_diffs = np.abs(xm_partner_ids - uid)
                for tag, rng in zip(CROSS_TAGS, CROSS_RANGES):
                    cic = float((xm_diffs <= rng).sum())
                    row[f'xm_cic_{tag}_{w_h}h']      = cic
                    row[f'xm_cic_frac_{tag}_{w_h}h'] = cic / (pc_xm + 1)
            else:
                for tag in CROSS_TAGS:
                    row[f'xm_cic_{tag}_{w_h}h']      = 0.0
                    row[f'xm_cic_frac_{tag}_{w_h}h'] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


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
    print(f"Total rows: {len(per_user)} (should be {len(df)})", flush=True)
    assert len(per_user) == len(df), "Row count mismatch — join bug not fixed!"

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

    print(f"\n=== Base ensemble: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
    all_probas = []
    for name, needs_fill, cfg_fn in configs:
        p = accumulate_oof(X, y, cfg_fn, SEEDS, N_FOLDS, needs_nan_fill=needs_fill)
        all_probas.append(p)
        print(f"  Done: {name}", flush=True)

    # ── Mean ensemble ──
    y_proba_mean = np.mean(all_probas, axis=0)
    thresh_mean  = find_best_threshold(y, y_proba_mean)
    y_pred_mean  = (y_proba_mean >= thresh_mean).astype(int)
    pm, rm, fm, _ = precision_recall_fscore_support(y, y_pred_mean, pos_label=1, average='binary', zero_division=0)
    print(f"\nMean ensemble (thresh={thresh_mean:.3f}): P={pm:.2%} R={rm:.2%} F1={fm:.2%}", flush=True)

    # ── Stacking: logistic regression on OOF probas ──
    X_meta = np.column_stack(all_probas)
    y_meta_proba = np.zeros(len(y))
    for ti, vi in StratifiedKFold(n_splits=15, shuffle=True, random_state=42).split(X_meta, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_meta[ti])
        X_va = scaler.transform(X_meta[vi])
        meta = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
        meta.fit(X_tr, y[ti])
        y_meta_proba[vi] = meta.predict_proba(X_va)[:, 1]
    thresh_stack = find_best_threshold(y, y_meta_proba)
    y_pred_stack = (y_meta_proba >= thresh_stack).astype(int)
    ps, rs, fs, _ = precision_recall_fscore_support(y, y_pred_stack, pos_label=1, average='binary', zero_division=0)
    print(f"Stacked (thresh={thresh_stack:.3f}):      P={ps:.2%} R={rs:.2%} F1={fs:.2%}", flush=True)

    fv = max(fm, fs)
    best_proba = y_meta_proba if fs >= fm else y_proba_mean
    best_thresh = thresh_stack if fs >= fm else thresh_mean

    print(f"\nThreshold sweep (stacked probas):")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for thresh in np.arange(0.3, 0.9, 0.05):
        yt = (y_meta_proba >= thresh).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y, yt, pos_label=1, average='binary', zero_division=0)
        print(f"  {thresh:>7.2f}  {pt:>9.2%}  {rt:>6.2%}  {ft:>6.2%}")

    # Feature importance via XGB
    Xf, _ = nan_fill(X, X)
    xgb_imp = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                             eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb_imp.fit(Xf, y)
    importances = sorted(zip(feats, xgb_imp.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop-15 feature importances:")
    for feat, imp in importances[:15]:
        print(f"  {feat:<48} {imp:.4f}")

    # Cross-model feature importance specifically
    xm_imps = [(f, i) for f, i in importances if f.startswith('xm_')]
    print(f"\nCross-model feature importances (top 5):")
    for feat, imp in xm_imps[:5]:
        print(f"  {feat:<48} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  v5 best:                    93.28%")
    print(f"  Mean ensemble:              {fm:.2%}")
    print(f"  Stacked:                    {fs:.2%}")
    print(f"  Best:                       {fv:.2%}")
    print(f"  vs v5: {fv - 0.9328:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.92:
        print(f"\n  Above 92%! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%}")
