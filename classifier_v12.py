"""
classifier_v12.py — partner risk label features (OOF-clean)

New idea: use the known risk labels of co-registering partners as features.
A user registering alongside 10 confirmed VH bots is suspicious even if
their own ID spread looks benign (the type-3 false negatives).

"partner_vh_count_8h" = count of VH-labeled TRAINING FOLD partners in 8h window
"partner_max_risk_8h" = max risk_score among training-fold partners in 8h window

Implemented OOF-clean: for each validation user, we only use partner labels
from the training fold (not from the validation fold itself). With 15-fold CV,
~93% of partners are in the training fold — minimal information loss.

This is valid in production: historical risk labels are known when classifying
new users.
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

ID_RANGES  = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['1k', '5k', '10k', '20k', '50k', '100k']

WINDOWS_MINUTES = [60, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]
KEY_WINDOWS_H   = {6, 8, 10, 12, 24}

PARTNER_WINDOWS_H = [4, 6, 8, 12, 24]

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

PARTNER_RISK_FEATURES = (
    [f'partner_vh_count_{w}h'   for w in PARTNER_WINDOWS_H] +
    [f'partner_max_risk_{w}h'   for w in PARTNER_WINDOWS_H] +
    [f'partner_mean_risk_{w}h'  for w in PARTNER_WINDOWS_H]
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
        # Store metadata needed for partner lookup
        row['subscribed_at'] = t
        row['tracking_model_name'] = model
        rows.append(row)

    per_user = pd.DataFrame(rows)
    per_user['cic_ratio_6h_24h']    = per_user['cic_50k_6h']  / (per_user['cic_50k_24h']  + 1)
    per_user['cic10k_ratio_6h_12h'] = per_user['cic_10k_6h']  / (per_user['cic_10k_12h']  + 1)
    per_user['pc_ratio_6h_24h']     = per_user['partner_count_6h'] / (per_user['partner_count_24h'] + 1)
    return per_user


def precompute_partner_lookup(subset):
    """
    For each row in subset, store the indices (in subset) of its partners
    at each PARTNER_WINDOWS_H window.
    Returns: list of dicts {w_h: np.array of partner indices in subset}
    """
    print("  Precomputing partner lookup tables...", flush=True)
    n = len(subset)

    # Build arrays for vectorized ops per model
    sub_times  = subset['subscribed_at'].values  # datetime64
    sub_models = subset['tracking_model_name'].values
    sub_names  = subset['user_name'].values
    sub_reset  = np.arange(n)

    # Result: for each user i, for each window, store partner indices in subset
    partner_idxs = [{w: [] for w in PARTNER_WINDOWS_H} for _ in range(n)]

    # Process model by model to reduce comparisons
    for model in np.unique(sub_models):
        model_mask = sub_models == model
        model_idxs = sub_reset[model_mask]
        model_times = sub_times[model_mask]
        model_names = sub_names[model_mask]

        n_m = len(model_idxs)
        if n_m < 2:
            continue

        # Compute pairwise time differences (seconds)
        t_ns = model_times.astype('int64')  # nanoseconds
        for i_local in range(n_m):
            i_global = model_idxs[i_local]
            dt_s = np.abs(t_ns - t_ns[i_local]) / 1e9
            for w_h in PARTNER_WINDOWS_H:
                w_s = w_h * 3600.0
                in_window = (dt_s <= w_s) & (model_names != model_names[i_local])
                partner_idxs[i_global][w_h] = model_idxs[in_window]

    return partner_idxs


def compute_partner_risk_feats(indices, partner_idxs, train_set, y_risk_scores):
    """
    For each user index in `indices`, compute partner risk features using
    only the partners in `train_set` (a set of trusted indices).
    Returns array of shape (len(indices), len(PARTNER_WINDOWS_H)*3)
    """
    rows = []
    for i in indices:
        row = []
        for w_h in PARTNER_WINDOWS_H:
            p_idxs = partner_idxs[i][w_h]
            # Only use training-set partners (OOF-clean)
            p_idxs_tr = [j for j in p_idxs if j in train_set]
            if p_idxs_tr:
                risks = y_risk_scores[p_idxs_tr]
                vh_count = float((risks >= 4).sum())  # VH=4, Extreme=5
                max_risk  = float(risks.max())
                mean_risk = float(risks.mean())
            else:
                vh_count = 0.0
                max_risk  = 1.0  # No risk baseline
                mean_risk = 1.0
            row.extend([vh_count, max_risk, mean_risk])
        rows.append(row)
    return np.array(rows, dtype=float)


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


def accumulate_oof_with_partner_risk(X, y, y_risk_scores, partner_idxs,
                                      make_fn, seeds, n_folds, needs_nan_fill=False):
    pool = np.zeros(len(y))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(len(y))
        for ti, vi in cv.split(X, y):
            train_set = set(ti)
            # Compute partner risk features using training fold labels
            prf_tr = compute_partner_risk_feats(ti, partner_idxs, train_set, y_risk_scores)
            prf_va = compute_partner_risk_feats(vi, partner_idxs, train_set, y_risk_scores)

            X_tr = np.hstack([X[ti], prf_tr])
            X_va = np.hstack([X[vi], prf_va])

            if needs_nan_fill:
                X_tr, X_va = nan_fill(X_tr, X_va)
            m = make_fn(seed)
            m.fit(X_tr, y[ti])
            proba[vi] = m.predict_proba(X_va)[:, 1]
        pool += proba
    return pool / len(seeds)


if __name__ == '__main__':
    df = fetch_df(selected=True)
    df = clean(df)

    per_user = compute_per_user(df)

    levels = ['Very High']
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy().reset_index(drop=True)
    subset['is_risky'] = subset['risk_level'].isin(levels)

    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values
    y = subset['is_risky'].astype(int).values
    # Risk scores for OOF-clean partner lookup (using RISK_MAP values)
    from base import RISK_MAP
    subset['risk_score_v'] = subset['risk_level'].map(RISK_MAP).fillna(1).astype(float)
    y_risk_scores = subset['risk_score_v'].values

    print(f"Dataset: {y.sum()} positives, {(y==0).sum()} negatives, {len(feats)} features", flush=True)

    # Precompute partner indices (within subset, for VH+No risk rows only)
    partner_idxs = precompute_partner_lookup(subset)

    # Quick sanity check: partner_vh_count distribution
    print("\nPartner VH count stats (8h, using full labels):", flush=True)
    full_prf = compute_partner_risk_feats(range(len(subset)), partner_idxs, set(range(len(subset))), y_risk_scores)
    vh8h_idx = PARTNER_WINDOWS_H.index(8)
    vh_counts_8h = full_prf[:, vh8h_idx * 3]  # vh_count column for 8h
    print(f"  VH users — mean partner_vh_count_8h: {vh_counts_8h[y==1].mean():.2f}, median: {np.median(vh_counts_8h[y==1]):.2f}", flush=True)
    print(f"  No risk  — mean partner_vh_count_8h: {vh_counts_8h[y==0].mean():.2f}, median: {np.median(vh_counts_8h[y==0]):.2f}", flush=True)

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

    print(f"\n=== v12 Ensemble: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
    print(f"    (with OOF-clean partner risk features: {len(PARTNER_RISK_FEATURES)} extra feats)", flush=True)
    all_probas = []
    for name, needs_fill, cfg_fn in configs:
        p = accumulate_oof_with_partner_risk(
            X, y, y_risk_scores, partner_idxs, cfg_fn, SEEDS, N_FOLDS, needs_nan_fill=needs_fill)
        all_probas.append(p)
        print(f"  Done: {name}", flush=True)

    y_proba = np.mean(all_probas, axis=0)
    best_thresh = find_best_threshold(y, y_proba)
    y_pred = (y_proba >= best_thresh).astype(int)
    pv, rv, fv, _ = precision_recall_fscore_support(y, y_pred, pos_label=1, average='binary', zero_division=0)

    print(f"\n{'='*60}")
    print(f"v12 Ensemble ({len(all_probas)} configs, opt threshold={best_thresh:.3f}):")
    print(f"  Precision: {pv:.2%}  Recall: {rv:.2%}  F1: {fv:.2%}")

    print(f"\nThreshold sweep:")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for thresh in np.arange(0.3, 0.9, 0.05):
        yt = (y_proba >= thresh).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y, yt, pos_label=1, average='binary', zero_division=0)
        print(f"  {thresh:>7.2f}  {pt:>9.2%}  {rt:>6.2%}  {ft:>6.2%}")

    # Feature importance with partner risk features
    all_feats = feats + PARTNER_RISK_FEATURES
    X_aug = np.hstack([X, full_prf])
    Xf, _ = nan_fill(X_aug, X_aug)
    xgb_imp = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                             eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb_imp.fit(Xf, y)
    importances = sorted(zip(all_feats, xgb_imp.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop-20 feature importances (XGB):")
    for feat, imp in importances[:20]:
        print(f"  {feat:<40} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  v5 best:   93.28%")
    print(f"  v12 F1:    {fv:.2%}")
    print(f"  vs v5: {fv - 0.9328:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.93:
        print(f"\n  Above 93%! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95%: {0.95 - fv:.2%}")
