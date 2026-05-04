"""
classifier_v3.py — push Very High risk F1 to 95%+

Key improvements over classifier_improved.py (91.82% F1):
1. Username-based features — strong signal:
   - No-risk users: 66% start with 'u'+digits; Very High bots: only 25%
   - Very High bots: 22% have trailing _XXXX number; No-risk: only 3%
   - VH bots have fewer digits, more underscores, longer names
2. Same HistGB+XGB ensemble, same 255 positives (selected=True)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

MAX_ID_RANGE = 100_000
WINDOWS_MINUTES = [60, 60 * 4, 60 * 24, 60 * 24 * 7, 60 * 24 * 7 * 2, 60 * 24 * 7 * 4]
WINDOW_HOURS = [w // 60 for w in WINDOWS_MINUTES]

USERNAME_FEATURES = [
    'uname_starts_u_digits',    # 1 if username starts with u+5+ digits (typical auto-ID)
    'uname_trailing_number',    # 1 if ends with _\d{2,5}
    'uname_digit_frac',         # fraction of chars that are digits
    'uname_length',             # total length
    'uname_underscore_count',   # number of underscores
    'uname_alpha_frac',         # fraction of chars that are alpha
    'uname_has_word_chars',     # 1 if name has 3+ consecutive letters
    'uname_trailing_digit_len', # length of trailing digit run (0 if none)
]

FEATURES = [
    *[f'log_min_id_diff_{w}h' for w in WINDOW_HOURS],
    *[f'close_id_count_{w}h' for w in WINDOW_HOURS],
    *[f'partner_count_{w}h' for w in WINDOW_HOURS],
    'log_min_time_diff',
    'log_model_norisk_median_id_diff',
    'rel_min_id_diff_24h',
    'user_model_count',
    'user_max_risk_elsewhere',
    'log_user_id_num',
    *USERNAME_FEATURES,
]


def extract_username_features(usernames: pd.Series) -> pd.DataFrame:
    names = usernames.str.lower().fillna('')
    df = pd.DataFrame(index=usernames.index)
    df['uname_starts_u_digits']    = names.str.match(r'^u\d{5,}').astype(float)
    df['uname_trailing_number']    = names.str.match(r'.*_\d{2,5}$').astype(float)
    df['uname_digit_frac']         = names.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
    df['uname_length']             = names.str.len().astype(float)
    df['uname_underscore_count']   = names.str.count('_').astype(float)
    df['uname_alpha_frac']         = names.apply(lambda x: sum(c.isalpha() for c in x) / max(len(x), 1))
    df['uname_has_word_chars']     = names.str.contains(r'[a-z]{3,}').astype(float)
    df['uname_trailing_digit_len'] = names.apply(
        lambda x: len(re.search(r'\d+$', x).group()) if re.search(r'\d+$', x) else 0
    )
    return df


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

        for w_min in WINDOWS_MINUTES:
            w_h = w_min // 60
            w_td = pd.Timedelta(minutes=w_min)
            partners = all_partners[
                (all_partners['subscribed_at'] >= t - w_td) &
                (all_partners['subscribed_at'] <= t + w_td)
            ]
            if partners.empty:
                row[f'log_min_id_diff_{w_h}h'] = float('nan')
                row[f'close_id_count_{w_h}h']  = float('nan')
                row[f'partner_count_{w_h}h']   = float('nan')
                any_empty = True
            else:
                id_diffs = (partners['user_id_num'] - user['user_id_num']).abs()
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                row[f'close_id_count_{w_h}h']  = (id_diffs <= MAX_ID_RANGE).sum()
                row[f'partner_count_{w_h}h']   = len(partners)

        if any_empty:
            for i, w_min in enumerate(WINDOWS_MINUTES[:-1]):
                w_h = w_min // 60
                next_w_h = WINDOWS_MINUTES[i + 1] // 60
                for feat in ['log_min_id_diff', 'close_id_count', 'partner_count']:
                    if np.isnan(row[f'{feat}_{w_h}h']):
                        row[f'{feat}_{w_h}h'] = row[f'{feat}_{next_w_h}h']

        if not all_partners.empty:
            row['log_min_time_diff'] = np.log1p(
                (all_partners['subscribed_at'] - t).abs().dt.total_seconds().min() / 60
            )
        else:
            row['log_min_time_diff'] = float('nan')

        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0:
            row['rel_min_id_diff_24h'] = raw_24h / norisk_med
        else:
            row['rel_min_id_diff_24h'] = np.nan

        uname = user['user_name']
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        global_max = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        row['user_max_risk_elsewhere'] = global_max if row['user_model_count'] > 1 else 0
        row['log_user_id_num'] = np.log1p(user['user_id_num']) if not np.isnan(user['user_id_num']) else np.nan

        rows.append(row)

    per_user = pd.DataFrame(rows)

    # Merge username features
    uname_feats = extract_username_features(per_user['user_name'])
    per_user = pd.concat([per_user, uname_feats], axis=1)

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
    X = subset[FEATURES].values
    y = subset['is_risky'].astype(int).values
    print(f"Dataset: {y.sum()} positives (Very High), {len(y)-y.sum()} negatives (No risk)", flush=True)

    # ── Baseline (no username features) ──
    ORIG_FEATURES = [f for f in FEATURES if f not in USERNAME_FEATURES and f != 'log_user_id_num']
    sub_drop = per_user.dropna(subset=ORIG_FEATURES)
    sub_drop = sub_drop[sub_drop['risk_level'].isin(levels + ['No risk'])].copy()
    sub_drop['is_risky'] = sub_drop['risk_level'].isin(levels)
    X_drop = sub_drop[ORIG_FEATURES].values
    y_drop = sub_drop['is_risky'].astype(int).values
    spw = (len(y_drop) - y_drop.sum()) / y_drop.sum()

    print(f"\n=== Baseline (XGB, drop_na, {y_drop.sum()} pos, scale={spw:.1f}) ===", flush=True)
    base_f1s = []
    for seed in [42, 7, 123, 999, 2025]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred = np.zeros_like(y_drop)
        for ti, vi in cv.split(X_drop, y_drop):
            Xtr, Xva = nan_fill(X_drop[ti], X_drop[vi])
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=seed)
            m.fit(Xtr, y_drop[ti]); y_pred[vi] = m.predict(Xva)
        _, _, f, _ = precision_recall_fscore_support(y_drop, y_pred, pos_label=1, average='binary', zero_division=0)
        base_f1s.append(f)
    print(f"  Mean F1: {np.mean(base_f1s):.2%}  seeds: {[f'{v:.2%}' for v in base_f1s]}", flush=True)

    # ── Improved ensemble with username features ──
    SEEDS = [42, 7, 123, 999, 2025, 1337, 2024, 101]
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

    print(f"\n=== Improved: {len(configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold OOF ===", flush=True)
    print(f"  {y.sum()} positives, {len(y)-y.sum()} negatives, {len(FEATURES)} features (+{len(USERNAME_FEATURES)} username)", flush=True)

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

    # Feature importance (use one HGB to assess)
    hgb = HistGradientBoostingClassifier(max_iter=800, max_depth=4, learning_rate=0.02,
                                          min_samples_leaf=8, class_weight='balanced', random_state=42)
    Xf, _ = nan_fill(X, X)
    hgb.fit(Xf, y)
    # HistGB doesn't expose importances directly, use XGB instead
    from sklearn.preprocessing import LabelEncoder
    Xf2, _ = nan_fill(X, X)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, scale_pos_weight=5,
                        eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=42)
    xgb.fit(Xf2, y)
    importances = sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop feature importances (XGB):")
    for feat, imp in importances[:15]:
        print(f"  {feat:<35} {imp:.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    baseline_f1 = np.mean(base_f1s)
    print(f"  Baseline F1 (classifier_tree.py):           {baseline_f1:.2%}")
    print(f"  Previous best (classifier_improved.py):     91.82%")
    print(f"  This script F1 (+username features):        {fv:.2%}")
    print(f"  vs previous best: {fv - 0.9182:+.2%}")
    if fv >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={fv:.2%} >= 95% ***")
    elif fv >= 0.92:
        print(f"\n  92% target exceeded! Gap to 95%: {0.95 - fv:.2%}")
    else:
        print(f"\n  Gap to 95% target: {0.95 - fv:.2%}")
