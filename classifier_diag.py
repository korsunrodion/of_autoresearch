"""
Diagnostic: identify the OOF false negatives from the v5 ensemble.
Outputs feature values for Very High users sorted by predicted probability,
so we can see what the missed bots look like vs the correctly caught ones.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

MAX_ID_RANGE = 100_000
WINDOWS_MINUTES = [60, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12, 60*24,
                   60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS = [w // 60 for w in WINDOWS_MINUTES]
ID_RANGES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['1k', '5k', '10k', '20k', '50k', '100k']
KEY_WH = {6, 8, 10, 12, 24}

FEATURES = (
    [f'log_min_id_diff_{w}h' for w in WINDOW_HOURS] +
    [f'partner_count_{w}h'   for w in WINDOW_HOURS] +
    [f'cic_{tag}_{w}h' for tag in RANGE_TAGS for w in WINDOW_HOURS] +
    [f'cic_frac_{tag}_{w}h' for tag in RANGE_TAGS for w in KEY_WH] +
    [f'log_id_span_{w}h'        for w in KEY_WH] +
    [f'log_id_std_{w}h'         for w in KEY_WH] +
    [f'log_min_consec_gap_{w}h' for w in KEY_WH] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num']
)


def _precompute_model_stats(df):
    max_window = pd.Timedelta(minutes=max(WINDOWS_MINUTES))
    times = df['subscribed_at']
    records = []
    for _, user in df[df['risk_level'] == 'No risk'].iterrows():
        t = user['subscribed_at']
        mask = ((times >= t - max_window) & (times <= t + max_window) &
                (df['tracking_model_name'] == user['tracking_model_name']) &
                (df['user_name'] != user['user_name']))
        partners = df[mask]
        if partners.empty: continue
        records.append({'tracking_model_name': user['tracking_model_name'],
                        'min_id_diff': (partners['user_id_num'] - user['user_id_num']).abs().min()})
    if not records: return pd.Series(dtype=float)
    return pd.DataFrame(records).groupby('tracking_model_name')['min_id_diff'].median()


def _precompute_cross_model(df):
    return pd.concat([
        df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count'),
        df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk')
    ], axis=1)


def compute_per_user(df):
    model_norisk_median = _precompute_model_stats(df)
    cross = _precompute_cross_model(df)
    max_window = pd.Timedelta(minutes=max(WINDOWS_MINUTES))
    times = df['subscribed_at']
    rows = []
    for _, user in df.iterrows():
        t, model, uname, uid = user['subscribed_at'], user['tracking_model_name'], user['user_name'], user['user_id_num']
        base_mask = ((times >= t - max_window) & (times <= t + max_window) &
                     (df['tracking_model_name'] == model) & (df['user_name'] != uname))
        all_partners = df[base_mask]
        row = {'user_name': uname, 'risk_level': user['risk_level']}
        any_empty = False
        for w_min, w_h in zip(WINDOWS_MINUTES, WINDOW_HOURS):
            partners = all_partners[(all_partners['subscribed_at'] >= t - pd.Timedelta(minutes=w_min)) &
                                    (all_partners['subscribed_at'] <= t + pd.Timedelta(minutes=w_min))]
            if partners.empty:
                row[f'log_min_id_diff_{w_h}h'] = np.nan
                row[f'partner_count_{w_h}h'] = np.nan
                for tag in RANGE_TAGS: row[f'cic_{tag}_{w_h}h'] = np.nan
                if w_h in KEY_WH:
                    for tag in RANGE_TAGS: row[f'cic_frac_{tag}_{w_h}h'] = np.nan
                    for k in ['log_id_span', 'log_id_std', 'log_min_consec_gap']:
                        row[f'{k}_{w_h}h'] = np.nan
                any_empty = True
            else:
                id_diffs = (partners['user_id_num'] - uid).abs().values
                pids = partners['user_id_num'].values
                pc = len(partners)
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                row[f'partner_count_{w_h}h'] = float(pc)
                for tag, rng in zip(RANGE_TAGS, ID_RANGES):
                    cic = float((id_diffs <= rng).sum())
                    row[f'cic_{tag}_{w_h}h'] = cic
                    if w_h in KEY_WH: row[f'cic_frac_{tag}_{w_h}h'] = cic / (pc + 1)
                if w_h in KEY_WH:
                    row[f'log_id_span_{w_h}h'] = np.log1p(float(pids.max() - pids.min()))
                    row[f'log_id_std_{w_h}h']  = np.log1p(float(np.std(pids)))
                    row[f'log_min_consec_gap_{w_h}h'] = np.log1p(float(np.diff(np.sort(pids)).min())) if pc >= 2 else np.nan
        if any_empty:
            for i in range(len(WINDOWS_MINUTES) - 1):
                wh, nwh = WINDOW_HOURS[i], WINDOW_HOURS[i+1]
                for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in RANGE_TAGS]:
                    k, nk = f'{base}_{wh}h', f'{base}_{nwh}h'
                    if np.isnan(row.get(k, np.nan)): row[k] = row.get(nk, np.nan)
        if not all_partners.empty:
            row['log_min_time_diff'] = np.log1p((all_partners['subscribed_at'] - t).abs().dt.total_seconds().min() / 60)
        else:
            row['log_min_time_diff'] = np.nan
        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        row['rel_min_id_diff_24h'] = (raw_24h / norisk_med if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0 else np.nan)
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
        row['log_user_id_num'] = np.log1p(uid) if not np.isnan(uid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def nan_fill(Xtr, Xva):
    med = np.nanmedian(Xtr, axis=0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xva), med, Xva)


def accumulate_oof(X, y, make_fn, seeds, n_folds, needs_nan_fill=False):
    pool = np.zeros(len(y))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(len(y))
        for ti, vi in cv.split(X, y):
            Xtr, Xva = (nan_fill(X[ti], X[vi]) if needs_nan_fill else (X[ti], X[vi]))
            m = make_fn(seed); m.fit(Xtr, y[ti]); proba[vi] = m.predict_proba(Xva)[:, 1]
        pool += proba
    return pool / len(seeds)


if __name__ == '__main__':
    df = fetch_df(selected=True); df = clean(df)
    per_user = compute_per_user(df)

    levels = ['Very High']
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)
    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values; y = subset['is_risky'].astype(int).values

    SEEDS = [42, 7, 123, 999, 2025, 1337, 2024, 101]; N_FOLDS = 15
    configs = [
        (False, lambda s: HistGradientBoostingClassifier(max_iter=800, max_depth=4, learning_rate=0.02, min_samples_leaf=8, l2_regularization=0.5, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=600, max_depth=6, learning_rate=0.03, min_samples_leaf=5, l2_regularization=0.1, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=3, learning_rate=0.01, min_samples_leaf=10, l2_regularization=1.0, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=500, max_depth=5, learning_rate=0.05, min_samples_leaf=5, l2_regularization=0.2, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=1000, max_depth=5, learning_rate=0.02, min_samples_leaf=3, l2_regularization=0.3, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=800, max_leaf_nodes=31, learning_rate=0.02, min_samples_leaf=8, l2_regularization=0.5, class_weight='balanced', random_state=s)),
        (False, lambda s: HistGradientBoostingClassifier(max_iter=600, max_leaf_nodes=15, learning_rate=0.03, min_samples_leaf=5, l2_regularization=0.2, class_weight='balanced', random_state=s)),
        (True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=3.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        (True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=5.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
        (True,  lambda s: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=10.0, eval_metric='logloss', verbosity=0, n_jobs=-1, random_state=s)),
    ]

    print("Running v5 ensemble for diagnostics...", flush=True)
    all_probas = [accumulate_oof(X, y, fn, SEEDS, N_FOLDS, nf) for nf, fn in configs]
    y_proba = np.mean(all_probas, axis=0)

    # Find optimal threshold
    prec, rec, thresholds = precision_recall_curve(y, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_thresh = float(thresholds[f1s.argmax()]) if f1s.argmax() < len(thresholds) else 0.5
    y_pred = (y_proba >= best_thresh).astype(int)
    pv, rv, fv, _ = precision_recall_fscore_support(y, y_pred, pos_label=1, average='binary', zero_division=0)
    print(f"F1={fv:.2%} P={pv:.2%} R={rv:.2%} thresh={best_thresh:.3f}")

    # Analyse VH users
    vh_mask = y == 1
    vh_proba = y_proba[vh_mask]
    vh_feats = X[vh_mask]
    vh_names = subset[subset['is_risky'] == True]['user_name'].values

    # Key features for analysis
    key_feats = ['cic_10k_8h', 'cic_20k_6h', 'cic_10k_6h', 'partner_count_8h',
                 'log_min_id_diff_8h', 'log_min_consec_gap_8h', 'user_model_count',
                 'user_max_risk_elsewhere', 'log_user_id_num', 'cic_1k_8h', 'cic_5k_8h']
    key_indices = {f: feats.index(f) for f in key_feats if f in feats}

    print(f"\n{'='*80}")
    print(f"VERY HIGH USERS sorted by OOF score (bottom = false negatives)")
    print(f"{'='*80}")
    header = f"{'score':>7} {'FP/TP':>5}  {'user_name':<30}"
    for f in key_indices: header += f" {f[:12]:>13}"
    print(header)
    print("-" * len(header))

    order = np.argsort(vh_proba)
    for i in order:
        score = vh_proba[i]
        tp_label = "FN" if score < best_thresh else "TP"
        row_feats = vh_feats[i]
        line = f"{score:>7.3f} {tp_label:>5}  {vh_names[i]:<30}"
        for f, idx in key_indices.items():
            v = row_feats[idx]
            line += f" {v:>13.2f}" if not np.isnan(v) else f" {'nan':>13}"
        print(line)

    # Stats: false negatives vs true positives
    fn_mask = vh_proba < best_thresh
    tp_mask = ~fn_mask
    print(f"\n{'='*60}")
    print(f"False Negatives (n={fn_mask.sum()}) vs True Positives (n={tp_mask.sum()})")
    print(f"{'feature':<35} {'FN mean':>10} {'TP mean':>10} {'diff':>10}")
    for f, idx in key_indices.items():
        fn_v = np.nanmean(vh_feats[fn_mask, idx])
        tp_v = np.nanmean(vh_feats[tp_mask, idx])
        print(f"  {f:<33} {fn_v:>10.2f} {tp_v:>10.2f} {fn_v-tp_v:>10.2f}")
