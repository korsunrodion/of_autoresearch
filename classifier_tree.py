import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from base import fetch_df, clean

MAX_ID_RANGE = 100000

WINDOWS_MINUTES = [60, 60 * 4, 60 * 24, 60 * 24 * 7, 60 * 24 * 7 * 2, 60 * 24 * 7 * 4]  # 1h, 4h, 24h, 7d, 14d, 28d

RISK_ORDER = ['Extreme', 'Very High', 'High', 'Low']

THRESHOLDS = [
    ['Very High']
    # RISK_ORDER[: i + 1] for i in range(len(RISK_ORDER))
]

FEATURES = [
    *[f'log_min_id_diff_{w}h' for w in [1, 4, 24, 168, 168 * 2, 168 * 4]],
    *[f'close_id_count_{w}h' for w in [1, 4, 24, 168, 168 * 2, 168 * 4]],
    *[f'partner_count_{w}h' for w in [1, 4, 24, 168, 168 * 2, 168 * 4]],
    'log_min_time_diff',
    # model-level
    'log_model_norisk_median_id_diff',
    'rel_min_id_diff_24h',           # min_id_diff_24h / model No risk median
    # cross-model
    'user_model_count',
    'user_max_risk_elsewhere',        # max risk_score on other models (0 if none)
]


def _precompute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median min_id_diff of No risk users per model (using all-time nearest neighbor)."""
    max_window = pd.Timedelta(minutes=max(WINDOWS_MINUTES))
    times = df['subscribed_at']
    norisk = df[df['risk_level'] == 'No risk']
    records = []
    for _, user in norisk.iterrows():
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


def _precompute_cross_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per user_name: how many models, and max risk_score on *other* models."""
    user_models = df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count')
    # max risk elsewhere: for each (user, model) pair, max risk on all other models
    user_max_risk = df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk')
    return pd.concat([user_models, user_max_risk], axis=1)


def compute_per_user(df: pd.DataFrame) -> pd.DataFrame:
    print("  Precomputing model stats...")
    model_norisk_median = _precompute_model_stats(df)
    print("  Precomputing cross-model stats...")
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
                row[f'close_id_count_{w_h}h'] = float('nan')
                row[f'partner_count_{w_h}h'] = float('nan')
                any_empty = True
            else:
                id_diffs = (partners['user_id_num'] - user['user_id_num']).abs()
                row[f'log_min_id_diff_{w_h}h'] = np.log1p(id_diffs.min())
                row[f'close_id_count_{w_h}h'] = (id_diffs <= MAX_ID_RANGE).sum()
                row[f'partner_count_{w_h}h'] = len(partners)

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

        # model-level
        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row[f'log_min_id_diff_24h']) if not np.isnan(row.get(f'log_min_id_diff_24h', np.nan)) else np.nan
        row['rel_min_id_diff_24h'] = (raw_24h / norisk_med) if (not np.isnan(norisk_med) and norisk_med > 0) else np.nan

        # cross-model
        uname = user['user_name']
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        global_max = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        # max risk on *other* models = global max if user appears elsewhere, else 0
        row['user_max_risk_elsewhere'] = global_max if row['user_model_count'] > 1 else 0

        rows.append(row)

    return pd.DataFrame(rows).dropna()

def weight_sweep(per_user: pd.DataFrame, levels: list[str], scales: list[float]):
    label = ' | '.join(levels)
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)

    X = subset[FEATURES].values
    y = subset['is_risky'].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n=== Weight sweep: {label} ===")
    print(f"  {'scale':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")

    for scale in scales:
        y_pred_cv = np.zeros_like(y)
        for train_idx, val_idx in cv.split(X, y):
            fold = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale,
                eval_metric='logloss', verbosity=0,
                random_state=42, n_jobs=-1,
            )
            fold.fit(X[train_idx], y[train_idx])
            y_pred_cv[val_idx] = fold.predict(X[val_idx])

        p, r, f, _ = precision_recall_fscore_support(y, y_pred_cv, pos_label=1, average='binary', zero_division=0)
        print(f"  {scale:>6.2f}  {p:>9.2%}  {r:>6.2%}  {f:>6.2%}")

def fit_threshold(per_user: pd.DataFrame, levels: list[str]):
    label = ' | '.join(levels)
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)

    X = subset[FEATURES].values
    y = subset['is_risky'].astype(int).values

    if y.sum() == 0 or y.sum() == len(y):
        print(f"\n[{label}] — skipped (all same class)")
        return

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = n_neg / n_pos  # scale_pos_weight = ratio of negatives to positives

    def make_model():
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='logloss', verbosity=0,
            random_state=42, n_jobs=-1,
        )

    model = make_model()

    seeds = [42, 7, 123, 999, 2025]
    all_p, all_r, all_f = [], [], []
    y_proba_cv = np.zeros(len(y))

    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred_cv = np.zeros_like(y)
        for train_idx, val_idx in cv.split(X, y):
            fold = make_model()
            fold.fit(X[train_idx], y[train_idx])
            y_pred_cv[val_idx] = fold.predict(X[val_idx])
            if seed == seeds[0]:
                y_proba_cv[val_idx] = fold.predict_proba(X[val_idx])[:, 1]
        p, r, f, _ = precision_recall_fscore_support(y, y_pred_cv, pos_label=1, average='binary', zero_division=0)
        all_p.append(p); all_r.append(r); all_f.append(f)

    model.fit(X, y)

    print(f"\n=== Threshold: {label} ===")
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances:
        print(f"  {feat:<28} importance: {imp:.4f}")

    print(f"\nSeed stability (5 seeds x 5-fold CV):")
    print(f"  {'seed':>6}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for i, seed in enumerate(seeds):
        print(f"  {seed:>6}  {all_p[i]:>9.2%}  {all_r[i]:>6.2%}  {all_f[i]:>6.2%}")
    print(f"  {'mean':>6}  {np.mean(all_p):>9.2%}  {np.mean(all_r):>6.2%}  {np.mean(all_f):>6.2%}")
    print(f"  {'std':>6}  {np.std(all_p):>9.2%}  {np.std(all_r):>6.2%}  {np.std(all_f):>6.2%}")

    print(f"\nProbability threshold sweep (seed={seeds[0]}):")
    print(f"  {'thresh':>7}  {'precision':>9}  {'recall':>6}  {'f1':>6}")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_t = (y_proba_cv >= thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, y_t, pos_label=1, average='binary', zero_division=0)
        print(f"  {thresh:>7.1f}  {p:>9.2%}  {r:>6.2%}  {f:>6.2%}")


def tree_to_json(model, feature_names: list[str]) -> dict:
    t = model.tree_
    classes = list(model.classes_)

    def node(i):
        if t.children_left[i] == -1:
            return {'label': int(classes[t.value[i][0].argmax()])}
        return {
            'feature': feature_names[t.feature[i]],
            'threshold': float(t.threshold[i]),
            'left': node(t.children_left[i]),
            'right': node(t.children_right[i]),
        }

    return node(0)


if __name__ == '__main__':
    df = fetch_df(selected=True)
    df = clean(df)

    per_user = compute_per_user(df)

    print(len(df))
    print(f"Total users with a same-model partner: {len(per_user)}")
    print(f"Risk distribution:\n{per_user['risk_level'].value_counts()}\n")

    for levels in THRESHOLDS:
        fit_threshold(per_user, levels)

    n_pos = (per_user['risk_level'] == 'Low').sum()
    n_neg = (per_user['risk_level'] == 'No risk').sum()
    full_ratio = n_neg / n_pos
    scales = [1.0, 2.0, 3.0, 5.0, round(np.sqrt(full_ratio), 1), round(full_ratio / 4, 1), round(full_ratio, 1)]
    weight_sweep(per_user, ['Very High'], sorted(set(scales)))