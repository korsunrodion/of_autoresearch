"""
classifier_v2_cpu_predict.py — CPU-only production train/predict script

Usage:
  # Train on full dataset and save models:
  python classifier_v2_cpu_predict.py --train [--model-dir models_cpu]

  # Predict for all users (or a subset):
  python classifier_v2_cpu_predict.py --predict [--users user1 user2] [--output results.csv]

Models saved per --model-dir:
  extreme_model.pkl, high_model.pkl, low_model.pkl,
  ordinal_model.pkl, vh_model.pkl,
  thresholds.json, features.json

JS export (optional):
  pip install m2cgen
  python classifier_v2_cpu_predict.py --train --export-js vh_lgb.js
"""
import os
import sys
import argparse
import json
import math
import pickle
import multiprocessing
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from base import fetch_df as _fetch_train, clean as _clean_train
from db import db as _db_train
from base_v2 import fetch_df as _fetch_verify, clean as _clean_verify, update_risk_levels
from db_v2 import db as _db_verify

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOWS_MINUTES = [30, 60, 60*4, 60*6, 60*7, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_STRS     = ['0h30m', '1h', '4h', '6h', '7h', '8h', '10h', '12h',
                   '24h', '168h', '336h', '672h']
ID_RANGES  = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
RANGE_TAGS = ['10k', '20k', '50k', '100k', '200k', '500k', '1m']
KEY_WS     = {'6h', '7h', '8h', '10h', '12h', '24h'}
SHORT_WS   = {'0h30m', '1h', '4h', '6h', '7h', '8h'}
DENSITY_WH = [1, 4, 6, 12, 24]

W_NS_LIST = [int(w * 60 * 1e9) for w in WINDOWS_MINUTES]
D_NS_LIST = [int(w * 3600 * 1e9) for w in DENSITY_WH]
MAX_W_NS  = W_NS_LIST[-1]
_5MIN_NS  = int(5 * 60 * 1e9)

FEATURES = (
    [f'log_min_id_diff_{w}' for w in WINDOW_STRS] +
    [f'partner_count_{w}'   for w in WINDOW_STRS] +
    [f'cic_{tag}_{w}'      for tag in RANGE_TAGS for w in WINDOW_STRS] +
    [f'cic_frac_{tag}_{w}' for tag in RANGE_TAGS for w in KEY_WS] +
    [f'log_id_span_{w}'        for w in sorted(KEY_WS)] +
    [f'log_id_std_{w}'         for w in sorted(KEY_WS)] +
    [f'log_min_consec_gap_{w}' for w in sorted(KEY_WS)] +
    [f'model_subs_{w}h'      for w in DENSITY_WH] +
    [f'model_subs_rate_{w}h' for w in DENSITY_WH] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num',
     'cic_ratio_6h_24h', 'pc_ratio_6h_24h', 'cic10k_ratio_6h_12h',
     'total_chargebacks', 'hour_sin', 'hour_cos', 'day_of_week',
     'frac_empty_short_windows', 'log_min_id_diff_global', 'log_global_id_span',
     'username_is_u_digits', 'username_len', 'model_norisk_rate',
     'username_digit_ratio', 'username_entropy',
     'model_vh_frac', 'model_extreme_frac', 'model_any_risk_frac',
     'id_percentile_in_model', 'time_percentile_in_model',
     'log_time_since_model_first_sub', 'n_subs_at_sub_time_5m']
)

RISK_ORDER = ['Extreme', 'Very High', 'High', 'Low']
N_JOBS = min(os.cpu_count() or 4, 8)

# Values to override for cold-start simulation (model with no labeled history).
# model_norisk_rate defaults to 1.0 at inference; the rest go to 0 or NaN.
COLD_OVERRIDE = {
    'model_vh_frac':                  0.0,
    'model_extreme_frac':             0.0,
    'model_any_risk_frac':            0.0,
    'model_norisk_rate':              1.0,
    'log_model_norisk_median_id_diff': np.nan,
    'rel_min_id_diff_24h':            np.nan,
    # Zeroing cross-model features in cold-sim prevents the VH model from
    # leaking user_max_risk_elsewhere (=4 for warm VH, =1 for cold-test VH)
    # as its primary signal, forcing it to learn from individual features.
    'user_max_risk_elsewhere':        0.0,
    'user_model_count':               1.0,
}

# ── Shared globals for multiprocessing workers ────────────────────────────────
_G_CROSS        = {}
_G_NORISK_MED   = {}
_G_NORISK_RATES = {}
_G_RISK_FRACS   = {}


def _pool_init(cross, norisk_med, norisk_rates, risk_fracs):
    global _G_CROSS, _G_NORISK_MED, _G_NORISK_RATES, _G_RISK_FRACS
    _G_CROSS        = cross
    _G_NORISK_MED   = norisk_med
    _G_NORISK_RATES = norisk_rates
    _G_RISK_FRACS   = risk_fracs


# ── Feature helpers ───────────────────────────────────────────────────────────
def _is_u_digits(name):
    return 1.0 if (isinstance(name, str) and len(name) > 1 and
                   name[0] == 'u' and name[1:].isdigit()) else 0.0


def _digit_ratio(name):
    if not isinstance(name, str) or len(name) == 0:
        return 0.0
    return sum(c.isdigit() for c in name) / len(name)


def _entropy(name):
    if not isinstance(name, str) or len(name) == 0:
        return 0.0
    from collections import Counter
    counts = Counter(name)
    n = len(name)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ── Worker: compute features for one model group ──────────────────────────────
def _compute_model_worker(pack):
    model_name, (times_ns, ids_arr, usernames, risk_lvls,
                 risk_scs, ts_arr, chargebacks) = pack

    norisk_med  = _G_NORISK_MED.get(model_name, np.nan)
    norisk_rate = _G_NORISK_RATES.get(model_name, 1.0)
    risk_fracs  = _G_RISK_FRACS.get(model_name, {'vh': 0., 'extreme': 0., 'any': 0.})
    n = len(times_ns)
    if n == 0:
        return []

    tdiff    = np.abs(times_ns[:, None] - times_ns[None, :])
    iddiff   = np.abs(ids_arr[:, None]  - ids_arr[None, :])
    not_self = ~np.eye(n, dtype=bool)
    sorted_ids = np.sort(ids_arr)
    sorted_ts  = np.sort(times_ns)
    model_first_sub_ns = times_ns.min()

    rows = []
    for i in range(n):
        row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
        n_empty_short = 0
        tdiff_i  = tdiff[i]
        iddiff_i = iddiff[i]

        for w_str, w_ns in zip(WINDOW_STRS, W_NS_LIST):
            mask = (tdiff_i <= w_ns) & not_self[i]
            is_key   = w_str in KEY_WS
            is_short = w_str in SHORT_WS
            if not mask.any():
                row[f'log_min_id_diff_{w_str}'] = np.nan
                row[f'partner_count_{w_str}']   = np.nan
                for tag in RANGE_TAGS:
                    row[f'cic_{tag}_{w_str}'] = np.nan
                if is_key:
                    for tag in RANGE_TAGS:
                        row[f'cic_frac_{tag}_{w_str}'] = np.nan
                    row[f'log_id_span_{w_str}']        = np.nan
                    row[f'log_id_std_{w_str}']         = np.nan
                    row[f'log_min_consec_gap_{w_str}'] = np.nan
                if is_short:
                    n_empty_short += 1
            else:
                diffs = iddiff_i[mask]
                pids  = ids_arr[mask]
                pc    = float(mask.sum())
                row[f'log_min_id_diff_{w_str}'] = np.log1p(diffs.min())
                row[f'partner_count_{w_str}']   = pc
                for tag, rng in zip(RANGE_TAGS, ID_RANGES):
                    cic = float((diffs <= rng).sum())
                    row[f'cic_{tag}_{w_str}'] = cic
                    if is_key:
                        row[f'cic_frac_{tag}_{w_str}'] = cic / (pc + 1)
                if is_key:
                    row[f'log_id_span_{w_str}'] = np.log1p(float(pids.max() - pids.min()))
                    row[f'log_id_std_{w_str}']  = (
                        np.log1p(float(np.std(pids))) if pc > 1 else 0.0)
                    row[f'log_min_consec_gap_{w_str}'] = (
                        np.log1p(float(np.diff(np.sort(pids)).min()))
                        if pc >= 2 else np.nan)

        for j in range(len(WINDOWS_MINUTES) - 1):
            ws, nws = WINDOW_STRS[j], WINDOW_STRS[j + 1]
            for base in (['log_min_id_diff', 'partner_count'] +
                         [f'cic_{t}' for t in RANGE_TAGS]):
                k, nk = f'{base}_{ws}', f'{base}_{nws}'
                if np.isnan(row.get(k, np.nan)):
                    row[k] = row.get(nk, np.nan)

        for w_h, d_ns in zip(DENSITY_WH, D_NS_LIST):
            n_subs = float((tdiff_i <= d_ns).sum())
            w_days = w_h * 2 / 24.0
            row[f'model_subs_{w_h}h']      = n_subs
            row[f'model_subs_rate_{w_h}h'] = n_subs / max(norisk_rate * w_days, 1.0)

        gm = not_self[i]
        if gm.any():
            gdiffs = iddiff_i[gm]
            gpids  = ids_arr[gm]
            row['log_min_id_diff_global'] = np.log1p(gdiffs.min())
            row['log_global_id_span']     = np.log1p(float(gpids.max() - gpids.min()))
        else:
            row['log_min_id_diff_global'] = np.nan
            row['log_global_id_span']     = np.nan

        all_mask_i = (tdiff_i <= MAX_W_NS) & not_self[i]
        row['log_min_time_diff'] = (
            np.log1p(tdiff_i[all_mask_i].min() / 1e9 / 60)
            if all_mask_i.any() else np.nan)
        row['log_model_norisk_median_id_diff'] = (
            np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan)
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        row['rel_min_id_diff_24h'] = (
            raw_24h / norisk_med
            if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0
            else np.nan)

        uname = usernames[i]
        mc, gmax_rs = _G_CROSS.get(uname, (1, float(risk_scs[i])))
        row['user_model_count']        = mc
        row['user_max_risk_elsewhere'] = gmax_rs if mc > 1 else 0
        row['log_user_id_num']         = (
            np.log1p(ids_arr[i]) if not np.isnan(ids_arr[i]) else np.nan)

        c6    = row.get('cic_10k_6h',  np.nan)
        c12   = row.get('cic_10k_12h', np.nan)
        c50_6  = row.get('cic_50k_6h',  np.nan)
        c50_24 = row.get('cic_50k_24h', np.nan)
        pc6   = row.get('partner_count_6h',  np.nan)
        pc24  = row.get('partner_count_24h', np.nan)
        row['cic_ratio_6h_24h']    = c50_6 / (c50_24 + 1) if not np.isnan(c50_6)  else np.nan
        row['cic10k_ratio_6h_12h'] = c6    / (c12    + 1) if not np.isnan(c6)     else np.nan
        row['pc_ratio_6h_24h']     = pc6   / (pc24   + 1) if not np.isnan(pc6)    else np.nan

        ts   = ts_arr[i]
        hour = (ts % 86400) / 3600.0
        row['hour_sin']   = float(np.sin(2 * np.pi * hour / 24))
        row['hour_cos']   = float(np.cos(2 * np.pi * hour / 24))
        row['day_of_week'] = float(int(ts // 86400 + 4) % 7)
        row['total_chargebacks']        = chargebacks[i]
        row['frac_empty_short_windows'] = n_empty_short / len(SHORT_WS)
        row['username_is_u_digits'] = _is_u_digits(uname)
        row['username_len']         = float(len(uname)) if isinstance(uname, str) else np.nan
        row['model_norisk_rate']    = norisk_rate
        row['username_digit_ratio'] = _digit_ratio(uname)
        row['username_entropy']     = _entropy(uname)
        row['model_vh_frac']        = risk_fracs['vh']
        row['model_extreme_frac']   = risk_fracs['extreme']
        row['model_any_risk_frac']  = risk_fracs['any']

        row['id_percentile_in_model']   = float(np.searchsorted(sorted_ids, ids_arr[i])) / n
        row['time_percentile_in_model'] = float(np.searchsorted(sorted_ts,  times_ns[i])) / n
        row['log_time_since_model_first_sub'] = np.log1p(
            (times_ns[i] - model_first_sub_ns) / 1e9 / 3600.0)
        row['n_subs_at_sub_time_5m'] = float((tdiff_i <= _5MIN_NS).sum())
        row['tracking_model_name'] = model_name

        rows.append(row)
    return rows


def _model_norisk_median_cpu(df, chunk_size=500):
    medians = {}
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if nr.empty:
            continue
        all_t     = mdf['subscribed_at'].values.astype(np.int64)
        all_id    = mdf['user_id_num'].values.astype(np.float64)
        all_names = np.array(mdf['user_name'], dtype=object)
        nr_t      = nr['subscribed_at'].values.astype(np.int64)
        nr_id     = nr['user_id_num'].values.astype(np.float64)
        nr_names  = np.array(nr['user_name'], dtype=object)
        min_diffs = []
        for i0 in range(0, len(nr_t), chunk_size):
            ct = nr_t[i0:i0 + chunk_size]
            ci = nr_id[i0:i0 + chunk_size]
            cn = nr_names[i0:i0 + chunk_size]
            td = np.abs(ct[:, None] - all_t[None, :])
            dd = np.abs(ci[:, None] - all_id[None, :])
            valid = (td <= MAX_W_NS) & (cn[:, None] != all_names[None, :])
            dd[~valid] = np.nan
            row_mins = np.nanmin(dd, axis=1)
            min_diffs.extend(row_mins[~np.isnan(row_mins)].tolist())
        if min_diffs:
            medians[model_name] = float(np.median(min_diffs))
    return pd.Series(medians)


def compute_per_user(df, n_jobs=N_JOBS):
    print("  Norisk medians (vectorized CPU)...", flush=True)
    norisk_med_dict = _model_norisk_median_cpu(df).to_dict()

    print("  Model rates and risk fractions...", flush=True)
    norisk_rates, risk_fracs = {}, {}
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        n_total = len(mdf)
        if len(nr) >= 2:
            t_range = (nr['subscribed_at'].max() - nr['subscribed_at'].min()).total_seconds() / 86400
            norisk_rates[model_name] = len(nr) / max(t_range, 1.0)
        else:
            norisk_rates[model_name] = 1.0
        risk_fracs[model_name] = {
            'vh':      (mdf['risk_level'] == 'Very High').sum() / n_total,
            'extreme': (mdf['risk_level'] == 'Extreme').sum() / n_total,
            'any':     (~mdf['risk_level'].isin(['No risk'])).sum() / n_total,
        }

    print("  Cross-model user stats...", flush=True)
    cross_mc  = df.groupby('user_name')['tracking_model_name'].nunique()
    cross_rs  = df.groupby('user_name')['risk_score'].max()
    cross_dict = {u: (int(cross_mc[u]), float(cross_rs[u])) for u in cross_mc.index}

    print(f"  Dispatching {df['tracking_model_name'].nunique()} model groups"
          f" to {n_jobs} workers...", flush=True)
    tasks = []
    for model_name, mdf in df.groupby('tracking_model_name'):
        mdf = mdf.reset_index(drop=True)
        tasks.append((model_name, (
            mdf['subscribed_at'].values.astype(np.int64),
            mdf['user_id_num'].values.astype(np.float64),
            np.array(mdf['user_name'], dtype=object),
            np.array(mdf['risk_level'], dtype=object),
            mdf['risk_score'].values.astype(np.float64),
            mdf['subscribed_ts'].values.astype(np.float64),
            mdf['total_chargebacks'].values.astype(np.float64),
        )))

    if n_jobs > 1:
        with multiprocessing.Pool(
            processes=n_jobs,
            initializer=_pool_init,
            initargs=(cross_dict, norisk_med_dict, norisk_rates, risk_fracs),
        ) as pool:
            results = pool.map(_compute_model_worker, tasks, chunksize=1)
    else:
        _pool_init(cross_dict, norisk_med_dict, norisk_rates, risk_fracs)
        results = [_compute_model_worker(t) for t in tasks]

    all_rows = [row for batch in results for row in batch]
    return pd.DataFrame(all_rows)


# ── Threshold ─────────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[f1.argmax()]) if f1.argmax() < len(thresholds) else 0.5


# ── Cold-start augmentation helpers ──────────────────────────────────────────
def make_cold_X(X, feat_names):
    """Return a copy of X with model-level context features overridden to cold-start values."""
    X_cold = X.copy()
    for feat, val in COLD_OVERRIDE.items():
        if feat in feat_names:
            X_cold[:, feat_names.index(feat)] = val
    return X_cold


# ── OOF training helper ───────────────────────────────────────────────────────
def build_oof_lgb(X, y, configs, seeds, n_folds, X_cold=None, model_ids=None,
                  cold_model_frac=0.25):
    """Train with model-level cold-start simulation.

    X_cold       : cold-start version of X (model-context features overridden to 0/NaN).
    model_ids    : 1-D array of model membership for each row (same length as X).
                   When provided, each fold randomly designates cold_model_frac of
                   training models as "cold": their rows use cold features instead of
                   warm features — no warm copies survive for those models. This forces
                   the classifier to learn cold patterns without a warm crutch.
                   For the remaining models, warm + cold copies are both included
                   (existing row-level augmentation).
    cold_model_frac: fraction of training models to treat as model-level cold per fold.

    Validation is always scored on warm features for honest OOF metrics.
    Cold OOF pool (for cold threshold calibration) is scored on cold features.

    Returns (results, cold_pool_arr).
    """
    results = []
    cold_agg = np.zeros(len(y)) if X_cold is not None else None
    rng = np.random.default_rng(0)

    for name, params in configs:
        pool_arr = np.zeros(len(y))
        cold_pool = np.zeros(len(y)) if X_cold is not None else None
        for seed in seeds:
            clf = LGBMClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y))
            cp  = np.zeros(len(y)) if X_cold is not None else None
            for ti, vi in cv.split(X, y):
                if X_cold is not None and model_ids is not None:
                    # Model-level cold simulation: pick cold_model_frac of training
                    # models and replace ALL their warm rows with cold features.
                    train_models = np.unique(model_ids[ti])
                    n_cm = max(1, int(len(train_models) * cold_model_frac))
                    cold_model_set = set(
                        rng.choice(train_models, n_cm, replace=False).tolist())
                    is_cold_model = np.array(
                        [model_ids[i] in cold_model_set for i in ti])

                    # Rows from cold models: cold features only (no warm copy)
                    # Rows from warm models: warm + cold copies
                    X_cm_cold = X_cold[ti][is_cold_model]
                    y_cm      = y[ti][is_cold_model]
                    X_wm_warm = X[ti][~is_cold_model]
                    X_wm_cold = X_cold[ti][~is_cold_model]
                    y_wm      = y[ti][~is_cold_model]

                    X_tr = np.vstack([X_wm_warm, X_wm_cold, X_cm_cold])
                    y_tr = np.concatenate([y_wm, y_wm, y_cm])
                elif X_cold is not None:
                    # Fallback: row-level augmentation (original behaviour)
                    X_tr = np.vstack([X[ti], X_cold[ti]])
                    y_tr = np.concatenate([y[ti], y[ti]])
                else:
                    X_tr, y_tr = X[ti], y[ti]

                clf.fit(X_tr, y_tr)
                pr[vi] = clf.predict_proba(X[vi])[:, 1]
                if cp is not None:
                    cp[vi] = clf.predict_proba(X_cold[vi])[:, 1]
            pool_arr += pr
            if cold_pool is not None:
                cold_pool += cp
        pool_arr /= len(seeds)
        if cold_pool is not None:
            cold_pool /= len(seeds)
            cold_agg += cold_pool
        t = find_best_threshold(y, pool_arr)
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, _ = precision_recall_fscore_support(
            y, (pool_arr >= t).astype(int),
            pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        results.append((name, pool_arr, f))
    if cold_agg is not None:
        cold_agg /= len(configs)
    return results, cold_agg


def _fit_augmented(clf, X, y, X_cold=None):
    """Fit on warm data + cold-start copies concatenated."""
    if X_cold is None:
        clf.fit(X, y)
    else:
        clf.fit(np.vstack([X, X_cold]), np.concatenate([y, y]))


# ── Train on full data and save models ────────────────────────────────────────
def train_and_save(model_dir, export_js=None):
    os.makedirs(model_dir, exist_ok=True)

    print("Loading dataset (selected=False)...", flush=True)
    df = _fetch_train(selected=False)
    df = _clean_train(df)
    try:
        _db_train.close()
    except Exception:
        pass
    print(f"  {len(df)} rows", flush=True)

    print("\nComputing features...", flush=True)
    per_user = compute_per_user(df)
    feats = [f for f in FEATURES if f in per_user.columns]
    print(f"  {len(feats)} features, {len(per_user)} users", flush=True)

    # Attach model membership for model-level cold simulation in build_oof_lgb
    _u2m = df.drop_duplicates('user_name').set_index('user_name')['tracking_model_name']
    per_user['_model_id'] = per_user['user_name'].map(_u2m)

    print("  Building cold-start augmentation...", flush=True)

    SEEDS_C   = [42, 7, 123]
    SEEDS_VH  = [42, 7, 123, 999, 2025]

    def lgb_c(nl, sw, lr=0.05, n=400, ss=0.80, cs=0.85, mcw=20):
        return dict(num_leaves=nl, learning_rate=lr, n_estimators=n,
                    scale_pos_weight=sw, subsample=ss, colsample_bytree=cs,
                    min_child_samples=mcw, subsample_freq=1, verbose=-1, n_jobs=-1)

    # ── Cascade OOF probas ──────────────────────────────────────────────────
    print("\n=== Cascade ===", flush=True)
    thresholds = {}

    def _cascade_oof(label, pos_level, configs, seeds, n_folds):
        sub = per_user[per_user['risk_level'].isin([pos_level, 'No risk'])].copy()
        sub['y'] = (sub['risk_level'] == pos_level).astype(int)
        X_, y_ = sub[feats].values, sub['y'].values
        spw = (y_ == 0).sum() / y_.sum()
        cfgs = [(n, lgb_c(*p, sw=spw) if isinstance(p, tuple) else {**lgb_c(63, spw), **p})
                for n, p in configs]
        # Build OOF pool
        pool = np.zeros(len(y_))
        all_clfs = []
        for name, params in [(f'{label}-{i}', c) for i, (_, c) in enumerate(cfgs)]:
            p_arr = np.zeros(len(y_))
            for seed in seeds:
                clf = LGBMClassifier(**params, random_state=seed)
                cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                pr  = np.zeros(len(y_))
                for ti, vi in cv.split(X_, y_):
                    clf.fit(X_[ti], y_[ti])
                    pr[vi] = clf.predict_proba(X_[vi])[:, 1]
                p_arr += pr
            p_arr /= len(seeds)
            pool += p_arr / len(cfgs)
            # Train final model on all data
            clf_final = LGBMClassifier(**params, random_state=42)
            clf_final.fit(X_, y_)
            all_clfs.append(clf_final)
        t = find_best_threshold(y_, pool)
        thresholds[label] = t
        return dict(zip(sub['user_name'].values, pool)), all_clfs, spw

    print("  Extreme...", flush=True)
    sub_e = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
    sub_e['y'] = (sub_e['risk_level'] == 'Extreme').astype(int)
    Xe, ye = sub_e[feats].values, sub_e['y'].values
    Xe_cold = make_cold_X(Xe, feats)
    spw_e = (ye == 0).sum() / ye.sum()
    res_e, _ = build_oof_lgb(Xe, ye, [
        (f'E-nl63-spw{spw_e:.0f}', lgb_c(63, spw_e, lr=0.05, n=300)),
    ], SEEDS_C, 5, X_cold=Xe_cold, model_ids=sub_e['_model_id'].values)
    p_extreme = np.mean([p for _, p, _ in res_e], axis=0)
    extreme_lookup = dict(zip(sub_e['user_name'].values, p_extreme))
    t_e = find_best_threshold(ye, p_extreme)
    thresholds['extreme'] = t_e
    clf_extreme = LGBMClassifier(**lgb_c(127, spw_e, lr=0.03, n=500), random_state=42)
    _fit_augmented(clf_extreme, Xe, ye, Xe_cold)

    print("  High...", flush=True)
    sub_h = per_user[per_user['risk_level'].isin(['High', 'No risk'])].copy()
    sub_h['y'] = (sub_h['risk_level'] == 'High').astype(int)
    Xh, yh = sub_h[feats].values, sub_h['y'].values
    Xh_cold = make_cold_X(Xh, feats)
    spw_h = (yh == 0).sum() / yh.sum()
    res_h, cold_h_pool = build_oof_lgb(Xh, yh, [(f'H-nl63-spw{spw_h:.0f}', lgb_c(63, spw_h, lr=0.05, n=300))],
                                      SEEDS_C, 5, X_cold=Xh_cold,
                                      model_ids=sub_h['_model_id'].values)
    p_high = np.mean([p for _, p, _ in res_h], axis=0)
    high_lookup = dict(zip(sub_h['user_name'].values, p_high))
    t_h = find_best_threshold(yh, p_high)
    thresholds['high'] = t_h
    # Constrain cold threshold ≤ warm threshold to ensure cold branch can fire
    thresholds['high_cold'] = float(min(find_best_threshold(yh, cold_h_pool), t_h))
    clf_high = LGBMClassifier(**lgb_c(127, spw_h, lr=0.03, n=500), random_state=42)
    _fit_augmented(clf_high, Xh, yh, Xh_cold)

    print("  Low...", flush=True)
    sub_l = per_user[per_user['risk_level'].isin(['Low', 'No risk'])].copy()
    sub_l['y'] = (sub_l['risk_level'] == 'Low').astype(int)
    Xl, yl = sub_l[feats].values, sub_l['y'].values
    Xl_cold = make_cold_X(Xl, feats)
    spw_l = (yl == 0).sum() / yl.sum()
    res_l, cold_l_pool = build_oof_lgb(Xl, yl, [(f'L-nl63-spw{spw_l:.0f}', lgb_c(63, spw_l, lr=0.05, n=300))],
                                       SEEDS_C, 5, X_cold=Xl_cold,
                                       model_ids=sub_l['_model_id'].values)
    p_low = np.mean([p for _, p, _ in res_l], axis=0)
    low_lookup = dict(zip(sub_l['user_name'].values, p_low))
    t_l = find_best_threshold(yl, p_low)
    thresholds['low'] = t_l
    # Constrain cold threshold ≤ warm threshold to ensure cold branch can fire
    thresholds['low_cold'] = float(min(find_best_threshold(yl, cold_l_pool), t_l))
    clf_low = LGBMClassifier(**lgb_c(127, spw_l, lr=0.03, n=500), random_state=42)
    _fit_augmented(clf_low, Xl, yl, Xl_cold)

    print("  Ordinal...", flush=True)
    sub_ord = per_user.copy()
    sub_ord['y'] = sub_ord['risk_level'].isin(['Very High', 'Extreme']).astype(int)
    X_ord, y_ord = sub_ord[feats].values, sub_ord['y'].values
    X_ord_cold = make_cold_X(X_ord, feats)
    spw_ord = (y_ord == 0).sum() / y_ord.sum()
    res_ord, _ = build_oof_lgb(X_ord, y_ord,
                               [(f'ORD-nl63-spw{spw_ord:.0f}', lgb_c(63, spw_ord, lr=0.05, n=300))],
                               SEEDS_C, 5, X_cold=X_ord_cold,
                               model_ids=sub_ord['_model_id'].values)
    p_ordinal = np.mean([p for _, p, _ in res_ord], axis=0)
    ordinal_lookup = dict(zip(sub_ord['user_name'].values, p_ordinal))
    clf_ordinal = LGBMClassifier(**lgb_c(127, spw_ord, lr=0.03, n=500), random_state=42)
    _fit_augmented(clf_ordinal, X_ord, y_ord, X_ord_cold)

    # ── VH model ────────────────────────────────────────────────────────────
    print("\n=== Very High (20s x 15-fold) ===", flush=True)
    sub_vh = (per_user[per_user['risk_level'].isin(['Very High', 'No risk'])]
              .copy().reset_index(drop=True))
    y_vh = (sub_vh['risk_level'] == 'Very High').astype(int).values
    X_vh = sub_vh[feats].values
    spw_vh = (y_vh == 0).sum() / y_vh.sum()

    # Warm cascade probas from OOF lookups
    casc_extreme = np.array([extreme_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    casc_high    = np.array([high_lookup.get(u, 0.0)    for u in sub_vh['user_name']])
    casc_low     = np.array([low_lookup.get(u, 0.0)     for u in sub_vh['user_name']])
    casc_ordinal = np.array([ordinal_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    X_vh_aug = np.column_stack([X_vh, casc_extreme, casc_high, casc_low, casc_ordinal])

    # Cold VH: override model-level features, recompute cascade probas from cold-trained models
    X_vh_cold_base = make_cold_X(X_vh, feats)
    casc_e_cold   = clf_extreme.predict_proba(X_vh_cold_base)[:, 1]
    casc_h_cold   = clf_high.predict_proba(X_vh_cold_base)[:, 1]
    casc_l_cold   = clf_low.predict_proba(X_vh_cold_base)[:, 1]
    casc_ord_cold = clf_ordinal.predict_proba(X_vh_cold_base)[:, 1]
    X_vh_aug_cold = np.column_stack([X_vh_cold_base, casc_e_cold, casc_h_cold,
                                     casc_l_cold, casc_ord_cold])

    print(f"  {y_vh.sum()} pos, {(y_vh==0).sum()} neg  spw={spw_vh:.1f}  feats={X_vh_aug.shape[1]}", flush=True)

    vh_results, cold_vh_pool = build_oof_lgb(X_vh_aug, y_vh, [
        (f'LGB-nl63-spw{spw_vh:.0f}',
         dict(num_leaves=63, learning_rate=0.05, n_estimators=400,
              scale_pos_weight=spw_vh, subsample=0.80, colsample_bytree=0.85,
              min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1)),
    ], SEEDS_VH, 7, X_cold=X_vh_aug_cold,
       model_ids=sub_vh['_model_id'].values,
       cold_model_frac=0.50)

    # OOF threshold for VH on warm data (use best individual config)
    best_name, best_pool, _ = max(vh_results, key=lambda x: x[2])
    t_vh = find_best_threshold(y_vh, best_pool)
    thresholds['vh'] = t_vh
    print(f"  VH threshold (OOF warm): {t_vh:.4f}", flush=True)

    # Cold-start VH threshold: calibrated on cold-simulated OOF val predictions.
    # cold_vh_pool scores each val fold on cold features — captures how confident
    # the model is about VH users when model-level context is unavailable.
    t_vh_cold = find_best_threshold(y_vh, cold_vh_pool)
    thresholds['vh_cold'] = float(t_vh_cold)
    from sklearn.metrics import precision_recall_fscore_support as _prf_vh
    _p, _r, _f, _ = _prf_vh(y_vh, (cold_vh_pool >= t_vh_cold).astype(int),
                             pos_label=1, average='binary', zero_division=0)
    print(f"  VH cold threshold: {t_vh_cold:.4f}  cold OOF F1={_f:.2%} P={_p:.2%} R={_r:.2%}",
          flush=True)

    # Train final VH model on ALL data (nl127 — best config)
    print("  Training final VH model on full data...", flush=True)
    clf_vh = LGBMClassifier(
        num_leaves=127, learning_rate=0.02, n_estimators=800,
        scale_pos_weight=spw_vh, subsample=0.75, colsample_bytree=0.80,
        min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1, random_state=42)
    _fit_augmented(clf_vh, X_vh_aug, y_vh, X_vh_aug_cold)

    # ── Cold-start VH vs Extreme secondary classifier ───────────────────────
    # Trained on VH+Extreme with cold-start features (model-level context zeroed).
    # Focused on the hard VH-vs-Extreme boundary so OOF threshold calibrates to
    # the same distribution seen at inference for cold users with high e_proba.
    print("\n=== Cold VH vs Extreme (secondary classifier) ===", flush=True)
    sub_cve = per_user[per_user['risk_level'].isin(['Very High', 'Extreme'])].copy()
    sub_cve['y'] = (sub_cve['risk_level'] == 'Very High').astype(int)
    X_cve, y_cve = sub_cve[feats].values, sub_cve['y'].values
    X_cve_cold = make_cold_X(X_cve, feats)
    spw_cve = (y_cve == 0).sum() / y_cve.sum()
    print(f"  VH={y_cve.sum()} Extreme={(y_cve==0).sum()} spw={spw_cve:.1f}", flush=True)

    SEEDS_CVE = [42, 7, 123, 999, 2025]
    cve_oof = np.zeros(len(y_cve))
    for seed in SEEDS_CVE:
        clf_tmp = LGBMClassifier(num_leaves=127, n_estimators=500, learning_rate=0.03,
                                 scale_pos_weight=spw_cve, n_jobs=-1, verbose=-1,
                                 subsample=0.80, colsample_bytree=0.85,
                                 min_child_samples=20, subsample_freq=1,
                                 random_state=seed)
        cv_cve = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
        pr_cve = np.zeros(len(y_cve))
        for ti, vi in cv_cve.split(X_cve_cold, y_cve):
            clf_tmp.fit(X_cve_cold[ti], y_cve[ti])
            pr_cve[vi] = clf_tmp.predict_proba(X_cve_cold[vi])[:, 1]
        cve_oof += pr_cve
    cve_oof /= len(SEEDS_CVE)
    t_cve = find_best_threshold(y_cve, cve_oof)
    from sklearn.metrics import precision_recall_fscore_support as _prf
    p_cve, r_cve, f_cve, _ = _prf(y_cve, (cve_oof >= t_cve).astype(int),
                                   pos_label=1, average='binary', zero_division=0)
    print(f"  Cold VH OOF: P={p_cve:.2%} R={r_cve:.2%} F1={f_cve:.2%} @ t={t_cve:.3f}",
          flush=True)
    thresholds['vh_extreme_cold'] = float(t_cve)

    # Ensemble final cold_vh (5 seeds) for stable probabilities at inference
    print("  Training final cold_vh ensemble...", flush=True)
    cold_vh_clfs = []
    for seed in SEEDS_CVE:
        clf_cvh = LGBMClassifier(num_leaves=127, n_estimators=800, learning_rate=0.03,
                                  scale_pos_weight=spw_cve, n_jobs=-1, verbose=-1,
                                  subsample=0.80, colsample_bytree=0.85,
                                  min_child_samples=20, subsample_freq=1,
                                  random_state=seed)
        clf_cvh.fit(X_cve_cold, y_cve)
        cold_vh_clfs.append(clf_cvh)

    # ── Cold High vs Extreme secondary classifier ────────────────────────────
    # Rescues true High users that fire the Extreme gate in cold-start.
    # total_chargebacks (preserved through COLD_OVERRIDE) is the primary signal.
    print("\n=== Cold High vs Extreme (secondary classifier) ===", flush=True)
    sub_che = per_user[per_user['risk_level'].isin(['High', 'Extreme'])].copy()
    sub_che['y'] = (sub_che['risk_level'] == 'High').astype(int)
    X_che, y_che = sub_che[feats].values, sub_che['y'].values
    X_che_cold = make_cold_X(X_che, feats)
    spw_che = (y_che == 0).sum() / max(y_che.sum(), 1)
    print(f"  High={y_che.sum()} Extreme={(y_che==0).sum()} spw={spw_che:.1f}", flush=True)

    SEEDS_CHE = [42, 7, 123, 999, 2025]
    che_oof = np.zeros(len(y_che))
    for seed in SEEDS_CHE:
        clf_tmp = LGBMClassifier(num_leaves=63, n_estimators=500, learning_rate=0.03,
                                 scale_pos_weight=spw_che, n_jobs=-1, verbose=-1,
                                 subsample=0.80, colsample_bytree=0.85,
                                 min_child_samples=10, subsample_freq=1,
                                 random_state=seed)
        cv_che = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pr_che = np.zeros(len(y_che))
        for ti, vi in cv_che.split(X_che_cold, y_che):
            clf_tmp.fit(X_che_cold[ti], y_che[ti])
            pr_che[vi] = clf_tmp.predict_proba(X_che_cold[vi])[:, 1]
        che_oof += pr_che
    che_oof /= len(SEEDS_CHE)
    from sklearn.metrics import precision_recall_fscore_support as _prf_che
    t_che = find_best_threshold(y_che, che_oof)
    p_che, r_che, f_che, _ = _prf_che(y_che, (che_oof >= t_che).astype(int),
                                       pos_label=1, average='binary', zero_division=0)
    print(f"  Cold High OOF: P={p_che:.2%} R={r_che:.2%} F1={f_che:.2%} @ t={t_che:.3f}",
          flush=True)
    thresholds['cold_high_extreme'] = float(t_che)

    print("  Training final cold_high ensemble...", flush=True)
    cold_high_clfs = []
    for seed in SEEDS_CHE:
        clf_ch = LGBMClassifier(num_leaves=63, n_estimators=700, learning_rate=0.03,
                                scale_pos_weight=spw_che, n_jobs=-1, verbose=-1,
                                subsample=0.80, colsample_bytree=0.85,
                                min_child_samples=10, subsample_freq=1,
                                random_state=seed)
        clf_ch.fit(X_che_cold, y_che)
        cold_high_clfs.append(clf_ch)

    # ── Cold Low vs No risk secondary classifier ─────────────────────────────
    # Catches Low users in cold-start where l_proba (trained warm) undershoots.
    # total_chargebacks (preserved through COLD_OVERRIDE) is the primary signal.
    print("\n=== Cold Low vs No risk (secondary classifier) ===", flush=True)
    sub_cln = per_user[per_user['risk_level'].isin(['Low', 'No risk'])].copy()
    sub_cln['y'] = (sub_cln['risk_level'] == 'Low').astype(int)
    X_cln, y_cln = sub_cln[feats].values, sub_cln['y'].values
    X_cln_cold = make_cold_X(X_cln, feats)
    spw_cln = (y_cln == 0).sum() / max(y_cln.sum(), 1)
    print(f"  Low={y_cln.sum()} No risk={(y_cln==0).sum()} spw={spw_cln:.1f}", flush=True)

    SEEDS_CLN = [42, 7, 123, 999, 2025]
    cln_oof = np.zeros(len(y_cln))
    for seed in SEEDS_CLN:
        clf_tmp = LGBMClassifier(num_leaves=63, n_estimators=500, learning_rate=0.03,
                                 scale_pos_weight=spw_cln, n_jobs=-1, verbose=-1,
                                 subsample=0.80, colsample_bytree=0.85,
                                 min_child_samples=10, subsample_freq=1,
                                 random_state=seed)
        cv_cln = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pr_cln = np.zeros(len(y_cln))
        for ti, vi in cv_cln.split(X_cln_cold, y_cln):
            clf_tmp.fit(X_cln_cold[ti], y_cln[ti])
            pr_cln[vi] = clf_tmp.predict_proba(X_cln_cold[vi])[:, 1]
        cln_oof += pr_cln
    cln_oof /= len(SEEDS_CLN)
    from sklearn.metrics import precision_recall_fscore_support as _prf_cln
    t_cln = find_best_threshold(y_cln, cln_oof)
    p_cln, r_cln, f_cln, _ = _prf_cln(y_cln, (cln_oof >= t_cln).astype(int),
                                       pos_label=1, average='binary', zero_division=0)
    print(f"  Cold Low OOF: P={p_cln:.2%} R={r_cln:.2%} F1={f_cln:.2%} @ t={t_cln:.3f}",
          flush=True)
    thresholds['cold_low_norisk'] = float(t_cln)

    print("  Training final cold_low ensemble...", flush=True)
    cold_low_clfs = []
    for seed in SEEDS_CLN:
        clf_cl = LGBMClassifier(num_leaves=63, n_estimators=700, learning_rate=0.03,
                                scale_pos_weight=spw_cln, n_jobs=-1, verbose=-1,
                                subsample=0.80, colsample_bytree=0.85,
                                min_child_samples=10, subsample_freq=1,
                                random_state=seed)
        clf_cl.fit(X_cln_cold, y_cln)
        cold_low_clfs.append(clf_cl)

    # ── Warm VH rescue classifier ────────────────────────────────────────────
    # Binary VH(1) vs {Extreme,High,Low}(0) on warm augmented features.
    # Used at predict time for warm users that fire the Extreme branch, so that
    # true VH users are not permanently masked as Extreme.
    print("\n=== Warm VH rescue classifier ===", flush=True)
    sub_wr = per_user[per_user['risk_level'].isin(['Very High', 'Extreme', 'High', 'Low'])].copy()
    sub_wr['y'] = (sub_wr['risk_level'] == 'Very High').astype(int)

    # Build augmented features (same as VH training)
    X_wr_base = sub_wr[feats].values
    casc_e_wr   = np.array([extreme_lookup.get(u, 0.0) for u in sub_wr['user_name']])
    casc_h_wr   = np.array([high_lookup.get(u, 0.0)    for u in sub_wr['user_name']])
    casc_l_wr   = np.array([low_lookup.get(u, 0.0)     for u in sub_wr['user_name']])
    casc_ord_wr = np.array([ordinal_lookup.get(u, 0.0) for u in sub_wr['user_name']])
    X_wr_aug = np.column_stack([X_wr_base, casc_e_wr, casc_h_wr, casc_l_wr, casc_ord_wr])
    y_wr = sub_wr['y'].values
    spw_wr = (y_wr == 0).sum() / max(y_wr.sum(), 1)
    print(f"  VH={y_wr.sum()} non-VH={(y_wr==0).sum()} spw={spw_wr:.1f}", flush=True)

    wr_oof = np.zeros(len(y_wr))
    for seed in [42, 7, 123, 999, 2025]:
        clf_tmp = LGBMClassifier(num_leaves=63, n_estimators=300, learning_rate=0.05,
                                 scale_pos_weight=spw_wr, n_jobs=-1, verbose=-1,
                                 random_state=seed)
        cv_wr = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pr_wr = np.zeros(len(y_wr))
        for ti, vi in cv_wr.split(X_wr_aug, y_wr):
            clf_tmp.fit(X_wr_aug[ti], y_wr[ti])
            pr_wr[vi] = clf_tmp.predict_proba(X_wr_aug[vi])[:, 1]
        wr_oof += pr_wr
    wr_oof /= 5
    t_wr = find_best_threshold(y_wr, wr_oof)
    from sklearn.metrics import precision_recall_fscore_support as _prf_wr
    p_wr, r_wr, f_wr, _ = _prf_wr(y_wr, (wr_oof >= t_wr).astype(int),
                                   pos_label=1, average='binary', zero_division=0)
    print(f"  Warm rescue OOF: P={p_wr:.2%} R={r_wr:.2%} F1={f_wr:.2%} @ t={t_wr:.4f}",
          flush=True)
    thresholds['warm_vh_rescue'] = float(t_wr)

    clf_warm_rescue = LGBMClassifier(num_leaves=63, n_estimators=300, learning_rate=0.05,
                                     scale_pos_weight=spw_wr, n_jobs=-1, verbose=-1,
                                     random_state=42)
    clf_warm_rescue.fit(X_wr_aug, y_wr)

    # ── Save everything ─────────────────────────────────────────────────────
    print(f"\nSaving models to {model_dir}/...", flush=True)
    for name, obj in [('extreme', clf_extreme), ('high', clf_high),
                      ('low', clf_low), ('ordinal', clf_ordinal), ('vh', clf_vh),
                      ('cold_vh', cold_vh_clfs), ('cold_high', cold_high_clfs),
                      ('cold_low', cold_low_clfs), ('warm_vh_rescue', clf_warm_rescue)]:
        with open(os.path.join(model_dir, f'{name}_model.pkl'), 'wb') as f:
            pickle.dump(obj, f)
    with open(os.path.join(model_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(model_dir, 'features.json'), 'w') as f:
        json.dump({'base_features': feats, 'spw': {
            'extreme': float(spw_e), 'high': float(spw_h),
            'low': float(spw_l), 'ordinal': float(spw_ord), 'vh': float(spw_vh),
        }}, f, indent=2)
    print("  Done.", flush=True)

    # ── Optional JS export ──────────────────────────────────────────────────
    if export_js:
        try:
            import m2cgen as m2c
            js_code = m2c.export_to_javascript(clf_vh)
            with open(export_js, 'w') as f:
                f.write(js_code)
            print(f"  JS model written to {export_js}", flush=True)
        except ImportError:
            print("  m2cgen not installed — skipping JS export. pip install m2cgen", flush=True)


# ── Predict using saved models ────────────────────────────────────────────────
def predict(model_dir, users=None, output=None):
    print("Loading models...", flush=True)
    models, thresholds, feat_info = {}, {}, {}
    for name in ['extreme', 'high', 'low', 'ordinal', 'vh', 'cold_vh', 'cold_high', 'cold_low', 'warm_vh_rescue']:
        path = os.path.join(model_dir, f'{name}_model.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    with open(os.path.join(model_dir, 'thresholds.json')) as f:
        thresholds = json.load(f)
    with open(os.path.join(model_dir, 'features.json')) as f:
        feat_info = json.load(f)
    feats = feat_info['base_features']

    print("Loading dataset...", flush=True)
    df = _fetch_verify()
    df = _clean_verify(df)
    try:
        _db_verify.close()
    except Exception:
        pass
    if users:
        df = df[df['user_name'].isin(users)]
        print(f"  Filtered to {len(users)} users: {len(df)} rows", flush=True)
    else:
        print(f"  {len(df)} rows", flush=True)

    print("Computing features...", flush=True)
    per_user = compute_per_user(df)

    # Prevent feedback loop: recompute model-context risk fracs using only
    # ground-truth labeled rows (is_internal_data=True). Without this, each
    # predict run would treat its own previous predictions as warm context,
    # shifting cold users out of the cold-start path and degrading recall.
    if 'is_internal_data' in df.columns:
        df_warm_ctx = df[df['is_internal_data'] == True]
        warm_risk_fracs = {}
        warm_norisk_rates = {}
        for model_name, mdf in df_warm_ctx.groupby('tracking_model_name'):
            n = len(mdf)
            warm_risk_fracs[model_name] = {
                'vh':      (mdf['risk_level'] == 'Very High').sum() / n,
                'extreme': (mdf['risk_level'] == 'Extreme').sum() / n,
                'any':     (~mdf['risk_level'].isin(['No risk'])).sum() / n,
            }
            warm_norisk_rates[model_name] = (mdf['risk_level'] == 'No risk').sum() / n

        # Map each per_user row to its model's warm-context fracs
        # (per_user may have one row per model appearance for multi-model users)
        pu_models = per_user.get('tracking_model_name', None)
        if pu_models is None:
            # Fallback: join on user_name -> tracking_model_name from df
            user_to_model = df.drop_duplicates('user_name').set_index('user_name')['tracking_model_name']
            pu_models = per_user['user_name'].map(user_to_model)

        per_user['model_vh_frac'] = pu_models.map(
            lambda m: warm_risk_fracs.get(m, {}).get('vh', 0.0))
        per_user['model_extreme_frac'] = pu_models.map(
            lambda m: warm_risk_fracs.get(m, {}).get('extreme', 0.0))
        per_user['model_any_risk_frac'] = pu_models.map(
            lambda m: warm_risk_fracs.get(m, {}).get('any', 0.0))
        per_user['model_norisk_rate'] = pu_models.map(
            lambda m: warm_norisk_rates.get(m, 1.0))
        print(f"  Model-context recomputed from {len(df_warm_ctx)} warm rows "
              f"({(~df['is_internal_data'].fillna(False)).sum()} cold rows excluded)",
              flush=True)

    X_base = per_user[feats].values

    # Cascade probas
    e_proba   = models['extreme'].predict_proba(X_base)[:, 1]
    h_proba   = models['high'].predict_proba(X_base)[:, 1]
    l_proba   = models['low'].predict_proba(X_base)[:, 1]
    ord_proba = models['ordinal'].predict_proba(X_base)[:, 1]

    X_aug = np.column_stack([X_base, e_proba, h_proba, l_proba, ord_proba])
    vh_proba = models['vh'].predict_proba(X_aug)[:, 1]

    X_base_cold = make_cold_X(X_base, feats)
    if 'cold_vh' in models:
        cold_vh_obj = models['cold_vh']
        if isinstance(cold_vh_obj, list):
            cold_vh_proba = np.mean([clf.predict_proba(X_base_cold)[:, 1]
                                     for clf in cold_vh_obj], axis=0)
        else:
            cold_vh_proba = cold_vh_obj.predict_proba(X_base_cold)[:, 1]
    else:
        cold_vh_proba = np.zeros(len(per_user))

    if 'cold_high' in models:
        cold_high_obj = models['cold_high']
        if isinstance(cold_high_obj, list):
            cold_high_proba = np.mean([clf.predict_proba(X_base_cold)[:, 1]
                                       for clf in cold_high_obj], axis=0)
        else:
            cold_high_proba = cold_high_obj.predict_proba(X_base_cold)[:, 1]
    else:
        cold_high_proba = np.zeros(len(per_user))

    if 'cold_low' in models:
        cold_low_obj = models['cold_low']
        if isinstance(cold_low_obj, list):
            cold_low_proba = np.mean([clf.predict_proba(X_base_cold)[:, 1]
                                      for clf in cold_low_obj], axis=0)
        else:
            cold_low_proba = cold_low_obj.predict_proba(X_base_cold)[:, 1]
    else:
        cold_low_proba = np.zeros(len(per_user))

    # Warm rescue: VH vs {Extreme,High,Low} on augmented features for warm users.
    warm_rescue_proba = (models['warm_vh_rescue'].predict_proba(X_aug)[:, 1]
                         if 'warm_vh_rescue' in models else np.zeros(len(per_user)))

    # Apply thresholds in risk-descending order
    t_e  = thresholds.get('extreme', 0.5)
    t_vh = thresholds.get('vh', 0.5)
    t_h  = thresholds.get('high', 0.5)
    t_l  = thresholds.get('low', 0.5)

    COLD_T_E  = thresholds.get('extreme_cold', 0.50)
    t_vh_cold = thresholds.get('vh_cold',   t_vh)
    t_h_cold  = thresholds.get('high_cold', t_h)
    t_l_cold  = thresholds.get('low_cold',  t_l)

    cold_mask = (per_user['model_any_risk_frac'].fillna(0) == 0).values

    t_cold_vh_e    = thresholds.get('vh_extreme_cold', 0.5)
    t_cold_high    = thresholds.get('cold_high_extreme', 0.5)
    t_cold_low     = thresholds.get('cold_low_norisk', 0.5)
    t_warm_rescue  = thresholds.get('warm_vh_rescue', 0.47)
    t_cold_vh_soft = thresholds.get('cold_vh_soft_gate', 0.01)
    t_cold_ord     = thresholds.get('cold_ord_gate', 0.10)

    n_cold = cold_mask.sum()
    print(f"  Cold-start users: {n_cold}/{len(per_user)}"
          f"  t_vh={t_vh:.4f}  t_vh_cold={t_vh_cold:.4f}", flush=True)

    predicted = []
    for i in range(len(per_user)):
        if cold_mask[i]:
            # Extreme gate first: VH users look Extreme in cold start (high e_proba).
            # cold_vh then rescues true VH from the Extreme pool.
            if e_proba[i] >= COLD_T_E:
                if cold_vh_proba[i] >= t_cold_vh_e:
                    predicted.append('Very High')
                elif cold_high_proba[i] >= t_cold_high:
                    predicted.append('High')
                elif cold_low_proba[i] >= t_cold_low:
                    predicted.append('Low')
                else:
                    predicted.append('Extreme')
            elif cold_vh_proba[i] >= t_cold_vh_e and (
                    e_proba[i] >= t_cold_vh_soft or ord_proba[i] >= t_cold_ord):
                # Moderate VH: cold_vh confident + minimal e_proba or ordinal risk signal.
                # No-risk FPs have e_proba < 0.009 AND ord_proba < 0.08 — blocked by both gates.
                predicted.append('Very High')
            elif h_proba[i] >= t_h_cold:
                predicted.append('High')
            elif cold_low_proba[i] >= t_cold_low:
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

    _RISK_NUM = {'No risk': 0, 'Low': 1, 'High': 2, 'Very High': 3, 'Extreme': 4}
    results = pd.DataFrame({
        'user_name':      per_user['user_name'].values,
        'predicted_risk': predicted,
        'risk_num':       [_RISK_NUM[r] for r in predicted],
        'is_cold':        cold_mask,
        'vh_proba':       np.round(vh_proba, 4),
        'extreme_proba':  np.round(e_proba, 4),
        'high_proba':     np.round(h_proba, 4),
        'low_proba':      np.round(l_proba, 4),
    })

    # For users with new (cold) subscriptions: use their cold-path prediction.
    # For warm-only users: use their highest-risk warm prediction.
    cold_r = results[results['is_cold']]
    warm_r = results[~results['is_cold']]
    if not cold_r.empty:
        cold_best = cold_r.loc[cold_r.groupby('user_name')['risk_num'].idxmax()]
    else:
        cold_best = cold_r
    cold_users = set(cold_best['user_name'])
    warm_only = warm_r[~warm_r['user_name'].isin(cold_users)]
    if not warm_only.empty:
        warm_best = warm_only.loc[warm_only.groupby('user_name')['risk_num'].idxmax()]
    else:
        warm_best = warm_only
    results = pd.concat([cold_best, warm_best], ignore_index=True)
    results = results.drop(columns=['risk_num', 'is_cold'])

    if output:
        results.to_csv(output, index=False)
        print(f"Saved {len(results)} predictions to {output}", flush=True)
    else:
        print(results.to_string(), flush=True)

    print("\nDistribution:", flush=True)
    print(results['predicted_risk'].value_counts().to_string(), flush=True)

    print("\nUpdating risk levels in DB...", flush=True)
    predictions_dict = dict(zip(results['user_name'], results['predicted_risk']))
    updated = update_risk_levels(predictions_dict)
    print(f"  Updated {updated} rows.", flush=True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--predict',    action='store_true')
    parser.add_argument('--model-dir',  default='models_cpu')
    parser.add_argument('--users',      nargs='+')
    parser.add_argument('--output',     default=None)
    parser.add_argument('--export-js',  default=None, metavar='PATH')
    args = parser.parse_args()

    if args.train:
        train_and_save(args.model_dir, export_js=args.export_js)
    elif args.predict:
        predict(args.model_dir, users=args.users, output=args.output)
    else:
        parser.print_help()
