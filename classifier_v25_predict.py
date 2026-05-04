"""
classifier_v25_predict.py — Train final models and classify users

Usage:
  python classifier_v25_predict.py --train
      Train all models on full dataset, save to models/ directory.
      Takes ~60-90 min on first run.

  python classifier_v25_predict.py --predict
      Classify all users currently in the database.

  python classifier_v25_predict.py --predict --users alice bob charlie
      Classify specific users by name.

  python classifier_v25_predict.py --predict --output results.csv
      Save predictions to CSV instead of printing.

Architecture (matches v25 evaluation):
  - Cascade models (XGB, GPU): Extreme vs NR, High vs NR, Low vs NR, Ordinal (VH+E vs rest)
  - VH model (LGB-nl127, CPU): trained on base features + 4 cascade probas
  - Each model saved as an ensemble of N seeds; inference averages probas
  - Thresholds found via OOF on training data
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = Path(__file__).parent / 'models'

WINDOWS_MINUTES = [30, 60, 60*4, 60*6, 60*7, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_STRS     = ['0h30m', '1h', '4h', '6h', '7h', '8h', '10h', '12h', '24h', '168h', '336h', '672h']
ID_RANGES  = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
RANGE_TAGS = ['10k', '20k', '50k', '100k', '200k', '500k', '1m']
KEY_WS     = {'6h', '7h', '8h', '10h', '12h', '24h'}
SHORT_WS   = {'0h30m', '1h', '4h', '6h', '7h', '8h'}
DENSITY_WH = [1, 4, 6, 12, 24]

FEATURES = (
    [f'log_min_id_diff_{w}' for w in WINDOW_STRS] +
    [f'partner_count_{w}'   for w in WINDOW_STRS] +
    [f'cic_{tag}_{w}'      for tag in RANGE_TAGS for w in WINDOW_STRS] +
    [f'cic_frac_{tag}_{w}' for tag in RANGE_TAGS for w in KEY_WS] +
    [f'log_id_span_{w}'        for w in sorted(KEY_WS)] +
    [f'log_id_std_{w}'         for w in sorted(KEY_WS)] +
    [f'log_min_consec_gap_{w}' for w in sorted(KEY_WS)] +
    [f'model_subs_{w}h'    for w in DENSITY_WH] +
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


# ── Feature helpers ────────────────────────────────────────────────────────────

def _is_u_digits(name):
    return 1.0 if (isinstance(name, str) and len(name) > 1 and name[0] == 'u' and name[1:].isdigit()) else 0.0

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


def _model_norisk_median_gpu(df):
    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    records = []
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if nr.empty:
            continue
        all_t  = torch.tensor(mdf['subscribed_at'].values.astype(np.int64), device=DEVICE)
        all_id = torch.tensor(mdf['user_id_num'].values, device=DEVICE, dtype=torch.float32)
        all_names = mdf['user_name'].values
        for t_ns, uid, uname in zip(nr['subscribed_at'].values.astype(np.int64),
                                     nr['user_id_num'].values, nr['user_name'].values):
            t_t = torch.tensor(t_ns, device=DEVICE)
            mask = ((all_t - t_t).abs() <= max_w_ns) & torch.tensor(all_names != uname, device=DEVICE)
            if mask.any():
                records.append({'tracking_model_name': model_name,
                                'min_id_diff': (all_id[mask] - uid).abs().min().item()})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).groupby('tracking_model_name')['min_id_diff'].median()


def compute_per_user(df):
    """Compute per-user feature matrix from a full DB snapshot."""
    print("  Computing model statistics...", flush=True)
    model_norisk_median = _model_norisk_median_gpu(df)
    cross = pd.concat([
        df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count'),
        df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk'),
    ], axis=1)
    model_norisk_rates = {}
    model_risk_fracs = {}
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        n_total = len(mdf)
        if len(nr) >= 2:
            t_range = (nr['subscribed_at'].max() - nr['subscribed_at'].min()).total_seconds() / 86400
            model_norisk_rates[model_name] = len(nr) / max(t_range, 1.0)
        else:
            model_norisk_rates[model_name] = 1.0
        model_risk_fracs[model_name] = {
            'vh':      (mdf['risk_level'] == 'Very High').sum() / n_total,
            'extreme': (mdf['risk_level'] == 'Extreme').sum()   / n_total,
            'any':     (~mdf['risk_level'].isin(['No risk'])).sum() / n_total,
        }

    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    w_ns_list     = [int(w * 60 * 1e9) for w in WINDOWS_MINUTES]
    density_ns    = [int(w * 3600 * 1e9) for w in DENSITY_WH]
    _5min_ns      = int(5 * 60 * 1e9)
    all_rows = []

    print("  Computing pairwise GPU features...", flush=True)
    for model_name, mdf in df.groupby('tracking_model_name'):
        n = len(mdf)
        if n == 0:
            continue
        mdf = mdf.reset_index(drop=True)
        times_ns    = mdf['subscribed_at'].values.astype(np.int64)
        ids_arr     = mdf['user_id_num'].values.astype(np.float64)
        usernames   = mdf['user_name'].values
        risk_lvls   = mdf['risk_level'].values
        risk_scs    = mdf['risk_score'].values
        ts_arr      = mdf['subscribed_ts'].values.astype(np.float64)
        chargebacks = mdf['total_chargebacks'].values.astype(np.float64)

        t_gpu  = torch.tensor(times_ns, device=DEVICE, dtype=torch.int64)
        id_gpu = torch.tensor(ids_arr,  device=DEVICE, dtype=torch.float32)
        tdiff_gpu  = (t_gpu.unsqueeze(0) - t_gpu.unsqueeze(1)).abs()
        iddiff_gpu = (id_gpu.unsqueeze(0) - id_gpu.unsqueeze(1)).abs()
        not_self   = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        w_ns_t     = torch.tensor(w_ns_list, device=DEVICE, dtype=torch.int64)
        all_masks  = (tdiff_gpu.unsqueeze(0) <= w_ns_t[:, None, None]) & not_self.unsqueeze(0)
        d_ns_t     = torch.tensor(density_ns, device=DEVICE, dtype=torch.int64)
        dens_masks = (tdiff_gpu.unsqueeze(0) <= d_ns_t[:, None, None])

        tdiff_cpu    = tdiff_gpu.cpu().numpy()
        iddiff_cpu   = iddiff_gpu.cpu().numpy()
        masks_cpu    = all_masks.cpu().numpy()
        not_self_cpu = not_self.cpu().numpy()
        dens_cpu     = dens_masks.cpu().numpy()
        norisk_med   = model_norisk_median.get(model_name, np.nan)
        norisk_rate  = model_norisk_rates.get(model_name, 1.0)
        risk_fracs   = model_risk_fracs.get(model_name, {'vh': 0., 'extreme': 0., 'any': 0.})
        sorted_ids   = np.sort(ids_arr)
        sorted_ts    = np.sort(times_ns)
        first_sub_ns = times_ns.min()

        for i in range(n):
            row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
            n_empty_short = 0
            for wi, (_, w_str) in enumerate(zip(WINDOWS_MINUTES, WINDOW_STRS)):
                mask   = masks_cpu[wi, i]
                is_key = w_str in KEY_WS
                is_sh  = w_str in SHORT_WS
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
                    if is_sh:
                        n_empty_short += 1
                else:
                    diffs = iddiff_cpu[i, mask]
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
                        row[f'log_id_std_{w_str}']  = np.log1p(float(np.std(pids))) if pc > 1 else 0.0
                        row[f'log_min_consec_gap_{w_str}'] = (
                            np.log1p(float(np.diff(np.sort(pids)).min())) if pc >= 2 else np.nan)

            for j in range(len(WINDOWS_MINUTES) - 1):
                ws, nws = WINDOW_STRS[j], WINDOW_STRS[j + 1]
                for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in RANGE_TAGS]:
                    k, nk = f'{base}_{ws}', f'{base}_{nws}'
                    if np.isnan(row.get(k, np.nan)):
                        row[k] = row.get(nk, np.nan)

            for di, w_h in enumerate(DENSITY_WH):
                n_subs = float(dens_cpu[di, i].sum())
                row[f'model_subs_{w_h}h']      = n_subs
                row[f'model_subs_rate_{w_h}h'] = n_subs / max(norisk_rate * w_h * 2 / 24.0, 1.0)

            gm = not_self_cpu[i]
            if gm.any():
                gdiffs = iddiff_cpu[i, gm]
                gpids  = ids_arr[gm]
                row['log_min_id_diff_global'] = np.log1p(gdiffs.min())
                row['log_global_id_span']     = np.log1p(float(gpids.max() - gpids.min()))
            else:
                row['log_min_id_diff_global'] = np.nan
                row['log_global_id_span']     = np.nan

            all_mask_i = (tdiff_cpu[i] <= max_w_ns) & not_self_cpu[i]
            row['log_min_time_diff'] = (
                np.log1p(tdiff_cpu[i, all_mask_i].min() / 1e9 / 60) if all_mask_i.any() else np.nan)
            row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
            raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
            row['rel_min_id_diff_24h'] = (
                raw_24h / norisk_med if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0 else np.nan)

            uname = usernames[i]
            row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
            gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else risk_scs[i]
            row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
            row['log_user_id_num'] = np.log1p(ids_arr[i]) if not np.isnan(ids_arr[i]) else np.nan

            c6     = row.get('cic_10k_6h',  np.nan)
            c12    = row.get('cic_10k_12h', np.nan)
            c50_6  = row.get('cic_50k_6h',  np.nan)
            c50_24 = row.get('cic_50k_24h', np.nan)
            pc6    = row.get('partner_count_6h',  np.nan)
            pc24   = row.get('partner_count_24h', np.nan)
            row['cic_ratio_6h_24h']    = c50_6 / (c50_24 + 1) if not np.isnan(c50_6)  else np.nan
            row['cic10k_ratio_6h_12h'] = c6    / (c12    + 1) if not np.isnan(c6)     else np.nan
            row['pc_ratio_6h_24h']     = pc6   / (pc24   + 1) if not np.isnan(pc6)    else np.nan

            ts   = ts_arr[i]
            hour = (ts % 86400) / 3600.0
            row['hour_sin']       = float(np.sin(2 * np.pi * hour / 24))
            row['hour_cos']       = float(np.cos(2 * np.pi * hour / 24))
            row['day_of_week']    = float(int(ts // 86400 + 4) % 7)
            row['total_chargebacks']        = chargebacks[i]
            row['frac_empty_short_windows'] = n_empty_short / len(SHORT_WS)
            row['username_is_u_digits']     = _is_u_digits(uname)
            row['username_len']             = float(len(uname)) if isinstance(uname, str) else np.nan
            row['model_norisk_rate']        = norisk_rate
            row['username_digit_ratio']     = _digit_ratio(uname)
            row['username_entropy']         = _entropy(uname)
            row['model_vh_frac']            = risk_fracs['vh']
            row['model_extreme_frac']       = risk_fracs['extreme']
            row['model_any_risk_frac']      = risk_fracs['any']
            row['id_percentile_in_model']   = float(np.searchsorted(sorted_ids, ids_arr[i])) / n
            row['time_percentile_in_model'] = float(np.searchsorted(sorted_ts,  times_ns[i])) / n
            row['log_time_since_model_first_sub'] = np.log1p(
                (times_ns[i] - first_sub_ns) / 1e9 / 3600.0)
            row['n_subs_at_sub_time_5m'] = float((tdiff_cpu[i] <= _5min_ns).sum())
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ── Model training helpers ─────────────────────────────────────────────────────

def find_threshold(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    return float(thr[idx]) if idx < len(thr) else 0.5, float(f1[idx])


def train_ensemble(X, y, clf_factory, seeds, label=""):
    """Train N models with different seeds on the full dataset. Returns list of fitted models."""
    models = []
    for seed in seeds:
        clf = clf_factory(seed)
        clf.fit(X, y)
        models.append(clf)
    return models


def oof_proba(X, y, clf_factory, seeds, n_folds=8):
    """OOF probas (for threshold calibration, no data leakage)."""
    pool = np.zeros(len(y))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        pr = np.zeros(len(y))
        for ti, vi in cv.split(X, y):
            clf = clf_factory(seed)
            clf.fit(X[ti], y[ti])
            pr[vi] = clf.predict_proba(X[vi])[:, 1]
        pool += pr
    return pool / len(seeds)


def predict_ensemble(models, X):
    """Average probas from an ensemble of models."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ── Train ──────────────────────────────────────────────────────────────────────

def train():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading data...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    print(f"  {len(df)} rows, {df['risk_level'].value_counts().to_dict()}", flush=True)

    print("Computing features...", flush=True)
    per_user = compute_per_user(df)
    feats = [f for f in FEATURES if f in per_user.columns]
    print(f"  {len(feats)} features, {len(per_user)} users", flush=True)

    # Save feature list
    with open(MODELS_DIR / 'features.json', 'w') as f:
        json.dump(feats, f)

    thresholds = {}
    SEEDS_C = [42, 7, 123, 999, 2025]

    xgb_device = 'cuda' if DEVICE.type == 'cuda' else 'cpu'

    def xgb_factory(d, lr, n, sw, ss=0.85, cs=0.85):
        return lambda seed: XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=lr, subsample=ss,
            colsample_bytree=cs, scale_pos_weight=sw, eval_metric='logloss',
            verbosity=0, device=xgb_device, random_state=seed)

    def lgb_factory(nl, lr, n, sw, ss=0.80, cs=0.80, mcw=20):
        return lambda seed: LGBMClassifier(
            n_estimators=n, num_leaves=nl, learning_rate=lr, subsample=ss,
            colsample_bytree=cs, scale_pos_weight=sw, min_child_samples=mcw,
            subsample_freq=1, verbose=-1, n_jobs=-1, random_state=seed)

    # ── Cascade: Extreme ──────────────────────────────────────────────────────
    print("\n[1/5] Training Extreme cascade...", flush=True)
    sub_e = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
    sub_e['y'] = (sub_e['risk_level'] == 'Extreme').astype(int)
    Xe, ye = sub_e[feats].values, sub_e['y'].values
    spw_e = (ye == 0).sum() / ye.sum()
    factory_e = xgb_factory(6, 0.03, 600, spw_e, 0.80, 0.85)
    p_e_oof = oof_proba(Xe, ye, factory_e, SEEDS_C, n_folds=8)
    t_e, f_e = find_threshold(ye, p_e_oof)
    thresholds['extreme'] = t_e
    models_e = train_ensemble(Xe, ye, factory_e, SEEDS_C)
    extreme_lookup = dict(zip(sub_e['user_name'].values,
                               predict_ensemble(models_e, Xe)))
    p, r, f, _ = precision_recall_fscore_support(ye, (p_e_oof >= t_e).astype(int),
                                                   pos_label=1, average='binary', zero_division=0)
    print(f"  Extreme OOF: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
    with open(MODELS_DIR / 'extreme_models.pkl', 'wb') as fh:
        pickle.dump({'models': models_e, 'feats': feats}, fh)

    # ── Cascade: High ─────────────────────────────────────────────────────────
    print("[2/5] Training High cascade...", flush=True)
    sub_h = per_user[per_user['risk_level'].isin(['High', 'No risk'])].copy()
    sub_h['y'] = (sub_h['risk_level'] == 'High').astype(int)
    Xh, yh = sub_h[feats].values, sub_h['y'].values
    spw_h = (yh == 0).sum() / yh.sum()
    factory_h = xgb_factory(6, 0.03, 500, spw_h, 0.75, 0.80)
    p_h_oof = oof_proba(Xh, yh, factory_h, SEEDS_C, n_folds=8)
    t_h, f_h = find_threshold(yh, p_h_oof)
    thresholds['high'] = t_h
    models_h = train_ensemble(Xh, yh, factory_h, SEEDS_C)
    high_lookup = dict(zip(sub_h['user_name'].values,
                            predict_ensemble(models_h, Xh)))
    p, r, f, _ = precision_recall_fscore_support(yh, (p_h_oof >= t_h).astype(int),
                                                   pos_label=1, average='binary', zero_division=0)
    print(f"  High OOF: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
    with open(MODELS_DIR / 'high_models.pkl', 'wb') as fh:
        pickle.dump({'models': models_h, 'feats': feats}, fh)

    # ── Cascade: Low ──────────────────────────────────────────────────────────
    print("[3/5] Training Low cascade...", flush=True)
    sub_l = per_user[per_user['risk_level'].isin(['Low', 'No risk'])].copy()
    sub_l['y'] = (sub_l['risk_level'] == 'Low').astype(int)
    Xl, yl = sub_l[feats].values, sub_l['y'].values
    spw_l = (yl == 0).sum() / yl.sum()
    factory_l = xgb_factory(6, 0.03, 500, spw_l, 0.75, 0.80)
    p_l_oof = oof_proba(Xl, yl, factory_l, SEEDS_C, n_folds=8)
    t_l, f_l = find_threshold(yl, p_l_oof)
    thresholds['low'] = t_l
    models_l = train_ensemble(Xl, yl, factory_l, SEEDS_C)
    low_lookup = dict(zip(sub_l['user_name'].values,
                           predict_ensemble(models_l, Xl)))
    p, r, f, _ = precision_recall_fscore_support(yl, (p_l_oof >= t_l).astype(int),
                                                   pos_label=1, average='binary', zero_division=0)
    print(f"  Low OOF: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
    with open(MODELS_DIR / 'low_models.pkl', 'wb') as fh:
        pickle.dump({'models': models_l, 'feats': feats}, fh)

    # ── Cascade: Ordinal ──────────────────────────────────────────────────────
    print("[4/5] Training Ordinal cascade (VH+Extreme vs rest)...", flush=True)
    sub_ord = per_user.copy()
    sub_ord['y'] = sub_ord['risk_level'].isin(['Very High', 'Extreme']).astype(int)
    X_ord, y_ord = sub_ord[feats].values, sub_ord['y'].values
    spw_ord = (y_ord == 0).sum() / y_ord.sum()
    factory_ord = xgb_factory(6, 0.03, 500, spw_ord, 0.75, 0.80)
    models_ord = train_ensemble(X_ord, y_ord, factory_ord, SEEDS_C)
    ordinal_lookup = dict(zip(sub_ord['user_name'].values,
                               predict_ensemble(models_ord, X_ord)))
    with open(MODELS_DIR / 'ordinal_models.pkl', 'wb') as fh:
        pickle.dump({'models': models_ord, 'feats': feats}, fh)
    print("  Ordinal cascade done.", flush=True)

    # ── VH model ──────────────────────────────────────────────────────────────
    print("[5/5] Training Very High model (LGB-nl127)...", flush=True)
    sub_vh = per_user[per_user['risk_level'].isin(['Very High', 'No risk'])].copy().reset_index(drop=True)
    y_vh = (sub_vh['risk_level'] == 'Very High').astype(int).values
    X_vh = sub_vh[feats].values
    spw_vh = (y_vh == 0).sum() / y_vh.sum()

    casc_e   = np.array([extreme_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    casc_h   = np.array([high_lookup.get(u, 0.0)    for u in sub_vh['user_name']])
    casc_l   = np.array([low_lookup.get(u, 0.0)     for u in sub_vh['user_name']])
    casc_ord = np.array([ordinal_lookup.get(u, 0.0)  for u in sub_vh['user_name']])
    X_vh_aug = np.column_stack([X_vh, casc_e, casc_h, casc_l, casc_ord])

    SEEDS_VH = [42, 7, 123, 999, 2025, 1337, 2024, 101]
    factory_vh = lgb_factory(127, 0.02, 800, spw_vh, 0.75, 0.80, mcw=20)

    # OOF for threshold calibration (uses cascade probas from full-data cascade models above)
    p_vh_oof = oof_proba(X_vh_aug, y_vh, factory_vh, SEEDS_VH, n_folds=12)
    t_vh, f_vh = find_threshold(y_vh, p_vh_oof)
    thresholds['vh'] = t_vh
    p, r, f, _ = precision_recall_fscore_support(y_vh, (p_vh_oof >= t_vh).astype(int),
                                                   pos_label=1, average='binary', zero_division=0)
    print(f"  VH OOF (calibration): P={p:.2%} R={r:.2%} F1={f:.2%} @ thr={t_vh:.3f}", flush=True)

    # Final models on full data
    models_vh = train_ensemble(X_vh_aug, y_vh, factory_vh, SEEDS_VH,
                                label="VH final")
    with open(MODELS_DIR / 'vh_models.pkl', 'wb') as fh:
        pickle.dump({'models': models_vh, 'feats': feats}, fh)

    with open(MODELS_DIR / 'thresholds.json', 'w') as fh:
        json.dump(thresholds, fh, indent=2)

    print(f"\nModels saved to {MODELS_DIR}/", flush=True)
    print(f"Thresholds: {thresholds}", flush=True)
    print("\nTraining complete.", flush=True)


# ── Predict ────────────────────────────────────────────────────────────────────

def load_models():
    """Load all saved models and thresholds."""
    assert MODELS_DIR.exists(), f"Models not found at {MODELS_DIR}. Run with --train first."
    with open(MODELS_DIR / 'thresholds.json') as f:
        thresholds = json.load(f)
    with open(MODELS_DIR / 'features.json') as f:
        feats = json.load(f)
    pkls = {}
    for name in ['extreme', 'high', 'low', 'ordinal', 'vh']:
        with open(MODELS_DIR / f'{name}_models.pkl', 'rb') as f:
            pkls[name] = pickle.load(f)
    return pkls, thresholds, feats


def predict(user_names=None):
    """
    Classify users. If user_names is None, classifies all users in DB.
    Returns a DataFrame with columns: user_name, risk_level, vh_proba,
    extreme_proba, high_proba, low_proba.
    """
    pkls, thresholds, feats = load_models()

    print("Fetching database...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    print(f"  {len(df)} rows loaded.", flush=True)

    print("Computing features...", flush=True)
    per_user = compute_per_user(df)
    feats_present = [f for f in feats if f in per_user.columns]

    # Get cascade probas for all users using saved cascade models
    def casc_proba(key, sub, feats_p):
        X = sub[feats_p].values
        return predict_ensemble(pkls[key]['models'], X)

    # Build cascade lookups
    sub_all = per_user
    X_all = sub_all[feats_present].values

    p_extreme_all = predict_ensemble(pkls['extreme']['models'], X_all)
    p_high_all    = predict_ensemble(pkls['high']['models'],    X_all)
    p_low_all     = predict_ensemble(pkls['low']['models'],     X_all)
    p_ord_all     = predict_ensemble(pkls['ordinal']['models'], X_all)

    extreme_lookup = dict(zip(sub_all['user_name'].values, p_extreme_all))
    high_lookup    = dict(zip(sub_all['user_name'].values, p_high_all))
    low_lookup     = dict(zip(sub_all['user_name'].values, p_low_all))
    ord_lookup     = dict(zip(sub_all['user_name'].values, p_ord_all))

    # VH probas (augmented features)
    casc_e   = np.array([extreme_lookup.get(u, 0.0) for u in sub_all['user_name']])
    casc_h   = np.array([high_lookup.get(u, 0.0)    for u in sub_all['user_name']])
    casc_l   = np.array([low_lookup.get(u, 0.0)     for u in sub_all['user_name']])
    casc_ord = np.array([ord_lookup.get(u, 0.0)      for u in sub_all['user_name']])
    X_aug    = np.column_stack([X_all, casc_e, casc_h, casc_l, casc_ord])
    p_vh_all = predict_ensemble(pkls['vh']['models'], X_aug)

    # Build result DataFrame
    result = pd.DataFrame({
        'user_name':     sub_all['user_name'].values,
        'vh_proba':      p_vh_all,
        'extreme_proba': p_extreme_all,
        'high_proba':    p_high_all,
        'low_proba':     p_low_all,
    })

    # Apply thresholds in risk-descending order
    def classify(row):
        if row['extreme_proba'] >= thresholds['extreme']:
            return 'Extreme'
        if row['vh_proba'] >= thresholds['vh']:
            return 'Very High'
        if row['high_proba'] >= thresholds['high']:
            return 'High'
        if row['low_proba'] >= thresholds['low']:
            return 'Low'
        return 'No risk'

    result['predicted_risk'] = result.apply(classify, axis=1)

    # Filter to requested users
    if user_names:
        result = result[result['user_name'].isin(user_names)]
        missing = set(user_names) - set(result['user_name'])
        if missing:
            print(f"  Warning: {len(missing)} user(s) not found in DB: {missing}", flush=True)

    return result[['user_name', 'predicted_risk', 'vh_proba', 'extreme_proba',
                   'high_proba', 'low_proba']].sort_values(
                       'predicted_risk',
                       key=lambda s: s.map({'Extreme': 0, 'Very High': 1, 'High': 2, 'Low': 3, 'No risk': 4})
                   ).reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',   action='store_true', help='Train and save models')
    parser.add_argument('--predict', action='store_true', help='Predict risk levels')
    parser.add_argument('--users',   nargs='+', default=None,
                        help='User names to classify (default: all)')
    parser.add_argument('--output',  default=None,
                        help='CSV output path (default: print to stdout)')
    args = parser.parse_args()

    if args.train:
        train()

    if args.predict:
        result = predict(user_names=args.users)
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"Saved {len(result)} predictions to {args.output}")
        else:
            pd.set_option('display.max_rows', 200)
            pd.set_option('display.float_format', '{:.3f}'.format)
            print(result.to_string(index=False))

    if not args.train and not args.predict:
        parser.print_help()
