"""
classifier_v22.py — Push VH past 95% with stronger cascade + optimization

v21 breakthrough: Extreme OOF proba as feature → VH jumped to 93.62%.
Gap to 95%: only 1.38%.

Strategy for v22:
1. Stronger Extreme model (more seeds/folds → better OOF proba signal)
2. Also add High + Low OOF probas as cascade features
3. More VH cascade configs with optimized depth/spw/colsample
4. Threshold at 0.80 sweep showed 93.04% — optimize the blend ratio/threshold
5. More seeds (12) for VH cascade models
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)

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
     'username_is_u_digits', 'username_len', 'model_norisk_rate']
)

RISK_ORDER = ['Extreme', 'Very High', 'High', 'Low']


def _is_u_digits(name):
    return 1.0 if (isinstance(name, str) and len(name) > 1 and name[0] == 'u' and name[1:].isdigit()) else 0.0


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
    print("  Precomputing model stats (GPU)...", flush=True)
    model_norisk_median = _model_norisk_median_gpu(df)
    print("  Precomputing cross-model stats...", flush=True)
    cross = pd.concat([
        df.groupby('user_name')['tracking_model_name'].nunique().rename('user_model_count'),
        df.groupby('user_name')['risk_score'].max().rename('user_global_max_risk'),
    ], axis=1)
    print("  Precomputing model subscription rates...", flush=True)
    model_norisk_rates = {}
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if len(nr) >= 2:
            t_range = (nr['subscribed_at'].max() - nr['subscribed_at'].min()).total_seconds() / 86400
            model_norisk_rates[model_name] = len(nr) / max(t_range, 1.0)
        else:
            model_norisk_rates[model_name] = 1.0

    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    w_ns_list = [int(w * 60 * 1e9) for w in WINDOWS_MINUTES]
    density_ns_list = [int(w * 3600 * 1e9) for w in DENSITY_WH]
    all_rows = []

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
        w_ns_t = torch.tensor(w_ns_list, device=DEVICE, dtype=torch.int64)
        all_masks  = (tdiff_gpu.unsqueeze(0) <= w_ns_t[:, None, None]) & not_self.unsqueeze(0)
        d_ns_t = torch.tensor(density_ns_list, device=DEVICE, dtype=torch.int64)
        density_masks = (tdiff_gpu.unsqueeze(0) <= d_ns_t[:, None, None])

        tdiff_cpu    = tdiff_gpu.cpu().numpy()
        iddiff_cpu   = iddiff_gpu.cpu().numpy()
        masks_cpu    = all_masks.cpu().numpy()
        not_self_cpu = not_self.cpu().numpy()
        global_cpu   = not_self.cpu().numpy()
        density_cpu  = density_masks.cpu().numpy()
        norisk_med   = model_norisk_median.get(model_name, np.nan)
        norisk_rate  = model_norisk_rates.get(model_name, 1.0)

        for i in range(n):
            row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
            n_empty_short = 0
            for wi, (w_min, w_str) in enumerate(zip(WINDOWS_MINUTES, WINDOW_STRS)):
                mask = masks_cpu[wi, i]
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
                n_subs = float(density_cpu[di, i].sum())
                w_days = w_h * 2 / 24.0
                row[f'model_subs_{w_h}h']      = n_subs
                row[f'model_subs_rate_{w_h}h'] = n_subs / max(norisk_rate * w_days, 1.0)

            gm = global_cpu[i]
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

            c6   = row.get('cic_10k_6h',  np.nan)
            c12  = row.get('cic_10k_12h', np.nan)
            c50_6  = row.get('cic_50k_6h',  np.nan)
            c50_24 = row.get('cic_50k_24h', np.nan)
            pc6  = row.get('partner_count_6h',  np.nan)
            pc24 = row.get('partner_count_24h', np.nan)
            row['cic_ratio_6h_24h']    = c50_6 / (c50_24 + 1) if not np.isnan(c50_6)  else np.nan
            row['cic10k_ratio_6h_12h'] = c6    / (c12    + 1) if not np.isnan(c6)     else np.nan
            row['pc_ratio_6h_24h']     = pc6   / (pc24   + 1) if not np.isnan(pc6)    else np.nan

            ts = ts_arr[i]
            hour = (ts % 86400) / 3600.0
            row['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
            row['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
            row['day_of_week'] = float(int(ts // 86400 + 4) % 7)
            row['total_chargebacks'] = chargebacks[i]
            row['frac_empty_short_windows'] = n_empty_short / len(SHORT_WS)
            row['username_is_u_digits'] = _is_u_digits(uname)
            row['username_len'] = float(len(uname)) if isinstance(uname, str) else np.nan
            row['model_norisk_rate'] = norisk_rate
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1  = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[f1.argmax()]) if f1.argmax() < len(thresholds) else 0.5


def build_oof(X, y, configs, seeds, n_folds, label=""):
    """OOF for a list of XGB param dicts. Returns mean proba."""
    pools = []
    for name, params in configs:
        pool = np.zeros(len(y))
        for seed in seeds:
            clf = XGBClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y))
            for ti, vi in cv.split(X, y):
                clf.fit(X[ti], y[ti])
                pr[vi] = clf.predict_proba(X[vi])[:, 1]
            pool += pr
        pool /= len(seeds)
        t_i = find_best_threshold(y, pool)
        p, r, f, _ = precision_recall_fscore_support(y, (pool >= t_i).astype(int), pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        pools.append(pool)
    return np.mean(pools, axis=0) if pools else np.zeros(len(y))


if __name__ == '__main__':
    print("Loading full dataset (selected=False)...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    print(f"  {len(df)} rows", flush=True)
    print(df['risk_level'].value_counts().to_string(), flush=True)

    print("\nComputing per-user features (GPU pairwise)...", flush=True)
    per_user = compute_per_user(df)
    print(f"Total per-user rows: {len(per_user)}", flush=True)

    feats = [f for f in FEATURES if f in per_user.columns]
    print(f"Features: {len(feats)}", flush=True)

    def xgb(d, lr, n, sw, ss=0.85, cs=0.85, mcw=1, g=0.0):
        return dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=ss,
                    colsample_bytree=cs, scale_pos_weight=sw, min_child_weight=mcw,
                    gamma=g, eval_metric='logloss', verbosity=0, device='cuda')

    # ── Step 1: Build cascade feature probas (Extreme, High, Low) ──────────────
    print("\n=== Step 1: Build cascade OOF probas ===", flush=True)
    cascade_probas = {}

    # Extreme vs No Risk
    print("  Extreme OOF (8 seeds × 10-fold):", flush=True)
    sub_e = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
    sub_e['y'] = (sub_e['risk_level'] == 'Extreme').astype(int)
    Xe, ye = sub_e[feats].values, sub_e['y'].values
    spw_e = (ye == 0).sum() / ye.sum()
    extreme_cfgs = [
        (f"E-d5-spw{spw_e:.0f}",   xgb(5, 0.03, 600, spw_e,      0.80, 0.85)),
        (f"E-d5-spw{spw_e*2:.0f}", xgb(5, 0.04, 500, min(spw_e*2, 200), 0.85, 0.75)),
        (f"E-d6-spw{spw_e:.0f}",   xgb(6, 0.03, 500, spw_e,      0.75, 0.80)),
    ]
    p_extreme_all = build_oof(Xe, ye, extreme_cfgs, [42, 7, 123, 999, 2025, 1337, 2024, 101], 10)
    # Build lookup: user_name → extreme proba
    extreme_lookup = dict(zip(sub_e.reset_index()['user_name'], p_extreme_all))
    print(f"  Extreme OOF done (range: {p_extreme_all.min():.3f}-{p_extreme_all.max():.3f})", flush=True)

    # High vs No Risk
    print("  High OOF (6 seeds × 8-fold):", flush=True)
    sub_h = per_user[per_user['risk_level'].isin(['High', 'No risk'])].copy()
    sub_h['y'] = (sub_h['risk_level'] == 'High').astype(int)
    Xh, yh = sub_h[feats].values, sub_h['y'].values
    spw_h = (yh == 0).sum() / yh.sum()
    high_cfgs = [
        (f"H-d5-spw{spw_h:.0f}", xgb(5, 0.03, 600, spw_h, 0.80, 0.85)),
        (f"H-d6-spw{spw_h:.0f}", xgb(6, 0.03, 500, spw_h, 0.75, 0.80)),
    ]
    p_high_all = build_oof(Xh, yh, high_cfgs, [42, 7, 123, 999, 2025, 1337], 8)
    high_lookup = dict(zip(sub_h.reset_index()['user_name'], p_high_all))
    print(f"  High OOF done.", flush=True)

    # Low vs No Risk
    print("  Low OOF (6 seeds × 8-fold):", flush=True)
    sub_l = per_user[per_user['risk_level'].isin(['Low', 'No risk'])].copy()
    sub_l['y'] = (sub_l['risk_level'] == 'Low').astype(int)
    Xl, yl = sub_l[feats].values, sub_l['y'].values
    spw_l = (yl == 0).sum() / yl.sum()
    low_cfgs = [
        (f"L-d5-spw{spw_l:.0f}", xgb(5, 0.03, 600, spw_l, 0.80, 0.85)),
        (f"L-d6-spw{spw_l:.0f}", xgb(6, 0.03, 500, spw_l, 0.75, 0.80)),
    ]
    p_low_all = build_oof(Xl, yl, low_cfgs, [42, 7, 123, 999, 2025, 1337], 8)
    low_lookup = dict(zip(sub_l.reset_index()['user_name'], p_low_all))
    print(f"  Low OOF done.", flush=True)

    # ── Step 2: Very High classifier with cascade features ─────────────────────
    print("\n=== Step 2: Very High + cascade features ===", flush=True)
    sub_vh = per_user[per_user['risk_level'].isin(['Very High', 'No risk'])].copy().reset_index()
    y_vh = (sub_vh['risk_level'] == 'Very High').astype(int).values
    X_vh = sub_vh[feats].values
    n_pos = y_vh.sum()
    n_neg = (y_vh == 0).sum()
    print(f"  {n_pos} pos, {n_neg} neg", flush=True)

    # Build cascade feature columns for VH+NoRisk subset
    casc_extreme = np.array([extreme_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    casc_high    = np.array([high_lookup.get(u, 0.0)    for u in sub_vh['user_name']])
    casc_low     = np.array([low_lookup.get(u, 0.0)     for u in sub_vh['user_name']])

    print(f"  Cascade stats — Extreme: mean={casc_extreme.mean():.3f} p99={np.percentile(casc_extreme, 99):.3f}", flush=True)
    print(f"                  High:    mean={casc_high.mean():.3f}    Low: mean={casc_low.mean():.3f}", flush=True)

    # Augment feature matrix
    X_vh_aug = np.column_stack([X_vh, casc_extreme, casc_high, casc_low])

    spw_vh = n_neg / n_pos
    spw_vh2 = min(spw_vh * 2, 200)
    SEEDS_VH = [42, 7, 123, 999, 2025, 1337, 2024, 101, 314, 271, 42424, 77777]
    N_FOLDS_VH = 12

    vh_cfgs = [
        (f"VH-d5-spw{spw_vh:.0f}",   xgb(5, 0.03, 600, spw_vh,  0.80, 0.85)),
        (f"VH-d5-spw{spw_vh2:.0f}",  xgb(5, 0.04, 500, spw_vh2, 0.85, 0.75)),
        (f"VH-d6-spw{spw_vh:.0f}",   xgb(6, 0.03, 500, spw_vh,  0.75, 0.80)),
        (f"VH-d6-spw{spw_vh2:.0f}",  xgb(6, 0.02, 600, spw_vh2, 0.75, 0.80)),
        (f"VH-d5-spw{spw_vh:.0f}-cs06", xgb(5, 0.03, 600, spw_vh,  0.80, 0.60)),
        (f"VH-d7-spw{spw_vh:.0f}",   xgb(7, 0.02, 600, spw_vh,  0.75, 0.80)),
    ]

    print(f"  VH cascade: {len(vh_cfgs)} configs × {len(SEEDS_VH)} seeds × {N_FOLDS_VH}-fold:", flush=True)
    vh_pools = []
    for name, params in vh_cfgs:
        pool = np.zeros(len(y_vh))
        for seed in SEEDS_VH:
            clf = XGBClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=N_FOLDS_VH, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y_vh))
            for ti, vi in cv.split(X_vh_aug, y_vh):
                clf.fit(X_vh_aug[ti], y_vh[ti])
                pr[vi] = clf.predict_proba(X_vh_aug[vi])[:, 1]
            pool += pr
        pool /= len(SEEDS_VH)
        t_i = find_best_threshold(y_vh, pool)
        p, r, f, _ = precision_recall_fscore_support(y_vh, (pool >= t_i).astype(int), pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        vh_pools.append(pool)

    y_vh_ens = np.mean(vh_pools, axis=0)
    t_vh = find_best_threshold(y_vh, y_vh_ens)
    yp_vh = (y_vh_ens >= t_vh).astype(int)
    p_vh, r_vh, f_vh, _ = precision_recall_fscore_support(y_vh, yp_vh, pos_label=1, average='binary', zero_division=0)
    print(f"\n  VH final: P={p_vh:.2%} R={r_vh:.2%} F1={f_vh:.2%} @ {t_vh:.3f}", flush=True)

    print(f"\n  Sweep:", flush=True)
    for thr in np.arange(0.2, 0.98, 0.02):
        yt = (y_vh_ens >= thr).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y_vh, yt, pos_label=1, average='binary', zero_division=0)
        print(f"    {thr:.2f}: P={pt:.2%} R={rt:.2%} F1={ft:.2%}", flush=True)

    # ── Step 3: Other categories ────────────────────────────────────────────────
    results = {'Very High': f_vh}

    for level in ['Extreme', 'High', 'Low']:
        if level not in per_user['risk_level'].values:
            continue
        print(f"\n=== {level} ===", flush=True)
        sub = per_user[per_user['risk_level'].isin([level, 'No risk'])].copy()
        sub['y'] = (sub['risk_level'] == level).astype(int)
        X_s, y_s = sub[feats].values, sub['y'].values
        n_p, n_n = y_s.sum(), (y_s == 0).sum()
        print(f"  {n_p} pos, {n_n} neg", flush=True)
        spw_s = n_n / n_p
        spw_s2 = min(spw_s * 2, 200)
        cfgs_s = [
            (f"d5-spw{spw_s:.0f}",  xgb(5, 0.03, 600, spw_s,  0.80, 0.85)),
            (f"d5-spw{spw_s2:.0f}", xgb(5, 0.04, 500, spw_s2, 0.85, 0.75)),
            (f"d6-spw{spw_s:.0f}",  xgb(6, 0.03, 500, spw_s,  0.75, 0.80)),
        ]
        p_s = build_oof(X_s, y_s, cfgs_s, [42, 7, 123, 999, 2025, 1337], 8)
        t_s = find_best_threshold(y_s, p_s)
        yp_s = (p_s >= t_s).astype(int)
        ps, rs, fs, _ = precision_recall_fscore_support(y_s, yp_s, pos_label=1, average='binary', zero_division=0)
        print(f"  Final: P={ps:.2%} R={rs:.2%} F1={fs:.2%}", flush=True)
        results[level] = fs

    print(f"\n{'='*60}")
    print(f"SUMMARY — v22 cascade+, selected=False")
    print(f"{'='*60}")
    for level in RISK_ORDER:
        if level in results:
            tag = " *** ACHIEVED ***" if results[level] >= 0.95 else f"  gap={0.95-results[level]:.2%}"
            print(f"  {level:<12}: F1={results[level]:.2%}{tag}")
