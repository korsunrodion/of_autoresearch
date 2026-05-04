"""
classifier_v21.py — Ordinal framing + multi-class cascade

The core issue: Very High FNs are structurally isolated (no ID proximity).
Standard binary VH vs No Risk has a ceiling ~89%.

New strategies:
1. Ordinal binary: (VH + Extreme) vs (No Risk + Low + High)
   — 7551 positives; Extreme bots share bot-detection signals with VH
   — model learns a wider "risky" boundary
2. Multi-class softmax XGBoost (5-class) → extract P(VH)
3. Cascade: use Extreme OOF proba as an extra feature for VH classifier
4. Blend ordinal, multi-class, and cascade probas
5. Reuse v20's rich feature set (197 features)
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
RISK_NUM   = {'No risk': 0, 'Low': 1, 'High': 2, 'Very High': 3, 'Extreme': 4}


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
        norisk_med  = model_norisk_median.get(model_name, np.nan)
        norisk_rate = model_norisk_rates.get(model_name, 1.0)

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


def xgb_oof(X, y, configs, seeds, n_folds, label=""):
    """Run OOF for list of XGB configs. Returns per-config OOF array and f1s."""
    oof_all, f1s = [], []
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
        yp  = (pool >= t_i).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, yp, pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        oof_all.append(pool)
        f1s.append(f)
    return np.array(oof_all), np.array(f1s)


def classify_vh_multiview(per_user, feats, n_seeds=10, n_folds=10):
    """Multiple views of Very High classification, blended."""
    all_rows_full = per_user.copy()
    all_rows_full['risk_num'] = all_rows_full['risk_level'].map(RISK_NUM).fillna(-1)

    SEEDS = [42, 7, 123, 999, 2025, 1337, 2024, 101, 314, 271][:n_seeds]

    # ── VIEW 1: Standard binary VH vs No Risk ────────────────────────────────
    print("\n  [VIEW 1] VH vs No Risk:", flush=True)
    sub1 = per_user[per_user['risk_level'].isin(['Very High', 'No risk'])].copy()
    sub1['y'] = (sub1['risk_level'] == 'Very High').astype(int)
    X1, y1 = sub1[feats].values, sub1['y'].values
    spw1 = (y1 == 0).sum() / y1.sum()
    spw1h = max(spw1 / 2, 5.0)
    spw1d = min(spw1 * 2, 200)

    def xgb(d, lr, n, sw, ss=0.85, cs=0.85):
        return dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=ss,
                    colsample_bytree=cs, scale_pos_weight=sw,
                    eval_metric='logloss', verbosity=0, device='cuda')

    cfgs1 = [
        (f"d5-spw{spw1:.0f}",   xgb(5, 0.03, 600, spw1,  0.80, 0.85)),
        (f"d5-spw{spw1d:.0f}",  xgb(5, 0.04, 500, spw1d, 0.85, 0.75)),
        (f"d6-spw{spw1:.0f}",   xgb(6, 0.03, 500, spw1,  0.75, 0.80)),
        (f"d6-spw{spw1d:.0f}",  xgb(6, 0.02, 600, spw1d, 0.75, 0.80)),
    ]
    oof1, f1s1 = xgb_oof(X1, y1, cfgs1, SEEDS, n_folds)
    p_view1 = np.average(oof1, axis=0, weights=f1s1 / f1s1.sum())
    t1 = find_best_threshold(y1, p_view1)
    _, _, f1_v1, _ = precision_recall_fscore_support(y1, (p_view1 >= t1).astype(int), pos_label=1, average='binary', zero_division=0)
    print(f"  View1 ensemble F1={f1_v1:.2%}", flush=True)

    # ── VIEW 2: Ordinal — (VH + Extreme) vs (No Risk + Low + High) ───────────
    print("\n  [VIEW 2] (VH+Extreme) vs (No Risk+Low+High):", flush=True)
    sub2 = per_user[per_user['risk_level'].isin(['Very High', 'Extreme', 'No risk', 'Low', 'High'])].copy()
    sub2['y'] = sub2['risk_level'].isin(['Very High', 'Extreme']).astype(int)
    X2_full, y2_full = sub2[feats].values, sub2['y'].values
    # We only care about OOF for VH rows; mask them
    vh_idx2 = sub2.index[sub2['risk_level'] == 'Very High'].tolist()
    sub2_reset = sub2.reset_index(drop=True)
    vh_mask2 = sub2_reset['risk_level'] == 'Very High'
    X2, y2 = X2_full, y2_full
    spw2 = (y2 == 0).sum() / max(y2.sum(), 1)
    cfgs2 = [
        (f"ord-d5-spw{spw2:.1f}",  xgb(5, 0.03, 600, spw2, 0.80, 0.85)),
        (f"ord-d6-spw{spw2:.1f}",  xgb(6, 0.02, 600, spw2, 0.75, 0.80)),
    ]
    oof2_all, _ = xgb_oof(X2, y2, cfgs2, SEEDS[:6], n_folds)
    p_view2_full = np.mean(oof2_all, axis=0)   # proba for all rows in sub2
    # Extract VH rows' probability (used as signal)
    p_view2_vh = p_view2_full[vh_mask2.values]
    # Also get No Risk rows for alignment
    norisk_mask2 = sub2_reset['risk_level'] == 'No risk'
    p_view2_nr   = p_view2_full[norisk_mask2.values]
    t2 = find_best_threshold(y1, np.concatenate([p_view2_vh, p_view2_nr]) if len(p_view2_vh) == len(y1) else p_view2_full[sub2_reset['risk_level'].isin(['Very High', 'No risk']).values])
    print(f"  View2 ordinal done ({len(sub2)} rows, {y2.sum()} pos)", flush=True)

    # ── VIEW 3: Cascade — Extreme OOF proba as extra feature for VH ──────────
    print("\n  [VIEW 3] Cascade: Extreme proba → VH feature:", flush=True)
    sub_extreme = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
    sub_extreme['y'] = (sub_extreme['risk_level'] == 'Extreme').astype(int)
    Xe, ye = sub_extreme[feats].values, sub_extreme['y'].values
    spw_e = (ye == 0).sum() / ye.sum()
    cfgs_e = [(f"extreme-d5-spw{spw_e:.0f}", xgb(5, 0.03, 600, spw_e, 0.80, 0.85))]
    oof_e, _ = xgb_oof(Xe, ye, cfgs_e, SEEDS[:4], n_folds)
    p_extreme_all = oof_e[0]
    # Map extreme OOF probas back to sub1 (VH + No Risk) rows
    sub1_reset = sub1.reset_index()
    extreme_proba_sub1 = np.zeros(len(sub1_reset))
    # Find index of each row in sub1 within sub_extreme (No Risk overlap)
    extreme_norisk_reset = sub_extreme[sub_extreme['risk_level'] == 'No risk'].reset_index()
    # Quick approximate: for the sub1 No Risk users that appear in sub_extreme, use their proba
    # Build lookup: user_name → extreme proba for No Risk rows
    extreme_lookup = dict(zip(sub_extreme.reset_index()['user_name'],
                               p_extreme_all)) if 'user_name' in sub_extreme.columns else {}
    # sub1 VH users don't appear in Extreme model (different class), use 0.0
    for idx, row in enumerate(sub1_reset.itertuples()):
        uname = getattr(row, 'user_name', None)
        extreme_proba_sub1[idx] = extreme_lookup.get(uname, 0.0)

    # Augment X1 with the extreme proba
    X1_aug = np.column_stack([X1, extreme_proba_sub1.reshape(-1, 1)])
    cfgs3 = [
        (f"casc-d5-spw{spw1:.0f}",  xgb(5, 0.03, 600, spw1,  0.80, 0.85)),
        (f"casc-d6-spw{spw1:.0f}",  xgb(6, 0.03, 500, spw1,  0.75, 0.80)),
    ]
    oof3, f1s3 = xgb_oof(X1_aug, y1, cfgs3, SEEDS[:6], n_folds)
    p_view3 = np.mean(oof3, axis=0)
    t3 = find_best_threshold(y1, p_view3)
    _, _, f1_v3, _ = precision_recall_fscore_support(y1, (p_view3 >= t3).astype(int), pos_label=1, average='binary', zero_division=0)
    print(f"  View3 cascade F1={f1_v3:.2%}", flush=True)

    # ── Ordinal-to-VH alignment: project view2 onto VH vs No Risk ────────────
    # Re-extract view2 OOF probas for the VH+NoRisk subset
    sub2_vhnr = sub2_reset[sub2_reset['risk_level'].isin(['Very High', 'No risk'])].copy()
    p_view2_vhnr = p_view2_full[sub2_reset['risk_level'].isin(['Very High', 'No risk']).values]
    t2_vhnr = find_best_threshold(y1, p_view2_vhnr)
    _, _, f1_v2, _ = precision_recall_fscore_support(y1, (p_view2_vhnr >= t2_vhnr).astype(int), pos_label=1, average='binary', zero_division=0)
    print(f"  View2 ordinal (on VH+NR) F1={f1_v2:.2%}", flush=True)

    # ── Final blend ──────────────────────────────────────────────────────────
    print("\n  Blend sweep (v1, v2, v3):", flush=True)
    best_f, best_combo = 0.0, None
    for a in np.arange(0.0, 1.05, 0.1):
        for b in np.arange(0.0, 1.0 - a + 0.05, 0.1):
            c = 1.0 - a - b
            if c < -0.01:
                continue
            c = max(c, 0.0)
            blend = a * p_view1 + b * p_view2_vhnr + c * p_view3
            tb = find_best_threshold(y1, blend)
            _, _, fb, _ = precision_recall_fscore_support(y1, (blend >= tb).astype(int), pos_label=1, average='binary', zero_division=0)
            if fb > best_f:
                best_f, best_combo = fb, (a, b, c, blend)

    a, b, c, best_blend = best_combo
    print(f"  Best blend: v1={a:.1f} v2={b:.1f} v3={c:.1f} → F1={best_f:.2%}", flush=True)

    best_t = find_best_threshold(y1, best_blend)
    yp = (best_blend >= best_t).astype(int)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y1, yp, pos_label=1, average='binary', zero_division=0)
    print(f"  Final: P={p_f:.2%} R={r_f:.2%} F1={f_f:.2%} @ {best_t:.3f}", flush=True)

    print(f"  Sweep (best blend):", flush=True)
    for thr in np.arange(0.2, 0.95, 0.05):
        yt = (best_blend >= thr).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y1, yt, pos_label=1, average='binary', zero_division=0)
        print(f"    {thr:.2f}: P={pt:.2%} R={rt:.2%} F1={ft:.2%}", flush=True)

    return f_f


def classify_binary(per_user, level, feats, n_seeds=6, n_folds=8):
    """Standard binary for non-VH categories."""
    subset = per_user[per_user['risk_level'].isin([level, 'No risk'])].copy()
    subset['is_risky'] = (subset['risk_level'] == level).astype(int)
    X = subset[feats].values
    y = subset['is_risky'].values
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"\n=== {level} | {n_pos} pos, {n_neg} neg ===", flush=True)
    if n_pos < 5:
        return 0.0

    spw = n_neg / n_pos
    spw2 = min(spw * 2, 200)
    SEEDS = [42, 7, 123, 999, 2025, 1337][:n_seeds]

    def xgb(d, lr, n, sw, ss=0.85, cs=0.85):
        return dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=ss,
                    colsample_bytree=cs, scale_pos_weight=sw,
                    eval_metric='logloss', verbosity=0, device='cuda')

    configs = [
        (f"d4-spw{spw:.0f}",   xgb(4, 0.02, 700, spw,  0.85, 0.85)),
        (f"d5-spw{spw:.0f}",   xgb(5, 0.03, 600, spw,  0.80, 0.85)),
        (f"d5-spw{spw2:.0f}",  xgb(5, 0.04, 500, spw2, 0.85, 0.75)),
        (f"d6-spw{spw:.0f}",   xgb(6, 0.03, 500, spw,  0.75, 0.80)),
    ]

    print(f"  XGB ({len(configs)} configs × {n_seeds} seeds × {n_folds}-fold):", flush=True)
    oof_all, f1s = [], []
    for name, params in configs:
        pool = np.zeros(len(y))
        for seed in SEEDS:
            clf = XGBClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y))
            for ti, vi in cv.split(X, y):
                clf.fit(X[ti], y[ti])
                pr[vi] = clf.predict_proba(X[vi])[:, 1]
            pool += pr
        pool /= n_seeds
        oof_all.append(pool)
        t_i = find_best_threshold(y, pool)
        p, r, f, _ = precision_recall_fscore_support(y, (pool >= t_i).astype(int), pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        f1s.append(f)

    y_ens = np.average(oof_all, axis=0, weights=np.array(f1s) / sum(f1s))
    t_f = find_best_threshold(y, y_ens)
    yp = (y_ens >= t_f).astype(int)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y, yp, pos_label=1, average='binary', zero_division=0)
    print(f"  Final: P={p_f:.2%} R={r_f:.2%} F1={f_f:.2%} @ {t_f:.3f}", flush=True)

    print(f"  Sweep:", flush=True)
    for thr in np.arange(0.2, 0.95, 0.05):
        yt = (y_ens >= thr).astype(int)
        pt, rt, ft, _ = precision_recall_fscore_support(y, yt, pos_label=1, average='binary', zero_division=0)
        print(f"    {thr:.2f}: P={pt:.2%} R={rt:.2%} F1={ft:.2%}", flush=True)

    return f_f


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

    results = {}

    # Very High: multi-view approach
    print("\n=== Very High (multi-view) ===", flush=True)
    results['Very High'] = classify_vh_multiview(per_user, feats, n_seeds=10, n_folds=10)

    # Others: standard binary
    for level in ['Extreme', 'High', 'Low']:
        if level not in per_user['risk_level'].values:
            continue
        results[level] = classify_binary(per_user, level, feats, n_seeds=6, n_folds=8)

    print(f"\n{'='*60}")
    print(f"SUMMARY — v21 multiview, selected=False")
    print(f"{'='*60}")
    for level in RISK_ORDER:
        if level in results:
            tag = " *** ACHIEVED ***" if results[level] >= 0.95 else f"  gap={0.95-results[level]:.2%}"
            print(f"  {level:<12}: F1={results[level]:.2%}{tag}")
