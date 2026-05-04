"""
classifier_v20.py — Subscription density + stacking meta-learner

Key new ideas:
1. Model-level subscription density features (on GPU):
   - model_subs_Xh: #subscriptions on same model in ±X hours (temporal wave)
   - model_subs_rate_Xh: rate vs model's baseline rate (anomaly signal)
   - model_norisk_rate: baseline rate of No-risk subs (per-model)
2. Stacking: XGB OOF probas → second-level XGB meta-learner
3. Higher diversity: vary colsample_bytree (0.6, 0.7, 0.85) and subsample
4. More seeds (10) for Very High
5. Include a logistic-regression level-2 model
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
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

# Density windows (just a few, focused on the hot zone)
DENSITY_WH = [1, 4, 6, 12, 24]   # hours

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

    # Per-model baseline subscription rate (No Risk only, per day)
    print("  Precomputing model subscription rates...", flush=True)
    model_norisk_rates = {}
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if len(nr) >= 2:
            t_range_days = (nr['subscribed_at'].max() - nr['subscribed_at'].min()).total_seconds() / 86400
            model_norisk_rates[model_name] = len(nr) / max(t_range_days, 1.0)  # subs/day
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

        # GPU pairwise
        t_gpu  = torch.tensor(times_ns, device=DEVICE, dtype=torch.int64)
        id_gpu = torch.tensor(ids_arr,  device=DEVICE, dtype=torch.float32)
        tdiff_gpu  = (t_gpu.unsqueeze(0) - t_gpu.unsqueeze(1)).abs()
        iddiff_gpu = (id_gpu.unsqueeze(0) - id_gpu.unsqueeze(1)).abs()
        not_self   = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        w_ns_t = torch.tensor(w_ns_list, device=DEVICE, dtype=torch.int64)
        all_masks  = (tdiff_gpu.unsqueeze(0) <= w_ns_t[:, None, None]) & not_self.unsqueeze(0)
        global_mask = not_self

        # Density masks: includes self
        d_ns_t = torch.tensor(density_ns_list, device=DEVICE, dtype=torch.int64)
        # tdiff including self (diagonal = 0), we count all subs in window
        tdiff_all = (t_gpu.unsqueeze(0) - t_gpu.unsqueeze(1)).abs()  # same as tdiff_gpu
        # density: count all in window (including self → subtract 1 later)
        density_masks = (tdiff_all.unsqueeze(0) <= d_ns_t[:, None, None])  # [D, n, n]

        tdiff_cpu    = tdiff_gpu.cpu().numpy()
        iddiff_cpu   = iddiff_gpu.cpu().numpy()
        masks_cpu    = all_masks.cpu().numpy()
        not_self_cpu = not_self.cpu().numpy()
        global_cpu   = global_mask.cpu().numpy()
        density_cpu  = density_masks.cpu().numpy()   # [D, n, n]

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

            # Forward-fill
            for j in range(len(WINDOWS_MINUTES) - 1):
                ws, nws = WINDOW_STRS[j], WINDOW_STRS[j + 1]
                for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in RANGE_TAGS]:
                    k, nk = f'{base}_{ws}', f'{base}_{nws}'
                    if np.isnan(row.get(k, np.nan)):
                        row[k] = row.get(nk, np.nan)

            # Density features: total model subscriptions in window ±Xh
            for di, w_h in enumerate(DENSITY_WH):
                # count all subscriptions (including self) in window
                n_subs = float(density_cpu[di, i].sum())
                w_days = w_h * 2 / 24.0  # window span in days (bidirectional)
                expected = norisk_rate * w_days
                row[f'model_subs_{w_h}h']      = n_subs
                row[f'model_subs_rate_{w_h}h'] = n_subs / max(expected, 1.0)

            # Global stats
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
    idx = f1.argmax()
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def classify_category(per_user, level, feats, n_seeds=6, n_folds=8):
    subset = per_user[per_user['risk_level'].isin([level, 'No risk'])].copy()
    subset['is_risky'] = (subset['risk_level'] == level).astype(int)
    X = subset[feats].values
    y = subset['is_risky'].values
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"\n=== {level} | {n_pos} pos, {n_neg} neg ===", flush=True)
    if n_pos < 5:
        print("  Skipped", flush=True)
        return 0.0

    spw = n_neg / n_pos
    spw2 = min(spw * 2, 200)
    spw_h = max(spw / 2, 5.0)
    SEEDS = [42, 7, 123, 999, 2025, 1337, 2024, 101, 314, 271][:n_seeds]

    def xgb(d, lr, n, sw, ss=0.85, cs=0.85, mcw=1, g=0.0):
        return dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=ss,
                    colsample_bytree=cs, scale_pos_weight=sw, min_child_weight=mcw,
                    gamma=g, eval_metric='logloss', verbosity=0, device='cuda')

    configs = [
        (f"d4-spw{spw_h:.0f}-cs06", xgb(4, 0.03, 700, spw_h, 0.85, 0.60)),
        (f"d4-spw{spw:.0f}",        xgb(4, 0.02, 700, spw,   0.85, 0.85)),
        (f"d5-spw{spw:.0f}-cs07",   xgb(5, 0.03, 600, spw,   0.80, 0.70)),
        (f"d5-spw{spw:.0f}",        xgb(5, 0.03, 600, spw,   0.80, 0.85)),
        (f"d5-spw{spw2:.0f}-cs06",  xgb(5, 0.04, 500, spw2,  0.85, 0.60)),
        (f"d6-spw{spw:.0f}",        xgb(6, 0.03, 500, spw,   0.75, 0.80)),
        (f"d6-spw{spw2:.0f}-g01",   xgb(6, 0.02, 600, spw2,  0.75, 0.80, g=0.1)),
        (f"d4-spw{spw:.0f}-mcw3",   xgb(4, 0.02, 700, spw,   0.85, 0.85, mcw=3)),
    ]

    print(f"  L1: {len(configs)} configs × {n_seeds} seeds × {n_folds}-fold:", flush=True)
    # L1 OOF probas (for stacking)
    oof_meta = np.zeros((len(y), len(configs)))
    xgb_f1s  = []
    for ci, (name, params) in enumerate(configs):
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
        oof_meta[:, ci] = pool
        t_i = find_best_threshold(y, pool)
        yp  = (pool >= t_i).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, yp, pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        xgb_f1s.append(f)

    # Mean ensemble
    y_mean = np.mean(oof_meta, axis=1)
    t_mean = find_best_threshold(y, y_mean)
    _, _, f_mean, _ = precision_recall_fscore_support(y, (y_mean >= t_mean).astype(int), pos_label=1, average='binary', zero_division=0)

    # F1-weighted ensemble
    wts = np.array(xgb_f1s); wts /= wts.sum()
    y_wt = np.average(oof_meta, axis=1, weights=wts)
    t_wt = find_best_threshold(y, y_wt)
    _, _, f_wt, _ = precision_recall_fscore_support(y, (y_wt >= t_wt).astype(int), pos_label=1, average='binary', zero_division=0)

    # L2: logistic regression meta-learner on OOF probas
    lr_meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
    y_lr_oof = np.zeros(len(y))
    cv_meta = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for ti, vi in cv_meta.split(oof_meta, y):
        lr_meta.fit(oof_meta[ti], y[ti])
        y_lr_oof[vi] = lr_meta.predict_proba(oof_meta[vi])[:, 1]
    t_lr = find_best_threshold(y, y_lr_oof)
    _, _, f_lr, _ = precision_recall_fscore_support(y, (y_lr_oof >= t_lr).astype(int), pos_label=1, average='binary', zero_division=0)

    # L2: XGB meta-learner on OOF probas + original features (top 20 by v19 importance)
    y_xgb2_oof = np.zeros(len(y))
    xgb2 = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                          scale_pos_weight=spw, eval_metric='logloss', verbosity=0, device='cuda')
    for ti, vi in cv_meta.split(oof_meta, y):
        meta_tr = np.column_stack([oof_meta[ti], X[ti]])
        meta_va = np.column_stack([oof_meta[vi], X[vi]])
        xgb2.fit(meta_tr, y[ti])
        y_xgb2_oof[vi] = xgb2.predict_proba(meta_va)[:, 1]
    t_xgb2 = find_best_threshold(y, y_xgb2_oof)
    _, _, f_xgb2, _ = precision_recall_fscore_support(y, (y_xgb2_oof >= t_xgb2).astype(int), pos_label=1, average='binary', zero_division=0)

    print(f"  Mean ensemble: F1={f_mean:.2%}", flush=True)
    print(f"  Wtd ensemble:  F1={f_wt:.2%}", flush=True)
    print(f"  LR meta:       F1={f_lr:.2%}", flush=True)
    print(f"  XGB meta+feat: F1={f_xgb2:.2%}", flush=True)

    # Pick best
    best_f1 = max(f_mean, f_wt, f_lr, f_xgb2)
    if best_f1 == f_xgb2:
        best_proba = y_xgb2_oof
    elif best_f1 == f_lr:
        best_proba = y_lr_oof
    elif best_f1 == f_wt:
        best_proba = y_wt
    else:
        best_proba = y_mean

    best_t = find_best_threshold(y, best_proba)
    yp = (best_proba >= best_t).astype(int)
    p_f, r_f, f_f, _ = precision_recall_fscore_support(y, yp, pos_label=1, average='binary', zero_division=0)
    print(f"  Final: P={p_f:.2%} R={r_f:.2%} F1={f_f:.2%} @ {best_t:.3f}", flush=True)

    print(f"  Sweep:", flush=True)
    for thr in np.arange(0.2, 0.95, 0.05):
        yt = (best_proba >= thr).astype(int)
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
    for level in RISK_ORDER:
        if level not in per_user['risk_level'].values:
            continue
        vh = (level == 'Very High')
        results[level] = classify_category(
            per_user, level, feats,
            n_seeds=10 if vh else 6,
            n_folds=10 if vh else 8,
        )

    print(f"\n{'='*60}")
    print(f"SUMMARY — v20 GPU stacking, selected=False")
    print(f"{'='*60}")
    for level in RISK_ORDER:
        if level in results:
            tag = " *** ACHIEVED ***" if results[level] >= 0.95 else f"  gap={0.95-results[level]:.2%}"
            print(f"  {level:<12}: F1={results[level]:.2%}{tag}")
