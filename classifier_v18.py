"""
classifier_v18.py — GPU + extended features + native NaN XGBoost

Changes from v17:
- Add total_chargebacks, hour_sin/cos (cyclical), day_of_week per user
- Add log_id_std for key windows, ratio features (cic_ratio, pc_ratio)
- Add frac_empty_short_windows (isolatedness signal)
- XGBoost native NaN (no nan_fill) — lets XGB learn optimal NaN direction
- More diverse spw values (5, 10, 20, 38, 77) for ensemble diversity
- 10-fold, 8 seeds for Very High; 8-fold, 6 seeds for others
- selected=False, classifies Extreme | Very High | High | Low
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from base import fetch_df, clean

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)

WINDOWS_MINUTES = [60, 60*4, 60*6, 60*7, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]

ID_RANGES  = [10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['10k', '20k', '50k', '100k']
KEY_WH     = {6, 7, 8, 10, 12, 24}
SHORT_WH   = {1, 4, 6, 7, 8}   # windows for frac_empty signal

FEATURES = (
    [f'log_min_id_diff_{w}h' for w in WINDOW_HOURS] +
    [f'partner_count_{w}h'   for w in WINDOW_HOURS] +
    [f'cic_{tag}_{w}h'      for tag in RANGE_TAGS for w in WINDOW_HOURS] +
    [f'cic_frac_{tag}_{w}h' for tag in RANGE_TAGS for w in KEY_WH] +
    [f'log_id_span_{w}h'        for w in sorted(KEY_WH)] +
    [f'log_id_std_{w}h'         for w in sorted(KEY_WH)] +
    [f'log_min_consec_gap_{w}h' for w in sorted(KEY_WH)] +
    ['log_min_time_diff', 'log_model_norisk_median_id_diff', 'rel_min_id_diff_24h',
     'user_model_count', 'user_max_risk_elsewhere', 'log_user_id_num',
     'cic_ratio_6h_24h', 'pc_ratio_6h_24h', 'cic10k_ratio_6h_12h',
     'total_chargebacks', 'hour_sin', 'hour_cos', 'day_of_week',
     'frac_empty_short_windows']
)

RISK_ORDER = ['Extreme', 'Very High', 'High', 'Low']


def _model_norisk_median_gpu(df):
    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    records = []
    for model_name, mdf in df.groupby('tracking_model_name'):
        nr = mdf[mdf['risk_level'] == 'No risk']
        if nr.empty:
            continue
        all_t = torch.tensor(mdf['subscribed_at'].values.astype(np.int64), device=DEVICE)
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

    max_w_ns = int(max(WINDOWS_MINUTES) * 60 * 1e9)
    w_ns_list = [int(w * 60 * 1e9) for w in WINDOWS_MINUTES]
    all_rows = []

    for model_name, mdf in df.groupby('tracking_model_name'):
        n = len(mdf)
        if n == 0:
            continue
        mdf = mdf.reset_index(drop=True)
        times_ns   = mdf['subscribed_at'].values.astype(np.int64)
        ids_arr    = mdf['user_id_num'].values.astype(np.float64)
        usernames  = mdf['user_name'].values
        risk_lvls  = mdf['risk_level'].values
        risk_scs   = mdf['risk_score'].values
        ts_arr     = mdf['subscribed_ts'].values.astype(np.float64)
        chargebacks = mdf['total_chargebacks'].values.astype(np.float64)

        # GPU pairwise
        t_gpu  = torch.tensor(times_ns, device=DEVICE, dtype=torch.int64)
        id_gpu = torch.tensor(ids_arr,  device=DEVICE, dtype=torch.float32)
        tdiff_gpu  = (t_gpu.unsqueeze(0) - t_gpu.unsqueeze(1)).abs()
        iddiff_gpu = (id_gpu.unsqueeze(0) - id_gpu.unsqueeze(1)).abs()
        not_self   = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        w_ns_t = torch.tensor(w_ns_list, device=DEVICE, dtype=torch.int64)
        all_masks = (tdiff_gpu.unsqueeze(0) <= w_ns_t[:, None, None]) & not_self.unsqueeze(0)

        tdiff_cpu  = tdiff_gpu.cpu().numpy()
        iddiff_cpu = iddiff_gpu.cpu().numpy()
        masks_cpu  = all_masks.cpu().numpy()
        not_self_cpu = not_self.cpu().numpy()
        norisk_med = model_norisk_median.get(model_name, np.nan)

        for i in range(n):
            row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
            n_empty_short = 0

            for wi, (w_min, w_h) in enumerate(zip(WINDOWS_MINUTES, WINDOW_HOURS)):
                mask = masks_cpu[wi, i]
                if not mask.any():
                    row[f'log_min_id_diff_{w_h}h'] = np.nan
                    row[f'partner_count_{w_h}h']   = np.nan
                    for tag in RANGE_TAGS:
                        row[f'cic_{tag}_{w_h}h'] = np.nan
                    if w_h in KEY_WH:
                        for tag in RANGE_TAGS:
                            row[f'cic_frac_{tag}_{w_h}h'] = np.nan
                        row[f'log_id_span_{w_h}h']        = np.nan
                        row[f'log_id_std_{w_h}h']         = np.nan
                        row[f'log_min_consec_gap_{w_h}h'] = np.nan
                    if w_h in SHORT_WH:
                        n_empty_short += 1
                else:
                    diffs = iddiff_cpu[i, mask]
                    pids  = ids_arr[mask]
                    pc    = float(mask.sum())
                    row[f'log_min_id_diff_{w_h}h'] = np.log1p(diffs.min())
                    row[f'partner_count_{w_h}h']   = pc
                    for tag, rng in zip(RANGE_TAGS, ID_RANGES):
                        cic = float((diffs <= rng).sum())
                        row[f'cic_{tag}_{w_h}h'] = cic
                        if w_h in KEY_WH:
                            row[f'cic_frac_{tag}_{w_h}h'] = cic / (pc + 1)
                    if w_h in KEY_WH:
                        row[f'log_id_span_{w_h}h'] = np.log1p(float(pids.max() - pids.min()))
                        row[f'log_id_std_{w_h}h']  = np.log1p(float(np.std(pids))) if pc > 1 else 0.0
                        row[f'log_min_consec_gap_{w_h}h'] = (
                            np.log1p(float(np.diff(np.sort(pids)).min())) if pc >= 2 else np.nan)

            # Forward-fill NaN from next larger window
            for j in range(len(WINDOWS_MINUTES) - 1):
                wh, nwh = WINDOW_HOURS[j], WINDOW_HOURS[j + 1]
                for base in ['log_min_id_diff', 'partner_count'] + [f'cic_{t}' for t in RANGE_TAGS]:
                    k, nk = f'{base}_{wh}h', f'{base}_{nwh}h'
                    if np.isnan(row.get(k, np.nan)):
                        row[k] = row.get(nk, np.nan)

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

            # Ratio features
            c6  = row.get('cic_10k_6h',  np.nan)
            c12 = row.get('cic_10k_12h', np.nan)
            c24 = row.get('cic_10k_24h', np.nan)
            c50_6  = row.get('cic_50k_6h',  np.nan)
            c50_24 = row.get('cic_50k_24h', np.nan)
            pc6  = row.get('partner_count_6h',  np.nan)
            pc24 = row.get('partner_count_24h', np.nan)
            row['cic_ratio_6h_24h']    = c50_6  / (c50_24 + 1) if not np.isnan(c50_6)  else np.nan
            row['cic10k_ratio_6h_12h'] = c6     / (c12    + 1) if not np.isnan(c6)     else np.nan
            row['pc_ratio_6h_24h']     = pc6    / (pc24   + 1) if not np.isnan(pc6)    else np.nan

            # Time features (cyclical hour, day_of_week)
            ts = ts_arr[i]
            hour = (ts % 86400) / 3600.0
            row['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
            row['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
            row['day_of_week'] = float(int(ts // 86400 + 4) % 7)

            # Chargebacks
            row['total_chargebacks'] = chargebacks[i]

            # Isolatedness
            row['frac_empty_short_windows'] = n_empty_short / len(SHORT_WH)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ── Lightweight PyTorch MLP ────────────────────────────────────────────────────

class BotMLP(nn.Module):
    def __init__(self, n_feats, hidden=(128, 64), dropout=0.35):
        super().__init__()
        layers, in_dim = [], n_feats
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp_oof(X_np, y_np, seeds, n_folds, hidden=(128, 64), dropout=0.35,
                  epochs=100, lr=5e-4, wd=1e-3, batch=512):
    pos_w = torch.tensor([(y_np == 0).sum() / y_np.sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    pool = np.zeros(len(y_np))
    for seed in seeds:
        torch.manual_seed(seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(len(y_np))
        for ti, vi in cv.split(X_np, y_np):
            mu  = np.nanmean(X_np[ti], axis=0)
            std = np.nanstd(X_np[ti], axis=0) + 1e-8
            Xtr = (np.where(np.isnan(X_np[ti]), mu, X_np[ti]) - mu) / std
            Xva = (np.where(np.isnan(X_np[vi]), mu, X_np[vi]) - mu) / std
            Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_np[ti], dtype=torch.float32, device=DEVICE)
            Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
            ds = TensorDataset(Xt, yt)
            dl = DataLoader(ds, batch_size=batch, shuffle=True)
            model = BotMLP(X_np.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    opt.zero_grad(); criterion(model(xb), yb).backward(); opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                proba[vi] = torch.sigmoid(model(Xv)).cpu().numpy()
        pool += proba
    return pool / len(seeds)


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1  = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def classify_category(per_user, level, feats, use_mlp=False, n_seeds=6, n_folds=8):
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
    SEEDS = [42, 7, 123, 999, 2025, 1337, 2024, 101][:n_seeds]

    def xgb_params(depth, lr, n_est, sw, ss=0.85, cs=0.85):
        return dict(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=ss, colsample_bytree=cs, scale_pos_weight=sw,
                    eval_metric='logloss', verbosity=0, device='cuda')

    spw2 = min(spw * 2, 200)
    spw_half = max(spw / 2, 5.0)
    # 5 configs: low/mid/true/2x spw across varied depths
    configs = [
        (f"XGB-d3-spw{spw:.0f}",   xgb_params(3, 0.01, 900, spw,      0.80, 0.80)),
        (f"XGB-d4-spw{spw_half:.0f}", xgb_params(4, 0.03, 700, spw_half, 0.85, 0.85)),
        (f"XGB-d4-spw{spw:.0f}",   xgb_params(4, 0.02, 700, spw,      0.85, 0.85)),
        (f"XGB-d5-spw{spw:.0f}",   xgb_params(5, 0.03, 600, spw,      0.80, 0.80)),
        (f"XGB-d5-spw{spw2:.0f}",  xgb_params(5, 0.04, 500, spw2,     0.85, 0.75)),
        (f"XGB-d6-spw{spw:.0f}",   xgb_params(6, 0.03, 500, spw,      0.75, 0.80)),
    ]

    print(f"  XGB GPU ({len(configs)} configs × {n_seeds} seeds × {n_folds}-fold, native NaN):", flush=True)
    xgb_probas = []
    for name, params in configs:
        pool = np.zeros(len(y))
        for seed in SEEDS:
            clf = XGBClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y))
            for ti, vi in cv.split(X, y):
                # Native NaN — XGBoost learns optimal direction for missing values
                clf.fit(X[ti], y[ti])
                pr[vi] = clf.predict_proba(X[vi])[:, 1]
            pool += pr
        pool /= n_seeds
        t_i = find_best_threshold(y, pool)
        yp  = (pool >= t_i).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, yp, pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        xgb_probas.append(pool)
    y_xgb = np.mean(xgb_probas, axis=0)
    t_xgb = find_best_threshold(y, y_xgb)
    yp_xgb = (y_xgb >= t_xgb).astype(int)
    px, rx, fx, _ = precision_recall_fscore_support(y, yp_xgb, pos_label=1, average='binary', zero_division=0)
    print(f"  XGB ensemble: P={px:.2%} R={rx:.2%} F1={fx:.2%}", flush=True)

    best_proba = y_xgb
    if use_mlp:
        print(f"  MLP GPU (2 configs × 3 seeds × {n_folds}-fold, 100ep):", flush=True)
        mlp_ps = []
        for hidden, drop in [((128, 64), 0.35), ((64, 32), 0.4)]:
            p = train_mlp_oof(X, y, SEEDS[:3], n_folds, hidden=hidden, dropout=drop, epochs=100)
            t_m = find_best_threshold(y, p)
            ym  = (p >= t_m).astype(int)
            pm, rm, fm, _ = precision_recall_fscore_support(y, ym, pos_label=1, average='binary', zero_division=0)
            print(f"    MLP{list(hidden)}: P={pm:.2%} R={rm:.2%} F1={fm:.2%}", flush=True)
            mlp_ps.append(p)
        y_mlp = np.mean(mlp_ps, axis=0)
        # Blend
        best_f1, best_blend = 0.0, y_xgb
        for alpha in np.arange(0.0, 1.05, 0.1):
            yb = (1 - alpha) * y_xgb + alpha * y_mlp
            t_b = find_best_threshold(y, yb)
            _, _, fb, _ = precision_recall_fscore_support(y, (yb >= t_b).astype(int), pos_label=1, average='binary', zero_division=0)
            if fb > best_f1:
                best_f1, best_blend = fb, yb
        best_proba = best_blend

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
            print(f"\n[{level}] not in data", flush=True)
            continue
        vh = (level == 'Very High')
        results[level] = classify_category(
            per_user, level, feats,
            use_mlp=vh,
            n_seeds=8 if vh else 6,
            n_folds=10 if vh else 8,
        )

    print(f"\n{'='*60}")
    print(f"SUMMARY — v18 GPU, selected=False")
    print(f"{'='*60}")
    for level in RISK_ORDER:
        if level in results:
            tag = " *** ACHIEVED ***" if results[level] >= 0.95 else f"  gap={0.95-results[level]:.2%}"
            print(f"  {level:<12}: F1={results[level]:.2%}{tag}")
