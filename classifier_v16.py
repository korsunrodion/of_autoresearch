"""
classifier_v16.py — Pure GPU: XGBoost (all configs) + PyTorch MLP

Replaces all CPU HGB configs from v15 with diverse XGBoost GPU configs.
Total: 12 XGB (device='cuda') + 3 MLP (PyTorch GPU) = 100% GPU.
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
print(f"Using device: {DEVICE}")

ID_RANGES  = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
RANGE_TAGS = ['1k', '5k', '10k', '20k', '50k', '100k']

WINDOWS_MINUTES = [60, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12,
                   60*24, 60*24*7, 60*24*7*2, 60*24*7*4]
WINDOW_HOURS    = [w // 60 for w in WINDOWS_MINUTES]
KEY_WINDOWS_H   = {6, 8, 10, 12, 24}

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
        t, model, uid = user['subscribed_at'], user['tracking_model_name'], user['user_id_num']
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
                        row[f'log_min_consec_gap_{w_h}h'] = np.log1p(float(np.diff(np.sort(partner_ids)).min()))
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
        row['log_min_time_diff'] = (
            np.log1p((all_partners['subscribed_at'] - t).abs().dt.total_seconds().min() / 60)
            if not all_partners.empty else np.nan
        )
        norisk_med = model_norisk_median.get(model, np.nan)
        row['log_model_norisk_median_id_diff'] = np.log1p(norisk_med) if not np.isnan(norisk_med) else np.nan
        raw_24h = np.expm1(row.get('log_min_id_diff_24h', np.nan))
        row['rel_min_id_diff_24h'] = (raw_24h / norisk_med if not np.isnan(raw_24h) and not np.isnan(norisk_med) and norisk_med > 0 else np.nan)
        uname = user['user_name']
        row['user_model_count'] = cross.loc[uname, 'user_model_count'] if uname in cross.index else 1
        gmax = cross.loc[uname, 'user_global_max_risk'] if uname in cross.index else user['risk_score']
        row['user_max_risk_elsewhere'] = gmax if row['user_model_count'] > 1 else 0
        row['log_user_id_num'] = np.log1p(uid) if not np.isnan(uid) else np.nan
        rows.append(row)
    per_user = pd.DataFrame(rows)
    per_user['cic_ratio_6h_24h']    = per_user['cic_50k_6h']  / (per_user['cic_50k_24h']  + 1)
    per_user['cic10k_ratio_6h_12h'] = per_user['cic_10k_6h']  / (per_user['cic_10k_12h']  + 1)
    per_user['pc_ratio_6h_24h']     = per_user['partner_count_6h'] / (per_user['partner_count_24h'] + 1)
    return per_user


# ── PyTorch MLP ──────────────────────────────────────────────────────────────

class BotMLP(nn.Module):
    def __init__(self, n_feats, hidden=(256, 128, 64), dropout=0.4):
        super().__init__()
        layers = []
        in_dim = n_feats
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp_oof(X_np, y_np, seeds, n_folds,
                  hidden=(256, 128, 64), dropout=0.4,
                  epochs=400, lr=5e-4, wd=1e-3, batch=256):
    n = len(y_np)
    pos_count = y_np.sum()
    neg_count = n - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pool = np.zeros(n)
    for seed in seeds:
        torch.manual_seed(seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(n)
        for ti, vi in cv.split(X_np, y_np):
            mu  = np.nanmean(X_np[ti], axis=0)
            std = np.nanstd(X_np[ti], axis=0) + 1e-8
            X_tr = np.where(np.isnan(X_np[ti]), mu, X_np[ti])
            X_va = np.where(np.isnan(X_np[vi]), mu, X_np[vi])
            X_tr = (X_tr - mu) / std
            X_va = (X_va - mu) / std

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
            y_tr_t = torch.tensor(y_np[ti], dtype=torch.float32, device=DEVICE)
            X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)

            ds = TensorDataset(X_tr_t, y_tr_t)
            dl = DataLoader(ds, batch_size=batch, shuffle=True)

            model = BotMLP(X_np.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
            opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    opt.step()
                sched.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_va_t).cpu().numpy()
            proba[vi] = torch.sigmoid(torch.tensor(logits)).numpy()
        pool += proba
    return pool / len(seeds)


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def nan_fill(X_train, X_val):
    col_med = np.nanmedian(X_train, axis=0)
    return np.where(np.isnan(X_train), col_med, X_train), np.where(np.isnan(X_val), col_med, X_val)


def accumulate_oof(X, y, make_fn, seeds, n_folds):
    pool = np.zeros(len(y))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        proba = np.zeros(len(y))
        for ti, vi in cv.split(X, y):
            X_tr, X_va = nan_fill(X[ti], X[vi])
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
    subset = per_user[per_user['risk_level'].isin(levels + ['No risk'])].copy()
    subset['is_risky'] = subset['risk_level'].isin(levels)
    elsewhere_mask = subset['user_max_risk_elsewhere'].values >= 4

    feats = [f for f in FEATURES if f in subset.columns]
    X = subset[feats].values
    y = subset['is_risky'].astype(int).values
    print(f"Dataset: {y.sum()} pos, {(y==0).sum()} neg, {len(feats)} features", flush=True)

    SEEDS   = [42, 7, 123, 999, 2025, 1337, 2024, 101, 314, 271, 1776, 2023]
    N_FOLDS = 20

    # All XGBoost configs on GPU — diverse depth/lr/spw/subsample for ensemble diversity
    def xgb(depth, lr, n_est, spw, ss=0.85, cs=0.85, mcw=1, gamma=0.0):
        return dict(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=ss, colsample_bytree=cs,
                    scale_pos_weight=spw, min_child_weight=mcw, gamma=gamma,
                    eval_metric='logloss', verbosity=0, device='cuda')

    tree_configs = [
        # Former HGB replacements: varied depth, lr, spw
        ("XGB-d3-lr01-spw35",  lambda s: XGBClassifier(**xgb(3, 0.01, 1000, 35.0, ss=0.8, cs=0.8), random_state=s)),
        ("XGB-d4-lr02-spw25",  lambda s: XGBClassifier(**xgb(4, 0.02, 800,  25.0, ss=0.85, cs=0.85), random_state=s)),
        ("XGB-d4-lr04-spw15",  lambda s: XGBClassifier(**xgb(4, 0.04, 600,  15.0, ss=0.9, cs=0.9), random_state=s)),
        ("XGB-d5-lr03-spw35",  lambda s: XGBClassifier(**xgb(5, 0.03, 700,  35.0, ss=0.8, cs=0.8), random_state=s)),
        ("XGB-d5-lr05-spw20",  lambda s: XGBClassifier(**xgb(5, 0.05, 500,  20.0, ss=0.85, cs=0.75), random_state=s)),
        ("XGB-d6-lr03-spw35",  lambda s: XGBClassifier(**xgb(6, 0.03, 600,  35.0, ss=0.75, cs=0.8), random_state=s)),
        ("XGB-d4-lr02-spw50",  lambda s: XGBClassifier(**xgb(4, 0.02, 800,  50.0, ss=0.8, cs=0.85, mcw=3), random_state=s)),
        # Original XGB configs from v15
        ("XGB-spw3",           lambda s: XGBClassifier(**xgb(4, 0.04, 600,  3.0), random_state=s)),
        ("XGB-spw5",           lambda s: XGBClassifier(**xgb(4, 0.04, 600,  5.0), random_state=s)),
        ("XGB-spw10",          lambda s: XGBClassifier(**xgb(4, 0.04, 600, 10.0), random_state=s)),
        ("XGB-spw20",          lambda s: XGBClassifier(**xgb(4, 0.04, 600, 20.0), random_state=s)),
        ("XGB-spw35",          lambda s: XGBClassifier(**xgb(4, 0.04, 600, 35.0), random_state=s)),
    ]

    print(f"\n=== XGBoost GPU ensemble: {len(tree_configs)} configs × {len(SEEDS)} seeds × {N_FOLDS}-fold ===", flush=True)
    tree_probas = []
    for name, cfg_fn in tree_configs:
        p = accumulate_oof(X, y, cfg_fn, SEEDS, N_FOLDS)
        tree_probas.append(p)
        t_i = find_best_threshold(y, p)
        yi = (p >= t_i).astype(int)
        pi, ri, fi, _ = precision_recall_fscore_support(y, yi, pos_label=1, average='binary', zero_division=0)
        print(f"  {name}: P={pi:.2%} R={ri:.2%} F1={fi:.2%} @ {t_i:.3f}", flush=True)
    y_proba_tree = np.mean(tree_probas, axis=0)

    t_tree = find_best_threshold(y, y_proba_tree)
    yp_tree = (y_proba_tree >= t_tree).astype(int)
    pt, rt, ft, _ = precision_recall_fscore_support(y, yp_tree, pos_label=1, average='binary', zero_division=0)
    print(f"\nXGB ensemble: P={pt:.2%} R={rt:.2%} F1={ft:.2%} @ {t_tree:.3f}", flush=True)

    # ── MLP on GPU ─────────────────────────────────────────────────────
    print(f"\n=== MLP (GPU) — 6 seeds × {N_FOLDS}-fold ===", flush=True)
    MLP_SEEDS = SEEDS[:6]
    mlp_configs = [
        ("MLP-256-128-64-d04", dict(hidden=(256, 128, 64), dropout=0.4, epochs=400, lr=5e-4, wd=1e-3)),
        ("MLP-128-64-d03",     dict(hidden=(128, 64),       dropout=0.3, epochs=400, lr=5e-4, wd=5e-4)),
        ("MLP-64-32-d05",      dict(hidden=(64, 32),         dropout=0.5, epochs=500, lr=3e-4, wd=2e-3)),
    ]
    mlp_probas = []
    for name, kw in mlp_configs:
        p = train_mlp_oof(X, y, MLP_SEEDS, N_FOLDS, **kw)
        mlp_probas.append(p)
        t_m = find_best_threshold(y, p)
        ym = (p >= t_m).astype(int)
        pm, rm, fm_i, _ = precision_recall_fscore_support(y, ym, pos_label=1, average='binary', zero_division=0)
        print(f"  {name}: P={pm:.2%} R={rm:.2%} F1={fm_i:.2%} @ {t_m:.3f}", flush=True)
    y_proba_mlp = np.mean(mlp_probas, axis=0)

    t_mlp = find_best_threshold(y, y_proba_mlp)
    yp_mlp = (y_proba_mlp >= t_mlp).astype(int)
    pm, rm, fm, _ = precision_recall_fscore_support(y, yp_mlp, pos_label=1, average='binary', zero_division=0)
    print(f"\nMLP ensemble: P={pm:.2%} R={rm:.2%} F1={fm:.2%} @ {t_mlp:.3f}", flush=True)

    # ── Blend sweep ─────────────────────────────────────────────────────
    print(f"\n=== Blend sweep (xgb_w, mlp_w) ===", flush=True)
    best_blend_f1, best_alpha, best_blend = 0.0, 0.0, None
    for alpha in np.arange(0.0, 1.05, 0.1):
        y_blend = (1 - alpha) * y_proba_tree + alpha * y_proba_mlp
        t_b = find_best_threshold(y, y_blend)
        yb = (y_blend >= t_b).astype(int)
        pb, rb, fb, _ = precision_recall_fscore_support(y, yb, pos_label=1, average='binary', zero_division=0)
        marker = " <<<"  if fb > best_blend_f1 else ""
        print(f"  xgb={1-alpha:.1f} mlp={alpha:.1f}: P={pb:.2%} R={rb:.2%} F1={fb:.2%}{marker}", flush=True)
        if fb > best_blend_f1:
            best_blend_f1, best_alpha, best_blend = fb, alpha, y_blend

    t_best = find_best_threshold(y, best_blend)

    # ── Hybrid rule ─────────────────────────────────────────────────────
    print(f"\nHybrid rule (elsewhere>=4) on best blend (xgb={1-best_alpha:.1f} mlp={best_alpha:.1f}):")
    for low_t in [0.25, 0.30, 0.35, 0.40]:
        yh = np.where(elsewhere_mask & (best_blend >= low_t), 1, (best_blend >= t_best).astype(int))
        ph, rh, fh, _ = precision_recall_fscore_support(y, yh, pos_label=1, average='binary', zero_division=0)
        print(f"  thresh={low_t:.2f}: P={ph:.2%} R={rh:.2%} F1={fh:.2%}")

    # ── FN analysis ─────────────────────────────────────────────────────
    print(f"\nFalse negatives in best blend (xgb={1-best_alpha:.1f} mlp={best_alpha:.1f}):")
    yp_best = (best_blend >= t_best).astype(int)
    fn_mask = (y == 1) & (yp_best == 0)
    fn_df = subset[fn_mask][['user_name', 'user_max_risk_elsewhere']].copy()
    fn_df['proba'] = best_blend[fn_mask]
    for _, r in fn_df.sort_values('proba').iterrows():
        print(f"  {r['user_name']}: score={r['proba']:.3f}  elsewhere={r['user_max_risk_elsewhere']:.0f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Previous best:    93.49%")
    print(f"  XGB ensemble:     {ft:.2%}")
    print(f"  MLP ensemble:     {fm:.2%}")
    print(f"  Best blend F1:    {best_blend_f1:.2%}  (xgb={1-best_alpha:.1f} mlp={best_alpha:.1f})")
    if best_blend_f1 >= 0.95:
        print(f"\n*** GOAL ACHIEVED: F1={best_blend_f1:.2%} >= 95% ***")
    else:
        print(f"  Gap to 95%: {0.95 - best_blend_f1:.2%}")
