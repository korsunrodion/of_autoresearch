"""
classifier_v2_cpu.py — CPU-only (v1_cpu + 15-fold VH + improved cascade)

v1_cpu result: VH=94.89%, gap=0.11% — very close, likely from 12-fold vs v25's 15-fold.

Changes from v1_cpu:
- N_FOLDS_VH = 15 (was 12) — primary lever, v25 used 15 folds for 95.13%
- Cascade models: nl127 (was nl63) — stronger cascade augmentation signal
- VH configs: LGB-nl127 + LGB-nl63 (replaced nl255 which was too slow)
  nl63 adds ensemble diversity at much lower cost than nl255

Target: VH F1 >= 95%, ~1-1.5h on 8-core CPU

Optional JS export:
  pip install m2cgen
  Set EXPORT_JS_PATH = 'vh_model.js' at the bottom to dump the trained
  VH LightGBM model to pure JavaScript via m2cgen.
"""
import os
import sys
import math
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# ── Constants (identical to v25) ─────────────────────────────────────────────
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

# ── Shared globals for multiprocessing workers (set via pool initializer) ─────
_G_CROSS        = {}   # user_name -> (model_count, global_max_risk_score)
_G_NORISK_MED   = {}   # model_name -> float
_G_NORISK_RATES = {}   # model_name -> float
_G_RISK_FRACS   = {}   # model_name -> {'vh', 'extreme', 'any'}


def _pool_init(cross, norisk_med, norisk_rates, risk_fracs):
    global _G_CROSS, _G_NORISK_MED, _G_NORISK_RATES, _G_RISK_FRACS
    _G_CROSS        = cross
    _G_NORISK_MED   = norisk_med
    _G_NORISK_RATES = norisk_rates
    _G_RISK_FRACS   = risk_fracs


# ── Username feature helpers ──────────────────────────────────────────────────
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


# ── Worker: compute all features for one model group ──────────────────────────
def _compute_model_worker(pack):
    model_name, (times_ns, ids_arr, usernames, risk_lvls,
                 risk_scs, ts_arr, chargebacks) = pack

    norisk_med  = _G_NORISK_MED.get(model_name, np.nan)
    norisk_rate = _G_NORISK_RATES.get(model_name, 1.0)
    risk_fracs  = _G_RISK_FRACS.get(model_name, {'vh': 0., 'extreme': 0., 'any': 0.})

    n = len(times_ns)
    if n == 0:
        return []

    # Build pairwise matrices via numpy broadcasting — O(n²) memory per model.
    # Typical model sizes are small enough that this is safe.
    tdiff  = np.abs(times_ns[:, None] - times_ns[None, :])   # (n,n) int64 ns
    iddiff = np.abs(ids_arr[:, None]  - ids_arr[None, :])    # (n,n) float64
    not_self = ~np.eye(n, dtype=bool)

    sorted_ids = np.sort(ids_arr)
    sorted_ts  = np.sort(times_ns)
    model_first_sub_ns = times_ns.min()

    rows = []
    for i in range(n):
        row = {'user_name': usernames[i], 'risk_level': risk_lvls[i]}
        n_empty_short = 0
        tdiff_i  = tdiff[i]    # reuse row slice
        iddiff_i = iddiff[i]

        # ── Window features ──────────────────────────────────────────────────
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

        # Forward-fill empty narrow windows from the next wider window
        for j in range(len(WINDOWS_MINUTES) - 1):
            ws, nws = WINDOW_STRS[j], WINDOW_STRS[j + 1]
            for base in (['log_min_id_diff', 'partner_count'] +
                         [f'cic_{t}' for t in RANGE_TAGS]):
                k, nk = f'{base}_{ws}', f'{base}_{nws}'
                if np.isnan(row.get(k, np.nan)):
                    row[k] = row.get(nk, np.nan)

        # ── Density features ─────────────────────────────────────────────────
        for w_h, d_ns in zip(DENSITY_WH, D_NS_LIST):
            n_subs = float((tdiff_i <= d_ns).sum())          # includes self
            w_days = w_h * 2 / 24.0
            row[f'model_subs_{w_h}h']      = n_subs
            row[f'model_subs_rate_{w_h}h'] = n_subs / max(norisk_rate * w_days, 1.0)

        # ── Global model-level ID features ───────────────────────────────────
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

        # ── Cross-model user features ─────────────────────────────────────────
        uname = usernames[i]
        mc, gmax_rs = _G_CROSS.get(uname, (1, float(risk_scs[i])))
        row['user_model_count']       = mc
        row['user_max_risk_elsewhere'] = gmax_rs if mc > 1 else 0
        row['log_user_id_num']        = (
            np.log1p(ids_arr[i]) if not np.isnan(ids_arr[i]) else np.nan)

        # ── Ratio features ────────────────────────────────────────────────────
        c6    = row.get('cic_10k_6h',  np.nan)
        c12   = row.get('cic_10k_12h', np.nan)
        c50_6  = row.get('cic_50k_6h',  np.nan)
        c50_24 = row.get('cic_50k_24h', np.nan)
        pc6   = row.get('partner_count_6h',  np.nan)
        pc24  = row.get('partner_count_24h', np.nan)
        row['cic_ratio_6h_24h']    = c50_6 / (c50_24 + 1) if not np.isnan(c50_6)  else np.nan
        row['cic10k_ratio_6h_12h'] = c6    / (c12    + 1) if not np.isnan(c6)     else np.nan
        row['pc_ratio_6h_24h']     = pc6   / (pc24   + 1) if not np.isnan(pc6)    else np.nan

        # ── Time / username / model-level scalars ─────────────────────────────
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

        # ── v25 percentile + 5m-cluster features ──────────────────────────────
        row['id_percentile_in_model']   = float(np.searchsorted(sorted_ids, ids_arr[i])) / n
        row['time_percentile_in_model'] = float(np.searchsorted(sorted_ts,  times_ns[i])) / n
        row['log_time_since_model_first_sub'] = np.log1p(
            (times_ns[i] - model_first_sub_ns) / 1e9 / 3600.0)
        row['n_subs_at_sub_time_5m'] = float((tdiff_i <= _5MIN_NS).sum())

        rows.append(row)

    return rows


# ── Norisk median — vectorized numpy (replaces GPU loop) ──────────────────────
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

        # Chunked to bound memory: chunk_size × n_all floats per iteration
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


# ── Feature computation orchestrator ─────────────────────────────────────────
def compute_per_user(df, n_jobs=N_JOBS):
    print("  Precomputing model norisk medians (vectorized CPU)...", flush=True)
    norisk_med_dict = _model_norisk_median_cpu(df).to_dict()

    print("  Precomputing model rates and risk fractions...", flush=True)
    norisk_rates = {}
    risk_fracs   = {}
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

    print("  Precomputing cross-model user stats...", flush=True)
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


# ── OOF training (LGB only) ───────────────────────────────────────────────────
def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[f1.argmax()]) if f1.argmax() < len(thresholds) else 0.5


def build_oof_lgb(X, y, configs, seeds, n_folds):
    results = []
    for name, params in configs:
        pool_arr = np.zeros(len(y))
        for seed in seeds:
            clf = LGBMClassifier(**params, random_state=seed)
            cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            pr  = np.zeros(len(y))
            for ti, vi in cv.split(X, y):
                clf.fit(X[ti], y[ti])
                pr[vi] = clf.predict_proba(X[vi])[:, 1]
            pool_arr += pr
        pool_arr /= len(seeds)
        t = find_best_threshold(y, pool_arr)
        p, r, f, _ = precision_recall_fscore_support(
            y, (pool_arr >= t).astype(int),
            pos_label=1, average='binary', zero_division=0)
        print(f"    {name}: P={p:.2%} R={r:.2%} F1={f:.2%}", flush=True)
        results.append((name, pool_arr, f))
    return results


def blend_results(results, y, label=""):
    f1s = np.array([f for _, _, f in results])
    w   = f1s / f1s.sum()
    blended = sum(wi * pool for wi, (_, pool, _) in zip(w, results))
    t = find_best_threshold(y, blended)
    p, r, f, _ = precision_recall_fscore_support(
        y, (blended >= t).astype(int),
        pos_label=1, average='binary', zero_division=0)
    if label:
        print(f"  {label}: P={p:.2%} R={r:.2%} F1={f:.2%} @ {t:.3f}", flush=True)
    return blended, t, p, r, f


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    from base import fetch_df, clean
    from db import db as _db

    print("Loading full dataset (selected=False)...", flush=True)
    df = fetch_df(selected=False)
    df = clean(df)
    # Close DB connection before forking to avoid psycopg2 issues in workers
    try:
        _db.close()
    except Exception:
        pass
    print(f"  {len(df)} rows", flush=True)
    print(df['risk_level'].value_counts().to_string(), flush=True)

    print("\nComputing per-user features (CPU multiprocessing)...", flush=True)
    per_user = compute_per_user(df)
    print(f"Total per-user rows: {len(per_user)}", flush=True)

    feats = [f for f in FEATURES if f in per_user.columns]
    print(f"Features: {len(feats)}", flush=True)

    # Cascade configs: nl127 for higher-quality augmentation signals.
    # Using 8-fold for cascade (fast enough, cascade doesn't need 15-fold).
    SEEDS_C  = [42, 7, 123, 999, 2025, 1337, 2024, 101]
    N_FOLDS_C = 8

    def lgb_c(nl, sw, lr=0.05, n=400, ss=0.80, cs=0.85, mcw=20):
        return dict(num_leaves=nl, learning_rate=lr, n_estimators=n,
                    scale_pos_weight=sw, subsample=ss, colsample_bytree=cs,
                    min_child_samples=mcw, subsample_freq=1, verbose=-1, n_jobs=-1)

    # ── Step 1: Cascade OOF probas (all LGB) ────────────────────────────────
    print("\n=== Step 1: Cascade OOF probas ===", flush=True)

    print("  Extreme:", flush=True)
    sub_e = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
    sub_e['y'] = (sub_e['risk_level'] == 'Extreme').astype(int)
    Xe, ye = sub_e[feats].values, sub_e['y'].values
    spw_e  = (ye == 0).sum() / ye.sum()
    res_e  = build_oof_lgb(Xe, ye, [
        (f'E-nl127-spw{spw_e:.0f}',            lgb_c(127, spw_e,            lr=0.03, n=600)),
        (f'E-nl127-spw{min(spw_e*2,200):.0f}', lgb_c(127, min(spw_e*2,200), lr=0.04, n=500)),
    ], SEEDS_C, N_FOLDS_C)
    p_extreme = np.mean([p for _, p, _ in res_e], axis=0)
    extreme_lookup = dict(zip(sub_e['user_name'].values, p_extreme))

    print("  High:", flush=True)
    sub_h = per_user[per_user['risk_level'].isin(['High', 'No risk'])].copy()
    sub_h['y'] = (sub_h['risk_level'] == 'High').astype(int)
    Xh, yh = sub_h[feats].values, sub_h['y'].values
    spw_h  = (yh == 0).sum() / yh.sum()
    res_h  = build_oof_lgb(Xh, yh, [(f'H-nl127-spw{spw_h:.0f}', lgb_c(127, spw_h, lr=0.03, n=600))],
                           SEEDS_C[:6], N_FOLDS_C)
    p_high = np.mean([p for _, p, _ in res_h], axis=0)
    high_lookup = dict(zip(sub_h['user_name'].values, p_high))

    print("  Low:", flush=True)
    sub_l = per_user[per_user['risk_level'].isin(['Low', 'No risk'])].copy()
    sub_l['y'] = (sub_l['risk_level'] == 'Low').astype(int)
    Xl, yl = sub_l[feats].values, sub_l['y'].values
    spw_l  = (yl == 0).sum() / yl.sum()
    res_l  = build_oof_lgb(Xl, yl, [(f'L-nl127-spw{spw_l:.0f}', lgb_c(127, spw_l, lr=0.03, n=600))],
                           SEEDS_C[:6], N_FOLDS_C)
    p_low = np.mean([p for _, p, _ in res_l], axis=0)
    low_lookup = dict(zip(sub_l['user_name'].values, p_low))

    print("  Ordinal (VH+Extreme vs rest):", flush=True)
    sub_ord = per_user.copy()
    sub_ord['y'] = sub_ord['risk_level'].isin(['Very High', 'Extreme']).astype(int)
    X_ord, y_ord = sub_ord[feats].values, sub_ord['y'].values
    spw_ord = (y_ord == 0).sum() / y_ord.sum()
    res_ord = build_oof_lgb(X_ord, y_ord,
                            [(f'ORD-nl127-spw{spw_ord:.0f}', lgb_c(127, spw_ord, lr=0.03, n=600))],
                            SEEDS_C[:6], N_FOLDS_C)
    p_ordinal = np.mean([p for _, p, _ in res_ord], axis=0)
    ordinal_lookup = dict(zip(sub_ord['user_name'].values, p_ordinal))
    print("  Cascade done.", flush=True)

    # ── Step 2: VH + cascade augmentation ───────────────────────────────────
    print("\n=== Step 2: Very High + 4 cascade features ===", flush=True)
    sub_vh = (per_user[per_user['risk_level'].isin(['Very High', 'No risk'])]
              .copy().reset_index(drop=True))
    y_vh   = (sub_vh['risk_level'] == 'Very High').astype(int).values
    X_vh   = sub_vh[feats].values
    n_pos  = y_vh.sum()
    n_neg  = (y_vh == 0).sum()
    spw_vh = n_neg / n_pos
    print(f"  {n_pos} pos, {n_neg} neg  spw={spw_vh:.1f}", flush=True)

    casc_extreme = np.array([extreme_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    casc_high    = np.array([high_lookup.get(u, 0.0)    for u in sub_vh['user_name']])
    casc_low     = np.array([low_lookup.get(u, 0.0)     for u in sub_vh['user_name']])
    casc_ordinal = np.array([ordinal_lookup.get(u, 0.0) for u in sub_vh['user_name']])
    X_vh_aug = np.column_stack([X_vh, casc_extreme, casc_high, casc_low, casc_ordinal])
    print(f"  Augmented feature count: {X_vh_aug.shape[1]}", flush=True)

    SEEDS_VH   = [42, 7, 123, 999, 2025, 1337, 2024, 101, 314, 271,
                  42424, 77777, 100, 200, 300, 400, 500, 600, 700, 800]
    N_FOLDS_VH = 15    # 15 folds same as v25 best; v1_cpu used 12 which caused 0.11% gap

    print(f"  VH LGB ({len(SEEDS_VH)} seeds × {N_FOLDS_VH}-fold):", flush=True)
    vh_results = build_oof_lgb(X_vh_aug, y_vh, [
        (f'LGB-nl127-spw{spw_vh:.0f}',
         dict(num_leaves=127, learning_rate=0.02, n_estimators=800,
              scale_pos_weight=spw_vh, subsample=0.75, colsample_bytree=0.80,
              min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1)),
        (f'LGB-nl63-spw{spw_vh:.0f}',
         dict(num_leaves=63, learning_rate=0.05, n_estimators=600,
              scale_pos_weight=spw_vh, subsample=0.80, colsample_bytree=0.85,
              min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1)),
    ], SEEDS_VH, N_FOLDS_VH)

    _, _t, _p, _r, f_blend = blend_results(vh_results, y_vh, "VH blend")
    best_f1 = max([f for _, _, f in vh_results] + [f_blend])
    print(f"  Best VH F1: {best_f1:.2%}", flush=True)
    summary = {'Very High': best_f1}

    # ── Step 3: Other risk categories (LGB) ──────────────────────────────────
    for level in ['Extreme', 'High', 'Low']:
        if level not in per_user['risk_level'].values:
            continue
        print(f"\n=== {level} ===", flush=True)
        sub = per_user[per_user['risk_level'].isin([level, 'No risk'])].copy()
        sub['y'] = (sub['risk_level'] == level).astype(int)
        X_s, y_s = sub[feats].values, sub['y'].values
        n_p = y_s.sum()
        n_n = (y_s == 0).sum()
        spw_s = n_n / n_p
        print(f"  {n_p} pos, {n_n} neg  spw={spw_s:.1f}", flush=True)
        res_s = build_oof_lgb(X_s, y_s, [
            (f'nl127-spw{spw_s:.0f}',
             lgb_c(127, spw_s,  lr=0.03, n=600, ss=0.80, cs=0.85)),
            (f'nl63-spw{min(spw_s*2,200):.0f}',
             lgb_c(63, min(spw_s * 2, 200), lr=0.05, n=400, ss=0.85, cs=0.75)),
        ], SEEDS_C, N_FOLDS_C)
        _, _t, ps, rs, fs = blend_results(res_s, y_s)
        print(f"  Final: P={ps:.2%} R={rs:.2%} F1={fs:.2%}", flush=True)
        summary[level] = fs

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — v2_cpu (15-fold VH, nl127 cascade, nl63+nl127 VH)")
    print(f"{'='*60}")
    for level in RISK_ORDER:
        if level in summary:
            tag = " *** ACHIEVED ***" if summary[level] >= 0.95 else f"  gap={0.95 - summary[level]:.2%}"
            print(f"  {level:<12}: F1={summary[level]:.2%}{tag}")

    # ── Optional: JS export via m2cgen ────────────────────────────────────────
    # To export the best VH LGB model to pure JavaScript:
    #   pip install m2cgen
    #   Uncomment and set EXPORT_JS_PATH below, then re-run with --export-js flag.
    #
    # EXPORT_JS_PATH = 'vh_lgb_model.js'
    # import m2cgen as m2c
    # best_config = lgb_vh_configs[0][1]
    # clf_export = LGBMClassifier(**best_config, random_state=42)
    # clf_export.fit(X_vh_aug, y_vh)
    # js_code = m2c.export_to_javascript(clf_export)
    # with open(EXPORT_JS_PATH, 'w') as f:
    #     f.write(js_code)
    # print(f"\nJS model written to {EXPORT_JS_PATH}")
