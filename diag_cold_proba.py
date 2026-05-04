"""
diag_cold_proba.py — diagnose production cold-start VH misclassification.

Reads directly from VERIFY_DB_URL subscriptions table (no DB seeding).
Selects top N models by subscriber count as holdout (matches test_accuracy.py).
Runs the cold cascade in-memory and prints probability distributions for:
  - Cold VH users predicted VH  (correct)
  - Cold VH users predicted Extreme  (the main problem)
  - Cold VH users predicted No risk  (missed)
  - Cold No-risk users predicted Extreme  (false positives)
  - Cold Extreme users predicted VH  (false VH)

Then sweeps ord_proba and warm_rescue_proba thresholds in the Extreme branch
to show which threshold, if any, cleanly rescues the missed VH users.

Usage:
  python diag_cold_proba.py [--n-holdout 17] [--model-dir models_cpu]
"""
import argparse, sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from db_verify import verify_db, Subscription
from classifier_v2_cpu_predict import compute_per_user, make_cold_X

_RISK_MAP = {'no risk': 1, 'low': 2, 'high': 3, 'very high': 4, 'extreme': 5}
_RISK_TITLE = {
    'no risk': 'No risk', 'low': 'Low', 'high': 'High',
    'very high': 'Very High', 'extreme': 'Extreme',
}


def _load_verify(min_rows: int, n_holdout: int):
    """Load subscriptions from verify DB, return (df_warm, df_cold_true, gt_per_user)."""
    verify_db.connect(reuse_if_open=True)
    rows = list(Subscription.select().dicts())
    verify_db.close()

    df = pd.DataFrame(rows)
    print(f"  Verify DB: {len(df)} rows, {df['tracking_model_name'].nunique()} models",
          flush=True)

    # Normalize
    df['subscribed_at'] = pd.to_datetime(df['subscribed_at'], errors='coerce')
    df = df.dropna(subset=['subscribed_at', 'user_name', 'tracking_model_name'])

    df['risk_level'] = (df['risk_level'].fillna('no risk')
                        .str.strip().str.lower()
                        .map(lambda r: _RISK_TITLE.get(r, 'No risk')))
    df['risk_score'] = df['risk_level'].map(
        {'No risk': 1, 'Low': 2, 'High': 3, 'Very High': 4, 'Extreme': 5}).fillna(1)

    df['user_id_num'] = pd.to_numeric(df['user_id'], errors='coerce')
    df = df.dropna(subset=['user_id_num'])
    df['user_id_num'] = df['user_id_num'].astype(int)

    df['total_chargebacks'] = pd.to_numeric(
        df['total_chargebacks'], errors='coerce').fillna(0).astype(int)
    df['subscribed_ts'] = df['subscribed_at'].astype('int64') // 10**9

    # Select top-N holdout models by subscriber count (mirrors test_accuracy.py)
    model_counts = df.groupby('tracking_model_name').size()
    eligible = model_counts[model_counts >= min_rows].sort_values(ascending=False)
    n = min(n_holdout, len(eligible))
    holdout_set = set(eligible.head(n).index)
    print(f"  Top-{n} holdout models by count:", flush=True)
    for m in list(holdout_set)[:5]:
        dist = df[df['tracking_model_name'] == m]['risk_level'].value_counts()
        print(f"    {m[:60]:<60} {int(eligible[m]):>4} rows  "
              + '  '.join(f"{k}={v}" for k, v in dist.items()))
    if n > 5:
        print(f"    ... ({n - 5} more)")

    keep_cols = ['user_name', 'tracking_model_name', 'subscribed_at', 'subscribed_ts',
                 'user_id_num', 'risk_level', 'risk_score', 'total_chargebacks']

    df_cold_true = df[df['tracking_model_name'].isin(holdout_set)][keep_cols].copy()
    df_warm      = df[~df['tracking_model_name'].isin(holdout_set)][keep_cols].copy()
    df_warm['is_internal_data'] = True

    # Ground-truth: highest risk label per user across holdout models
    gt = {}
    for _, r in df_cold_true.iterrows():
        u = r['user_name']
        if u not in gt or _RISK_MAP.get(r['risk_level'].lower(), 0) > _RISK_MAP.get(gt[u].lower(), 0):
            gt[u] = r['risk_level']

    # Cold rows injected with No risk + is_internal_data=False
    df_cold_sim = df_cold_true.copy()
    df_cold_sim['risk_level']       = 'No risk'
    df_cold_sim['risk_score']       = 1
    df_cold_sim['is_internal_data'] = False

    df_sim = pd.concat([df_warm, df_cold_sim], ignore_index=True)
    print(f"  Sim: {len(df_warm)} warm + {len(df_cold_sim)} cold = {len(df_sim)} total",
          flush=True)
    print(f"  Holdout risk dist: { {k: v for k, v in df_cold_true['risk_level'].value_counts().items()} }",
          flush=True)
    return df_sim, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-holdout', type=int, default=17)
    parser.add_argument('--model-dir',  default='models_cpu')
    parser.add_argument('--min-rows',   type=int, default=50)
    args = parser.parse_args()

    print("Loading verify DB...", flush=True)
    df_sim, gt_per_user = _load_verify(args.min_rows, args.n_holdout)

    # ── Features ──────────────────────────────────────────────────────────────
    per_user = compute_per_user(df_sim)

    df_w = df_sim[df_sim['is_internal_data'] == True]
    warm_rf, warm_nr = {}, {}
    for mn, mdf in df_w.groupby('tracking_model_name'):
        n = len(mdf)
        warm_rf[mn] = {
            'vh':      (mdf['risk_level'] == 'Very High').sum() / n,
            'extreme': (mdf['risk_level'] == 'Extreme').sum() / n,
            'any':     (~mdf['risk_level'].isin(['No risk'])).sum() / n,
        }
        warm_nr[mn] = (mdf['risk_level'] == 'No risk').sum() / n

    utm = df_sim.drop_duplicates('user_name').set_index('user_name')['tracking_model_name']
    pu_m = per_user['user_name'].map(utm)
    per_user['model_vh_frac']       = pu_m.map(lambda m: warm_rf.get(m, {}).get('vh', 0.0))
    per_user['model_extreme_frac']  = pu_m.map(lambda m: warm_rf.get(m, {}).get('extreme', 0.0))
    per_user['model_any_risk_frac'] = pu_m.map(lambda m: warm_rf.get(m, {}).get('any', 0.0))
    per_user['model_norisk_rate']   = pu_m.map(lambda m: warm_nr.get(m, 1.0))

    # ── Load models ───────────────────────────────────────────────────────────
    mdir = args.model_dir
    models = {}
    for name in ['extreme', 'high', 'low', 'ordinal', 'vh', 'cold_vh', 'warm_vh_rescue']:
        p = os.path.join(mdir, f'{name}_model.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                models[name] = pickle.load(f)
    with open(os.path.join(mdir, 'thresholds.json')) as f:
        thresholds = json.load(f)
    with open(os.path.join(mdir, 'features.json')) as f:
        feats = json.load(f)['base_features']

    # ── Probabilities ─────────────────────────────────────────────────────────
    X_b       = per_user[feats].values
    e_proba   = models['extreme'].predict_proba(X_b)[:, 1]
    h_proba   = models['high'].predict_proba(X_b)[:, 1]
    l_proba   = models['low'].predict_proba(X_b)[:, 1]
    ord_proba = models['ordinal'].predict_proba(X_b)[:, 1]
    X_aug     = np.column_stack([X_b, e_proba, h_proba, l_proba, ord_proba])
    vh_proba  = models['vh'].predict_proba(X_aug)[:, 1]

    X_cold = make_cold_X(X_b, feats)
    cvh = models.get('cold_vh')
    if cvh is None:
        cold_vh_proba = np.zeros(len(per_user))
    elif isinstance(cvh, list):
        cold_vh_proba = np.mean([c.predict_proba(X_cold)[:, 1] for c in cvh], axis=0)
    else:
        cold_vh_proba = cvh.predict_proba(X_cold)[:, 1]

    wr = models.get('warm_vh_rescue')
    warm_rescue_proba = wr.predict_proba(X_aug)[:, 1] if wr else np.zeros(len(per_user))

    cold_mask = (per_user['model_any_risk_frac'].fillna(0) == 0).values

    t_e           = thresholds.get('extreme', 0.5)
    t_vh          = thresholds.get('vh', 0.5)
    t_h           = thresholds.get('high', 0.5)
    t_l           = thresholds.get('low', 0.5)
    COLD_T_E      = thresholds.get('extreme_cold', 0.50)
    t_h_cold      = thresholds.get('high_cold', t_h)
    t_l_cold      = thresholds.get('low_cold', t_l)
    t_cold_vh_e   = thresholds.get('vh_extreme_cold', 0.5)
    t_warm_rescue = thresholds.get('warm_vh_rescue', 0.47)
    t_soft        = thresholds.get('cold_vh_soft_gate', 0.01)
    t_ord         = thresholds.get('cold_ord_gate', 0.10)

    per_user['e_proba']           = e_proba
    per_user['h_proba']           = h_proba
    per_user['ord_proba']         = ord_proba
    per_user['vh_proba']          = vh_proba
    per_user['cold_vh_proba']     = cold_vh_proba
    per_user['warm_rescue_proba'] = warm_rescue_proba
    per_user['is_cold']           = cold_mask

    # ── Current cascade ───────────────────────────────────────────────────────
    predicted = []
    for i in range(len(per_user)):
        if cold_mask[i]:
            if e_proba[i] >= COLD_T_E:
                if cold_vh_proba[i] >= t_cold_vh_e:
                    predicted.append('Very High')
                else:
                    predicted.append('Extreme')
            elif cold_vh_proba[i] >= t_cold_vh_e and (
                    e_proba[i] >= t_soft or ord_proba[i] >= t_ord):
                predicted.append('Very High')
            elif h_proba[i] >= t_h_cold:
                predicted.append('High')
            elif l_proba[i] >= t_l_cold:
                predicted.append('Low')
            else:
                predicted.append('No risk')
        else:
            if e_proba[i] >= t_e:
                predicted.append('Very High' if warm_rescue_proba[i] >= t_warm_rescue else 'Extreme')
            elif vh_proba[i] >= t_vh:
                predicted.append('Very High')
            elif h_proba[i] >= t_h:
                predicted.append('High')
            elif l_proba[i] >= t_l:
                predicted.append('Low')
            else:
                predicted.append('No risk')
    per_user['predicted'] = predicted

    # ── Merge with ground truth ───────────────────────────────────────────────
    cold_pu = per_user[cold_mask].copy()
    gt_df = pd.DataFrame({'user_name': list(gt_per_user),
                          'true_risk':  list(gt_per_user.values())})
    cold_pu = cold_pu.merge(gt_df, on='user_name', how='inner')

    n_vh  = (cold_pu['true_risk'] == 'Very High').sum()
    n_ext = (cold_pu['true_risk'] == 'Extreme').sum()
    n_nr  = (cold_pu['true_risk'] == 'No risk').sum()
    print(f"\n  Cold users evaluated: {len(cold_pu)}  "
          f"(VH={n_vh}, Extreme={n_ext}, No risk={n_nr})")

    # ── Print probability tables ──────────────────────────────────────────────
    hdr = f"  {'e_proba':>10} {'cold_vh':>9} {'ord_proba':>10} {'vh_proba':>9} {'warm_rsc':>9}"

    def _show(label, mask):
        grp = cold_pu[mask].sort_values('e_proba', ascending=False)
        if grp.empty:
            print(f"\n{label}: (none)")
            return
        print(f"\n{label}  (n={len(grp)})")
        print(hdr)
        for _, r in grp.iterrows():
            print(f"  {r['e_proba']:>10.5f} {r['cold_vh_proba']:>9.4f} "
                  f"{r['ord_proba']:>10.4f} {r['vh_proba']:>9.4f} "
                  f"{r['warm_rescue_proba']:>9.4f}")

    print(f"\n{'='*70}")
    print(f"COLD CASCADE DIAGNOSTICS  "
          f"(t_cold_vh_e={t_cold_vh_e:.4f}  COLD_T_E={COLD_T_E:.2f}  "
          f"ord_gate={t_ord:.2f}  soft={t_soft:.3f})")
    print(f"{'='*70}")

    _show("VH → VH   [correct]",
          (cold_pu['true_risk'] == 'Very High') & (cold_pu['predicted'] == 'Very High'))
    _show("VH → Extreme   [MISSED — Extreme-gate, cold_vh rescue failed]",
          (cold_pu['true_risk'] == 'Very High') & (cold_pu['predicted'] == 'Extreme'))
    _show("VH → No risk   [MISSED]",
          (cold_pu['true_risk'] == 'Very High') & (cold_pu['predicted'] == 'No risk'))
    _show("No risk → Extreme   [false Extreme]",
          (cold_pu['true_risk'] == 'No risk') & (cold_pu['predicted'] == 'Extreme'))
    _show("Extreme → VH   [false VH]",
          (cold_pu['true_risk'] == 'Extreme') & (cold_pu['predicted'] == 'Very High'))

    # ── Threshold sweep: what helps in the Extreme branch? ────────────────────
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP — adding ord/warm_rescue to Extreme-branch rescue")
    print(f"{'Gate':>28}  VH_caught  No-risk→VH  Extreme→VH")

    vh_m   = cold_pu['true_risk'] == 'Very High'
    nr_m   = cold_pu['true_risk'] == 'No risk'
    ext_m  = cold_pu['true_risk'] == 'Extreme'
    in_ext = cold_pu['e_proba'] >= COLD_T_E   # users in the Extreme branch

    baseline_vh = (vh_m & (cold_pu['predicted'] == 'Very High')).sum()

    for tag, ord_gate, wr_gate in [
        ('baseline (current)',   None,  None),
        ('ord >= 0.50',          0.50,  None),
        ('ord >= 0.30',          0.30,  None),
        ('ord >= 0.20',          0.20,  None),
        ('ord >= 0.10',          0.10,  None),
        ('ord >= 0.05',          0.05,  None),
        ('warm_rescue >= 0.30',  None,  0.30),
        ('warm_rescue >= 0.20',  None,  0.20),
        ('warm_rescue >= 0.10',  None,  0.10),
        ('warm_rescue >= 0.05',  None,  0.05),
        ('ord>=0.10 OR wr>=0.10', 0.10, 0.10),
        ('ord>=0.05 OR wr>=0.05', 0.05, 0.05),
    ]:
        def _rescued(row):
            if not row['is_cold'] or row['e_proba'] < COLD_T_E:
                return False
            if row['cold_vh_proba'] >= t_cold_vh_e:
                return True
            if ord_gate is not None and row['ord_proba'] >= ord_gate:
                return True
            if wr_gate is not None and row['warm_rescue_proba'] >= wr_gate:
                return True
            return False

        rescued = cold_pu.apply(_rescued, axis=1)
        vh_new   = (vh_m  & in_ext & ~(cold_pu['predicted'] == 'Very High') & rescued).sum()
        fp_nr    = (nr_m  & in_ext & rescued).sum()
        fp_ext   = (ext_m & in_ext & rescued).sum()
        total_vh = baseline_vh + vh_new
        pct      = total_vh / max(vh_m.sum(), 1)
        print(f"  {tag:>28}:  {total_vh}/{vh_m.sum()}={pct:.0%}   "
              f"no-risk FP={fp_nr}   ext→VH={fp_ext}")


if __name__ == '__main__':
    main()
