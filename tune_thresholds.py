"""
tune_thresholds.py — offline threshold sweep using diag CSV from e2e_cold_start_v2.

Vectorized replay of predict.py's full cascade. Lets us iterate on thresholds
without re-running the slow full e2e (~25 min/model).
"""
import argparse, json
import numpy as np
import pandas as pd
from itertools import product


# Risk codes for vectorized decisions
NO_RISK, LOW, HIGH, VH, EXTREME = 0, 1, 2, 3, 4
LABELS = {NO_RISK: 'No risk', LOW: 'Low', HIGH: 'High', VH: 'Very High', EXTREME: 'Extreme'}


def replay_vec(df, t):
    """Vectorized cascade replay. Returns array of int risk codes."""
    e   = df['extreme_proba'].values
    cvh = df['cold_vh_proba'].values
    chh = df['cold_high_proba'].values
    cll = df['cold_low_proba'].values
    op  = df['ord_proba'].values
    hp  = df['high_proba'].values
    vp  = df['vh_proba'].values
    lp  = df['low_proba'].values
    wr  = df['warm_rescue_proba'].values
    cold_mask = df['cold_mask'].values.astype(bool)
    is_actual_cold = df['is_cold'].values.astype(bool)
    mvf = df['model_vh_frac'].values.astype(float)
    mwc = df['model_warm_count'].values.astype(int)

    t_e          = t.get('extreme', 0.605)
    t_vh         = t.get('vh', 0.43)
    t_h          = t.get('high', 0.397)
    t_l          = t.get('low', 0.648)
    COLD_T_E     = t.get('extreme_cold', 0.50)
    t_h_cold     = t.get('high_cold', t_h)
    t_l_cold     = t.get('low_cold', t_l)
    t_cold_vh_e  = t.get('vh_extreme_cold', 0.5)
    t_cold_vh_ex = t.get('cold_vh_in_extreme', t_cold_vh_e)
    t_cold_high  = t.get('cold_high_extreme', 0.5)
    t_cold_low   = t.get('cold_low_norisk', 0.5)
    t_warm_rescue = t.get('warm_vh_rescue', 0.47)
    t_warm_rescue_vhfrac = t.get('warm_vh_rescue_vhfrac', t_warm_rescue)
    t_cold_vh_vhfrac_gate = t.get('cold_vh_vhfrac_gate', 0.90)
    t_cold_vh_soft = t.get('cold_vh_soft_gate', 0.01)
    t_cold_ord = t.get('cold_ord_gate', 0.10)
    MIN_WARM_ROWS = t.get('min_warm_rows', 55)

    n = len(df)
    out = np.full(n, -1, dtype=np.int8)

    # Cold path branch
    cm = cold_mask
    # Per-branch low rescue thresholds (set very high to disable a rescue)
    t_low_in_extreme = t.get('cold_low_in_extreme', t_cold_low)
    t_low_in_vh      = t.get('cold_low_in_vh',      t_cold_low)
    t_low_in_high    = t.get('cold_low_in_high',    t_cold_low)
    # New: warm h_proba High rescue inside Extreme branch (catches Highs with low chh)
    t_warm_h_in_extreme = t.get('warm_h_in_extreme', 1.01)
    # branch 1: e >= COLD_T_E
    b1 = cm & (e >= COLD_T_E)
    out[b1 & (cvh >= t_cold_vh_ex)] = VH
    rest = b1 & (out == -1)
    out[rest & (hp >= t_warm_h_in_extreme)] = HIGH
    rest = b1 & (out == -1)
    out[rest & (chh >= t_cold_high)] = HIGH
    rest = b1 & (out == -1)
    out[rest & (cll >= t_low_in_extreme)] = LOW
    out[b1 & (out == -1)] = EXTREME
    # branch 2: cvh >= t_cold_vh_e and (e >= soft or op >= cold_ord)
    rest_cm = cm & (out == -1)
    b2 = rest_cm & (cvh >= t_cold_vh_e) & ((e >= t_cold_vh_soft) | (op >= t_cold_ord))
    out[b2 & (cll >= t_low_in_vh)] = LOW
    out[b2 & (out == -1)] = VH
    # branch 3: hp >= t_h_cold
    rest_cm = cm & (out == -1)
    b3 = rest_cm & (hp >= t_h_cold)
    out[b3 & (cll >= t_low_in_high)] = LOW
    out[b3 & (out == -1)] = HIGH
    # branch 4: cll >= t_cold_low (with optional ord/e gate)
    cl_ord_gate = t.get('cold_low_ord_gate', 0.0)
    cl_e_gate = t.get('cold_low_e_gate', 0.0)
    cl_signal = (op >= cl_ord_gate) | (e >= cl_e_gate)
    rest_cm = cm & (out == -1)
    b4 = rest_cm & (cll >= t_cold_low) & cl_signal
    out[b4] = LOW
    # default for cold: No risk
    out[cm & (out == -1)] = NO_RISK

    # Warm path
    wm = ~cm
    # NEW universal high-cvh + ord rescue (regardless of mvf)
    t_cvh_uni = t.get('cvh_universal_gate', 1.01)  # disabled by default
    t_ord_uni = t.get('cvh_universal_ord', 0.05)
    g_uni = wm & (cvh >= t_cvh_uni) & (op >= t_ord_uni)
    out[g_uni] = VH
    # Secondary: medium cvh + high ord rescue
    t_cvh_uni2 = t.get('cvh_universal2_gate', 1.01)
    t_ord_uni2 = t.get('cvh_universal2_ord', 0.50)
    g_uni2 = wm & (out == -1) & (cvh >= t_cvh_uni2) & (op >= t_ord_uni2)
    out[g_uni2] = VH
    # high-confidence cold_vh in vhfrac model
    g0 = wm & (out == -1) & (mvf > 0.10) & (cvh >= t_cold_vh_vhfrac_gate) & (e >= t_cold_vh_soft)
    out[g0] = VH
    rest_w = wm & (out == -1)
    # extreme branch
    b1w = rest_w & (e >= t_e)
    # sub-branch when vh_frac > 0.01
    # cvh-VH gate optionally requires h_proba above some floor (avoid Extreme FP)
    t_cvh_h_floor = t.get('cvh_in_extreme_h_floor', 0.0)
    b1w_vh = b1w & (mvf > 0.01)
    out[b1w_vh & (cvh >= t_cold_vh_ex) & (hp >= t_cvh_h_floor)] = VH
    out[b1w_vh & (out == -1) & (wr >= t_warm_rescue_vhfrac)] = VH
    out[b1w_vh & (out == -1) & (chh >= t_cold_high)] = HIGH
    out[b1w_vh & (out == -1)] = EXTREME
    # sub-branch when vh_frac <= 0.01
    b1w_nv = b1w & (mvf <= 0.01)
    out[b1w_nv & (wr >= t_warm_rescue)] = VH
    out[b1w_nv & (out == -1)] = EXTREME

    # warm path remaining
    suppress = is_actual_cold & (mwc < MIN_WARM_ROWS)
    rest_w = wm & (out == -1)
    out[rest_w & (vp >= t_vh) & ~suppress] = VH
    rest_w = wm & (out == -1)
    out[rest_w & (hp >= t_h) & ~suppress] = HIGH
    rest_w = wm & (out == -1)
    out[rest_w & (lp >= t_l) & ~suppress] = LOW
    out[wm & (out == -1)] = NO_RISK

    # Fallback: if predicted No risk on warm path AND ord_proba is very high,
    # escalate based on cvh (high cvh → VH, else → Extreme)
    t_ord_fallback = t.get('ord_fallback_gate', 1.01)
    t_cvh_fb_split = t.get('cvh_fallback_split', 0.50)
    fb_mask = wm & (out == NO_RISK) & (op >= t_ord_fallback)
    out[fb_mask & (cvh >= t_cvh_fb_split)] = VH
    out[fb_mask & (out == NO_RISK)] = EXTREME

    return out


def gt_codes(df):
    m = {'No risk': NO_RISK, 'Low': LOW, 'High': HIGH, 'Very High': VH, 'Extreme': EXTREME}
    return df['gt_risk'].map(m).values


def score(df, t, gt=None):
    pred = replay_vec(df, t)
    if gt is None:
        gt = gt_codes(df)
    correct = (pred == gt).astype(int)

    # Per-batch
    per_batch = []
    keys = list(zip(df['eval_model'].values, df['batch'].values))
    bydict = {}
    for k, c in zip(keys, correct):
        bydict.setdefault(k, [0, 0])
        bydict[k][0] += int(c)
        bydict[k][1] += 1
    for (em, b), (c, n) in sorted(bydict.items()):
        per_batch.append((em, b, c, n, c / n))

    overall = correct.sum() / len(correct)
    return overall, per_batch, pred


def report(t, df, label=''):
    overall, per_batch, _ = score(df, t)
    n_pass = sum(1 for _, _, _, _, a in per_batch if a >= 0.90)
    min_b = min(a for _, _, _, _, a in per_batch)
    print(f"\n=== {label} ===")
    print(f"Overall: {overall:.3%} | min_batch: {min_b:.3%} | passing: {n_pass}/{len(per_batch)}")
    for em, b, c, n, a in per_batch:
        marker = '+' if a >= 0.90 else '-'
        print(f"  {marker} {em[:30]:30}  B{b}  {c}/{n} = {a:.3%}")
    return overall, min_b, n_pass


def show_misses(df, t, label=''):
    pred = replay_vec(df, t)
    gt = gt_codes(df)
    miss_mask = pred != gt
    df = df.copy()
    df['_pred'] = [LABELS[p] for p in pred]
    miss = df[miss_mask]
    print(f"\n--- {label} misses ({len(miss)}) ---")
    cols = ['eval_model','batch','user_name','gt_risk','_pred','cold_mask',
            'extreme_proba','vh_proba','high_proba','low_proba','ord_proba',
            'cold_vh_proba','cold_high_proba','cold_low_proba','warm_rescue_proba',
            'model_vh_frac']
    cols = [c for c in cols if c in miss.columns]
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 250)
    print(miss[cols].to_string(max_colwidth=18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag-csv', default='diag_baseline.csv')
    ap.add_argument('--thresholds', default='models_cpu/thresholds.json')
    ap.add_argument('--show-misses', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--sweep2', action='store_true', help='Larger sweep')
    ap.add_argument('--write', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.diag_csv)
    df = df[df['is_explicit_cold'] == True].copy()
    print(f"loaded {len(df)} explicit-cold rows from {args.diag_csv}")
    print(f"models: {df['eval_model'].unique().tolist()}")
    bg = df.groupby(['eval_model', 'batch']).size()
    print("rows per batch:")
    print(bg.to_string())
    print(f"cold_mask values: {df['cold_mask'].value_counts().to_dict()}")

    with open(args.thresholds) as f:
        t = json.load(f)
    t.setdefault('extreme_cold', 0.50)
    t.setdefault('cold_vh_soft_gate', 0.01)
    t.setdefault('cold_ord_gate', 0.10)
    t.setdefault('min_warm_rows', 55)

    report(t, df, label='current thresholds')

    if args.show_misses:
        show_misses(df, t, label='current')

    if args.sweep:
        gt = gt_codes(df)
        keys_b = list(zip(df['eval_model'].values, df['batch'].values))
        unique_batches = sorted(set(keys_b))
        batch_idx_lists = {b: np.array([i for i, k in enumerate(keys_b) if k == b]) for b in unique_batches}

        grid = {
            'extreme_cold':       [0.40, 0.50, 0.60],
            'cold_vh_in_extreme': [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
            'cold_high_extreme':  [0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
            'vh_extreme_cold':    [0.50, 0.60, 0.63, 0.70],
            'high_cold':          [0.20, 0.30, 0.38, 0.50],
            'cold_low_norisk':    [0.50, 0.65, 0.80, 0.92, 0.95],
            'warm_vh_rescue_vhfrac': [0.50, 0.65, 0.75, 0.85],
        }
        keys = list(grid.keys())
        n = 1
        for v in grid.values(): n *= len(v)
        print(f"\nsweeping {n} combinations...")

        best_score = -1.0
        best_overall = -1.0
        best_t = None
        best_breakdown = None
        for vals in product(*[grid[k] for k in keys]):
            tt = dict(t)
            for k, v in zip(keys, vals):
                tt[k] = v
            pred = replay_vec(df, tt)
            correct = (pred == gt).astype(int)
            min_b = 1.0
            per_batch = []
            for b, idx in batch_idx_lists.items():
                c = int(correct[idx].sum())
                nb = len(idx)
                acc = c / nb
                if acc < min_b:
                    min_b = acc
                per_batch.append((b[0], b[1], c, nb, acc))
            overall = correct.sum() / len(correct)
            if (min_b > best_score) or (min_b == best_score and overall > best_overall):
                best_score = min_b
                best_overall = overall
                best_t = tt
                best_breakdown = per_batch
        print(f"\n=== best per-batch min: {best_score:.3%} (overall {best_overall:.3%}) ===")
        print(json.dumps({k: best_t[k] for k in keys}, indent=2))
        for em, b, c, n, a in best_breakdown:
            marker = '+' if a >= 0.90 else '-'
            print(f"  {marker} {em[:30]:30}  B{b}  {c}/{n} = {a:.3%}")

        if args.write:
            with open(args.write, 'w') as f:
                json.dump(best_t, f, indent=2)
            print(f"\nwritten best thresholds to {args.write}")


if __name__ == '__main__':
    main()
