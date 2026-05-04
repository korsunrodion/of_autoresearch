"""
Accuracy test for the cold-start prediction pipeline.

Seeds DATABASE_URL from DB_VERIFY_URL:
  - All subscriptions *not* in the N highest-count models → warm context
    (is_internal_data=True, is_processed=True, risk_level preserved)
  - The N highest-count models → cold test batch
    (is_internal_data=False, is_processed=False, risk_level cleared)

Then runs predict and compares predictions against the ground-truth labels
from DB_VERIFY_URL.

Usage:
  python test_accuracy.py [--n-holdout 17] [--model-dir models_cpu] [--output results.csv]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from db_verify import verify_db, Subscription
from db import db, TrackingLinkSubscriber

RISK_LABELS  = ['No risk', 'Low', 'High', 'Very High', 'Extreme']
_RISK_ORDER  = {r: i for i, r in enumerate(RISK_LABELS)}
_CHUNK       = 500


def _normalize_risk(r):
    if not isinstance(r, str) or not r.strip():
        return 'No risk'
    t = r.strip().title()
    return 'No risk' if t == 'No Risk' else (t if t in _RISK_ORDER else 'No risk')


def _ensure_col():
    """Add is_internal_data to tracking_links_subscriber if missing."""
    db.connect(reuse_if_open=True)
    db.execute_sql("""
        ALTER TABLE tracking_links_subscriber
        ADD COLUMN IF NOT EXISTS is_internal_data BOOLEAN DEFAULT FALSE
    """)


def _seed(warm: pd.DataFrame, cold: pd.DataFrame):
    db.connect(reuse_if_open=True)

    print(f"  Clearing tracking_links_subscriber...", flush=True)
    TrackingLinkSubscriber.delete().execute()

    def _rows(df, is_internal_data, clear_risk):
        out = []
        for _, r in df.iterrows():
            uid = str(r.get('user_id') or '').strip()
            try:
                user_id_int = int(uid) if uid else None
            except ValueError:
                user_id_int = None
            try:
                chargebacks = int(float(str(r.get('total_chargebacks') or 0)))
            except (ValueError, TypeError):
                chargebacks = 0
            out.append({
                'id':               f"test_{r['id']}",
                'tracking_link_id': str(r.get('tracking_model_name') or ''),
                'username':         r.get('user_name'),
                'user_id':          user_id_int,
                'subscription_date': r.get('subscribed_at'),
                'risk_level':       'no risk' if clear_risk else (r.get('risk_level') or 'no risk'),
                'total_chargebacks': chargebacks,
                'is_processed':     not clear_risk,
                'is_internal_data': is_internal_data,
            })
        return out

    warm_rows = _rows(warm, is_internal_data=True,  clear_risk=False)
    cold_rows = _rows(cold, is_internal_data=False, clear_risk=True)

    for batch in [warm_rows, cold_rows]:
        for i in range(0, len(batch), _CHUNK):
            TrackingLinkSubscriber.insert_many(batch[i:i + _CHUNK]).execute()

    print(f"  Seeded {len(warm_rows)} warm + {len(cold_rows)} cold rows", flush=True)


def run_test(n_holdout: int = 17, model_dir: str = 'models_cpu',
             output: str | None = None, min_rows: int = 50):

    # ── 1. Load ground-truth data ─────────────────────────────────────────────
    print("Loading verify DB...", flush=True)
    verify_db.connect(reuse_if_open=True)
    all_subs = pd.DataFrame(list(Subscription.select().dicts()))
    print(f"  {len(all_subs)} subscriptions, "
          f"{all_subs['tracking_model_name'].nunique()} models", flush=True)

    if all_subs.empty:
        print("ERROR: verify DB is empty", flush=True)
        return

    # ── 2. Pick holdout models ────────────────────────────────────────────────
    model_counts = all_subs.groupby('tracking_model_name').size()
    eligible = model_counts[model_counts >= min_rows].sort_values(ascending=False)

    if eligible.empty:
        print(f"ERROR: no models with >= {min_rows} rows in verify DB", flush=True)
        return

    # Per-model risk counts for greedy balanced selection
    eligible_subs = all_subs[all_subs['tracking_model_name'].isin(eligible.index)]
    risk_counts = (eligible_subs
                   .groupby(['tracking_model_name', 'risk_level']).size()
                   .unstack(fill_value=0))
    for lvl in ['Low', 'High', 'Very High']:
        if lvl not in risk_counts.columns:
            risk_counts[lvl] = 0

    # Greedy: for each required level, keep adding the richest model until target met
    LEVEL_TARGETS = [('Very High', 120), ('High', 40), ('Low', 80)]
    holdout_models: set[str] = set()
    totals: dict[str, int] = {lvl: 0 for lvl, _ in LEVEL_TARGETS}

    for level, target in LEVEL_TARGETS:
        candidates = sorted(
            [m for m in eligible.index if m not in holdout_models],
            key=lambda m: int(risk_counts.loc[m, level]) if m in risk_counts.index else 0,
            reverse=True,
        )
        for m in candidates:
            if totals[level] >= target:
                break
            holdout_models.add(m)
            for lvl, _ in LEVEL_TARGETS:
                totals[lvl] += int(risk_counts.loc[m, lvl]) if m in risk_counts.index else 0

    # If n_holdout is larger, pad with the highest-count eligible models not yet selected
    for m in eligible.index:
        if len(holdout_models) >= n_holdout and all(
                totals[lvl] >= tgt for lvl, tgt in LEVEL_TARGETS):
            break
        holdout_models.add(m)

    actual_holdout = len(holdout_models)
    warm_subs = all_subs[~all_subs['tracking_model_name'].isin(holdout_models)].copy()
    cold_subs = all_subs[ all_subs['tracking_model_name'].isin(holdout_models)].copy()

    print(f"  Warm : {len(warm_subs)} rows, "
          f"{warm_subs['tracking_model_name'].nunique()} models", flush=True)
    print(f"  Cold : {len(cold_subs)} rows in {actual_holdout} holdout models"
          f"  (VH={totals['Very High']}, High={totals['High']}, Low={totals['Low']})",
          flush=True)
    print(f"  Holdout risk dist:\n"
          f"{cold_subs['risk_level'].value_counts().to_string()}", flush=True)

    # ── 3. Build ground-truth map (max risk per user across holdout models) ───
    gt_per_user: dict[str, str] = {}
    for _, row in cold_subs.iterrows():
        uname = row.get('user_name')
        if not isinstance(uname, str) or not uname.strip():
            continue
        r = _normalize_risk(row.get('risk_level'))
        prev = gt_per_user.get(uname)
        if prev is None or _RISK_ORDER[r] > _RISK_ORDER[prev]:
            gt_per_user[uname] = r

    print(f"  {len(gt_per_user)} unique users in holdout set", flush=True)

    # ── 4. Seed main DB ───────────────────────────────────────────────────────
    _ensure_col()
    _seed(warm_subs, cold_subs)

    # ── 5. Run predict ────────────────────────────────────────────────────────
    print("\nRunning predict...", flush=True)
    from predict import predict as _predict
    results = _predict(model_dir=model_dir)

    if results is None or results.empty:
        print("ERROR: predict returned no results", flush=True)
        return

    # ── 6. Accuracy report ───────────────────────────────────────────────────
    pred_map = dict(zip(results['user_name'], results['predicted_risk']))

    y_true, y_pred, missing = [], [], 0
    for uname, gt in gt_per_user.items():
        pred = pred_map.get(uname)
        if pred is None:
            missing += 1
            continue
        y_true.append(gt)
        y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        print("ERROR: no overlap between ground truth and predict output", flush=True)
        return

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    if missing:
        print(f"  Warning: {missing} ground-truth users not found in predict output",
              flush=True)

    labels_present = [l for l in RISK_LABELS if l in set(y_true) | set(y_pred)]

    print(f"\n{'='*60}", flush=True)
    print(f"Accuracy: {correct}/{n} = {correct/n:.1%}  "
          f"({actual_holdout} holdout models, {n} users)", flush=True)
    print(f"{'='*60}", flush=True)

    print("\nClassification report:", flush=True)
    print(classification_report(y_true, y_pred,
                                labels=labels_present, zero_division=0), flush=True)

    print("Confusion matrix (rows=truth, cols=predicted):", flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=labels_present, columns=labels_present)
    print(cm_df.to_string(), flush=True)

    # Per-model breakdown
    cold_results = results[results['user_name'].isin(gt_per_user)]
    if not cold_results.empty:
        print("\nPredicted distribution (cold batch):", flush=True)
        print(cold_results['predicted_risk'].value_counts().to_string(), flush=True)

    if output:
        rows = []
        for uname, gt in gt_per_user.items():
            rows.append({
                'user_name':    uname,
                'ground_truth': gt,
                'predicted':    pred_map.get(uname, 'MISSING'),
                'correct':      pred_map.get(uname) == gt,
            })
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"\nDetailed comparison saved to {output}", flush=True)

    return y_true, y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cold-start accuracy test')
    parser.add_argument('--n-holdout', type=int, default=17,
                        help='Number of high-count models to hold out as cold batch')
    parser.add_argument('--model-dir',  default='models_cpu')
    parser.add_argument('--output',     default=None,
                        help='CSV path for per-user comparison output')
    parser.add_argument('--min-rows',   type=int, default=50,
                        help='Minimum subscription count for a model to be holdout-eligible')
    args = parser.parse_args()

    run_test(
        n_holdout=args.n_holdout,
        model_dir=args.model_dir,
        output=args.output,
        min_rows=args.min_rows,
    )
