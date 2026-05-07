"""
e2e_cold_start_v2.py — E2E cold-start test with cumulative re-evaluation.

Pipeline per batch:
  1. Insert batch as cold rows (is_internal_data=False, risk_level='no risk')
  2. Run predict() — returns predictions for ALL users (new cold + previously warm)
  3. Evaluate new batch (cold path) and ALL accumulated users (warm re-prediction)
     so we can see how old-batch accuracy improves as context grows
  4. Promote batch to warm context (is_internal_data=True, ground-truth labels)

Usage:
  python e2e_cold_start_v2.py [--model "Amanda 🎀 GG swaps"] [--batches 5] [--model-dir models_cpu]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.metrics import classification_report

from db_verify import verify_db, Subscription
from db import db, TrackingLinkSubscriber

RISK_LABELS = ['No risk', 'Low', 'High', 'Very High', 'Extreme']
_RISK_ORDER = {r: i for i, r in enumerate(RISK_LABELS)}
_CHUNK = 500

RISK_TO_DB = {
    'No risk':   'no risk',
    'Low':       'low',
    'High':      'high',
    'Very High': 'very high',
    'Extreme':   'extreme',
}


def _normalize_risk(r):
    if not isinstance(r, str) or not r.strip():
        return 'No risk'
    t = r.strip().title()
    return 'No risk' if t == 'No Risk' else (t if t in _RISK_ORDER else 'No risk')


def _load_model_rows(model_name: str) -> pd.DataFrame:
    verify_db.connect(reuse_if_open=True)
    rows = list(
        Subscription.select()
        .where(Subscription.tracking_model_name == model_name)
        .dicts()
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['subscribed_at'] = pd.to_datetime(df['subscribed_at'], errors='coerce')
    df = df.dropna(subset=['subscribed_at', 'user_name'])
    df = df.sort_values('subscribed_at').reset_index(drop=True)
    df['risk_level'] = df['risk_level'].apply(_normalize_risk)
    return df


def _ensure_col():
    db.connect(reuse_if_open=True)
    db.execute_sql("""
        ALTER TABLE tracking_links_subscriber
        ADD COLUMN IF NOT EXISTS is_internal_data BOOLEAN DEFAULT FALSE
    """)


def _remove_model_rows(model_name: str):
    db.connect(reuse_if_open=True)
    deleted = (
        TrackingLinkSubscriber
        .delete()
        .where(TrackingLinkSubscriber.tracking_link_id == model_name)
        .execute()
    )
    if deleted:
        print(f"  Removed {deleted} existing rows for '{model_name}'", flush=True)


def _insert_cold_batch(batch: pd.DataFrame, model_name: str):
    db.connect(reuse_if_open=True)
    rows = []
    for _, r in batch.iterrows():
        uid = str(r.get('user_id') or '').strip()
        try:
            user_id_int = int(uid) if uid else None
        except ValueError:
            user_id_int = None
        try:
            chargebacks = int(float(str(r.get('total_chargebacks') or 0)))
        except (ValueError, TypeError):
            chargebacks = 0
        rows.append({
            'id':                f"e2ev2_{r['id']}",
            'tracking_link_id':  model_name,
            'username':          r.get('user_name'),
            'user_id':           user_id_int,
            'subscription_date': str(r.get('subscribed_at')),
            'risk_level':        'no risk',
            'total_chargebacks': chargebacks,
            'is_processed':      False,
            'is_internal_data':  False,
        })
    for i in range(0, len(rows), _CHUNK):
        TrackingLinkSubscriber.insert_many(rows[i:i + _CHUNK]).execute()


def _promote_to_warm(batch: pd.DataFrame):
    db.connect(reuse_if_open=True)
    for _, r in batch.iterrows():
        risk_db = RISK_TO_DB.get(r['risk_level'], 'no risk')
        (TrackingLinkSubscriber
         .update(is_internal_data=True, is_processed=True, risk_level=risk_db)
         .where(TrackingLinkSubscriber.id == f"e2ev2_{r['id']}")
         .execute())


def _max_risk_gt(df: pd.DataFrame) -> dict[str, str]:
    """Max ground-truth risk per username."""
    gt: dict[str, str] = {}
    for _, row in df.iterrows():
        uname = row.get('user_name')
        if not isinstance(uname, str) or not uname.strip():
            continue
        r = row['risk_level']
        prev = gt.get(uname)
        if prev is None or _RISK_ORDER[r] > _RISK_ORDER[prev]:
            gt[uname] = r
    return gt


def _score(gt_map: dict, pred_map: dict, label: str):
    y_true, y_pred, missing = [], [], 0
    for uname, gt in gt_map.items():
        pred = pred_map.get(uname)
        if pred is None:
            missing += 1
            continue
        y_true.append(gt)
        y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        print(f"  [{label}] ERROR: no predictions found", flush=True)
        return [], []

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    if missing:
        print(f"  [{label}] Warning: {missing} users missing from predict output", flush=True)
    print(f"\n  [{label}] accuracy: {correct}/{n} = {correct/n:.1%}", flush=True)
    labels_present = [l for l in RISK_LABELS if l in set(y_true) | set(y_pred)]
    print(classification_report(y_true, y_pred, labels=labels_present, zero_division=0),
          flush=True)
    return y_true, y_pred


def run_e2e_v2(model_name: str, n_batches: int = 5, model_dir: str = 'models_cpu',
               diag_csv: str = None):
    print(f"\n{'='*60}", flush=True)
    print(f"E2E cold-start v2: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    df = _load_model_rows(model_name)
    if df.empty:
        print(f"ERROR: No rows found for '{model_name}'", flush=True)
        return

    print(f"\nTotal rows: {len(df)}", flush=True)
    print("Ground-truth distribution:", flush=True)
    print(df['risk_level'].value_counts().to_string(), flush=True)

    _ensure_col()
    _remove_model_rows(model_name)

    batch_size = len(df) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (start + batch_size) if i < n_batches - 1 else len(df)
        batches.append(df.iloc[start:end].copy())

    print(f"\nBatch sizes: {[len(b) for b in batches]}", flush=True)

    from predict import predict as _predict

    # Ground-truth maps, grown each batch
    cumulative_gt: dict[str, str] = {}   # all users seen so far
    batch_gt_maps: list[dict] = []       # per-batch ground-truth

    final_y_true, final_y_pred = [], []

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n{'─'*60}", flush=True)
        print(f"Batch {batch_idx}/{n_batches}  ({len(batch)} rows)", flush=True)
        risk_dist = batch['risk_level'].value_counts().to_dict()
        print(f"  Risk: {risk_dist}", flush=True)
        date_range = (batch['subscribed_at'].min().date(),
                      batch['subscribed_at'].max().date())
        print(f"  Dates: {date_range[0]} → {date_range[1]}", flush=True)

        # Build ground-truth for this batch and add to cumulative
        batch_gt = _max_risk_gt(batch)
        batch_gt_maps.append(batch_gt)
        for uname, risk in batch_gt.items():
            prev = cumulative_gt.get(uname)
            if prev is None or _RISK_ORDER[risk] > _RISK_ORDER[prev]:
                cumulative_gt[uname] = risk

        _insert_cold_batch(batch, model_name)
        print(f"  Inserted {len(batch)} cold rows", flush=True)

        print(f"\nRunning predict (batch {batch_idx})...", flush=True)
        results = _predict(model_dir=model_dir)

        if results is None or results.empty:
            print("ERROR: predict returned no results", flush=True)
            _promote_to_warm(batch)
            continue

        pred_map = dict(zip(results['user_name'], results['predicted_risk']))

        if diag_csv is not None:
            diag = results[results['user_name'].isin(batch_gt.keys())].copy()
            diag['gt_risk'] = diag['user_name'].map(batch_gt)
            diag['batch'] = batch_idx
            diag['eval_model'] = model_name
            mode = 'w' if (batch_idx == 1 and not os.path.exists(diag_csv)) else 'a'
            diag.to_csv(diag_csv, mode=mode, header=(mode == 'w'), index=False)

        # ── Evaluate this batch (cold path) ───────────────────────────────────
        _score(batch_gt, pred_map, f"Batch {batch_idx} cold")

        # ── Re-evaluate each previous batch (now warm-path predictions) ───────
        for prev_idx, prev_gt in enumerate(batch_gt_maps[:-1], 1):
            _score(prev_gt, pred_map, f"Batch {prev_idx} re-eval (warm)")

        # ── Cumulative: all batches 1..N ──────────────────────────────────────
        cum_yt, cum_yp = _score(
            cumulative_gt, pred_map,
            f"Cumulative batches 1-{batch_idx} ({len(cumulative_gt)} users)"
        )
        final_y_true, final_y_pred = cum_yt, cum_yp

        # Promote to warm context for next batch
        print(f"  Promoting batch {batch_idx} to warm context...", flush=True)
        _promote_to_warm(batch)

    # Final summary (= cumulative after last batch)
    if final_y_true:
        n_total = len(final_y_true)
        correct_total = sum(t == p for t, p in zip(final_y_true, final_y_pred))
        print(f"\n{'='*60}", flush=True)
        print(f"FINAL ACCURACY: {correct_total}/{n_total} = {correct_total/n_total:.1%}"
              f"  ({n_batches} batches, all users)", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E2E cold-start v2 with cumulative eval')
    parser.add_argument('--model',     default='Amanda 🎀 GG swaps')
    parser.add_argument('--batches',   type=int, default=5)
    parser.add_argument('--model-dir', default='models_cpu')
    parser.add_argument('--diag-csv',  default=None,
                        help='If set, append per-batch predictions+probas+gt to this CSV')
    args = parser.parse_args()

    run_e2e_v2(
        model_name=args.model,
        n_batches=args.batches,
        model_dir=args.model_dir,
        diag_csv=args.diag_csv,
    )
