"""
cold_start_test.py — Incremental cold-start batch test for a single model.

Loads all rows for the target model from DB_VERIFY_URL, splits them into
N chronological batches, and inserts them into DATABASE_URL one at a time.
After each predict() run, the batch is promoted to warm context
(is_internal_data=True, ground-truth labels) so subsequent batches benefit
from the accumulated history.

Usage:
  python cold_start_test.py [--model "Amanda 🎀 GG swaps"] [--batches 5] [--model-dir models_cpu]
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
    """Remove all rows for this model (cleanup for re-runs)."""
    db.connect(reuse_if_open=True)
    deleted = (
        TrackingLinkSubscriber
        .delete()
        .where(TrackingLinkSubscriber.tracking_link_id == model_name)
        .execute()
    )
    if deleted:
        print(f"  Removed {deleted} existing rows for '{model_name}'", flush=True)


def _reset_stale_cold_rows(model_name: str):
    """Promote leftover cold rows from previous test runs to warm.

    test_accuracy.py seeds cold rows for holdout models and may leave them behind.
    Those rows contaminate cold_start_test deduplication: they appear as explicit cold
    rows for other models and can override the Amanda batch predictions via max-risk dedup.
    Reset them to warm (is_internal_data=True) so only the current test's cold rows matter.
    """
    db.connect(reuse_if_open=True)
    n = (
        TrackingLinkSubscriber
        .update(is_internal_data=True)
        .where(
            (TrackingLinkSubscriber.is_internal_data == False) &
            (TrackingLinkSubscriber.tracking_link_id != model_name)
        )
        .execute()
    )
    if n:
        print(f"  Reset {n} stale cold rows (other models) to warm", flush=True)


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
            'id':                f"cs_{r['id']}",
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
    """Set is_internal_data=True and restore ground-truth labels for this batch."""
    db.connect(reuse_if_open=True)
    for _, r in batch.iterrows():
        risk_db = RISK_TO_DB.get(r['risk_level'], 'no risk')
        (TrackingLinkSubscriber
         .update(is_internal_data=True, is_processed=True, risk_level=risk_db)
         .where(TrackingLinkSubscriber.id == f"cs_{r['id']}")
         .execute())


def _batch_gt(batch: pd.DataFrame) -> dict:
    """Max ground-truth risk per username in this batch."""
    gt: dict[str, str] = {}
    for _, row in batch.iterrows():
        uname = row.get('user_name')
        if not isinstance(uname, str) or not uname.strip():
            continue
        r = row['risk_level']
        prev = gt.get(uname)
        if prev is None or _RISK_ORDER[r] > _RISK_ORDER[prev]:
            gt[uname] = r
    return gt


def run_cold_start_test(model_name: str, n_batches: int = 5,
                        model_dir: str = 'models_cpu'):
    print(f"\n{'='*60}", flush=True)
    print(f"Cold-start batch test: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    df = _load_model_rows(model_name)
    if df.empty:
        print(f"ERROR: No rows found for '{model_name}'", flush=True)
        return

    print(f"\nTotal rows: {len(df)}", flush=True)
    print("Ground-truth distribution:", flush=True)
    print(df['risk_level'].value_counts().to_string(), flush=True)

    _ensure_col()
    _reset_stale_cold_rows(model_name)
    _remove_model_rows(model_name)

    # Chronological split
    batch_size = len(df) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (start + batch_size) if i < n_batches - 1 else len(df)
        batches.append(df.iloc[start:end].copy())

    print(f"\nBatch sizes: {[len(b) for b in batches]}", flush=True)

    from predict import predict as _predict

    all_y_true, all_y_pred = [], []

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n{'─'*60}", flush=True)
        print(f"Batch {batch_idx}/{n_batches}  ({len(batch)} rows)", flush=True)
        risk_dist = batch['risk_level'].value_counts().to_dict()
        print(f"  Risk: {risk_dist}", flush=True)
        date_range = (batch['subscribed_at'].min().date(),
                      batch['subscribed_at'].max().date())
        print(f"  Dates: {date_range[0]} → {date_range[1]}", flush=True)

        _insert_cold_batch(batch, model_name)
        print(f"  Inserted {len(batch)} cold rows", flush=True)

        print(f"\nRunning predict (batch {batch_idx})...", flush=True)
        results = _predict(model_dir=model_dir)

        if results is None or results.empty:
            print("ERROR: predict returned no results", flush=True)
            _promote_to_warm(batch)
            continue

        pred_map = dict(zip(results['user_name'], results['predicted_risk']))
        batch_gt_map = _batch_gt(batch)

        y_true, y_pred, missing = [], [], 0
        for uname, gt in batch_gt_map.items():
            pred = pred_map.get(uname)
            if pred is None:
                missing += 1
                continue
            y_true.append(gt)
            y_pred.append(pred)
            all_y_true.append(gt)
            all_y_pred.append(pred)

        n = len(y_true)
        if n == 0:
            print(f"  ERROR: No predictions for batch {batch_idx} users", flush=True)
        else:
            correct = sum(t == p for t, p in zip(y_true, y_pred))
            if missing:
                print(f"  Warning: {missing} users missing from predict output", flush=True)
            print(f"\n  Batch {batch_idx} accuracy: {correct}/{n} = {correct/n:.1%}", flush=True)

            labels_present = [l for l in RISK_LABELS if l in set(y_true) | set(y_pred)]
            print(f"  Classification report (batch {batch_idx}):", flush=True)
            print(classification_report(y_true, y_pred,
                                        labels=labels_present, zero_division=0),
                  flush=True)

        # Promote to warm context so next batch benefits from this batch's history
        print(f"  Promoting batch {batch_idx} to warm context (ground-truth labels)...",
              flush=True)
        _promote_to_warm(batch)

    # Final summary
    if all_y_true:
        n_total = len(all_y_true)
        correct_total = sum(t == p for t, p in zip(all_y_true, all_y_pred))
        print(f"\n{'='*60}", flush=True)
        print(f"TOTAL ACCURACY: {correct_total}/{n_total} = {correct_total/n_total:.1%}"
              f"  ({n_batches} batches)", flush=True)
        print(f"{'='*60}", flush=True)
        labels_present = [l for l in RISK_LABELS if l in set(all_y_true) | set(all_y_pred)]
        print("\nOverall classification report:", flush=True)
        print(classification_report(all_y_true, all_y_pred,
                                    labels=labels_present, zero_division=0),
              flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental cold-start batch test')
    parser.add_argument('--model',     default='Amanda 🎀 GG swaps',
                        help='Model name to test (must exist in DB_VERIFY_URL)')
    parser.add_argument('--batches',   type=int, default=5,
                        help='Number of chronological batches to split the model into')
    parser.add_argument('--model-dir', default='models_cpu',
                        help='Directory with trained model .pkl files')
    args = parser.parse_args()

    run_cold_start_test(
        model_name=args.model,
        n_batches=args.batches,
        model_dir=args.model_dir,
    )
