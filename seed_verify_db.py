"""
seed_verify_db.py — populate verify DB for cold-start testing.

Steps:
1. Load all rows from main DB (subscriptions table)
2. Transfer ALL models except N cold-start models (with real risk labels, is_internal_data=True)
3. Insert cold-start models as 'no risk' (is_internal_data=False)
4. Print true risk distribution for cold models

Usage:
  uv run seed_verify_db.py [--n-cold 15] [--seed 42]
  uv run seed_verify_db.py [--cold-model MODEL_NAME]   # single specific model
"""
import sys
import io
import os
import uuid
import argparse
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from db import db as db_main, Subscription
from db_v2 import db as db_verify, TrackingLinkSubscriber

RISK_TO_VERIFY = {
    'No risk':   'no risk',
    'Low':       'low',
    'High':      'high',
    'Very High': 'very high',
    'Extreme':   'extreme',
}

CHUNK = 500


def clear_verify_db():
    db_verify.connect(reuse_if_open=True)
    deleted = TrackingLinkSubscriber.delete().execute()
    print(f"  Cleared {deleted} existing rows from verify DB.", flush=True)


def insert_rows(rows, is_internal):
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        data = []
        for r in chunk:
            risk_db = RISK_TO_VERIFY.get(r['risk_level'], 'no risk')
            if not is_internal:
                risk_db = 'no risk'   # cold-start: reset to default
            data.append({
                'id':               str(uuid.uuid4()),
                'tracking_link_id': r['tracking_model_name'],
                'username':         r['user_name'],
                'user_id':          int(r['user_id']) if r['user_id'] else None,
                'subscription_date': str(r['subscribed_at']),
                'risk_level':       risk_db,
                'is_internal_data': is_internal,
            })
        TrackingLinkSubscriber.insert_many(data).execute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cold-model', default=None,
                        help='Specific single model name to use as cold-start target')
    parser.add_argument('--n-cold', type=int, default=15,
                        help='Number of cold-start models (picked automatically)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load main DB ──────────────────────────────────────────────────────────
    print("Loading main DB...", flush=True)
    db_main.connect(reuse_if_open=True)
    rows = list(Subscription.select().dicts())
    db_main.close()
    print(f"  {len(rows)} rows loaded.", flush=True)

    from collections import defaultdict, Counter
    by_model = defaultdict(list)
    for r in rows:
        if r.get('tracking_model_name') and r.get('user_name') and r.get('user_id'):
            by_model[r['tracking_model_name']].append(r)

    models = list(by_model.keys())
    print(f"  {len(models)} models.", flush=True)

    # ── Pick cold-start models ────────────────────────────────────────────────
    if args.cold_model:
        if args.cold_model not in by_model:
            print(f"ERROR: model '{args.cold_model}' not found.", flush=True)
            sys.exit(1)
        cold_models = [args.cold_model]
    else:
        # Prefer models with VH/Extreme users and >=20 rows for meaningful test
        candidates = [
            m for m in models
            if any(r['risk_level'] in ('Very High', 'Extreme') for r in by_model[m])
            and len(by_model[m]) >= 20
        ]
        if len(candidates) < args.n_cold:
            candidates = [m for m in models if len(by_model[m]) >= 10]
        n = min(args.n_cold, len(candidates))
        cold_models = random.sample(candidates, n)

    print(f"\nCold-start models ({len(cold_models)}):", flush=True)
    total_cold_rows = 0
    for cm in cold_models:
        dist = Counter(r['risk_level'] for r in by_model[cm])
        n_users = len(by_model[cm])
        total_cold_rows += n_users
        risk_str = '  '.join(f"{lvl}={cnt}" for lvl, cnt in sorted(dist.items()) if cnt > 0)
        print(f"  {cm[:55]:<55} ({n_users} users)  {risk_str}", flush=True)

    # ── Transfer ALL remaining models with real labels ────────────────────────
    cold_set = set(cold_models)
    transfer_models = [m for m in models if m not in cold_set]
    transfer_rows = [r for m in transfer_models for r in by_model[m]]
    cold_rows = [r for m in cold_models for r in by_model[m]]

    print(f"\nWarm transfer: {len(transfer_models)} models, {len(transfer_rows)} rows", flush=True)
    print(f"Cold insert:   {len(cold_models)} models, {len(cold_rows)} rows", flush=True)

    # ── Populate verify DB ────────────────────────────────────────────────────
    print("\nClearing verify DB...", flush=True)
    clear_verify_db()

    db_verify.connect(reuse_if_open=True)
    print("Inserting warm rows (is_internal_data=True)...", flush=True)
    insert_rows(transfer_rows, is_internal=True)
    print(f"  Done: {len(transfer_rows)} rows.", flush=True)

    print("Inserting cold-start rows (all 'no risk', is_internal_data=False)...", flush=True)
    insert_rows(cold_rows, is_internal=False)
    print(f"  Done: {len(cold_rows)} rows.", flush=True)

    db_verify.close()

    print(f"\nVerify DB ready.")
    print(f"  Warm:  {len(transfer_rows)} rows across {len(transfer_models)} models")
    print(f"  Cold:  {len(cold_rows)} rows across {len(cold_models)} models (all 'no risk')")
    print(f"\nRun prediction:")
    print(f"  python classifier_v2_cpu_predict.py --predict --output results.csv")


if __name__ == '__main__':
    main()
