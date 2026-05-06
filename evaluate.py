"""
evaluate.py — autoresearch evaluator for the risk classifier.

Simulates production flow: for each test model, insert cold-row batches one
at a time and run predict() between each insertion. Cold rows stay cold
throughout (no ground-truth warm promotion); their *predicted* labels from
the previous round persist in the DB so the next round's compute_per_user
sees the model's evolving cold context. Final per-user prediction = the
prediction made in the LAST round, when full context is available.

Aggregation: y_true / y_pred concatenated across all test models'
final-round predictions; metrics computed once on the union.

Score (maximize): min(macro_f1_4risk - 0.95, f1_vh - 0.90, f1_high - 0.90, f1_low - 0.90)
Pass: score >= 0

Per-round per-model F1 trace is included in details so the user can verify
that accuracy improves as more cold context arrives.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

from db_verify import verify_db, Subscription
from db import db, TrackingLinkSubscriber

import classifier_v2_cpu_predict as _clf
from classifier_v2_cpu_predict import predict as _predict


# Monkey-patch update_risk_levels: the original does ~38k row-by-row UPDATEs
# via peewee (one per user in the predictions dict, even though only ~250 are
# cold rows for the test models). We replace it with a fast bulk variant that
# updates only the cold rows we know about, grouped by predicted label.
_TARGET_COLD_USERS = set()  # filled per-round in run_one_model

def _fast_bulk_update(predictions):
    if not _TARGET_COLD_USERS:
        return 0
    db.connect(reuse_if_open=True)
    by_label = {}
    for uname, risk_title in predictions.items():
        if uname not in _TARGET_COLD_USERS:
            continue
        risk_db_val = RISK_TO_DB.get(risk_title)
        if risk_db_val is None:
            continue
        by_label.setdefault(risk_db_val, []).append(uname)
    n = 0
    for label, users in by_label.items():
        # Chunk to avoid massive IN clauses
        for i in range(0, len(users), 500):
            chunk = users[i:i+500]
            n += (TrackingLinkSubscriber
                  .update(risk_level=label)
                  .where(
                      (TrackingLinkSubscriber.username.in_(chunk)) &
                      (TrackingLinkSubscriber.is_internal_data == False)
                  )
                  .execute())
    return n
_clf.update_risk_levels = _fast_bulk_update


RISK_LABELS = ['No risk', 'Low', 'High', 'Very High', 'Extreme']
FOUR_RISK = ['Very High', 'Extreme', 'High', 'Low']
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


def _ensure_col():
    db.connect(reuse_if_open=True)
    db.execute_sql("""
        ALTER TABLE tracking_links_subscriber
        ADD COLUMN IF NOT EXISTS is_internal_data BOOLEAN DEFAULT FALSE
    """)


def _select_models(n_models: int) -> list:
    """Pick top-N models (by row count) that contain all 4 risk classes."""
    verify_db.connect(reuse_if_open=True)
    rows = list(Subscription.select(
        Subscription.tracking_model_name,
        Subscription.risk_level,
    ).dicts())
    df = pd.DataFrame(rows)
    df['risk_level'] = df['risk_level'].apply(_normalize_risk)

    grouped = df.groupby('tracking_model_name')
    candidates = []
    for name, g in grouped:
        present = set(g['risk_level'])
        if all(c in present for c in FOUR_RISK):
            candidates.append((name, len(g)))
    candidates.sort(key=lambda x: -x[1])
    chosen = [c[0] for c in candidates[:n_models]]
    return chosen


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


def _remove_model_rows(model_name: str):
    db.connect(reuse_if_open=True)
    (TrackingLinkSubscriber
     .delete()
     .where(TrackingLinkSubscriber.tracking_link_id == model_name)
     .execute())


def _reset_stale_cold_rows(model_name: str):
    db.connect(reuse_if_open=True)
    (TrackingLinkSubscriber
     .update(is_internal_data=True)
     .where(
        (TrackingLinkSubscriber.is_internal_data == False) &
        (TrackingLinkSubscriber.tracking_link_id != model_name)
     )
     .execute())


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
    db.connect(reuse_if_open=True)
    for _, r in batch.iterrows():
        risk_db = RISK_TO_DB.get(r['risk_level'], 'no risk')
        (TrackingLinkSubscriber
         .update(is_internal_data=True, is_processed=True, risk_level=risk_db)
         .where(TrackingLinkSubscriber.id == f"cs_{r['id']}")
         .execute())


def _batch_gt(batch: pd.DataFrame) -> dict:
    gt = {}
    for _, row in batch.iterrows():
        uname = row.get('user_name')
        if not isinstance(uname, str) or not uname.strip():
            continue
        r = row['risk_level']
        prev = gt.get(uname)
        if prev is None or _RISK_ORDER[r] > _RISK_ORDER[prev]:
            gt[uname] = r
    return gt


def run_one_model(model_name: str, n_batches: int, model_dir: str):
    """Production-flow simulation: insert one batch at a time, run predict()
    after each insertion, persist predicted labels back to DB so the next
    round's compute_per_user sees them, never promote to warm.

    Returns:
        y_true_final, y_pred_final  — last-round prediction per user
        round_trace                  — list of {round, n_inserted, acc, f1_macro_4, per_class_f1}
    """
    global _TARGET_COLD_USERS
    df = _load_model_rows(model_name)
    if df.empty:
        return [], [], []

    _ensure_col()
    _reset_stale_cold_rows(model_name)
    _remove_model_rows(model_name)

    batch_size = max(1, len(df) // n_batches)
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (start + batch_size) if i < n_batches - 1 else len(df)
        batches.append(df.iloc[start:end].copy())

    truth_map = _batch_gt(df)  # max-risk per user across all rows

    accumulated_users = set()
    final_pred_map = {}
    round_trace = []

    for round_idx, batch in enumerate(batches, 1):
        _insert_cold_batch(batch, model_name)
        for _, r in batch.iterrows():
            uname = r.get('user_name')
            if isinstance(uname, str) and uname.strip():
                accumulated_users.add(uname)

        # Tell the patched update_risk_levels which users we care about
        # (the cold pool of THIS test model, accumulated across rounds).
        _TARGET_COLD_USERS = accumulated_users

        results = _predict(model_dir=model_dir, output='/tmp/_eval_preds.csv')
        if results is None or len(results) == 0:
            continue

        pred_map = dict(zip(results['user_name'], results['predicted_risk']))

        # Last-prediction-wins per user (final accumulator).
        round_y_true, round_y_pred = [], []
        for uname in accumulated_users:
            if uname in pred_map and uname in truth_map:
                final_pred_map[uname] = pred_map[uname]
                round_y_true.append(truth_map[uname])
                round_y_pred.append(pred_map[uname])

        # Per-round metric snapshot for the trace (accuracy follows).
        if round_y_true:
            round_acc = float(accuracy_score(round_y_true, round_y_pred))
            round_macro_f1_4 = float(f1_score(
                round_y_true, round_y_pred,
                labels=FOUR_RISK, average='macro', zero_division=0))
            round_per_class = {
                lvl: float(f1_score(
                    round_y_true, round_y_pred,
                    labels=[lvl], average='macro', zero_division=0))
                for lvl in RISK_LABELS
            }
        else:
            round_acc, round_macro_f1_4, round_per_class = 0.0, 0.0, {}

        round_trace.append({
            'round':        round_idx,
            'n_inserted':   len(accumulated_users),
            'accuracy':     round_acc,
            'macro_f1_4':   round_macro_f1_4,
            'per_class_f1': round_per_class,
        })

    # Final eval set: every user we ever inserted, with their LAST round's prediction.
    y_true_final, y_pred_final = [], []
    for uname, gt in truth_map.items():
        pred = final_pred_map.get(uname)
        if pred is None:
            continue
        y_true_final.append(gt)
        y_pred_final.append(pred)

    _remove_model_rows(model_name)
    _TARGET_COLD_USERS = set()
    return y_true_final, y_pred_final, round_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-models',  type=int, default=3)
    ap.add_argument('--n-batches', type=int, default=5)
    ap.add_argument('--model-dir', default='models_cpu')
    ap.add_argument('--models',    nargs='*', default=None,
                    help='Override model selection (use these names instead of top-N)')
    ap.add_argument('--macro-target',     type=float, default=0.95)
    ap.add_argument('--per-class-target', type=float, default=0.90)
    args = ap.parse_args()

    t0 = time.time()
    if args.models:
        models = args.models[:args.n_models]
    else:
        models = _select_models(args.n_models)
    print(f"[evaluate] models: {models}", flush=True)

    y_true_all, y_pred_all = [], []
    per_model = {}
    for m in models:
        yt, yp, trace = run_one_model(m, args.n_batches, args.model_dir)
        y_true_all.extend(yt)
        y_pred_all.extend(yp)
        per_model[m] = {
            'n':     len(yt),
            'acc':   float(accuracy_score(yt, yp)) if yt else 0.0,
            'rounds': trace,
        }
        # Compact one-line per-round trace for the eval log.
        if trace:
            tline = ' | '.join(f"r{t['round']}:n={t['n_inserted']},acc={t['accuracy']:.3f},mF1={t['macro_f1_4']:.3f}"
                                for t in trace)
            print(f"[evaluate] {m[:40]:<40} final_n={len(yt):>4}  final_acc={per_model[m]['acc']:.3f}",
                  flush=True)
            print(f"[evaluate]   trace: {tline}", flush=True)
        else:
            print(f"[evaluate] {m[:40]:<40} n={len(yt):>4}  acc={per_model[m]['acc']:.3f}",
                  flush=True)

    if not y_true_all:
        out = {"pass": False, "score": -999.0, "details": {"error": "no predictions"}}
        print(json.dumps(out))
        return

    overall_acc = float(accuracy_score(y_true_all, y_pred_all))
    macro_f1_5  = float(f1_score(y_true_all, y_pred_all,
                                  labels=RISK_LABELS, average='macro', zero_division=0))
    macro_f1_4  = float(f1_score(y_true_all, y_pred_all,
                                  labels=FOUR_RISK, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true_all, y_pred_all,
                                  labels=RISK_LABELS, average='weighted', zero_division=0))

    per_class_f1 = {}
    for lvl in RISK_LABELS:
        per_class_f1[lvl] = float(f1_score(
            y_true_all, y_pred_all, labels=[lvl], average='macro', zero_division=0))

    f1_vh   = per_class_f1['Very High']
    f1_high = per_class_f1['High']
    f1_low  = per_class_f1['Low']

    margins = [
        macro_f1_4 - args.macro_target,
        f1_vh     - args.per_class_target,
        f1_high   - args.per_class_target,
        f1_low    - args.per_class_target,
    ]
    score = min(margins)
    passed = score >= 0.0

    elapsed = time.time() - t0
    details = {
        "macro_f1_4risk":     macro_f1_4,
        "macro_f1_all5":      macro_f1_5,
        "weighted_f1":        weighted_f1,
        "overall_accuracy":   overall_acc,
        "per_class_f1":       per_class_f1,
        "n_predictions":      len(y_true_all),
        "n_models":           len(models),
        "n_batches":          args.n_batches,
        "models":             models,
        "per_model":          per_model,
        "targets": {
            "macro_f1_4risk": args.macro_target,
            "per_class":      args.per_class_target,
        },
        "margins": {
            "macro_f1_4risk": margins[0],
            "f1_very_high":   margins[1],
            "f1_high":        margins[2],
            "f1_low":         margins[3],
        },
        "elapsed_s": round(elapsed, 1),
    }

    print("\n[evaluate] classification_report:")
    print(classification_report(y_true_all, y_pred_all,
                                 labels=RISK_LABELS, zero_division=0))
    print(json.dumps({"pass": passed, "score": round(score, 4), "details": details}))


if __name__ == '__main__':
    main()
