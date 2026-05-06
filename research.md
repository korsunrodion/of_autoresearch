# Research: User Risk-Level Classifier — F1 Optimization

## Goal
Improve the risk-level classifier in `classifier_v2_cpu_predict.py` so that, when evaluated by `evaluate.py` against `models_cpu_baseline/`, **macro-F1 over the four risk classes (Very High / Extreme / High / Low) ≥ 0.95** AND **per-class F1 for Very High / High / Low each ≥ 0.90**, all measured on the cumulative final-round predictions across 3 top-row-count test models (each containing all 4 risk classes).

## Success Metric
- **Metric:** `score = min(macro_f1_4risk - 0.95, f1_very_high - 0.90, f1_high - 0.90, f1_low - 0.90)`
- **Target:** `score ≥ 0.0` (all four floors met)
- **Direction:** maximize

The score is the minimum margin to the per-component target; this forces the agent to lift the *weakest* class rather than inflate one already-good number. `pass = score ≥ 0`.

Sub-metrics tracked in `details`:
- `macro_f1_4risk` (target ≥ 0.95)
- `f1_very_high`, `f1_high`, `f1_low` (each target ≥ 0.90)
- `f1_extreme`, `macro_f1_all5`, `weighted_f1`, `overall_accuracy` (informational)

## Constraints
- **Max iterations:** 30
- **Time budget per experiment:** 25 minutes (covers retrain ~15 min + eval)
- **Pause for review every:** never (fully unattended)
- **Evaluator:** `uv run --no-sync python -u evaluate.py --n-models 3 --n-batches 5 --model-dir <model_dir>`
- **Keep policy:** `score_improvement` — only keep a change if `score` strictly improves over the best so far
- **Guard:** retraining must complete before predict is run; no DB schema changes
- **Noise runs:** 1 (eval is deterministic given the same DB state and `models_cpu_baseline/` weights)
- **Min delta:** 0.001 (treat improvements smaller than 0.1% as noise)

## Current Approach
A multi-stage LightGBM cascade (heads: `extreme`, `high`, `low`, `ordinal`, `vh`, `cold_vh`, `cold_high`, `cold_low`, `warm_vh_rescue`) with separate warm and cold prediction paths in `predict()`. The cold path uses `make_cold_X` to override `model_*_frac` features to 0 (an intentional feedback-loop fix). Trained baseline weights live in `models_cpu_baseline/` (~15 min full retrain). 206 features, ~48k rows, 1119 tracking-model groups.

Baseline OOF F1 (training-time, NOT the eval metric):
- Extreme 96.90, High 79.94, Low 89.41, Ordinal 95.03, VH 93.66
- Cold VH 87.90, Cold High 72.16, Cold Low 89.69
- Warm VH rescue 90.48

The High and Cold High heads are the obvious weak links.

## Search Space
- **Allowed changes:** ANY file under `project/of_autoresearch/` — full autonomy granted. Practical priorities:
  1. **Cascade logic + thresholds** in `classifier_v2_cpu_predict.py` and `thresholds.json` — fastest iteration (no retrain)
  2. **Feature engineering** (add/remove columns in `FEATURES`) — requires retrain
  3. **Model hyperparameters** (LGBM configs, n_folds, ensemble structure, `build_oof_lgb` configs) — requires retrain
  4. **`COLD_OVERRIDE` set** (which features get zeroed in cold path) — likely requires retrain to change cold-trained heads
  5. **Cold-path model_*_frac feedback handling** (line 905-922) — risky, see hypothesis below
- **Forbidden changes:** None declared by user. Implicit common sense:
  - Do NOT modify `evaluate.py` itself (gaming the metric is pointless — verify with diff after each iteration)
  - Do NOT modify `.env` or DB connection strings
  - Do NOT delete `models_cpu_baseline/` (this is the iteration-0 reference; copy to a new dir for retrain)
- **Guard rail:** If macro_f1_4risk regresses by more than 5 percentage points vs baseline AND no per-class target improves, revert.

## Hypotheses to Test Early
1. **Threshold tuning only** (no retrain). Sweep `vh_cold`, `high_cold`, `low_cold`, `vh_extreme_cold`, `cold_vh_in_extreme`, `warm_vh_rescue` on the eval set. Expected to be the cheapest first win.
2. **Cold High is the weakest head (F1 72.16).** Likely candidates:
   - Increase Cold High head capacity (more boosting rounds, deeper trees) or use a richer feature subset for cold path
   - Add ordinal-style monotonicity constraints
3. **Predictor-side enrichment hypothesis (the "bug" the user flagged).** The feedback-loop fix at `predict_df` line 905-922 (or the `make_cold_X` `COLD_OVERRIDE` of `model_vh_frac` etc.) blinds cold predictions to each other's class. Removing/relaxing this *might* improve accuracy when many cold rows arrive together — but it can also re-introduce the feedback loop. Test by:
   - (a) replacing the warm-only override with a warm + previously-predicted-cold blended estimate
   - (b) running A/B with `COLD_OVERRIDE` removed for `model_*_frac` only
4. **Ensemble cold heads via stacking** instead of averaging. Cold VH already uses an ensemble — extend to Cold High / Cold Low.
5. **Class-weight tuning** in the LGBM heads — High class is under-recalled (R=81.73) and may benefit from explicit upweighting.
6. **Feature subset for cold path** — train cold heads on a smaller, more stable feature set that doesn't depend on warm-context columns.

## Context & References
- Predictor: `classifier_v2_cpu_predict.py` (1184 lines, all-in-one train + predict)
- Evaluator: `evaluate.py` (this directory; wraps the cold-start eval, emits JSON)
- Train command: `uv run --no-sync python classifier_v2_cpu_predict.py --train --model-dir <dir>`
- Baseline weights: `models_cpu_baseline/` (do not modify; copy to `models_cpu_iter_<N>/` per experiment)
- DB: Railway Postgres via `.env` (DATABASE_URL == DB_VERIFY_URL); each eval inserts and removes ~thousands of cold rows for the test models
- Per-experiment training cost: ~15 min full retrain on 8-CPU machine; threshold-only changes: ~0 retrain
- Per-experiment eval cost: ~85 s compute / ~120 s wallclock measured on baseline (3 models × 1 predict pass)
- Total per-iteration budget: ~17 min if retrain, ~2 min if threshold/cascade-only

---

## History

> **NOTE:** Iterations 0/1/2 below are the *re-run* against the corrected production-flow evaluator. The original eval (insert-all-cold-then-predict-once) under-stated difficulty by skipping the temporal flow; user flagged it and `evaluate.py` was rewritten to insert one batch per round, run predict between rounds, and persist predicted labels to DB so `compute_per_user`'s `norisk_med` and risk_fracs see the model's evolving cold context. Last-round prediction per user is what the score uses.

| # | Change | macro_f1_4risk | f1_VH | f1_High | f1_Low | score | Result | Timestamp |
|---|--------|----------------|-------|---------|--------|-------|--------|-----------|
| 0 | Baseline (`models_cpu_baseline/`) — production-flow eval | 0.8686 | 0.9147 | 0.9524 | **0.6415** | **−0.2585** | fail (Low precision dominates) | 2026-05-05 |
| 1 | cold_low_norisk 0.4896 → 0.65 | 0.8721 | 0.9147 | 0.9524 | 0.6538 | **−0.2462** | **keep** (small gain; persistence dampens) | 2026-05-05 |
| 2 | cold_low_norisk 0.65 → 0.85 | 0.9055 | 0.9147 | 0.9524 | 0.7727 | **−0.1273** | **keep** (Low F1 +0.12, still fails target) | 2026-05-05 |
| 3 | cold_low_norisk 0.85 → 0.92 | 0.9404 | 0.9231 | 0.9524 | 0.8947 | **−0.0096** | **keep** (margins macro −0.01, low −0.005) | 2026-05-05 |
| 4 | cold_low_norisk 0.92 → 0.95 | 0.9404 | 0.9231 | 0.9524 | 0.8947 | **−0.0096** | **revert** (identical — sweep plateaued) | 2026-05-05 |
| 5 | disable all 3 Low rescues in cold cascade | 0.9373 | 0.9323 | 0.9524 | 0.8750 | **−0.0250** | **revert** (Low TPs lost > VH gain) | 2026-05-05 |
| 6 | disable VH→Low rescue only (keep Extreme/High→Low) | **0.9476** | **0.9323** | **0.9524** | **0.9143** | **−0.0024** | **keep** (per-class all pass; macro 0.0024 short) | 2026-05-05 |
| 7 | also disable High→Low rescue (no effect) | 0.9476 | 0.9323 | 0.9524 | 0.9143 | **−0.0024** | **revert** (rescue wasn't firing) | 2026-05-05 |
| 8 | cold_high_extreme 0.328 → 0.20 (no effect) | 0.9476 | 0.9323 | 0.9524 | 0.9143 | **−0.0024** | **revert** (classifier bimodal) | 2026-05-05 |
| 9 | high_cold 0.359 → 0.20 (no effect) | 0.9476 | 0.9323 | 0.9524 | 0.9143 | **−0.0024** | **revert** (h_proba < 0.20 on missed Highs) | 2026-05-05 |
| 10 | vh_cold 0.812 → 0.65 (no effect) | 0.9476 | 0.9323 | 0.9524 | 0.9143 | **−0.0024** | **revert** (missed VHs not in branch 2) | 2026-05-05 |
| 11 | vh_extreme_cold 0.561 → 0.30 (no effect at final) | 0.9476 | 0.9323 | 0.9524 | 0.9143 | **−0.0024** | **revert** (mid-rounds shift, final converges) | 2026-05-05 |
| 12 | retrain 2× cold_high+cold_low spw, OOF thresholds | 0.8976 | 0.9466 | 0.9524 | 0.7111 | **−0.1889** | **revert** (overall acc gained, but Low collapsed at OOF threshold 0.65) | 2026-05-05 |
| 12b | iter12 weights + cold_low_norisk forced 0.92 | 0.9396 | 0.9466 | 0.9524 | 0.8649 | **−0.0351** | **revert** (Low below floor) | 2026-05-05 |
| 13 | iter12 weights + cold_low_norisk = 0.85 | 0.9277 | 0.9466 | 0.9524 | 0.8205 | **−0.0795** | **revert** (worse) | 2026-05-05 |
| **F** | **iter6 final — VH→Low rescue removed + cold_low_norisk = 0.92** | **0.9476** | **0.9323** | **0.9524** | **0.9143** | **−0.0024** | **WINNER** (per-class all pass, macro 0.0024 short) | 2026-05-05 |

## Baseline diagnostics (iteration 0)
**Eval runtime:** 85.5 s of compute, ~120 s wallclock for 3 models × 1 predict pass each.
**Aggregate confusion (n=768 across 3 test models):** No risk 379, Extreme 287, Very High 62, High 22, Low 18.

Per-class precision / recall / F1 from the classification report:
- No risk: 0.99 / 0.99 / **0.991**
- Low: **0.50 / 0.94 / 0.654**  ← biggest gap
- High: 1.00 / 0.91 / 0.952
- Very High: 0.91 / 0.95 / 0.929
- Extreme: 1.00 / 0.94 / 0.968

**Highest-leverage fix:** Low has near-perfect recall but precision only 0.50 — the predictor over-fires Low. Tightening the Low cascade gate (`thresholds.json: low`, `low_cold`, `cold_low_norisk`) and/or improving the cold_low classifier should close most of the macro-F1 gap.
