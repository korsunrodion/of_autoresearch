# Final Report — Risk Classifier F1 Optimization (production-flow eval)

> **Result: NEAR-TARGET.** Best score = **−0.0024** at iteration 6 / 30-budget. All four per-class F1 floors cleared (≥ 0.90); macro-F1 over the four risk classes lands **0.0024 below the strict 0.95 macro target**. Overall accuracy = **98.18%** (well above the original literal "95% total accuracy" goal).

> **Earlier "TARGET MET at iter 2" report is invalid.** That was on the wrong evaluator (insert-all-cold-then-predict-once). User flagged the missing temporal flow; `evaluate.py` was rewritten to simulate production (one batch per round, predict between rounds, persist predicted labels to DB), and the loop was restarted from a fresh baseline. Numbers in this report are all on the corrected evaluator.

## Headline numbers (corrected eval, 3 test models × 5 batches)

| Metric              | Baseline | iter6 (final) | Δ        | Target | Margin   |
|---------------------|----------|---------------|----------|--------|----------|
| **score**           | −0.2585  | **−0.0024**   | +0.2561  | ≥ 0    | −0.0024  |
| macro_f1_4risk      | 0.8686   | **0.9476**    | +0.0789  | ≥ 0.95 | −0.0024  |
| f1_Very High        | 0.9147   | **0.9323**    | +0.0177  | ≥ 0.90 | +0.0323  |
| f1_High             | 0.9524   | 0.9524        |  0.0000  | ≥ 0.90 | +0.0524  |
| f1_Low              | 0.6415   | **0.9143**    | +0.2728  | ≥ 0.90 | +0.0143  |
| f1_Extreme          | 0.9658   | 0.9912        | +0.0254  | —      | —        |
| overall_accuracy    | 0.9609   | **0.9818**    | +0.0209  | —      | —        |

Per-test-model accuracy at iter6:
- Amanda 🎀 GG swaps: 0.956 → **0.997**
- @allesGGswap:        0.992 → **0.992**
- @asya_marketer:      0.936 → **0.953**

Per-round trace (Amanda): 0.741 → 0.879 → 0.937 → 0.918 → **0.997** — accuracy follows context as cold rows accumulate, exactly the production behavior the evaluator is designed to simulate.

## What changed (the entire winning diff)

1. **Threshold:** `models_cpu_baseline/thresholds.json: cold_low_norisk` 0.4896 → **0.92**
2. **Cascade code:** `classifier_v2_cpu_predict.py` lines 1055-1062 — the **VH→Low rescue** is removed. When `cold_vh_proba ≥ t_cold_vh_e` triggers the VH branch, the prediction is now Very High unconditionally; the prior `cold_low_proba ≥ t_cold_low → Low` rescue inside that branch was costing VH true-positives without recovering Low TPs. The Extreme→Low and High→Low rescues are **kept** (iter5 showed they catch real Low users).

That's the entire diff. **No retraining was needed.** Final weights: `models_cpu_final/`.

## Iteration trace

| #  | Change                                              | macro_f1_4 | f1_Low | score    | Decision |
|----|-----------------------------------------------------|------------|--------|----------|----------|
| 0  | baseline                                            | 0.8686     | 0.6415 | −0.2585  | start    |
| 1  | cold_low_norisk 0.4896 → 0.65                       | 0.8721     | 0.6538 | −0.2462  | keep     |
| 2  | cold_low_norisk 0.65  → 0.85                        | 0.9055     | 0.7727 | −0.1273  | keep     |
| 3  | cold_low_norisk 0.85  → 0.92                        | 0.9404     | 0.8947 | −0.0096  | keep     |
| 4  | cold_low_norisk 0.92  → 0.95                        | 0.9404     | 0.8947 | −0.0096  | revert (no change — sweep saturated) |
| 5  | disable all 3 Low rescues in cold cascade           | 0.9373     | 0.8750 | −0.0250  | revert (lost Low TPs > VH gain) |
| **6** | **disable VH→Low rescue only**                  | **0.9476** | **0.9143** | **−0.0024** | **KEEP — winner** |
| 7  | also disable High→Low rescue                        | 0.9476     | 0.9143 | −0.0024  | revert (rescue wasn't firing) |
| 8  | cold_high_extreme 0.328 → 0.20                      | 0.9476     | 0.9143 | −0.0024  | revert (classifier bimodal) |
| 9  | high_cold 0.359 → 0.20                              | 0.9476     | 0.9143 | −0.0024  | revert (h_proba<0.20 on misses) |
| 10 | vh_cold 0.812 → 0.65                                | 0.9476     | 0.9143 | −0.0024  | revert (no effect) |
| 11 | vh_extreme_cold 0.561 → 0.30                        | 0.9476     | 0.9143 | −0.0024  | revert (final converges) |
| 12 | retrain with 2× cold_high+cold_low spw              | 0.8976     | 0.7111 | −0.1889  | revert (cold_low calibration shifted, OOF threshold 0.65 too low) |
| 12b| iter12 weights + cold_low_norisk forced 0.92        | 0.9396     | 0.8649 | −0.0351  | revert (Low below floor) |
| 13 | iter12 weights + cold_low_norisk 0.85               | 0.9277     | 0.8205 | −0.0795  | revert (worse) |

See `progress.png`.

## Why we didn't strictly pass macro = 0.95

The remaining 0.0024 gap is a hard plateau on this evaluator. After iter6, every threshold tweak (4, 7-11) returned **identical aggregate F1** because the residual errors cluster in saturated probability regions:

- **2 missed Highs** have `h_proba < 0.20` AND `cold_high_proba < 0.20` (lowering either threshold to 0.20 didn't reach them). The classifiers just don't see them as High.
- **3 missed Very Highs** are predicted as Extreme; lowering `vh_cold` (0.812 → 0.65) and `vh_extreme_cold` (0.561 → 0.30) didn't budge them either — the cold_vh classifier rates them below 0.30 too.
- **2 Low FPs at threshold 0.92** have `cold_low_proba > 0.95`, so any further threshold raise (tested up to 0.95) doesn't reach them.

These are classifier-level failures, not threshold-tuning failures. Iter12 attempted to break the plateau by retraining with 2× scale_pos_weight on the cold heads. That genuinely lifted overall accuracy (98.18% → 98.70%) and recovered @allesGGswap (0.992 → 1.00), but the cold_low classifier's *calibration* shifted such that no threshold reproduced the Low precision/recall balance iter6 had. Net regression on the score, so reverted.

## Why this differs from the earlier "TARGET MET at iter 2" claim

The earlier report was on a different evaluator that inserted all 5 batches of cold rows simultaneously and ran predict() once. That gave the *final-state* prediction with maximum context but skipped the temporal flow entirely. Predicted labels never persisted between rounds, so:

- `compute_per_user`'s `norisk_med` and risk_fracs saw only freshly-inserted "no risk" labels
- The cold cascade got a clean signal on every user
- Cold_low FPs were trivially trimmed by raising `cold_low_norisk` 0.4896 → 0.85 — which gave score = +0.0068 ✓

Once the eval was rewritten to persist predicted labels between rounds (production-flow), the EARLIER round's predicted-Low FPs end up in the DB and contaminate later rounds' feature engineering, which is precisely why the eval is harder. The threshold that worked on the simultaneous-insert eval (0.85) doesn't work on the production-flow eval (Low F1 only 0.77 there) — we needed 0.92 plus the cascade fix to clear all per-class floors.

## Where to push further

To close the last 0.0024 macro gap (or build a robust buffer above it):

1. **Retrain `cold_high` with feature-engineered "High-likeness" signals** that capture what makes the 2 missed Highs look-not-Extreme — e.g., specific time-of-day or chargeback patterns. The current cold_high is bimodal at ~0 and ~1 with little middle ground.
2. **Calibrate cold_low predictions** (Platt or isotonic) so that threshold tuning on the production-flow eval has a smoother surface. Iter12 hinted that wider distribution helps overall accuracy; the right balance probably exists between 1× and 2× spw.
3. **Cold-only ordinal head**: train an ordinal multiclass head specifically on cold examples instead of relying on warm-trained binary heads with COLD_OVERRIDE. The current architecture forces every cold prediction through binary heads that were never trained jointly.
4. **Per-test-model debug**: @asya_marketer's r1-r2 accuracy stays under 0.30 for the first 2 batches across every iteration. Targeted fixes for that early-cold regime (e.g., a "no warm context yet" early-detection branch) would lift the trace without touching the late-round behavior.

To resume the loop on this same `research.md` later, re-invoke `/autoresearch` — it will pick up at iter6 and probe directions 1-4 above with fresh budget.

## Files

| Path                                   | Purpose                                              |
|----------------------------------------|------------------------------------------------------|
| `models_cpu_baseline/`                 | iter-0 reference weights (do not touch)              |
| `models_cpu_iter3/`                    | iter-3 weights — same as final, kept for traceability|
| **`models_cpu_final/`**                | **deployable copy of iter-3 weights w/ cold_low_norisk = 0.92** |
| `classifier_v2_cpu_predict.py`         | predictor with iter-6 cascade fix (VH→Low rescue removed) |
| `classifier_v2_cpu_predict.py.iter3_backup` | backup of original cascade (all 3 rescues intact)  |
| `classifier_v2_cpu_predict.py.iter6_backup` | backup of iter-6 cascade state                     |
| `evaluate.py`                          | production-flow evaluator (5 rounds, persistent predicted labels) |
| `research.md`                          | living research document with full History          |
| `research_log.md`                      | per-iteration audit trail                           |
| `autoresearch-results.tsv`             | machine-readable run log                            |
| `progress.png`                         | convergence plot (kept = filled, revert = hollow)   |

## Reproduce

```bash
# Score the final weights
uv run --no-sync python -u evaluate.py --n-models 3 --n-batches 5 \
  --model-dir models_cpu_final
# Expect: {"pass": false, "score": -0.0024, ...}
# (score < 0 because macro_f1_4risk = 0.9476 vs strict 0.95 floor;
#  every per-class F1 is above the 0.90 floor)
```
