"""
Export all 5 trained LGB models to a single JavaScript module.

Output: models_cpu/classifier.js
  - One function per model: scoreExtreme, scoreHigh, scoreLow, scoreOrdinal, scoreVH
  - A top-level predict(features) function that runs the full cascade
  - Features list and thresholds embedded as constants

Usage from Node.js:
  const { predict, FEATURES } = require('./classifier.js');
  // features = array of numbers (NaN for missing), same order as FEATURES
  const result = predict(features);
  // result = { risk: 'Very High', vh: 0.82, extreme: 0.03, high: 0.01, low: 0.05 }
"""
import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import m2cgen as m2c

MODEL_DIR = 'models_cpu'

print("Loading models...", flush=True)
models, thresholds, feat_info = {}, {}, {}
for name in ['extreme', 'high', 'low', 'ordinal', 'vh']:
    with open(os.path.join(MODEL_DIR, f'{name}_model.pkl'), 'rb') as f:
        models[name] = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'thresholds.json')) as f:
    thresholds = json.load(f)
with open(os.path.join(MODEL_DIR, 'features.json')) as f:
    feat_info = json.load(f)

feats = feat_info['base_features']
# VH model takes base features + 4 cascade probas appended at end
vh_feats = feats + ['cascade_extreme', 'cascade_high', 'cascade_low', 'cascade_ordinal']

print("Exporting models to JS (this may take a moment for large trees)...", flush=True)

def export_model(clf, fn_name):
    """Export LGB model to a JS function returning raw score (log-odds)."""
    js = m2c.export_to_javascript(clf, function_name=fn_name)
    return js

js_extreme = export_model(models['extreme'], 'scoreExtreme')
print("  extreme done", flush=True)
js_high    = export_model(models['high'],    'scoreHigh')
print("  high done", flush=True)
js_low     = export_model(models['low'],     'scoreLow')
print("  low done", flush=True)
js_ordinal = export_model(models['ordinal'], 'scoreOrdinal')
print("  ordinal done", flush=True)
js_vh      = export_model(models['vh'],      'scoreVH')
print("  vh done", flush=True)

# Sigmoid helper and thresholds
t_e   = thresholds.get('extreme', 0.5)
t_vh  = thresholds.get('vh', 0.5)
t_h   = thresholds.get('high', 0.5)
t_l   = thresholds.get('low', 0.5)

# m2cgen for LGB outputs raw leaf value sums (log-odds for binary).
# Apply sigmoid to get probability.
wrapper = f"""
// ── Feature lists ─────────────────────────────────────────────────────────────
const FEATURES = {json.dumps(feats, indent=2)};
// VH model expects base features + 4 cascade probas appended at indices [-4..-1]
// cascade_extreme, cascade_high, cascade_low, cascade_ordinal
const VH_FEATURES = {json.dumps(vh_feats, indent=2)};

// ── Thresholds (calibrated on OOF) ───────────────────────────────────────────
const THRESHOLDS = {{
  extreme: {t_e},
  vh:      {t_vh},
  high:    {t_h},
  low:     {t_l},
}};

// ── Sigmoid ───────────────────────────────────────────────────────────────────
function sigmoid(x) {{
  return 1.0 / (1.0 + Math.exp(-x));
}}

// ── Cascade predict ───────────────────────────────────────────────────────────
/**
 * Classify a single user.
 *
 * @param {{number[]}} baseFeatures  Array of length FEATURES.length.
 *                                   Use NaN for missing values.
 * @returns {{{{risk: string, extreme: number, vh: number, high: number, low: number}}}}
 */
function predict(baseFeatures) {{
  const eProba   = sigmoid(scoreExtreme(baseFeatures));
  const hProba   = sigmoid(scoreHigh(baseFeatures));
  const lProba   = sigmoid(scoreLow(baseFeatures));
  const ordProba = sigmoid(scoreOrdinal(baseFeatures));

  // Augment with cascade probas for VH model
  const vhFeatures = baseFeatures.concat([eProba, hProba, lProba, ordProba]);
  const vhProba    = sigmoid(scoreVH(vhFeatures));

  let risk;
  if (eProba  >= THRESHOLDS.extreme) risk = 'Extreme';
  else if (vhProba >= THRESHOLDS.vh) risk = 'Very High';
  else if (hProba  >= THRESHOLDS.high) risk = 'High';
  else if (lProba  >= THRESHOLDS.low)  risk = 'Low';
  else                                  risk = 'No risk';

  return {{ risk, extreme: eProba, vh: vhProba, high: hProba, low: lProba }};
}}

module.exports = {{ predict, FEATURES, VH_FEATURES, THRESHOLDS }};
"""

out_path = os.path.join(MODEL_DIR, 'classifier.js')
print(f"\nWriting {out_path}...", flush=True)
with open(out_path, 'w') as f:
    f.write(js_extreme + '\n')
    f.write(js_high    + '\n')
    f.write(js_low     + '\n')
    f.write(js_ordinal + '\n')
    f.write(js_vh      + '\n')
    f.write(wrapper)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"Done — {out_path} ({size_mb:.1f} MB)", flush=True)

# Quick sanity check: score first row of training data
print("\nSanity check (first 3 users from results.csv):", flush=True)
import pandas as pd
results = pd.read_csv('results.csv')
print(results.head(3)[['user_name', 'predicted_risk', 'vh_proba', 'extreme_proba']].to_string())
