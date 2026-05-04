"""Quick diagnostic: inspect feature values and raw model outputs for verify DB."""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

# Import predict-side deps
from classifier_v2_cpu_predict import compute_per_user, FEATURES
from base_v2 import fetch_df, clean
from db_v2 import db as _db

MODEL_DIR = 'models_cpu'

print("Loading verify DB...", flush=True)
df = fetch_df()
df = clean(df)
try:
    _db.close()
except Exception:
    pass
print(f"  {len(df)} rows, {df['tracking_model_name'].nunique()} models", flush=True)
print("risk_level distribution:")
print(df['risk_level'].value_counts().to_string())

print("\nComputing features...", flush=True)
per_user = compute_per_user(df)

with open(os.path.join(MODEL_DIR, 'features.json')) as f:
    feat_info = json.load(f)
feats = feat_info['base_features']

X_base = per_user[feats].values
print(f"\nX_base shape: {X_base.shape}")
print(f"NaN fraction per feature (top 10 most NaN):")
nan_fracs = np.isnan(X_base).mean(axis=0)
top_nan = np.argsort(nan_fracs)[::-1][:10]
for i in top_nan:
    print(f"  {feats[i]}: {nan_fracs[i]:.1%} NaN")

print(f"\nAll-NaN rows: {np.all(np.isnan(X_base), axis=1).sum()}")
print(f"Rows with >50% NaN: {(np.isnan(X_base).mean(axis=1) > 0.5).sum()}")

# Load models and check raw outputs
print("\nLoading cascade models...", flush=True)
models = {}
for name in ['extreme', 'high', 'low', 'ordinal', 'vh']:
    with open(os.path.join(MODEL_DIR, f'{name}_model.pkl'), 'rb') as f:
        models[name] = pickle.load(f)

e_proba = models['extreme'].predict_proba(X_base)[:, 1]
print(f"\nextreme proba — min={e_proba.min():.4f} max={e_proba.max():.4f} mean={e_proba.mean():.4f}")
print(f"  nonzero: {(e_proba > 0).sum()}/{len(e_proba)}")

h_proba = models['high'].predict_proba(X_base)[:, 1]
print(f"high proba    — min={h_proba.min():.4f} max={h_proba.max():.4f} mean={h_proba.mean():.4f}")

l_proba = models['low'].predict_proba(X_base)[:, 1]
print(f"low proba     — min={l_proba.min():.4f} max={l_proba.max():.4f} mean={l_proba.mean():.4f}")

ord_proba = models['ordinal'].predict_proba(X_base)[:, 1]
print(f"ordinal proba — min={ord_proba.min():.4f} max={ord_proba.max():.4f} mean={ord_proba.mean():.4f}")

X_aug = np.column_stack([X_base, e_proba, h_proba, l_proba, ord_proba])
vh_proba = models['vh'].predict_proba(X_aug)[:, 1]
print(f"vh proba      — min={vh_proba.min():.4f} max={vh_proba.max():.4f} mean={vh_proba.mean():.4f}")
print(f"  nonzero: {(vh_proba > 0).sum()}/{len(vh_proba)}")

# Show a few sample rows of actual feature values
print("\nSample feature values (first 3 users):")
sample_feats = ['log_min_id_diff_24h', 'partner_count_24h', 'cic_10k_24h',
                'model_vh_frac', 'model_extreme_frac', 'model_any_risk_frac',
                'model_norisk_rate', 'log_model_norisk_median_id_diff',
                'username_is_u_digits', 'log_user_id_num', 'n_subs_at_sub_time_5m']
for fname in sample_feats:
    if fname in feats:
        idx = feats.index(fname)
        vals = X_base[:3, idx]
        print(f"  {fname}: {vals}")

with open(os.path.join(MODEL_DIR, 'thresholds.json')) as f:
    thresholds = json.load(f)
print(f"\nThresholds: {thresholds}")
