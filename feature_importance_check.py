"""
Quick feature importance extraction from a single LGB-nl127 VH model fit.
Groups features into intra-model (context-relative) vs individual.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from base import fetch_df, clean
from db import db as _db

# Reuse the feature/compute machinery from v2_cpu
exec(open('classifier_v2_cpu.py').read().split('if __name__')[0])

print("Loading data...", flush=True)
df = fetch_df(selected=False)
df = clean(df)
try:
    _db.close()
except Exception:
    pass

print("Computing features...", flush=True)
per_user = compute_per_user(df)
feats = [f for f in FEATURES if f in per_user.columns]

# Build VH dataset with a simple extreme OOF cascade (1 seed, 5-fold — just for importances)
sub_e = per_user[per_user['risk_level'].isin(['Extreme', 'No risk'])].copy()
sub_e['y'] = (sub_e['risk_level'] == 'Extreme').astype(int)
Xe, ye = sub_e[feats].values, sub_e['y'].values
spw_e = (ye == 0).sum() / ye.sum()
from sklearn.model_selection import StratifiedKFold
clf_e = LGBMClassifier(num_leaves=127, learning_rate=0.03, n_estimators=600,
                       scale_pos_weight=spw_e, subsample=0.80, colsample_bytree=0.85,
                       min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
p_e = np.zeros(len(ye))
for ti, vi in cv.split(Xe, ye):
    clf_e.fit(Xe[ti], ye[ti])
    p_e[vi] = clf_e.predict_proba(Xe[vi])[:, 1]
extreme_lookup = dict(zip(sub_e['user_name'].values, p_e))

# Simple ordinal cascade
sub_ord = per_user.copy()
sub_ord['y'] = sub_ord['risk_level'].isin(['Very High', 'Extreme']).astype(int)
X_ord, y_ord = sub_ord[feats].values, sub_ord['y'].values
spw_ord = (y_ord == 0).sum() / y_ord.sum()
clf_ord = LGBMClassifier(num_leaves=63, learning_rate=0.05, n_estimators=400,
                         scale_pos_weight=spw_ord, subsample=0.80, colsample_bytree=0.85,
                         verbose=-1, n_jobs=-1, random_state=42)
p_ord = np.zeros(len(y_ord))
for ti, vi in cv.split(X_ord, y_ord):
    clf_ord.fit(X_ord[ti], y_ord[ti])
    p_ord[vi] = clf_ord.predict_proba(X_ord[vi])[:, 1]
ordinal_lookup = dict(zip(sub_ord['user_name'].values, p_ord))

# VH
sub_vh = per_user[per_user['risk_level'].isin(['Very High', 'No risk'])].copy().reset_index(drop=True)
y_vh = (sub_vh['risk_level'] == 'Very High').astype(int).values
X_vh = sub_vh[feats].values
spw_vh = (y_vh == 0).sum() / y_vh.sum()
casc_e   = np.array([extreme_lookup.get(u, 0.0) for u in sub_vh['user_name']])
casc_ord = np.array([ordinal_lookup.get(u, 0.0) for u in sub_vh['user_name']])
# Use 2 cascade features (extreme + ordinal) — enough for importance signal
X_aug = np.column_stack([X_vh, casc_e, casc_ord])
feat_names = feats + ['cascade_extreme', 'cascade_ordinal']

clf_vh = LGBMClassifier(num_leaves=127, learning_rate=0.02, n_estimators=800,
                        scale_pos_weight=spw_vh, subsample=0.75, colsample_bytree=0.80,
                        min_child_samples=20, subsample_freq=1, verbose=-1, n_jobs=-1, random_state=42)
clf_vh.fit(X_aug, y_vh)

imp = clf_vh.feature_importances_   # gain-based by default in LGB
total = imp.sum()
fi = pd.DataFrame({'feature': feat_names, 'importance': imp, 'pct': imp / total * 100})
fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)

# ── Classify features ──────────────────────────────────────────────────────────
INTRA_MODEL_PREFIXES = (
    'log_min_id_diff_', 'partner_count_', 'cic_', 'log_id_span_', 'log_id_std_',
    'log_min_consec_gap_', 'model_subs_', 'model_subs_rate_', 'rel_min_id_diff_',
    'log_model_norisk_median_id_diff', 'log_global_id_span', 'log_min_id_diff_global',
    'log_min_time_diff', 'model_norisk_rate', 'frac_empty_short_windows',
    'model_vh_frac', 'model_extreme_frac', 'model_any_risk_frac',
    'id_percentile_in_model', 'time_percentile_in_model',
    'log_time_since_model_first_sub', 'n_subs_at_sub_time_5m',
    'cascade_extreme', 'cascade_ordinal',
)
INDIVIDUAL_PREFIXES = (
    'username_', 'log_user_id_num', 'user_model_count', 'user_max_risk_elsewhere',
    'total_chargebacks', 'hour_sin', 'hour_cos', 'day_of_week',
)

def classify(feat):
    for p in INTRA_MODEL_PREFIXES:
        if feat.startswith(p) or feat == p:
            return 'intra_model'
    for p in INDIVIDUAL_PREFIXES:
        if feat.startswith(p) or feat == p:
            return 'individual'
    return 'other'

fi['group'] = fi['feature'].apply(classify)

print("\n=== GROUP SUMMARY ===")
gs = fi.groupby('group')['pct'].sum().sort_values(ascending=False)
for g, pct in gs.items():
    print(f"  {g:<15}: {pct:.1f}%")

print("\n=== TOP 30 FEATURES ===")
for _, row in fi.head(30).iterrows():
    print(f"  {row['group']:<15} {row['pct']:5.2f}%  {row['feature']}")

print("\n=== INTRA-MODEL SUBGROUPS ===")
intra = fi[fi['group'] == 'intra_model'].copy()
def subgroup(feat):
    if feat.startswith('cascade_'):         return 'cascade'
    if feat.startswith('model_vh') or feat.startswith('model_extreme') or feat.startswith('model_any'): return 'model_risk_fracs'
    if feat in ('id_percentile_in_model', 'time_percentile_in_model',
                'log_time_since_model_first_sub', 'n_subs_at_sub_time_5m'): return 'percentile/timing'
    if feat.startswith('model_subs'):       return 'model_density'
    if feat.startswith('log_model_norisk') or feat == 'rel_min_id_diff_24h' or feat == 'model_norisk_rate': return 'norisk_median'
    if feat.startswith('cic_frac'):         return 'cic_frac'
    if feat.startswith('cic_'):             return 'cic_count'
    if feat.startswith('partner_count'):    return 'partner_count'
    if feat.startswith('log_min_id_diff'):  return 'log_min_id_diff'
    if feat.startswith('log_id_span') or feat.startswith('log_id_std') or feat.startswith('log_min_consec'): return 'id_spread'
    if feat.startswith('log_global') or feat == 'log_min_id_diff_global': return 'global_id'
    if feat in ('log_min_time_diff', 'frac_empty_short_windows'): return 'time_gap'
    return 'other'
intra = intra.copy()
intra['subgroup'] = intra['feature'].apply(subgroup)
sg = intra.groupby('subgroup')['pct'].sum().sort_values(ascending=False)
for sg_name, pct in sg.items():
    print(f"  {sg_name:<22}: {pct:.1f}%")
