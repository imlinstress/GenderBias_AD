"""
triage_eod.py — Phase 1: EOD fix strategies for AD (seed=49).
Tests threshold + calibration strategies, picks best trade-off.
Outputs: optimal config → ../model/thresholds.json
"""

import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_4mod_MCIvsAD.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(OUT_DIR, exist_ok=True)

METADATA_COLS = ['PTID','CDRSB','PTHAND','PTCOGBEG','DXDSEV','DXMDUE','DXMCI',
                 'DXAD','DXAPP','DXDDUE','PTADDX','ICV']
DXMPTR_COLS = ['DXMPTR1','DXMPTR2','DXMPTR3','DXMPTR4','DXMPTR5','DXMPTR6']
UNPRIVILEGED_GROUPS = [{'Sex': 1}]
PRIVILEGED_GROUPS = [{'Sex': 0}]


def compute_metrics(y_true, y_pred, sex):
    metrics = {}
    for grp, label in [(0, 'M'), (1, 'F')]:
        m = sex == grp
        tp = np.sum((y_true[m] == 1) & (y_pred[m] == 1))
        tn = np.sum((y_true[m] == 0) & (y_pred[m] == 0))
        fp = np.sum((y_true[m] == 0) & (y_pred[m] == 1))
        fn = np.sum((y_true[m] == 1) & (y_pred[m] == 0))
        metrics[f'tpr_{label}'] = tp / (tp + fn) if (tp + fn) else 0
        metrics[f'tnr_{label}'] = tn / (tn + fp) if (tn + fp) else 0
        metrics[f'ba_{label}'] = 0.5 * (metrics[f'tpr_{label}'] + metrics[f'tnr_{label}'])
    metrics['ba'] = 0.5 * (metrics['ba_M'] + metrics['ba_F'])
    metrics['eod'] = metrics['tpr_F'] - metrics['tpr_M']
    p_pos_m = np.mean(y_pred[sex == 0])
    p_pos_f = np.mean(y_pred[sex == 1])
    metrics['di'] = p_pos_f / p_pos_m if p_pos_m > 0 else 9.99
    metrics['aod'] = metrics['eod'] + (metrics['tnr_F'] - metrics['tnr_M'])
    metrics['aod'] /= 2
    return metrics


# ── Load & prepare ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=METADATA_COLS + DXMPTR_COLS, errors='ignore')

dataset = BinaryLabelDataset(
    df=df, label_names=['DIAGNOSIS'], protected_attribute_names=['Sex'],
    favorable_label=1, unfavorable_label=0)
dataset_train, dataset_val = dataset.split([0.66], seed=42)
feature_names = list(dataset_train.feature_names)

X_tr_raw = dataset_train.features
y_tr_raw = dataset_train.labels.ravel()
X_val_raw = dataset_val.features
y_val_true = dataset_val.labels.ravel()
sex_val = dataset_val.protected_attributes.ravel()

smote = SMOTE(random_state=2)
X_tr_imb, y_tr_imb = smote.fit_resample(X_tr_raw, y_tr_raw)

scaler = StandardScaler()
scaler.fit(X_tr_imb)

# ── Train AD ───────────────────────────────────────────────────────
print("Training AdversarialDebiasing (seed=49)...")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()
ad_model = AdversarialDebiasing(
    unprivileged_groups=UNPRIVILEGED_GROUPS,
    privileged_groups=PRIVILEGED_GROUPS,
    scope_name='deb_ad', debias=True, sess=sess, seed=49)
ad_model.fit(dataset_train)
ds_pred_val = ad_model.predict(dataset_val)
scores_val = np.asarray(ds_pred_val.scores).flatten()
sess.close()
tf.reset_default_graph()

# ── Score statistics by sex ────────────────────────────────────────
print("\nScore distribution by sex:")
for grp, label in [(0, 'Male'), (1, 'Female')]:
    m = sex_val == grp
    s = scores_val[m]
    print(f"  {label}: mean={s.mean():.4f}, median={np.median(s):.4f}, "
          f"min={s.min():.4f}, max={s.max():.4f}")

# ── Calibration: align female scores ──────────────────────────────
calib_delta = np.median(scores_val[sex_val == 0]) - np.median(scores_val[sex_val == 1])
print(f"\nCalibration delta (median M - median F): {calib_delta:.4f}")
scores_calibrated = scores_val.copy()
scores_calibrated[sex_val == 1] += calib_delta

# ── Strategies ────────────────────────────────────────────────────
thresholds = np.linspace(0.01, 0.99, 200)
deltas = np.linspace(0, 0.5, 51)


def eval_strategy(scores):
    """Return best per objective for given scores."""
    best_ba = {'t': 0.5, 'ba': 0, 'eod': 0, 'di': 0, 'tnr': 0, 'tpr': 0}
    best_eod = {'t': 0.5, 'ba': 0, 'eod': 99, 'di': 0, 'tnr': 0, 'tpr': 0}
    best_comb = {'t': 0.5, 'ba': 0, 'eod': 0, 'di': 0, 'tnr': 0, 'tpr': 0, 'score': 99}

    for t in thresholds:
        yp = (scores > t).astype(int)
        m = compute_metrics(y_val_true, yp, sex_val)
        tnr = 0.5 * (m['tnr_M'] + m['tnr_F'])

        if m['ba'] > best_ba['ba']:
            best_ba = {**{k: round(float(v), 4) for k, v in m.items() if not k.startswith('ba_')},
                       't': round(float(t), 4), 'ba': round(m['ba'], 4), 'tnr': round(tnr, 4)}

        if abs(m['eod']) < abs(best_eod['eod']):
            best_eod = {**{k: round(float(v), 4) for k, v in m.items() if not k.startswith('ba_')},
                        't': round(float(t), 4), 'ba': round(m['ba'], 4), 'tnr': round(tnr, 4)}

        combined = abs(m['eod']) + abs(1 - m['di']) + max(0, 0.84 - m['ba']) * 3
        if combined < best_comb['score']:
            best_comb = {**{k: round(float(v), 4) for k, v in m.items() if not k.startswith('ba_')},
                         't': round(float(t), 4), 'ba': round(m['ba'], 4),
                         'tnr': round(tnr, 4), 'score': round(combined, 4)}

    return {'ba_max': best_ba, 'eod_min': best_eod, 'tradeoff': best_comb}


def show_one(name, cfg):
    print(f"  {name:31s} | {str(cfg.get('t', 'N/A')):>8s} | {cfg['ba']:.4f} | {cfg['di']:.4f} | "
          f"{cfg['eod']:.4f} | {cfg.get('tpr_M',0):.3f} | {cfg.get('tpr_F',0):.3f} | "
          f"{cfg.get('tnr_M',0):.3f} | {cfg.get('tnr_F',0):.3f}")


# 1. Raw scores
raw = eval_strategy(scores_val)

# 2. Median-calibrated scores (existing)
scores_calib = scores_val.copy()
scores_calib[sex_val == 1] += calib_delta
cal = eval_strategy(scores_calib)

# 3. Delta sweep — find the best calibration delta + threshold tradeoff
print("\n[3/3] Sweeping calibration deltas 0–0.5...")
best_delta_cfg = None
best_delta_score = 999
best_delta_val = 0
for d in deltas:
    s = scores_val.copy()
    s[sex_val == 1] += d
    r = eval_strategy(s)
    cfg = r['tradeoff']
    score = abs(cfg['eod']) + abs(1 - cfg['di']) + max(0, 0.84 - cfg['ba']) * 3
    if score < best_delta_score:
        best_delta_score = score
        best_delta_val = d
        best_delta_cfg = cfg

# Re-evaluate with best delta
scores_best_delta = scores_val.copy()
scores_best_delta[sex_val == 1] += best_delta_val
delta_opt = eval_strategy(scores_best_delta)

# 4. Per-sex thresholds with DI constraint [0.8, 1.2]
print("  Per-sex threshold sweep (DI ∈ [0.8, 1.2])...")
best_ps = {'t_m': 0.5, 't_f': 0.5, 'ba': 0, 'eod': 1, 'di': 0, 'tnr': 0, 'score': 999}
for t_m in thresholds:
    for t_f in thresholds:
        yp = scores_val.copy()
        yp[sex_val == 0] = (scores_val[sex_val == 0] > t_m).astype(int)
        yp[sex_val == 1] = (scores_val[sex_val == 1] > t_f).astype(int)
        m = compute_metrics(y_val_true, yp, sex_val)
        if not (0.8 <= m['di'] <= 1.2):
            continue
        tnr = 0.5 * (m['tnr_M'] + m['tnr_F'])
        score = abs(m['eod']) + abs(1 - m['di']) + max(0, 0.84 - m['ba']) * 3
        if score < best_ps['score']:
            best_ps = {'t_m': round(float(t_m), 4), 't_f': round(float(t_f), 4),
                       'ba': round(m['ba'], 4), 'di': round(m['di'], 4),
                       'eod': round(m['eod'], 4), 'tnr': round(tnr, 4),
                       'score': round(score, 4),
                       'tpr_M': round(m['tpr_M'], 4), 'tpr_F': round(m['tpr_F'], 4),
                       'tnr_M': round(m['tnr_M'], 4), 'tnr_F': round(m['tnr_F'], 4)}

# ── Pick winner ────────────────────────────────────────────────────
candidates = [
    ('raw: BA-max', raw['ba_max']),
    ('raw: trade-off', raw['tradeoff']),
    ('median-cal: BA-max', cal['ba_max']),
    ('median-cal: trade-off', cal['tradeoff']),
    (f'delta={best_delta_val:.3f}: BA-max', delta_opt['ba_max']),
    (f'delta={best_delta_val:.3f}: trade-off', delta_opt['tradeoff']),
]
if best_ps['ba'] > 0:
    candidates.append((f'per-sex (DI∈[0.8,1.2])', best_ps))

def pick_winner(candidates):
    best_name, best_cfg, best_score = None, None, 999
    for name, cfg in candidates:
        score = abs(cfg['eod']) + abs(1 - cfg['di']) + max(0, 0.84 - cfg['ba']) * 5
        if score < best_score:
            best_score = score
            best_name = name
            best_cfg = {**cfg, 'mode': name}
    return best_name, best_cfg

winner_name, winner = pick_winner(candidates)

# ── Display ────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print(f"  {'Strategy':31s} {'Thresh':>8s} {'BA':8s} {'DI':8s} {'EOD':8s} {'TPR_M':6s} {'TPR_F':6s} {'TNR_M':6s} {'TNR_F':6s}")
print("=" * 95)
print("  RAW SCORES:")
show_one("  BA-max", raw['ba_max'])
show_one("  trade-off", raw['tradeoff'])
show_one("  |EOD|-min", raw['eod_min'])
print("  MEDIAN-CALIBRATED (Δ={:.4f}):".format(calib_delta))
show_one("  BA-max", cal['ba_max'])
show_one("  trade-off", cal['tradeoff'])
print(f"  DELTA SWEEP (best Δ={best_delta_val:.3f}):")
show_one("  BA-max", delta_opt['ba_max'])
show_one("  trade-off", delta_opt['tradeoff'])
if best_ps['ba'] > 0:
    print("  PER-SEX (DI constrained):")
    print(f"    M={best_ps['t_m']:.4f} F={best_ps['t_f']:.4f} | BA={best_ps['ba']:.4f} "
          f"DI={best_ps['di']:.4f} EOD={best_ps['eod']:.4f} TNR={best_ps['tnr']:.4f}")

print(f"\n  🏆 Winner: {winner_name}")
print(f"     Config: ", end="")
if winner_name.startswith('per-sex'):
    print(f"M={winner['t_m']:.4f}, F={winner['t_f']:.4f}")
elif 'delta' in winner_name or 'cal' in winner_name:
    print(f"threshold={winner.get('t', '?')}, calibration required")
else:
    print(f"threshold={winner.get('t', '?')}")
print(f"     BA={winner['ba']:.4f}  DI={winner['di']:.4f}  EOD={winner['eod']:.4f}")
print(f"     TPR: M={winner['tpr_M']:.3f}  F={winner['tpr_F']:.3f}")
print(f"     TNR: M={winner['tnr_M']:.3f}  F={winner['tnr_F']:.3f}")

# ── Save ───────────────────────────────────────────────────────────
output = {
    'calibration_delta': round(float(calib_delta), 4),
    'winner': winner_name,
    'deployed': {
        'use_calibration': 'cal' in winner_name or 'delta' in winner_name,
        'calibration_delta': round(float(best_delta_val), 4) if 'delta' in winner_name else (
            round(float(calib_delta), 4) if 'cal' in winner_name else 0),
        'threshold': winner.get('t'),
        'threshold_male': winner.get('t_m'),
        'threshold_female': winner.get('t_f'),
        'metrics': {
            'ba': winner['ba'], 'di': winner['di'], 'eod': winner['eod'],
            'tpr_m': winner.get('tpr_M'), 'tpr_f': winner.get('tpr_F'),
            'tnr_m': winner.get('tnr_M'), 'tnr_f': winner.get('tnr_F'),
        }
    },
    'strategies': {
        'raw_ba_max': raw['ba_max'],
        'raw_tradeoff': raw['tradeoff'],
        'raw_eod_min': raw['eod_min'],
        'calibrated_ba_max': cal['ba_max'],
        'calibrated_tradeoff': cal['tradeoff'],
        f'delta_{best_delta_val:.3f}_ba_max': delta_opt['ba_max'],
        f'delta_{best_delta_val:.3f}_tradeoff': delta_opt['tradeoff'],
        'per_sex_di_constrained': best_ps if best_ps['ba'] > 0 else None,
    },
}
with open(os.path.join(OUT_DIR, 'thresholds.json'), 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved to {OUT_DIR}/thresholds.json")
