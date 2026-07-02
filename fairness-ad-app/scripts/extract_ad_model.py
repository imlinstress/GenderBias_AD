"""
extract_ad_model.py — Phase 2: Extract trained AD model artifacts.
Saves: checkpoint, config, scaler, feature names, SHAP background, training data
All go to ../model/ for the Docker app.
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

print("=" * 60)
print("  EXTRACT AD MODEL ARTIFACTS")
print("=" * 60)

# ── 1. Load data ──
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=METADATA_COLS + DXMPTR_COLS, errors='ignore')
dataset = BinaryLabelDataset(
    df=df, label_names=['DIAGNOSIS'], protected_attribute_names=['Sex'],
    favorable_label=1, unfavorable_label=0)
dataset_train, dataset_val = dataset.split([0.66], seed=42)
feature_names = list(dataset_train.feature_names)
print(f"  Features: {len(feature_names)}, Train: {dataset_train.features.shape[0]}, "
      f"Val: {dataset_val.features.shape[0]}")

# ── 2. SMOTE + Scaler ──
print("\n[2/6] Applying SMOTE + StandardScaler...")
X_tr_raw = dataset_train.features
y_tr_raw = dataset_train.labels.ravel()
smote = SMOTE(random_state=2)
X_tr_imb, y_tr_imb = smote.fit_resample(X_tr_raw, y_tr_raw)
scaler = StandardScaler()
scaler.fit(X_tr_imb)
print(f"  After SMOTE: {np.bincount(y_tr_imb.astype(int))}")

# ── 3. Train AD ──
print("\n[3/6] Training AdversarialDebiasing (seed=49)...")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()
ad_model = AdversarialDebiasing(
    unprivileged_groups=UNPRIVILEGED_GROUPS,
    privileged_groups=PRIVILEGED_GROUPS,
    scope_name='deb_ad', debias=True, sess=sess, seed=49)
ad_model.fit(dataset_train)
print("  Training complete.")

# ── 4. Save TF checkpoint + config ──
print("\n[4/6] Saving TF checkpoint and config...")
saver = tf.train.Saver(var_list=tf.trainable_variables())
ckpt_path = os.path.join(OUT_DIR, 'ad_model.ckpt')
saver.save(sess, ckpt_path)
print(f"  Checkpoint: {ckpt_path}")

config = {
    'scope_name': ad_model.scope_name,
    'seed': ad_model.seed,
    'unprivileged_groups': ad_model.unprivileged_groups,
    'privileged_groups': ad_model.privileged_groups,
    'protected_attribute_name': ad_model.protected_attribute_name,
    'adversary_loss_weight': float(ad_model.adversary_loss_weight),
    'num_epochs': ad_model.num_epochs,
    'batch_size': ad_model.batch_size,
    'classifier_num_hidden_units': ad_model.classifier_num_hidden_units,
    'debias': ad_model.debias,
    'features_dim': ad_model.features_dim,
}
with open(os.path.join(OUT_DIR, 'ad_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"  Config: {OUT_DIR}/ad_config.json")

sess.close()
tf.reset_default_graph()

# ── 5. Save scaler + feature names ──
print("\n[5/6] Saving scaler, feature names, train data...")
import joblib
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))
print(f"  Scaler: {OUT_DIR}/scaler.pkl")

with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f)
print(f"  Feature names: {OUT_DIR}/feature_names.json")

# ── 6. Save training data for retraining ──
np.savez(os.path.join(OUT_DIR, 'train_data.npz'),
         X_train_raw=X_tr_raw,
         y_train_raw=y_tr_raw,
         X_train_smote=X_tr_imb,
         y_train_smote=y_tr_imb,
         X_val_raw=dataset_val.features,
         y_val_true=dataset_val.labels.ravel(),
         sex_val=dataset_val.protected_attributes.ravel())
print(f"  Train data: {OUT_DIR}/train_data.npz")

# SHAP background (50 random training samples in scaled space)
rng = np.random.RandomState(42)
bg_indices = rng.choice(X_tr_imb.shape[0], min(50, X_tr_imb.shape[0]), replace=False)
bg = scaler.transform(X_tr_imb)[bg_indices]
np.save(os.path.join(OUT_DIR, 'shap_background.npy'), bg)
print(f"  SHAP background: {OUT_DIR}/shap_background.npy ({bg.shape[0]} samples)")

print("\n" + "=" * 60)
print("  ALL ARTIFACTS SAVED")
print("=" * 60)
print(f"\n  {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    {f:35s} {size/1024:.1f} KB")
print()
