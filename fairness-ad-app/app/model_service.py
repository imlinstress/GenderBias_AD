"""
model_service.py — AD model lifecycle: load, predict, retrain.

Loads the AdversarialDebiasing model from saved artifacts (TF checkpoint + config),
wraps it for sklearn-like inference, and exposes predict/predict_proba.

Supports score calibration (per-sex delta) and per-sex thresholding.
Retrained checkpoints persist to /app/data/ (named volume) across restarts.
"""

import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

BUILTIN_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')


def load_ad_model(model_dir=None):
    """Load AD model from saved artifacts. Returns ModelService instance."""
    return ModelService()


class ModelService:
    """AD model inference service with calibration and thresholding."""

    def __init__(self, model_dir=None):
        self.builtin_dir = BUILTIN_MODEL_DIR
        self.persistent_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        # Configs always from built-in (baked into image)
        self.model_dir = model_dir or BUILTIN_MODEL_DIR

        # Check for retrained checkpoint in persistent volume
        retrained_ckpt = os.path.join(self.persistent_dir, 'ad_model_retrained.ckpt.index')
        self._using_retrained = os.path.exists(retrained_ckpt)

        self._load_config()
        self._load_artifacts()
        self._rebuild_model()

    def _load_config(self):
        with open(os.path.join(self.model_dir, 'ad_config.json')) as f:
            self.config = json.load(f)
        with open(os.path.join(self.model_dir, 'thresholds.json')) as f:
            self.thresholds = json.load(f)
        self.threshold = self.thresholds['deployed']['threshold']
        self.calibration_delta = self.thresholds['deployed'].get('calibration_delta', 0.0)
        self.use_calibration = self.thresholds['deployed'].get('use_calibration', False)

    def _load_artifacts(self):
        import joblib
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        with open(os.path.join(self.model_dir, 'feature_names.json')) as f:
            self.feature_names = json.load(f)

    def _rebuild_model(self):
        import tensorflow.compat.v1 as tf
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        from aif360.datasets import BinaryLabelDataset
        from app.fairness_audit import Aif360ModelWrapper

        tf.disable_v2_behavior()
        self.sess = tf.Session()

        # Need a dummy dataset to build the TF graph (any dataset with right feature dim)
        dummy_df = pd.DataFrame(
            np.zeros((1, self.config['features_dim'])),
            columns=self.feature_names[:self.config['features_dim']])
        dummy_df['Sex'] = 0
        dummy_df['DIAGNOSIS'] = 0
        dummy_dataset = BinaryLabelDataset(
            df=dummy_df, label_names=['DIAGNOSIS'],
            protected_attribute_names=['Sex'],
            favorable_label=1, unfavorable_label=0)

        self.model = AdversarialDebiasing(
            unprivileged_groups=self.config['unprivileged_groups'],
            privileged_groups=self.config['privileged_groups'],
            scope_name=self.config['scope_name'],
            debias=self.config['debias'],
            sess=self.sess,
            seed=self.config['seed'])
        self.model.fit(dummy_dataset)

        # Restore trained weights — check for retrained checkpoint first
        restore_saver = tf.train.Saver(
            var_list=tf.trainable_variables(scope=self.config['scope_name']))
        if self._using_retrained:
            ckpt_path = os.path.join(self.persistent_dir, 'ad_model_retrained.ckpt')
        else:
            ckpt_path = os.path.join(self.model_dir, 'ad_model.ckpt')
        restore_saver.restore(self.sess, ckpt_path)

        self.wrapper = Aif360ModelWrapper(
            self.model, self.scaler, self.feature_names,
            protected_attribute_names=['Sex'],
            _tf_session=self.sess)

    def predict_proba(self, X, sex=None):
        """Get calibrated probabilities.

        Parameters
        ----------
        X : array-like, shape (n, n_features) — scaled features
        sex : array-like, optional, shape (n,) — 0=male, 1=female

        Returns
        -------
        proba : ndarray, shape (n, 2) — [P(MCI), P(AD)]
        """
        X = np.asarray(X)
        proba = self.wrapper.predict_proba(X)

        # Apply calibration: add delta to female AD scores
        if self.use_calibration and sex is not None:
            sex = np.asarray(sex).ravel()
            female_mask = sex == 1
            proba[female_mask, 1] = np.clip(
                proba[female_mask, 1] + self.calibration_delta, 0, 1)
            proba[female_mask, 0] = 1 - proba[female_mask, 1]

        return proba

    def predict(self, X, sex=None):
        """Predict labels with per-sex (or single) threshold.

        Parameters
        ----------
        X : array-like, shape (n, n_features)
        sex : array-like, optional

        Returns
        -------
        labels : ndarray, shape (n,)
        scores : ndarray, shape (n,)
        """
        X = np.asarray(X)
        proba = self.predict_proba(X, sex=sex)
        scores = proba[:, 1]

        if sex is not None and self.thresholds['deployed'].get('threshold_male') is not None:
            sex = np.asarray(sex).ravel()
            labels = np.zeros(len(scores))
            labels[(sex == 0) & (scores > self.thresholds['deployed']['threshold_male'])] = 1
            labels[(sex == 1) & (scores > self.thresholds['deployed']['threshold_female'])] = 1
        else:
            labels = (scores > self.threshold).astype(int)

        return labels, scores

    def retrain(self, X_new, y_new):
        """Retrain the AD model on original + new data.

        Note: AD retraining requires a full TF graph rebuild + checkpoint save.
        """
        import tensorflow.compat.v1 as tf
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        from aif360.datasets import BinaryLabelDataset
        from imblearn.over_sampling import SMOTE

        # Load original training data
        data = np.load(os.path.join(self.model_dir, 'train_data.npz'))
        X_orig = data['X_train_raw']
        y_orig = data['y_train_raw']

        # Combine with new data
        X_combined = np.vstack([X_orig, np.asarray(X_new)])
        y_combined = np.hstack([y_orig, np.asarray(y_new)])

        # Rebuild BinaryLabelDataset
        df_combined = pd.DataFrame(X_combined, columns=self.feature_names[:X_combined.shape[1]])
        df_combined['Sex'] = 0  # Placeholder; real sex comes from input features
        df_combined['DIAGNOSIS'] = y_combined
        dataset_combined = BinaryLabelDataset(
            df=df_combined, label_names=['DIAGNOSIS'],
            protected_attribute_names=['Sex'],
            favorable_label=1, unfavorable_label=0)

        # SMOTE
        smote = SMOTE(random_state=2)
        X_sm, y_sm = smote.fit_resample(X_combined, y_combined)
        df_sm = pd.DataFrame(X_sm, columns=self.feature_names[:X_sm.shape[1]])
        df_sm['Sex'] = 0
        df_sm['DIAGNOSIS'] = y_sm
        dataset_sm = BinaryLabelDataset(
            df=df_sm, label_names=['DIAGNOSIS'],
            protected_attribute_names=['Sex'],
            favorable_label=1, unfavorable_label=0)

        # Retrain
        tf.disable_v2_behavior()
        new_sess = tf.Session()
        new_model = AdversarialDebiasing(
            unprivileged_groups=self.config['unprivileged_groups'],
            privileged_groups=self.config['privileged_groups'],
            scope_name='deb_ad', debias=True, sess=new_sess,
            seed=self.config['seed'] + 1)
        new_model.fit(dataset_sm)

        # Save new checkpoint to persistent directory
        new_ckpt = os.path.join(self.persistent_dir, 'ad_model_retrained.ckpt')
        new_saver.save(new_sess, new_ckpt)

        # Update config in persistent directory
        self.config['seed'] += 1
        with open(os.path.join(self.persistent_dir, 'ad_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # Swap in-memory
        if hasattr(self, 'sess') and self.sess is not None:
            old_sess = self.sess
        self.sess = new_sess
        self.model = new_model

        restore_saver = tf.train.Saver(var_list=tf.trainable_variables(scope='deb_ad'))
        restore_saver.restore(self.sess, new_ckpt)

        self.wrapper = Aif360ModelWrapper(
            self.model, self.scaler, self.feature_names,
            protected_attribute_names=['Sex'],
            _tf_session=self.sess)

        if 'old_sess' in dir():
            old_sess.close()
            tf.reset_default_graph()

        # Save updated train_data.npz to persistent directory
        np.savez(os.path.join(self.persistent_dir, 'train_data.npz'),
                 X_train_raw=X_combined, y_train_raw=y_combined,
                 X_train_smote=X_sm, y_train_smote=y_sm,
                 X_val_raw=data['X_val_raw'],
                 y_val_true=data['y_val_true'],
                 sex_val=data['sex_val'])

        self._using_retrained = True

        return {'n_original': len(X_orig), 'n_new': len(X_new), 'n_total': len(X_combined)}

    def close(self):
        if hasattr(self, 'wrapper') and self.wrapper is not None:
            self.wrapper.close()
