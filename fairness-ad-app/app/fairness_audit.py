"""
fairness_audit.py — Single-module fairness audit library.

Provides FairnessAuditor class with:
  - Fairness metrics (BA, DI, AOD, SPD, EOD, Theil)  — numpy-only, aif360 optional
  - Threshold tuning (single + per-sex)
  - Improved/corrected case analysis
  - LIME + SHAP explainability (with Aif360ModelWrapper bridge)
  - Per-group t-tests with Bonferroni correction
  - Fairness dashboard + corrected case plots
"""

import numpy as np
import pandas as pd
import warnings
from collections import OrderedDict

# ── Optional deps ─────────────────────────────────────────────────
try:
    from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
    HAS_AIF360 = True
except ImportError:
    HAS_AIF360 = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class Aif360ModelWrapper:
    """Wraps an aif360 inprocessing model with sklearn-like predict_proba.

    aif360 models (AdversarialDebiasing, PrejudiceRemover) expect
    BinaryLabelDataset inputs.  LIME/SHAP call models via predict_proba(numpy array).
    This wrapper bridges the two: it stores the scaler used to transform
    LIME's perturbed samples, inverse-transforms them back to the original
    (unscaled) feature space, converts to BinaryLabelDataset, and forwards
    to the aif360 model's predict().
    """

    def __init__(self, model, scaler, feature_names, protected_attribute_names,
                 label_names=None, favorable_label=1, unfavorable_label=0,
                 _tf_session=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.protected_attribute_names = list(protected_attribute_names)
        self.label_names = label_names if label_names is not None else ['label']
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self._tf_sess = _tf_session

    def predict_proba(self, X):
        from aif360.datasets import BinaryLabelDataset
        X_unscaled = self.scaler.inverse_transform(X)
        n = X.shape[0]
        df = pd.DataFrame(X_unscaled, columns=self.feature_names)
        for label in self.label_names:
            df[label] = self.unfavorable_label
        ds = BinaryLabelDataset(
            df=df, label_names=self.label_names,
            protected_attribute_names=self.protected_attribute_names,
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label)
        try:
            pred = self.model.predict(ds)
            scores = np.asarray(pred.scores).flatten()
        except Exception:
            scores = np.zeros(n)
            for i in range(n):
                row = df.iloc[i:i+1].copy()
                ds_i = BinaryLabelDataset(
                    df=row, label_names=self.label_names,
                    protected_attribute_names=self.protected_attribute_names,
                    favorable_label=self.favorable_label,
                    unfavorable_label=self.unfavorable_label)
                pred_i = self.model.predict(ds_i)
                scores[i] = np.asarray(pred_i.scores).flatten()[0]
        proba = np.zeros((n, 2))
        proba[:, 1] = scores
        proba[:, 0] = 1 - scores
        return proba

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def close(self):
        if hasattr(self, '_tf_sess') and self._tf_sess is not None:
            self._tf_sess.close()
            import tensorflow.compat.v1 as tf
            tf.reset_default_graph()
            self._tf_sess = None


class FairnessAuditor:
    """Main fairness audit orchestrator.

    Parameters
    ----------
    sensitive_feature_name : str
        Name of the protected attribute (e.g., 'Sex').
    privileged_value : int or str
        Value of the privileged group (e.g., 0 for male).
    unprivileged_value : int or str
        Value of the unprivileged group (e.g., 1 for female).
    favorable_label : int
        The favorable outcome label (e.g., 1 for AD diagnosis).
    """

    def __init__(self, sensitive_feature_name='Sex',
                 privileged_value=0, unprivileged_value=1,
                 favorable_label=1):
        self.sensitive_feature_name = sensitive_feature_name
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.favorable_label = favorable_label

    # ── Metrics ───────────────────────────────────────────────────

    def compute_fairness_metrics(self, y_true, y_pred, sensitive_attr,
                                 use_aif360=False, dataset_true=None,
                                 dataset_pred=None):
        """Compute fairness metrics: BA, DI, AOD, SPD, EOD, Theil.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        sensitive_attr : array-like
            Sensitive attribute values per sample.
        use_aif360 : bool, optional
            If True and aif360 available, use ClassificationMetric.
            Otherwise uses numpy-only computation.
        dataset_true, dataset_pred : aif360 BinaryLabelDataset, optional
            Required if use_aif360=True.

        Returns
        -------
        OrderedDict of metric_name -> value.
        """
        if use_aif360 and HAS_AIF360 and dataset_true is not None and dataset_pred is not None:
            return self._metrics_via_aif360(dataset_true, dataset_pred)

        return self._metrics_numpy(y_true, y_pred, sensitive_attr)

    def _metrics_numpy(self, y_true, y_pred, sensitive_attr):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        sens = np.asarray(sensitive_attr).ravel()

        mask_priv = sens == self.privileged_value
        mask_unpriv = sens == self.unprivileged_value

        def _rates(y, pred, mask):
            tp = np.sum((y[mask] == self.favorable_label) & (pred[mask] == self.favorable_label))
            tn = np.sum((y[mask] != self.favorable_label) & (pred[mask] != self.favorable_label))
            fp = np.sum((y[mask] != self.favorable_label) & (pred[mask] == self.favorable_label))
            fn = np.sum((y[mask] == self.favorable_label) & (pred[mask] != self.favorable_label))
            tpr = tp / (tp + fn) if (tp + fn) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0
            return tpr, tnr

        tpr_priv, tnr_priv = _rates(y_true, y_pred, mask_priv)
        tpr_unpriv, tnr_unpriv = _rates(y_true, y_pred, mask_unpriv)

        ba = 0.5 * (tpr_priv + tnr_priv + tpr_unpriv + tnr_unpriv) / 2
        p_pos_priv = np.mean(y_pred[mask_priv])
        p_pos_unpriv = np.mean(y_pred[mask_unpriv])

        metrics = OrderedDict()
        metrics['BA'] = round(ba, 4)
        metrics['DI'] = round(p_pos_unpriv / p_pos_priv, 4) if p_pos_priv > 0 else 0.0
        metrics['AOD'] = round(0.5 * ((tpr_unpriv - tpr_priv) + (tnr_unpriv - tnr_priv)), 4)
        metrics['SPD'] = round(p_pos_unpriv - p_pos_priv, 4)
        metrics['EOD'] = round(tpr_unpriv - tpr_priv, 4)
        metrics['TPR_priv'] = round(tpr_priv, 4)
        metrics['TPR_unpriv'] = round(tpr_unpriv, 4)
        metrics['TNR_priv'] = round(tnr_priv, 4)
        metrics['TNR_unpriv'] = round(tnr_unpriv, 4)
        return metrics

    def _metrics_via_aif360(self, dataset_true, dataset_pred):
        cm = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=[{self.sensitive_feature_name: self.unprivileged_value}],
                                  privileged_groups=[{self.sensitive_feature_name: self.privileged_value}])
        metrics = OrderedDict()
        metrics['BA'] = round(0.5 * (cm.true_positive_rate() + cm.true_negative_rate()), 4)
        metrics['DI'] = round(cm.disparate_impact(), 4)
        metrics['AOD'] = round(cm.average_odds_difference(), 4)
        metrics['SPD'] = round(cm.statistical_parity_difference(), 4)
        metrics['EOD'] = round(cm.equal_opportunity_difference(), 4)
        metrics['Theil'] = round(cm.theil_index(), 4)
        return metrics

    # ── Threshold tuning ──────────────────────────────────────────

    def tune_threshold(self, y_true, scores, threshold_grid=None):
        """Find the threshold that maximizes balanced accuracy.

        Parameters
        ----------
        y_true : array-like of {0, 1}
        scores : array-like of float, predicted probabilities for class 1.
        threshold_grid : array-like, optional
            Thresholds to try. Default: np.linspace(0.01, 0.99, 200).

        Returns
        -------
        best_threshold : float
        best_ba : float
        """
        if threshold_grid is None:
            threshold_grid = np.linspace(0.01, 0.99, 200)
        y_true = np.asarray(y_true).ravel()
        scores = np.asarray(scores).ravel()

        best_t, best_ba = 0.5, 0.0
        for t in threshold_grid:
            yp = (scores > t).astype(int)
            tp = np.sum((y_true == 1) & (yp == 1))
            tn = np.sum((y_true == 0) & (yp == 0))
            fn = np.sum((y_true == 1) & (yp == 0))
            fp = np.sum((y_true == 0) & (yp == 1))
            tpr = tp / (tp + fn) if (tp + fn) else 0
            tnr = tn / (tn + fp) if (tn + fp) else 0
            ba = 0.5 * (tpr + tnr)
            if ba > best_ba:
                best_ba = ba
                best_t = t
        return float(best_t), float(best_ba)

    def tune_threshold_per_sex(self, y_true, scores, sensitive_attr,
                               threshold_grid=None, di_range=(0.8, 1.2)):
        """Find per-sex thresholds maximizing BA subject to DI constraint.

        Parameters
        ----------
        y_true, scores, sensitive_attr : arrays
        threshold_grid : array-like, optional
        di_range : tuple of (min, max) for DI constraint.

        Returns
        -------
        config : dict with 'threshold_male', 'threshold_female', 'ba', 'di', 'eod'
        """
        if threshold_grid is None:
            threshold_grid = np.linspace(0.01, 0.99, 100)
        y_true = np.asarray(y_true).ravel()
        scores = np.asarray(scores).ravel()
        sens = np.asarray(sensitive_attr).ravel()
        mask_m = sens == self.privileged_value
        mask_f = sens == self.unprivileged_value

        best = {'threshold_male': 0.5, 'threshold_female': 0.5,
                'ba': 0, 'di': 0, 'eod': 1}

        for t_m in threshold_grid:
            for t_f in threshold_grid:
                yp = scores.copy()
                yp[mask_m] = (scores[mask_m] > t_m).astype(int)
                yp[mask_f] = (scores[mask_f] > t_f).astype(int)
                m = self._metrics_numpy(y_true, yp, sens)
                if di_range[0] <= m['DI'] <= di_range[1] and m['BA'] > best['ba']:
                    best = {'threshold_male': float(t_m), 'threshold_female': float(t_f),
                            'ba': m['BA'], 'di': m['DI'], 'eod': m['EOD']}
        return best

    # ── Corrected cases ───────────────────────────────────────────

    def find_improved_cases(self, y_true, y_pred_baseline, y_pred_debiased,
                            sensitive_attr=None):
        """Return indices where debiased model corrected baseline errors.

        Parameters
        ----------
        y_true : array-like
        y_pred_baseline : array-like
        y_pred_debiased : array-like
        sensitive_attr : array-like, optional
            If provided, returns sex breakdown.

        Returns
        -------
        dict with keys 'indices', 'count',
               and optionally 'count_privileged', 'count_unprivileged'.
        """
        y_true = np.asarray(y_true).ravel()
        base = np.asarray(y_pred_baseline).ravel()
        deb = np.asarray(y_pred_debiased).ravel()
        idx = np.where((base != y_true) & (deb == y_true))[0]
        result = {'indices': idx, 'count': len(idx)}
        if sensitive_attr is not None:
            sens = np.asarray(sensitive_attr).ravel()
            result['count_privileged'] = int(np.sum(sens[idx] == self.privileged_value))
            result['count_unprivileged'] = int(np.sum(sens[idx] == self.unprivileged_value))
        return result

    def compute_overlap(self, improved_a, improved_b):
        """Return overlap and unique counts between two sets of improved indices."""
        set_a = set(improved_a['indices'])
        set_b = set(improved_b['indices'])
        return {
            'overlap': len(set_a & set_b),
            'only_a': len(set_a - set_b),
            'only_b': len(set_b - set_a),
        }

    # ── SHAP ──────────────────────────────────────────────────────

    def explain_with_shap(self, model, X_background, X_instances,
                          feature_names=None, model_type='classifier'):
        """Compute SHAP values for given instances.

        Parameters
        ----------
        model : callable with .predict_proba(X) or .predict(X)
        X_background : array, background for KernelExplainer
        X_instances : array, instances to explain
        feature_names : list of str, optional
        model_type : str, ignored (for compatibility)

        Returns
        -------
        shap_values : ndarray of shape (n_instances, n_features)
        explanation : SHAP Explanation object
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required: pip install shap")
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        sv = explainer.shap_values(X_instances)
        if isinstance(sv, list):
            sv = sv[1]  # class 1 (favorable)
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        if feature_names is not None:
            pass  # shap values retain shape
        return sv, explainer

    def shap_to_dataframe(self, shap_values, feature_names):
        """Convert SHAP values to DataFrame."""
        return pd.DataFrame(shap_values, columns=feature_names)

    # ── LIME ──────────────────────────────────────────────────────

    def explain_with_lime(self, model, X_train, X_instances, feature_names,
                          class_names=None, discretize_continuous=True):
        """Compute LIME explanations for given instances.

        Parameters
        ----------
        model : callable with .predict_proba(X)
        X_train : array, training data for LIME explainer
        X_instances : array, instances to explain
        feature_names : list of str
        class_names : list of str, optional
        discretize_continuous : bool

        Returns
        -------
        lime_df : DataFrame of LIME weights (instances × features)
        """
        if not HAS_LIME:
            raise ImportError("LIME is required: pip install lime")
        if class_names is None:
            class_names = [0, 1]
        X_train = np.asarray(X_train)
        X_instances = np.asarray(X_instances)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names, class_names=class_names,
            discretize_continuous=discretize_continuous)
        all_values = []
        for i, x in enumerate(X_instances):
            exp = explainer.explain_instance(x, model.predict_proba,
                                             num_features=len(feature_names))
            d = dict(exp.as_list())
            all_values.append([d.get(f, 0) for f in feature_names])
        return pd.DataFrame(all_values, columns=feature_names)

    # ── T-tests ───────────────────────────────────────────────────

    def group_stratified_ttest(self, explanation_df, sensitive_attr,
                               instance_indices=None, alpha=0.05):
        """Per-group one-sample t-tests with Bonferroni correction.

        Tests whether mean SHAP/LIME value for each feature differs from 0
        within each demographic group.

        Parameters
        ----------
        explanation_df : DataFrame, instances × features of SHAP/LIME values
        sensitive_attr : array-like, per-instance group labels
        instance_indices : array-like, optional, indices to subset
        alpha : float, significance level before Bonferroni

        Returns
        -------
        results : dict with 'privileged' and 'unprivileged' entries,
                  each containing {feature: {'t': float, 'p': float, 'significant': bool}}
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required: pip install scipy")
        df = explanation_df.copy()
        sens = np.asarray(sensitive_attr).ravel()
        if instance_indices is not None:
            sens = sens[list(instance_indices)]
            df = df.iloc[list(instance_indices)]

        results = {}
        for grp_val, grp_name in [(self.privileged_value, 'privileged'),
                                   (self.unprivileged_value, 'unprivileged')]:
            mask = sens == grp_val
            grp_df = df[mask].copy()
            n_tests = len(grp_df.columns)
            alpha_bonf = alpha / n_tests if n_tests > 0 else alpha
            grp_results = {}
            for col in grp_df.columns:
                t_stat, p_val = stats.ttest_1samp(grp_df[col], 0)
                grp_results[col] = {
                    't': round(t_stat, 4),
                    'p': round(p_val, 4),
                    'significant': p_val < alpha_bonf,
                    'mean': round(grp_df[col].mean(), 4),
                }
            results[grp_name] = grp_results
        return results

    # ── Plotting ──────────────────────────────────────────────────

    def plot_fairness_dashboard(self, metrics_history, save_path=None):
        """Plot fairness metrics over time across multiple audits.

        Parameters
        ----------
        metrics_history : list of dict, each from compute_fairness_metrics()
        save_path : str, optional
        """
        if not HAS_MPL:
            raise ImportError("matplotlib is required: pip install matplotlib")
        df = pd.DataFrame(metrics_history)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        ax = axes[0, 0]
        if 'BA' in df.columns:
            ax.plot(df['BA'], marker='o', color='#0072B2')
            ax.set_ylabel('Balanced Accuracy')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        ax = axes[0, 1]
        if 'DI' in df.columns:
            ax.plot(df['DI'], marker='s', color='#E69F00')
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
                       label='Parity')
            ax.set_ylabel('Disparate Impact')

        ax = axes[1, 0]
        if 'EOD' in df.columns:
            ax.plot(df['EOD'], marker='^', color='#D55E00')
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
            ax.set_ylabel('Equal Opportunity Diff')

        ax = axes[1, 1]
        if 'AOD' in df.columns:
            ax.plot(df['AOD'], marker='v', color='#009E73')
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
            ax.set_ylabel('Average Odds Diff')

        for ax in axes.flat:
            ax.set_xlabel('Audit #')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_corrected_cases(self, improved_dicts, labels, save_path=None):
        """Bar chart of corrected cases per method with sex breakdown.

        Parameters
        ----------
        improved_dicts : list of dict from find_improved_cases()
        labels : list of str
        save_path : str, optional
        """
        if not HAS_MPL:
            raise ImportError("matplotlib is required")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(labels))
        w = 0.3
        for i, (d, label) in enumerate(zip(improved_dicts, labels)):
            priv = d.get('count_privileged', 0)
            unpriv = d.get('count_unprivileged', 0)
            ax.bar(i - w/2, priv, w, label=f'{label} (M)', color='#56B4E9')
            ax.bar(i + w/2, unpriv, w, label=f'{label} (F)', color='#D55E00')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Corrected cases')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
