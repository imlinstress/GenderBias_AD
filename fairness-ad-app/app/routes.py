"""routes.py — FastAPI route handlers for the Fair AD Predictor app."""

import json, os, numpy as np, pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from app.database import (
    insert_prediction, correct_prediction,
    get_all_predictions, get_correction_data, log_event,
    save_model_version, get_fairness_summary)
from app.fairness_audit import FairnessAuditor

router = APIRouter()
auditor = FairnessAuditor(sensitive_feature_name='Sex',
                          privileged_value=0, unprivileged_value=1,
                          favorable_label=1)

# Set by main.py after model is loaded
model_service = None


class PredictRequest(BaseModel):
    features: Dict[str, float]
    sex: int  # 0=male, 1=female


class PredictResponse(BaseModel):
    prediction_id: int
    score: float
    predicted_label: int
    threshold: float
    shap_values: Optional[Dict[str, float]] = None
    calibration_applied: bool = False
    calibration_delta: float = 0.0


class CorrectRequest(BaseModel):
    prediction_id: int
    true_label: int


class RetrainResponse(BaseModel):
    status: str
    version: str
    n_original: int
    n_new: int
    metrics_before: Dict
    metrics_after: Dict


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model_service is None:
        raise HTTPException(503, "Model not loaded")

    # Build feature vector (must match training feature order)
    feats = [req.features.get(f, 0.0) for f in model_service.feature_names]
    X = np.array([feats])

    # Scale
    if hasattr(model_service, 'scaler') and model_service.scaler is not None:
        X_scaled = model_service.scaler.transform(X)
    else:
        X_scaled = X

    # Predict
    sex_arr = np.array([req.sex])
    labels, scores = model_service.predict(X_scaled, sex=sex_arr)

    # SHAP
    shap_dict = None
    if hasattr(model_service, 'wrapper') and model_service.wrapper is not None:
        try:
            bg_path = os.path.join(model_service.model_dir, 'shap_background.npy')
            if os.path.exists(bg_path):
                bg = np.load(bg_path)
                sv, _ = auditor.explain_with_shap(
                    model_service.wrapper, bg, X_scaled,
                    feature_names=model_service.feature_names)
                sv_1d = sv[0] if sv.ndim > 1 else sv
                top_idx = np.argsort(np.abs(sv_1d))[-10:]
                shap_dict = {model_service.feature_names[i]: round(float(sv_1d[i]), 4)
                             for i in top_idx}
        except Exception as e:
            pass  # SHAP failure shouldn't block prediction

    # Store
    pred_id = insert_prediction(
        sex=int(req.sex), score=float(scores[0]),
        predicted_label=int(labels[0]),
        threshold=model_service.threshold,
        features=req.features,
        shap_values=list(shap_dict.values()) if shap_dict else None)

    log_event('prediction', {
        'id': pred_id, 'sex': req.sex, 'score': float(scores[0]),
        'label': int(labels[0])})

    return PredictResponse(
        prediction_id=pred_id,
        score=float(scores[0]),
        predicted_label=int(labels[0]),
        threshold=model_service.threshold,
        shap_values=shap_dict,
        calibration_applied=model_service.use_calibration,
        calibration_delta=model_service.calibration_delta)


@router.post("/correct")
async def correct(req: CorrectRequest):
    correct_prediction(req.prediction_id, req.true_label)
    log_event('correction', {'id': req.prediction_id, 'true_label': req.true_label})
    return {"status": "ok", "prediction_id": req.prediction_id}


@router.post("/retrain")
async def retrain():
    if model_service is None:
        raise HTTPException(503, "Model not loaded")

    X_new, y_new = get_correction_data()
    if len(X_new) == 0:
        raise HTTPException(400, "No corrections to train on")

    metrics_before = _compute_current_metrics()

    result = model_service.retrain(X_new, y_new)

    metrics_after = _compute_current_metrics()

    version = f"v{result['n_total']}"
    save_model_version(version, metrics=metrics_after, n_train=result['n_total'])
    log_event('retrain', {
        'version': version, 'n_new': result['n_new'],
        'metrics_before': metrics_before, 'metrics_after': metrics_after})

    return RetrainResponse(
        status='ok', version=version,
        n_original=result['n_original'], n_new=result['n_new'],
        metrics_before=metrics_before, metrics_after=metrics_after)


@router.get("/dashboard")
async def dashboard():
    data = get_fairness_summary()
    if not data or len(data['sex']) == 0:
        return {"n_predictions": 0, "metrics": {}}

    # Reconstruct y_pred from stored labels
    y_pred = data['labels']
    y_true = np.full_like(y_pred, -1)  # Unknown for uncorrected
    sensitive = data['sex']

    metrics = auditor.compute_fairness_metrics(
        y_true, y_pred, sensitive) if np.any(y_true >= 0) else {}

    # Compute per-sex stats
    hist = {}
    for grp, name in [(0, 'male'), (1, 'female')]:
        mask = sensitive == grp
        n = int(mask.sum())
        hist[name] = {
            'n': n,
            'n_ad_predicted': int(y_pred[mask].sum()) if n > 0 else 0,
            'mean_score': round(float(data['scores'][mask].mean()), 4) if n > 0 else None,
        }

    return {
        'n_predictions': len(y_pred),
        'sex_distribution': hist,
        'metrics': metrics,
        'config': {
            'threshold': model_service.threshold,
            'calibration_delta': model_service.calibration_delta,
            'use_calibration': model_service.use_calibration,
        } if model_service else {},
    }


@router.get("/history")
async def history(limit: int = 100):
    preds = get_all_predictions()
    return preds[:limit]


def _compute_current_metrics():
    """Compute fairness metrics on the current validation set."""
    # Check persistent directory first (retrained data), fall back to built-in
    data_path = os.path.join(model_service.persistent_dir, 'train_data.npz')
    if not os.path.exists(data_path):
        data_path = os.path.join(model_service.model_dir, 'train_data.npz')
    if os.path.exists(data_path):
        d = np.load(data_path)
        y_true = d['y_val_true']
        sex = d['sex_val']
        X_val = d['X_val_raw']

        # Get model predictions
        if hasattr(model_service, 'scaler') and model_service.scaler is not None:
            X_scaled = model_service.scaler.transform(X_val)
        else:
            X_scaled = X_val
        labels, _ = model_service.predict(X_scaled, sex=sex)
        m = auditor.compute_fairness_metrics(y_true, labels, sex)
        return m
    return {}
