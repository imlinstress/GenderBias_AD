# Fair AD Predictor

Deployable web app for fair Alzheimer's disease prediction using AdversarialDebiasing (seed=49) with per-sex calibration, SHAP explanations, and a self-updating model via user corrections.

Built as part of the [GenderBias_AD](https://github.com/imlinstress/GenderBias_AD) research project.

## Quick Start

```bash
git clone git@github.com:imlinstress/GenderBias_AD.git
cd GenderBias_AD/fairness-ad-app
docker compose up --build
```

Open http://localhost:8000

## Usage

### Predict
Fill in patient features (Sex, Age, MMSE, FAQ, CDR, APOE4, hippocampal volumes) and click **Run Assessment**. The app returns:

- **Diagnosis**: AD (Alzheimer's Disease) or MCI (Mild Cognitive Impairment)
- **Confidence**: raw AD probability after calibration (0–1)
- **Calibration**: per-sex score adjustment applied (active: +0.010 for females)
- **SHAP explanation**: top features driving the prediction, with bar chart

### Correct
After each prediction, confirm or correct the diagnosis using the buttons. Corrections are stored and used for model retraining.

### Dashboard
Cumulative fairness metrics across all predictions:
- **BA** — Balanced Accuracy (average of TPR and TNR)
- **DI** — Disparate Impact (ratio of AD prediction rates: Female / Male)
- **EOD** — Equal Opportunity Difference (TPR difference: Female − Male)
- **AOD** — Average Odds Difference (average of TPR and FPR differences)
- Per-sex statistics (count, AD predictions, mean score)

Metrics only appear after corrections are submitted (ground truth needed).

### Retrain
Click **Retrain Now** to train a new AD model on the original data plus all accumulated corrections. The retrained checkpoint is saved to the persistent volume and loaded automatically on next startup.

### History
Table of all predictions with score, label, correction status, and timestamp.

## Fairness Config

The deployed model uses calibrated per-sex scoring:

| Metric | Value |
|---|---|
| Female calibration delta | +0.010 |
| Threshold (single, after calibration) | 0.0346 |
| Equal Opportunity Difference (EOD) | **0.0000** |
| Balanced Accuracy (BA) | 0.8575 |
| Disparate Impact (DI) | 1.0012 |
| TPR (both sexes) | 1.0000 |
| TNR | 0.7149 |

Full strategy comparison in `model/thresholds.json`.

## Architecture

```
fairness-ad-app/
├── Dockerfile              # python:3.10-slim, OCI labels
├── docker-compose.yml      # named volume app-data for persistence
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI entry point (uvicorn)
│   ├── routes.py           # /api/* endpoints
│   ├── database.py         # SQLite schema + queries
│   ├── model_service.py    # AD model lifecycle (load/predict/retrain)
│   └── fairness_audit.py   # Fairness metrics + SHAP/LIME
│   ├── static/             # CSS, JS
│   └── templates/          # index.html
└── model/                  # Baked into image (not overridable)
    ├── ad_model.ckpt.*     # TF1 checkpoint
    ├── ad_config.json
    ├── thresholds.json
    ├── scaler.pkl
    ├── feature_names.json
    ├── shap_background.npy
    └── train_data.npz
```

### Persistence

| Data | Location | Persists across restarts? | Cleared by `down -v`? |
|---|---|---|---|
| Predictions + corrections | `app-data` named volume → `/app/data/predictions.db` | Yes | Yes |
| Retrained model checkpoints | `app-data` named volume → `/app/data/ad_model_retrained.ckpt` | Yes | Yes |
| Original model artifacts | Image layer → `/app/model/` | Always (immutable) | No |

On startup, the app checks for a retrained checkpoint in the `app-data` volume. If found, it loads the retrained model; otherwise it falls back to the baked-in model from the image.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/predict` | POST | Submit patient features → AD prediction + SHAP |
| `/api/correct` | POST | Submit correct label for a past prediction |
| `/api/retrain` | POST | Retrain AD model on all accumulated corrections |
| `/api/dashboard` | GET | Cumulative fairness metrics and per-sex stats |
| `/api/history` | GET | Prediction log with correction status |

### Example: predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":78,"MMSCORE_x":23,"FAQTOTAL":12,"CDGLOBAL":4.5,"GENOTYPE_3/4":1,"LEFT_HIPP_NS":2800,"RIGHT_HIPP_NS":2900,"Sex":1},"sex":1}'
```

### Example: correct

```bash
curl -X POST http://localhost:8000/api/correct \
  -H "Content-Type: application/json" \
  -d '{"prediction_id":1,"true_label":1}'
```

## Publish to Docker Hub

```bash
docker build -t imlinstress/fairness-ad-app:2.0.0 .
docker tag imlinstress/fairness-ad-app:2.0.0 imlinstress/fairness-ad-app:latest
docker push imlinstress/fairness-ad-app:2.0.0
docker push imlinstress/fairness-ad-app:latest
```

Users can then run with:

```bash
docker run -d -p 8000:8000 imlinstress/fairness-ad-app:latest
```
