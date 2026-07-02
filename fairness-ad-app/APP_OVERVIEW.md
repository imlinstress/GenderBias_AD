# Fair AD Predictor — App Overview

## What It Does

A deployable web app for fair Alzheimer's disease screening. It takes patient
features (cognitive scores, demographics, MRI volumes), runs them through an
AdversarialDebiasing model trained on ADNI data, and returns an AD/MCI
prediction with a SHAP explanation. Female scores are calibrated by +0.010 to
close the equal‑opportunity gap. Users can submit corrections when the model is
wrong, and retrain the model on the accumulated data — making it a
self‑improving screening tool.

## Key Decisions

| Decision | Rationale |
|---|---|
| Per‑sex score calibration vs per‑sex thresholds | Calibration preserves DI ≈ 1.0; per‑sex thresholds broke demographic parity |
| EOD=0 as primary target | Equal opportunity is clinically critical for a screening tool — both sexes should have the same detection rate |
| TPR=1.0 at ~28.5% FPR | Missing an AD case is far costlier than a false positive (which gets caught at follow‑up) |
| FastAPI + vanilla HTML/JS vs Streamlit | Lighter, no full‑script rerun, mobile‑friendly, REST API accessible from any client |
| Named volume + baked‑in model | Model lives in the image (immutable, no missing artifacts); only DB + retrained checkpoints persist in volume |
| SHAP for local explanations | Standardized, model‑agnostic, gives per‑feature contribution per prediction |
| TF1 compat for AdversarialDebiasing | aif360's AD implementation uses TF1; compat.v1 bridge works on TF 2.x |

## Key Features

- **Per‑sex calibration**: +0.010 added to every female AD score. Closes EOD
  from −0.165 → 0.000 at BA=0.857.
- **Self‑updating**: Submit corrections via the UI; click "Retrain Now" to train
  a new AD model on original + corrected data. Retrained checkpoint persists in
  the named volume.
- **SHAP explanations**: Top 10 features driving each prediction, displayed as
  horizontal bars (orange = pushes toward AD, blue = pushes toward MCI).
- **Fairness dashboard**: Per‑sex counts, mean scores, and cumulative BA/DI/EOD/AOD
  metrics (visible after corrections provide ground truth).
- **Tooltip hints**: `?` icon next to every label explains the field or metric
  in plain language.
- **REST API + web UI**: Predict, correct, retrain, dashboard, and history
  available both via browser and curl.
- **Single‑command Docker deployment**: `docker compose up --build` — model
  artifacts are baked into the image, no external downloads.
- **Self‑contained**: No cloud dependencies, no API keys, no external services.
  Runs entirely on the local machine.

## Shortcomings and Context

### 1. Web form exposes only 8 of 128 features (API for the rest)

> **Discussion.** The 8 fields in the web form (Age, Sex, MMSE, FAQ, CDR, APOE4,
> L/R hippocampal volume) were chosen for clinical usability — they are things a
> doctor can collect during an exam or from routine tests. The remaining 120
> features are automated MRI subfield volumes (e.g., `LEFT_BA35_NS`,
> `RIGHT_CA1_VOL`) that hippocampal segmentation software like FreeSurfer
> outputs — a clinician won't type those by hand. The web form fills missing
> features with 0.0, which works because the model was trained on centered/scaled
> data. For best accuracy, send the full 128‑feature vector via the API:
>
> ```bash
> curl -X POST ... -d '{"features":{"age":78,"MMSCORE_x":23,...,"LEFT_BA35_NS":0.42}, "sex":1}'
> ```

### 2. SHAP latency ~2–3 seconds per prediction

> **Discussion.** Each prediction runs the background dataset through the TF
> graph to compute Shapley values. This is the standard SHAP trade‑off:
> explainability vs speed. For a screening tool that processes one patient at a
> time, 2–3 s is acceptable — the clinician sees the result with explanation in
> one interaction. Speed could be improved with KernelShap approximations or
> cached background summaries, but the current latency keeps the implementation
> transparent.

### 3. No batch endpoint, no authentication, no HTTPS

> **Discussion.** Designed as a single‑user or small‑clinic tool running on a
> trusted local network. Authentication and HTTPS add complexity with no benefit
> in this context. For deployment behind a reverse proxy (nginx, Caddy), adding
> TLS + basic auth takes minutes. A batch `/api/predict-batch` endpoint could be
> added for validation studies if needed.

### 4. SQLite — single‑user, no concurrent writes

> **Discussion.** SQLite is the right choice here: no concurrent write
> contention (one user, one retrain at a time), zero configuration, backup by
> copying one file. The database lives in the persistent Docker named volume.
> For multi‑user or high‑throughput scenarios, swapping to PostgreSQL requires
> only changes to `app/database.py` — the rest of the app is DB‑agnostic.

### 5. Retrain is manual (button click)

> **Discussion.** Deliberate design. Automatic retraining after every correction
> could cause model instability with few samples. The manual trigger lets the
> clinician review corrections and retrain when confident (e.g., after 10–20
> cases). Automation could be added later with a minimum‑correction threshold.

### 6. 28.5% false‑positive rate for MCI cases

> **Discussion.** This is an intentional trade‑off: catch every AD case
> (TPR=1.0 for both sexes) at the cost of ~28.5% of MCI patients being
> incorrectly flagged. In a screening tool, missing an AD diagnosis is far more
> harmful than a false positive (which is caught at follow‑up). The model is a
> sensitive screen, not a definitive diagnostic — 71.5 % of MCI patients are
> correctly identified as non‑AD.

### 7. TensorFlow 1.x dependency — future deprecation risk

> **Discussion.** aif360's AdversarialDebiasing uses TF1 APIs. TF2 still
> supports compat.v1 mode (as used in `model_service.py`), but future versions
> may drop it. If that happens, the model can be reimplemented in pure sklearn
> or PyTorch — the `FairnessAuditor` library (`fairness_audit.py`) is already
> TF‑agnostic and unaffected.

## How to Use

### Quick start

```bash
git clone git@github.com:imlinstress/GenderBias_AD.git
cd GenderBias_AD/fairness-ad-app
docker compose up --build
```

Open http://localhost:8000

### Predict workflow

1. Fill patient data in the Predict tab (Sex, Age, MMSE, FAQ, CDR, APOE4,
   hippocampal volumes)
2. Click **Run Assessment**
3. Read the diagnosis badge (AD / MCI), confidence score, and SHAP bar chart
4. Confirm or correct the prediction using the buttons below the result

### Correct + retrain workflow

1. After each prediction, click **Yes, AD** / **Yes, MCI** to confirm, or
   **Wrong — submit correction** to flag an error
2. After accumulating several corrections, switch to the Dashboard tab
3. Click **Retrain Now** — the app trains a new AD model on original data +
   all corrections
4. On next startup, the retrained checkpoint is loaded automatically from the
   persistent volume

### Dashboard interpretation

| Metric | Meaning | Ideal |
|---|---|---|
| Balanced Accuracy | Average of TPR and TNR | 1.0 |
| Disparate Impact | AD prediction rate ratio (Female/Male) | 1.0 |
| EOD | TPR difference (Female − Male) | 0.0 |
| AOD | Average of TPR and FPR differences | 0.0 |

Metrics appear only after corrections are submitted (ground truth required).

### API examples

```bash
# Predict
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":78,"MMSCORE_x":23,"FAQTOTAL":12,"CDGLOBAL":4.5},"sex":1}'

# Correct
curl -X POST http://localhost:8000/api/correct \
  -H "Content-Type: application/json" \
  -d '{"prediction_id":1,"true_label":1}'

# Dashboard
curl http://localhost:8000/api/dashboard

# History
curl http://localhost:8000/api/history
```
