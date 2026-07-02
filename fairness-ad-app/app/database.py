"""database.py — SQLite for predictions and corrections."""

import sqlite3
import json
import os
import numpy as np
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.db')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            features_json TEXT NOT NULL,
            sex INTEGER NOT NULL,
            score REAL NOT NULL,
            predicted_label INTEGER NOT NULL,
            threshold REAL NOT NULL,
            shap_json TEXT,
            corrected_label INTEGER,
            correction_timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL,
            trained_at TEXT NOT NULL,
            metrics_json TEXT,
            n_train_samples INTEGER,
            active INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details_json TEXT
        );
    """)
    conn.commit()
    conn.close()


def insert_prediction(sex, score, predicted_label, threshold, features, shap_values=None):
    conn = get_connection()
    if shap_values is not None:
        shap_values = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
    shap_json = json.dumps(shap_values) if shap_values is not None else None
    cur = conn.execute(
        "INSERT INTO predictions (timestamp, features_json, sex, score, predicted_label, threshold, shap_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(features), int(sex), float(score),
         int(predicted_label), float(threshold), shap_json))
    pred_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def correct_prediction(pred_id, true_label):
    conn = get_connection()
    conn.execute(
        "UPDATE predictions SET corrected_label = ?, correction_timestamp = ? WHERE id = ?",
        (int(true_label), datetime.utcnow().isoformat(), pred_id))
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_uncorrected():
    """Return predictions where the model was wrong and no correction exists yet."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE corrected_label IS NULL AND predicted_label != ?",
        (0,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_correction_data():
    """Return all (features, true_label) pairs from corrections for retraining."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT features_json, corrected_label FROM predictions WHERE corrected_label IS NOT NULL"
    ).fetchall()
    conn.close()
    if not rows:
        return [], []
    X = []
    y = []
    for r in rows:
        feats = json.loads(r['features_json'])
        if isinstance(feats, dict):
            feats = list(feats.values())
        X.append(feats)
        y.append(int(r['corrected_label']))
    return X, y


def log_event(event_type, details=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_log (timestamp, event_type, details_json) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), event_type, json.dumps(details) if details else None))
    conn.commit()
    conn.close()


def save_model_version(version, metrics=None, n_train=0):
    conn = get_connection()
    conn.execute("UPDATE model_versions SET active = 0")
    conn.execute(
        "INSERT INTO model_versions (version, trained_at, metrics_json, n_train_samples, active) "
        "VALUES (?, ?, ?, ?, 1)",
        (version, datetime.utcnow().isoformat(),
         json.dumps(metrics) if metrics else None, n_train))
    conn.commit()
    conn.close()


def get_active_model_version():
    conn = get_connection()
    row = conn.execute("SELECT * FROM model_versions WHERE active = 1 ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def get_fairness_summary():
    """Compute cumulative fairness metrics from all predictions."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT sex, score, predicted_label, threshold FROM predictions"
    ).fetchall()
    conn.close()
    if not rows:
        return {}
    sex = np.array([r['sex'] for r in rows])
    scores = np.array([r['score'] for r in rows])
    labels = np.array([r['predicted_label'] for r in rows])
    return {'sex': sex, 'scores': scores, 'labels': labels}
