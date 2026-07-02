/* app.js — Fair AD Predictor frontend */

let currentPredictionId = null;

document.addEventListener('DOMContentLoaded', () => {
  // Tab switching
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      const tab = document.getElementById('tab-' + btn.dataset.tab);
      if (tab) tab.classList.add('active');
      if (btn.dataset.tab === 'dashboard') loadDashboard();
      if (btn.dataset.tab === 'history') loadHistory();
    });
  });

  // Predict
  document.getElementById('btn-predict').addEventListener('click', predict);

  // Correction buttons
  document.querySelectorAll('.btn-correct').forEach(btn => {
    btn.addEventListener('click', () => {
      if (currentPredictionId) {
        submitCorrection(currentPredictionId, parseInt(btn.dataset.label));
      }
    });
  });
  document.querySelector('.btn-wrong')?.addEventListener('click', () => {
    const label = document.getElementById('result-badge').classList.contains('ad') ? 0 : 1;
    if (currentPredictionId) {
      submitCorrection(currentPredictionId, label);
    }
  });

  // Retrain
  document.getElementById('btn-retrain').addEventListener('click', retrainModel);
});

// ── Build feature vector ──
function buildFeatures() {
  const sex = parseInt(document.getElementById('input-sex').value);
  const age = parseFloat(document.getElementById('input-age').value) || 75;
  const mmse = parseFloat(document.getElementById('input-mmse').value) || 26;
  const faq = parseFloat(document.getElementById('input-faq').value) || 5;
  const cdr = parseFloat(document.getElementById('input-cdr').value) || 2.5;
  const apoe = parseFloat(document.getElementById('input-apoe').value) || 1;
  const lhippo = parseFloat(document.getElementById('input-lhippo').value) || 3200;
  const rhippo = parseFloat(document.getElementById('input-rhippo').value) || 3300;

  return { sex, age, mmse, faq, cdr, apoe, lhippo, rhippo };
}

// ── Predict ──
async function predict() {
  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';

  const features = buildFeatures();
  const sex = parseInt(document.getElementById('input-sex').value);

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, sex }),
    });
    const data = await resp.json();

    currentPredictionId = data.prediction_id;

    // Show result area
    document.getElementById('result-area').classList.remove('hidden');

    // Badge
    const badge = document.getElementById('result-badge');
    if (data.predicted_label === 1) {
      badge.textContent = 'AD — Alzheimer\'s Disease';
      badge.className = 'result-badge ad';
    } else {
      badge.textContent = 'MCI — Mild Cognitive Impairment';
      badge.className = 'result-badge mci';
    }

    // Metrics
    document.getElementById('result-confidence').textContent =
      (data.score * 100).toFixed(1) + '%';
    document.getElementById('result-threshold').textContent =
      data.threshold.toFixed(4);
    document.getElementById('result-calibration').textContent =
      data.calibration_applied ? 'Yes (Δ=' + data.calibration_delta.toFixed(3) + ')' : 'No';

    // SHAP
    const shapArea = document.getElementById('shap-area');
    const shapBars = document.getElementById('shap-bars');
    if (data.shap_values && Object.keys(data.shap_values).length > 0) {
      shapArea.classList.remove('hidden');
      shapBars.innerHTML = '';
      const entries = Object.entries(data.shap_values)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8);
      const maxAbs = Math.max(...entries.map(e => Math.abs(e[1])));
      entries.forEach(([name, val]) => {
        const pct = (Math.abs(val) / maxAbs) * 100;
        const color = val > 0 ? 'var(--orange)' : 'var(--blue)';
        const row = document.createElement('div');
        row.className = 'shap-bar-row';
        row.innerHTML = `
          <span class="shap-bar-label">${name}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width:${pct}%;background:${color};left:${val > 0 ? '50%' : (50 - pct) + '%'}"></div>
          </div>
          <span class="shap-bar-value">${val > 0 ? '+' : ''}${val.toFixed(3)}</span>
        `;
        shapBars.appendChild(row);
      });
    } else {
      shapArea.classList.add('hidden');
    }

    // Correction
    document.getElementById('correction-feedback').classList.add('hidden');

  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Assessment';
  }
}

// ── Submit correction ──
async function submitCorrection(predictionId, trueLabel) {
  try {
    await fetch('/api/correct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction_id: predictionId, true_label: trueLabel }),
    });
    const fb = document.getElementById('correction-feedback');
    fb.textContent = '✓ Correction saved. The model will improve with more data.';
    fb.className = '';
    fb.classList.remove('hidden');
  } catch (err) {
    alert('Correction failed: ' + err.message);
  }
}

// ── Dashboard ──
async function loadDashboard() {
  const loading = document.getElementById('dashboard-loading');
  const content = document.getElementById('dashboard-content');
  loading.classList.remove('hidden');
  content.classList.add('hidden');

  try {
    const resp = await fetch('/api/dashboard');
    const data = await resp.json();

    loading.classList.add('hidden');
    content.classList.remove('hidden');

    document.getElementById('stat-n').textContent = data.n_predictions || 0;
    const m = data.metrics || {};
    const hasMetrics = m.BA != null || m.DI != null || m.EOD != null || m.AOD != null;

    const placeholder = document.getElementById('metrics-placeholder');
    const metricBoxes = document.querySelectorAll('.stat-box:not(:first-child)');
    if (hasMetrics) {
      placeholder.classList.add('hidden');
      metricBoxes.forEach(el => el.classList.remove('hidden'));
      document.getElementById('stat-ba').textContent = m.BA != null ? m.BA.toFixed(4) : '--';
      document.getElementById('stat-di').textContent = m.DI != null ? m.DI.toFixed(4) : '--';
      document.getElementById('stat-eod').textContent = m.EOD != null ? m.EOD.toFixed(4) : '--';
      document.getElementById('stat-aod').textContent = m.AOD != null ? m.AOD.toFixed(4) : '--';

      const eodEl = document.getElementById('stat-eod');
      if (m.EOD != null) {
        eodEl.style.color = Math.abs(m.EOD) < 0.05 ? 'var(--green)' : 'var(--red)';
      }
    } else if (data.n_predictions > 0) {
      placeholder.classList.remove('hidden');
      metricBoxes.forEach(el => el.classList.add('hidden'));
    }

    // Sex stats
    const s = data.sex_distribution || {};
    if (s.male) {
      document.getElementById('sex-male-n').textContent = s.male.n;
      document.getElementById('sex-male-ad').textContent = s.male.n_ad_predicted;
      document.getElementById('sex-male-score').textContent = s.male.mean_score.toFixed(3);
    }
    if (s.female) {
      document.getElementById('sex-female-n').textContent = s.female.n;
      document.getElementById('sex-female-ad').textContent = s.female.n_ad_predicted;
      document.getElementById('sex-female-score').textContent = s.female.mean_score.toFixed(3);
    }

    // Config
    const cfg = data.config || {};
    document.getElementById('cfg-threshold').textContent = cfg.threshold != null ? cfg.threshold.toFixed(4) : '--';
    document.getElementById('cfg-delta').textContent = cfg.calibration_delta != null ? cfg.calibration_delta.toFixed(3) : '--';

  } catch (err) {
    loading.textContent = 'Failed to load dashboard.';
  }
}

// ── History ──
async function loadHistory() {
  const loading = document.getElementById('history-loading');
  const content = document.getElementById('history-content');
  const empty = document.getElementById('history-empty');
  const tbody = document.getElementById('history-body');

  loading.classList.remove('hidden');
  content.classList.add('hidden');
  empty.classList.add('hidden');

  try {
    const resp = await fetch('/api/history?limit=100');
    const data = await resp.json();

    loading.classList.add('hidden');

    if (!data || data.length === 0) {
      empty.classList.remove('hidden');
      return;
    }

    content.classList.remove('hidden');
    tbody.innerHTML = '';
    data.forEach(p => {
      const tr = document.createElement('tr');
      const predLabel = p.predicted_label === 1 ? 'AD' : 'MCI';
      const corrected = p.corrected_label != null
        ? (p.corrected_label === 1 ? 'AD' : 'MCI')
        : '—';
      tr.innerHTML = `
        <td>${p.id}</td>
        <td>${p.sex === 0 ? 'M' : 'F'}</td>
        <td>${p.score.toFixed(3)}</td>
        <td>${predLabel}</td>
        <td>${corrected}</td>
        <td>${p.timestamp ? p.timestamp.slice(0, 19).replace('T', ' ') : ''}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    loading.textContent = 'Failed to load history.';
  }
}

// ── Retrain ──
async function retrainModel() {
  const btn = document.getElementById('btn-retrain');
  const result = document.getElementById('retrain-result');
  btn.disabled = true;
  btn.textContent = 'Retraining...';
  result.classList.add('hidden');

  try {
    const resp = await fetch('/api/retrain', { method: 'POST' });
    const data = await resp.json();

    result.classList.remove('hidden');
    if (data.status === 'ok') {
      result.innerHTML = `
        <p style="color:var(--green);font-weight:600;">✓ Retrain complete — ${data.version}</p>
        <p>Trained on ${data.n_original} original + ${data.n_new} new samples (${data.n_total} total).</p>
      `;
    } else {
      result.textContent = 'Retrain failed.';
    }
  } catch (err) {
    result.classList.remove('hidden');
    result.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Retrain Now';
  }
}
