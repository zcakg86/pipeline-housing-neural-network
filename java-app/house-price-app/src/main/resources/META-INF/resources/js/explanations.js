// Format model inputs and contribution outputs for safe, readable popup HTML.
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}

function modelFeatureLabel(name) {
  return String(name).replaceAll('_', ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
}

function modelFeatureValue(feature, value) {
  const numeric = Number(value);
  if (feature === 'distance_to_water_m') {
    return numeric < 1000 ? `${Math.round(numeric)} m` : `${(numeric / 1000).toFixed(2)} km`;
  }
  if (feature === 'water_proximity') return numeric.toFixed(4);
  if (['mortgage_rate', 'unemployment_rate'].includes(feature)) return `${numeric.toFixed(2)}%`;
  if (['sqft', 'sqft_lot', 'beds'].includes(feature) ||
      feature.startsWith('community_')) return Math.round(numeric).toLocaleString();
  return numeric.toFixed(3);
}

function modelSignedEffect(effect) {
  const numeric = Number(effect);
  const css = numeric >= 0 ? 'shap-positive' : 'shap-negative';
  return `<span class="${css}">${numeric >= 0 ? '+' : ''}${numeric.toFixed(2)}%</span>`;
}

function fullModelExplanationHtml(observation, payload) {
  const tree = payload.explanation;
  const neural = payload.neuralExplanation;
  const gnn = payload.gnnExplanation;
  const treeFeatures = new Map(tree.contributions.map(item => [item.feature, item]));
  const neuralFeatures = new Map((neural.features || []).map(item => [item.feature, item]));
  const gnnFeatures = new Map((gnn?.features || []).map(item => [item.feature, item]));
  const orderedFeatures = [...new Set([
    ...tree.contributions.map(item => item.feature),
    ...(neural.features || []).map(item => item.feature),
    ...(gnn?.features || []).map(item => item.feature)
  ])];
  const featureRows = orderedFeatures.map(feature => {
    const treeFeature = treeFeatures.get(feature);
    const neuralFeature = neuralFeatures.get(feature);
    const gnnFeature = gnnFeatures.get(feature);
    const value = treeFeature?.value ?? neuralFeature?.value ?? gnnFeature?.value;
    const treeEffect = treeFeature ? modelSignedEffect(treeFeature.priceEffectPct) : '—';
    const stability = neuralFeature && Number.isFinite(Number(neuralFeature.samplingStdErrorPct))
      ? `<br><span class="shap-value">±${Number(neuralFeature.samplingStdErrorPct).toFixed(2)} pp sampling SE</span>`
      : '';
    const neuralEffect = neuralFeature
      ? `${modelSignedEffect(neuralFeature.priceEffectPct)}${stability}` : '—';
    const gnnEffect = gnnFeature ? modelSignedEffect(gnnFeature.priceEffectPct) : '—';
    const group = neuralFeature?.group || gnnFeature?.group || '';
    const displayValue = feature === 'spatial_h3_embedding'
      ? '<span class="shap-value">dated precomputed graph embedding</span>'
      : escapeHtml(modelFeatureValue(feature, value));
    return `<tr><td>${escapeHtml(modelFeatureLabel(feature))}` +
      (group ? `<br><span class="shap-value">${escapeHtml(group)}</span>` : '') + `</td>` +
      `<td>${displayValue}</td>` +
      `<td>${treeEffect}</td><td>${neuralEffect}</td><td>${gnnEffect}</td></tr>`;
  }).join('');
  const rows = featureRows;
  const label = observation.address || observation.h3Index || payload.address || payload.h3Index;
  const date = observation.saleDate || payload.saleDate;
  const historicalNote = payload.historicalSnapshotReconstruction
    ? `<div class="explanation-note shap-negative"><b>Historical reconstruction:</b> ` +
      `the exported Java snapshot cannot recreate row-specific local-market features from this old date. ` +
      `These effects recompute the current deployed models using the snapshot in demonstration mode and ` +
      `may not match the predictions stored in the historical CSV.</div>` : '';
  const gnnSummary = gnn
    ? `<br><b>Spatial GNN prediction:</b> $${Math.round(gnn.predictedPrice).toLocaleString()} ` +
      `<span class="shap-value">(reference $${Math.round(gnn.referencePrice).toLocaleString()})</span><br>`
    : '';
  return `<div><b>Observation:</b> ${escapeHtml(label)} on ${escapeHtml(date)}<br>` +
    `<b>LightGBM prediction:</b> $${Math.round(tree.predictedPrice).toLocaleString()} ` +
    `<span class="shap-value">(baseline $${Math.round(tree.baselinePrice).toLocaleString()})</span><br>` +
    `<b>Neural prediction:</b> $${Math.round(neural.predictedPrice).toLocaleString()} ` +
    `<span class="shap-value">(reference $${Math.round(neural.referencePrice).toLocaleString()})</span></div>` +
    gnnSummary +
    historicalNote +
    `<div class="explanation-note">LightGBM values are exact per-feature TreeSHAP. Neural feature ` +
    `values use ${neural.sampledFeatureCoalitions} deterministic sampled KernelSHAP coalitions, then ` +
    `are constrained to the exact totals from ${neural.evaluatedCoalitions} seven-group coalitions. ` +
    `Feature effects therefore sum exactly to the neural prediction. Sampling SE describes stability ` +
    `of the within-group split.</div>` +
    `<table class="feature-explanation-table"><thead><tr>` +
    `<th>Model feature / group</th><th>Observation value</th>` +
    `<th>LightGBM TreeSHAP</th><th>Neural feature Shapley</th><th>Spatial GNN Shapley</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
}

async function openPointExplanation(source, id, title) {
  const popup = document.getElementById('featurePopup');
  const popupTitle = document.getElementById('featurePopupTitle');
  const body = document.getElementById('featurePopupBody');
  popupTitle.textContent = `All model features — ${title || id}`;
  body.innerHTML = '<span class="shap-value">Calculating TreeSHAP, neural feature Shapley, and 1,024 exact Spatial GNN feature coalitions…</span>';
  popup.style.visibility = 'visible';
  reportMapInteraction('feature contributions requested', title || id);
  const progressKey = `point-explanation-${source}-${id}`;
  beginMapEvent(progressKey, 'Calculating model feature contributions…');
  const cacheKey = `${source}|${id}`;
  try {
    let payload = pointExplanationCache.get(cacheKey);
    if (!payload) {
      const query = new URLSearchParams({ source, id });
      const response = await fetch(`/api/points/explanation?${query}`);
      if (!response.ok) throw new Error(await response.text());
      payload = await response.json();
      pointExplanationCache.set(cacheKey, payload);
    }
    body.innerHTML = fullModelExplanationHtml(payload, payload);
    completeMapEvent(progressKey, 'Feature contributions ready');
  } catch (error) {
    completeMapEvent(progressKey);
    body.innerHTML = `<span class="shap-negative">Feature explanation failed: ` +
      `${escapeHtml(error.message)}</span>`;
  }
}

function attachPointExplanationButton(layer, properties) {
  layer.on('popupopen', event => {
    const button = event.popup.getElement()?.querySelector('.feature-explanation-button');
    if (button && button.dataset.bound !== 'true') {
      button.dataset.bound = 'true';
      button.addEventListener('click', () => openPointExplanation(
        properties.source, properties.id, properties.address
      ));
    }
    const timeButton = event.popup.getElement()?.querySelector(
      '.property-time-series-button'
    );
    if (timeButton && timeButton.dataset.bound !== 'true') {
      timeButton.dataset.bound = 'true';
      timeButton.addEventListener('click', () => openPropertyTimeSeries(
        properties.source, properties.id, properties.address
      ));
    }
  });
}
