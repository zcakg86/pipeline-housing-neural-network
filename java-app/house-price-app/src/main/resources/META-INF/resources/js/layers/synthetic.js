// Date-driven synthetic predictions, feature display, and cached explanations.
// ── Synthetic H3 L8 layer ───────────────────────────────────────────────────
let syntheticRequestId = 0;
let syntheticDates = [localIsoDate(new Date())];
let syntheticDateDebounce = null;
let syntheticCoordinateMarker = null;

function localIsoDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function subtractOneMonth(date) {
  const result = new Date(date);
  const originalDay = result.getDate();
  result.setDate(1);
  result.setMonth(result.getMonth() - 1);
  const lastDay = new Date(result.getFullYear(), result.getMonth() + 1, 0).getDate();
  result.setDate(Math.min(originalDay, lastDay));
  return result;
}

function syntheticDateForOffset(monthOffset) {
  return syntheticDates[Number(monthOffset)] || syntheticDates[0];
}

const syntheticDateControl = L.control({ position: 'topleft' });
syntheticDateControl.onAdd = () => {
  const div = L.DomUtil.create('div', 'synthetic-date-control');
  div.id = 'syntheticDateControl';
  div.style.display = document.getElementById('showSynthetic').checked ? 'block' : 'none';
  div.innerHTML =
    `<strong>Synthetic sale date</strong><br>` +
    `<span id="syntheticDateLabel">${syntheticDateForOffset(0)}</span>` +
    `<input id="syntheticDateSlider" type="range" min="0" max="0" value="0" step="1" disabled ` +
      `list="syntheticMonthTicks" aria-label="Synthetic sale date, months backwards"/>` +
    `<datalist id="syntheticMonthTicks"></datalist>` +
    `<div id="syntheticYearTicks" class="synthetic-year-ticks"></div>` +
    `<div class="synthetic-date-direction"><span>Today</span><span>Older →</span></div>` +
    `<div id="syntheticDateStatus" class="status">Loading available dates…</div>`;
  L.DomEvent.disableClickPropagation(div);
  L.DomEvent.disableScrollPropagation(div);
  return div;
};
syntheticDateControl.addTo(map);

async function initializeSyntheticDateControl() {
  const status = document.getElementById('syntheticDateStatus');
  try {
    const response = await fetch('/api/synthetic/meta');
    if (!response.ok) throw new Error(await response.text());
    const metadata = await response.json();
    const isoDate = /^\d{4}-\d{2}-\d{2}$/;
    if (!isoDate.test(metadata.defaultSaleDate) ||
        !isoDate.test(metadata.minimumSaleDate)) {
      throw new Error('Synthetic metadata did not contain valid dates');
    }
    syntheticDates = [metadata.defaultSaleDate];
    let cursor = new Date(`${metadata.defaultSaleDate}T12:00:00`);
    while (syntheticDates[syntheticDates.length - 1] !== metadata.minimumSaleDate) {
      cursor = subtractOneMonth(cursor);
      const candidate = localIsoDate(cursor);
      if (candidate <= metadata.minimumSaleDate) {
        syntheticDates.push(metadata.minimumSaleDate);
      } else {
        syntheticDates.push(candidate);
      }
    }
    const slider = document.getElementById('syntheticDateSlider');
    const maximumMonths = syntheticDates.length - 1;
    slider.max = maximumMonths;
    slider.disabled = maximumMonths === 0;
    const datalist = document.getElementById('syntheticMonthTicks');
    datalist.innerHTML = syntheticDates.map((date, index) =>
      `<option value="${index}" label="${date.slice(0, 7)}"></option>`
    ).join('');
    const yearTicks = document.getElementById('syntheticYearTicks');
    const renderedYears = new Set();
    yearTicks.innerHTML = syntheticDates.map((date, index) => {
      const year = date.slice(0, 4);
      if (date.slice(5, 7) !== '01' || renderedYears.has(year)) return '';
      renderedYears.add(year);
      const left = maximumMonths ? index / maximumMonths * 100 : 0;
      return `<span style="left:${left}%">${year}</span>`;
    }).join('');
    status.textContent = `${metadata.count.toLocaleString()} cells · monthly steps`;
    slider.addEventListener('input', () => {
      document.getElementById('syntheticDateLabel').textContent =
        syntheticDateForOffset(slider.value);
      clearTimeout(syntheticDateDebounce);
      if (document.getElementById('showSynthetic').checked) {
        syntheticDateDebounce = setTimeout(loadSynthetic, 300);
      }
    });
  } catch (error) {
    status.textContent = `Synthetic dates unavailable: ${error.message}`;
  }
}

function syntheticVariableValue(properties, variable, model = displayedModel()) {
  switch (variable) {
    case 'predicted_price': return model === 'lightgbm'
      ? properties.lightgbmPredictedPrice
      : model === 'gnn' ? properties.gnnPredictedPrice : properties.neuralPredictedPrice;
    case 'sqft': return properties.sqft;
    case 'sqft_lot': return properties.sqftLot;
    case 'pred_std': return (properties.neuralUpper95 - properties.neuralLower95) / 3.92;
    case 'pred_cv_pct': return (
      (properties.neuralUpper95 - properties.neuralLower95) /
      properties.neuralPredictedPrice * 100
    );
    case 'attn_community': return properties.attnCommunity;
    case 'attn_property': return properties.attnProperty;
    case 'attn_time': return properties.attnTime;
    case 'attn_market': return properties.attnMarket;
    default: return model === 'gnn' ? properties.gnnPredictedPrice : properties.neuralPredictedPrice;
  }
}

async function loadSynthetic() {
  const requestId = ++syntheticRequestId;
  syntheticLayer.clearLayers();
  syntheticCoordinateMarker = null;
  const visible = document.getElementById('showSynthetic').checked;
  setLayerGroupVisible(syntheticLayer, visible);
  if (!visible) return;

  const slider = document.getElementById('syntheticDateSlider');
  const candidateDate = syntheticDateForOffset(slider.value);
  const saleDate = /^\d{4}-\d{2}-\d{2}$/.test(candidateDate)
    ? candidateDate : localIsoDate(new Date());
  const status = document.getElementById('syntheticDateStatus');
  status.textContent = `Calculating predictions for ${saleDate}…`;
  const progressKey = `synthetic-${requestId}`;
  beginMapEvent(progressKey, `Calculating synthetic predictions for ${saleDate}…`);
  try {
    const query = new URLSearchParams({ saleDate });
    const response = await fetch(`/api/synthetic?${query}`);
    if (!response.ok) throw new Error(await response.text());
    const geojson = await response.json();
    if (requestId !== syntheticRequestId) {
      completeMapEvent(progressKey);
      return;
    }

    const requestedVariable = document.getElementById('variable').value;
    const variable = ['pct_error', 'sale_price', 'num_sales'].includes(requestedVariable)
      ? 'predicted_price' : requestedVariable;
    const model = displayedModel();
    const values = geojson.features.map(feature =>
      syntheticVariableValue(feature.properties, variable, model)
    );
    const colorFn = getColorFn(variable, values, null);
    const money = value => `$${Math.round(value).toLocaleString()}`;
    const attentionRow = (label, value) => `${label}: ${(value * 100).toFixed(1)}%<br>`;

    const featureLabel = name => name.replaceAll('_', ' ')
      .replace(/\b\w/g, character => character.toUpperCase());
    const featureValue = contribution => {
      const value = Number(contribution.value);
      if (contribution.feature === 'distance_to_water_m') {
        return value < 1000 ? `${Math.round(value)} m` : `${(value / 1000).toFixed(2)} km`;
      }
      if (contribution.feature === 'water_proximity') return value.toFixed(4);
      if (['mortgage_rate', 'unemployment_rate'].includes(contribution.feature)) {
        return `${value.toFixed(2)}%`;
      }
      if (['sqft', 'sqft_lot', 'beds'].includes(contribution.feature) ||
          contribution.feature.startsWith('community_')) {
        return Math.round(value).toLocaleString();
      }
      return value.toFixed(3);
    };
    const effectRow = (contribution, className) => {
      const effect = Number(contribution.priceEffectPct);
      const sign = effect >= 0 ? '+' : '';
      return `<span class="${className}">${sign}${effect.toFixed(1)}%</span> ` +
        `${escapeHtml(featureLabel(contribution.feature))} ` +
        `<span class="shap-value">(${escapeHtml(featureValue(contribution))})</span><br>`;
    };
    const neuralGroupHtml = payload => {
      const neural = payload.neuralExplanation;
      if (!neural) return '';
      const groups = [...neural.groups]
        .sort((a, b) => Math.abs(b.logContribution) - Math.abs(a.logContribution))
        .map(group => {
          const effect = Number(group.priceEffectPct);
          const css = effect >= 0 ? 'shap-positive' : 'shap-negative';
          return `<span class="${css}">${escapeHtml(group.group)} ` +
            `${effect >= 0 ? '+' : ''}${effect.toFixed(1)}%</span>`;
        }).join('<br>');
      return `<br><b>Neural exact grouped Shapley:</b><br>` +
        `<div class="shap-groups">${groups}</div>`;
    };
    const explanationHtml = payload => {
      const explanation = payload.explanation;
      const contributions = explanation.contributions.filter(item =>
        Number.isFinite(Number(item.logContribution)) &&
        Math.abs(Number(item.logContribution)) > 1e-10
      );
      const positive = [...contributions]
        .filter(item => item.logContribution > 0)
        .sort((a, b) => b.logContribution - a.logContribution)
        .slice(0, 5);
      const negative = [...contributions]
        .filter(item => item.logContribution < 0)
        .sort((a, b) => a.logContribution - b.logContribution)
        .slice(0, 5);
      const groups = [...explanation.groups]
        .sort((a, b) => Math.abs(b.logContribution) - Math.abs(a.logContribution))
        .map(group => {
          const effect = Number(group.priceEffectPct);
          return `${escapeHtml(group.group)} ${effect >= 0 ? '+' : ''}${effect.toFixed(1)}%`;
        }).join('<br>');
      return `<b>LightGBM exact TreeSHAP:</b><br>` +
        `<div class="shap-groups">Groups:<br>${groups}</div>` +
        `<span class="shap-positive">Top positive effects</span><br>` +
        (positive.length ? positive.map(item => effectRow(item, 'shap-positive')).join('') : 'None<br>') +
        `<span class="shap-negative">Top negative effects</span><br>` +
        (negative.length ? negative.map(item => effectRow(item, 'shap-negative')).join('') : 'None<br>') +
        neuralGroupHtml(payload);
    };
    const neuralGroupForFeature = feature => {
      if (feature.startsWith('community_')) return 'Community';
      if (['sqft', 'sqft_lot', 'beds', 'distance_to_water_m', 'water_proximity'].includes(feature)) {
        return 'Property';
      }
      if (['time_trend', 'annual_sin', 'annual_cos'].includes(feature)) return 'Time';
      if (['mortgage_rate', 'unemployment_rate'].includes(feature)) return 'Economics';
      if (feature.startsWith('center_local_') || feature.startsWith('neighbor_')) {
        return 'Local market';
      }
      return 'Other';
    };
    const signedEffect = effect => {
      const numeric = Number(effect);
      const css = numeric >= 0 ? 'shap-positive' : 'shap-negative';
      return `<span class="${css}">${numeric >= 0 ? '+' : ''}${numeric.toFixed(2)}%</span>`;
    };
    const detailedExplanationHtml = (p, payload) => {
      return fullModelExplanationHtml(p, payload);
    };
    const loadExplanation = async p => {
      const cacheKey = `${p.saleDate}|${p.h3Index}|${p.lat}|${p.lng}`;
      let explanation = lightgbmExplanationCache.get(cacheKey);
      if (!explanation) {
        const query = new URLSearchParams({
          saleDate: p.saleDate, h3Index: p.h3Index, lat: p.lat, lng: p.lng
        });
        const response = await fetch(`/api/synthetic/explanation?${query}`);
        if (!response.ok) throw new Error(await response.text());
        explanation = await response.json();
        lightgbmExplanationCache.set(cacheKey, explanation);
        if (lightgbmExplanationCache.size > 500) {
          lightgbmExplanationCache.delete(lightgbmExplanationCache.keys().next().value);
        }
      }
      return explanation;
    };
    const tooltipHtml = p =>
      `<div class="synthetic-tooltip"><b>Synthetic H3 L8</b>: ${p.h3Index}<br>` +
      `Sale date: ${p.saleDate}<br>` +
      `Sqft: ${p.sqft.toLocaleString()} | Lot Sqft: ${p.sqftLot.toLocaleString()} | Beds: ${p.beds}<br>` +
      waterFeatureRows(p) + `<br>` +
      `<b>Neural: ${money(p.neuralPredictedPrice)}</b><br>` +
      `&nbsp;90% CI: ${money(p.neuralLower90)} – ${money(p.neuralUpper90)}<br>` +
      `&nbsp;95% CI: ${money(p.neuralLower95)} – ${money(p.neuralUpper95)}<br><br>` +
      `<b>Spatial GNN: ${money(p.gnnPredictedPrice)}</b><br>` +
      `&nbsp;90% conformal CI: ${money(p.gnnLower90)} – ${money(p.gnnUpper90)}<br>` +
      `&nbsp;95% conformal CI: ${money(p.gnnLower95)} – ${money(p.gnnUpper95)}<br><br>` +
      `<b>LightGBM: ${money(p.lightgbmPredictedPrice)}</b><br>` +
      `&nbsp;90% conformal CI: ${money(p.lightgbmLower90)} – ${money(p.lightgbmUpper90)}<br>` +
      `&nbsp;95% conformal CI: ${money(p.lightgbmLower95)} – ${money(p.lightgbmUpper95)}<br><br>` +
      `<span class="shap-value">Click the cell for full model feature effects.</span></div>`;

    const recalculateMovedPoint = async (p, polygonLayer, latlng, marker) => {
      const progressKey = `synthetic-marker-${p.h3Index}`;
      reportMapInteraction(
        'synthetic marker moved', `${p.h3Index} → ${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}`
      );
      beginMapEvent(progressKey, 'Recalculating synthetic prediction at adjusted coordinates…');
      marker.bindTooltip('Recalculating at adjusted coordinates…').openTooltip();
      const query = new URLSearchParams({
        h3Index: p.h3Index,
        saleDate: p.saleDate,
        lat: latlng.lat,
        lng: latlng.lng
      });
      try {
        const response = await fetch(`/api/synthetic/point?${query}`);
        if (!response.ok) throw new Error(await response.text());
        Object.assign(p, await response.json());
        polygonLayer.setTooltipContent(tooltipHtml(p));
        polygonLayer.setStyle({ fillColor: colorFn(syntheticVariableValue(p, variable, model)) });
        marker.bindTooltip(
          `Adjusted coordinate<br>${p.lat.toFixed(5)}, ${p.lng.toFixed(5)}<br>` +
          `Drag to recalculate · double-click to reset`, { direction: 'top' }
        ).openTooltip();
        completeMapEvent(progressKey, 'Adjusted synthetic prediction ready');
      } catch (error) {
        completeMapEvent(progressKey);
        throw error;
      }
    };

    const activateCoordinateMarker = (p, polygonLayer) => {
      if (syntheticCoordinateMarker) syntheticLayer.removeLayer(syntheticCoordinateMarker);
      if (!Number.isFinite(Number(p.centerLat))) {
        p.centerLat = Number(p.lat);
        p.centerLng = Number(p.lng);
      }
      const icon = L.divIcon({
        className: '',
        html: '<div class="synthetic-coordinate-icon"></div>',
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });
      syntheticCoordinateMarker = L.marker([p.lat, p.lng], {
        draggable: true,
        icon,
        keyboard: true,
        title: 'Drag to adjust synthetic property coordinates'
      }).addTo(syntheticLayer);
      syntheticCoordinateMarker.bindTooltip(
        `Cell center<br>${Number(p.lat).toFixed(5)}, ${Number(p.lng).toFixed(5)}<br>` +
        `Drag to recalculate · double-click to reset`,
        { direction: 'top' }
      ).openTooltip();
      syntheticCoordinateMarker.on('dragend', async event => {
        try {
          await recalculateMovedPoint(
            p, polygonLayer, event.target.getLatLng(), event.target
          );
        } catch (error) {
          event.target.bindTooltip(`Prediction failed: ${error.message}`).openTooltip();
        }
      });
      syntheticCoordinateMarker.on('dblclick', async event => {
        if (event.originalEvent) L.DomEvent.stop(event.originalEvent);
        const center = L.latLng(p.centerLat, p.centerLng);
        event.target.setLatLng(center);
        try {
          await recalculateMovedPoint(p, polygonLayer, center, event.target);
        } catch (error) {
          event.target.bindTooltip(`Reset failed: ${error.message}`).openTooltip();
        }
      });
    };

    await ensureCommunityColorMap();
    if (requestId !== syntheticRequestId) return;

    L.geoJSON(geojson, {
      renderer: syntheticRenderer,
      style: feature => ({
        fillColor: colorFn(syntheticVariableValue(feature.properties, variable, model)),
        fillOpacity: 0.42,
        color: communityColorMap.get(String(feature.properties.community)) || '#888',
        opacity: 1,
        weight: 2,
        lineCap: 'round',
        lineJoin: 'round'
      }),
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        layer.bindTooltip(
          tooltipHtml(p),
          { sticky: true, className: 'synthetic-tooltip-container' }
        );
        layer.on('click', async () => {
          reportMapInteraction('synthetic cell selected', p.h3Index);
          activateCoordinateMarker(p, layer);
          const popup = document.getElementById('featurePopup');
          const title = document.getElementById('featurePopupTitle');
          const body = document.getElementById('featurePopupBody');
          title.textContent = `All model features — ${p.h3Index}`;
          body.innerHTML = '<span class="shap-value">Calculating TreeSHAP, neural feature Shapley, and 1,024 exact Spatial GNN feature coalitions…</span>';
          popup.style.visibility = 'visible';
          const progressKey = `synthetic-explanation-${p.h3Index}`;
          beginMapEvent(progressKey, 'Calculating feature contributions…');
          try {
            const explanation = await loadExplanation(p);
            body.innerHTML = detailedExplanationHtml(p, explanation);
            completeMapEvent(progressKey, 'Feature contributions ready');
          } catch (error) {
            completeMapEvent(progressKey);
            body.innerHTML = `<span class="shap-negative">Feature explanation failed: ` +
              `${escapeHtml(error.message)}</span>`;
          }
        });
      }
    }).addTo(syntheticLayer);
    buildLegend(variable, values, null);
    const indicatorDateNote = geojson.marketIndicatorDate !== saleDate
      ? ` · indicators as of ${geojson.marketIndicatorDate}` : '';
    status.textContent = `${geojson.count.toLocaleString()} predictions · ${saleDate}` +
      ` · Mortgage ${Number(geojson.mortgageRate).toFixed(2)}%` +
      ` · Unemployment ${Number(geojson.unemploymentRate).toFixed(1)}%` +
      indicatorDateNote;
    completeMapEvent(progressKey, `Synthetic predictions ready for ${saleDate}`);
  } catch (error) {
    completeMapEvent(progressKey);
    if (requestId !== syntheticRequestId) return;
    status.textContent = `Prediction failed: ${error.message}`;
  }
}
