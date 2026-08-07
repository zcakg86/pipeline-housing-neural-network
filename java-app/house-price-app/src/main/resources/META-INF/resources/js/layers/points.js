// Detailed historical/RentCast/Zillow points, popups, and on-demand explanations.
// ── Sales points layer (RentCast / historical / both) ────────────────────────
function waterFeatureRows(properties) {
  const distance = Number(properties.distanceToWaterM);
  if (!Number.isFinite(distance)) return '';
  const formattedDistance = distance < 1000
    ? `${Math.round(distance).toLocaleString()} m`
    : `${(distance / 1000).toFixed(2)} km`;
  return `Nearest water boundary: ${formattedDistance}<br>` +
    `Water proximity (τ=100 m): ${Math.exp(-distance / 100).toFixed(4)}<br>`;
}

function modelIntervalRows(properties, model) {
  const money = value => `$${Math.round(Number(value)).toLocaleString()}`;
  const prefix = model === 'neural' ? 'neural' : model === 'lightgbm' ? 'lightgbm' : 'gnn';
  const title = model === 'neural' ? '' : 'conformal ';
  return `&nbsp;90% ${title}CI: ${money(properties[`${prefix}Lower90`])} – ${money(properties[`${prefix}Upper90`])}<br>` +
    `&nbsp;95% ${title}CI: ${money(properties[`${prefix}Lower95`])} – ${money(properties[`${prefix}Upper95`])}<br>`;
}

async function loadRentcast() {
  rentcastLayer.clearLayers();
  const visible = document.getElementById('showRentcast').checked;
  setLayerGroupVisible(rentcastLayer, visible);
  if (!visible) { abortLayerRequest('rentcast'); return; }

  const variable  = document.getElementById('variable').value;
  const model = displayedModel();
  const {min: minErr, max: maxErr} = getErrorRange();
  const minSqft   = document.getElementById('minSqft').value;
  const maxSqft   = document.getElementById('maxSqft').value;
  const homeType  = document.getElementById('salesType').value;
  const dateFrom  = document.getElementById('dateFrom').value;
  const dateTo    = document.getElementById('dateTo').value;
  const source    = document.getElementById('salesPointSource').value;
  const firstLoad = source !== 'sales' && !loadedLiveSources.has('rentcast');

  if (firstLoad) {
    beginMapEvent('rentcast-load', 'Loading saved RentCast records and computing predictions…');
  }

  let gj;
  try {
    gj = await fetchLayerJson('rentcast',
      `/api/rentcast?source=${source}&minError=${minErr}&maxError=${maxErr}` +
      `&minSqft=${minSqft}&maxSqft=${maxSqft}&homeType=${homeType}` +
      `&dateFrom=${dateFrom}&dateTo=${dateTo}&model=${model}&${mapViewportQuery()}`
    );
  } finally {
    if (firstLoad) completeMapEvent('rentcast-load', gj ? 'RentCast layer ready' : '');
  }
  if (!gj) return;
  if (!loadedLiveSources.has('rentcast')) {
    loadedLiveSources.add('rentcast');
    loadHomeTypes();
    loadStats();
  }
  reportPointResponse('Sales points', gj);
  if (!gj.features || gj.features.length === 0) return;

  // Compute colour function from the loaded data (same approach as H3 layer)
  const values  = gj.features.map(f => pointVariableValue(f.properties, variable, model));
  const anchors = variable === 'pct_error' ? computeErrorAnchors(values) : null;
  const colorFn = getColorFn(variable, values, anchors);

  L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius:      f.properties.cluster
        ? Math.min(15, 5 + Math.log1p(f.properties.count || 1) * 1.5) : 6,
      fillColor:   colorFn(pointVariableValue(f.properties, variable, model)),
      fillOpacity: 0.85,
      color:       '#00cc66',
      weight:      1.5
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      if (p.cluster) {
        layer.bindPopup(
          `<b>${Number(p.count).toLocaleString()} sales in this map cell</b><br>` +
          `Average sale: $${Math.round(p.salePrice || 0).toLocaleString()}<br>` +
          `Average neural: $${Math.round(p.predictedPrice || 0).toLocaleString()}<br>` +
          `Average LightGBM: $${Math.round(p.lightgbmPredictedPrice || 0).toLocaleString()}<br>` +
          `Average Lot Sqft: ${Math.round(p.sqftLot || 0).toLocaleString()}<br>` +
          `Zoom in to inspect individual sales.`,
          { autoPan: false }
        );
        return;
      }
      const srcLabel = p.source === 'sales' ? 'Historical Sale' : 'RentCast Sale';
      const localMarketWarning = p.localMarketPredictionIssue === 'rolling_snapshot_look_ahead'
        ? `<div class="local-market-warning">⚠ Local market context includes completed sales through ` +
          `${escapeHtml(p.rentcastLocalMarketLatestSaleDate || 'the snapshot date')}, after this sale. ` +
          `Prediction is displayed for comparison, not a causal day-of-sale estimate.</div>`
        : '';
      layer.bindPopup(
        `<b>${p.address || srcLabel}</b><br>` +
        `<i style="color:#00cc66">${srcLabel}</i><br>` +
        (p.saleDate ? `Sold: ${p.saleDate}<br>` : '') +
        `Sale Price: $${Math.round(p.salePrice).toLocaleString()}<br>` +
        `<b>Neural:</b> $${Math.round(p.predictedPrice).toLocaleString()} (${p.pctError.toFixed(1)}%)<br>` + modelIntervalRows(p, 'neural') +
        `<b>LightGBM:</b> $${Math.round(p.lightgbmPredictedPrice).toLocaleString()} (${p.lightgbmPctError.toFixed(1)}%)<br>` + modelIntervalRows(p, 'lightgbm') +
        `<b>Spatial GNN:</b> $${Math.round(p.gnnPredictedPrice || 0).toLocaleString()} (${(p.gnnPctError || 0).toFixed(1)}%)<br>` + modelIntervalRows(p, 'gnn') +
        `Beds: ${p.beds} | Baths: ${p.baths} | Sqft: ${Math.round(p.sqft).toLocaleString()}<br>` +
        `Lot Sqft: ${Math.round(p.sqftLot || 0).toLocaleString()}<br>` +
        waterFeatureRows(p) +
        localMarketWarning +
        `<button class="feature-explanation-button" type="button">Show feature contributions</button>` +
        `<button class="property-time-series-button" type="button">Show predicted value history</button>`,
        { autoPan: false }
      );
      attachPointExplanationButton(layer, p);
    }
  }).addTo(rentcastLayer);
  // Detailed sales use the same active map-colour scale as the H3 layer.
  buildLegend(variable, values, anchors);
}

// ── Zillow layer ──────────────────────────────────────────────────────────────
async function loadZillow() {
  zillowLayer.clearLayers();
  const visible = document.getElementById('showZillow').checked;
  setLayerGroupVisible(zillowLayer, visible);
  if (!visible) { abortLayerRequest('zillow'); return; }

  const variable  = document.getElementById('variable').value;
  const model = displayedModel();
  const {min: minErr, max: maxErr} = getErrorRange();
  const minSqft   = document.getElementById('minSqft').value;
  const maxSqft   = document.getElementById('maxSqft').value;
  const homeType  = document.getElementById('zillowType').value;
  const firstLoad = !loadedLiveSources.has('zillow');

  if (firstLoad) {
    beginMapEvent('zillow-load', 'Loading saved Zillow listings and computing predictions…');
  }

  let gj;
  try {
    gj = await fetchLayerJson('zillow',
      `/api/zillow?minError=${minErr}&maxError=${maxErr}` +
      `&minSqft=${minSqft}&maxSqft=${maxSqft}&homeType=${homeType}&model=${model}` +
      `&${mapViewportQuery()}`
    );
  } finally {
    if (firstLoad) completeMapEvent('zillow-load', gj ? 'Zillow layer ready' : '');
  }
  if (!gj) return;
  if (!loadedLiveSources.has('zillow')) {
    loadedLiveSources.add('zillow');
    loadHomeTypes();
    loadStats();
  }
  reportPointResponse('Zillow', gj);
  if (!gj.features || !gj.features.length) return;

  const values  = gj.features.map(f => pointVariableValue(f.properties, variable, model));
  const anchors = variable === 'pct_error' ? computeErrorAnchors(values) : null;
  const colorFn = getColorFn(variable, values, anchors);

  L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius:      f.properties.cluster
        ? Math.min(15, 5 + Math.log1p(f.properties.count || 1) * 1.5) : 7,
      fillColor:   colorFn(pointVariableValue(f.properties, variable, model)),
      fillOpacity: 0.85,
      color:       '#3399ff',
      weight:      1.5
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      if (p.cluster) {
        layer.bindPopup(
          `<b>${Number(p.count).toLocaleString()} listings in this map cell</b><br>` +
          `Average list: $${Math.round(p.listPrice || 0).toLocaleString()}<br>` +
          (p.zestimate > 0 ? `Average Zestimate: $${Math.round(p.zestimate).toLocaleString()}<br>` : '') +
          `Average neural: $${Math.round(p.predictedPrice || 0).toLocaleString()}<br>` +
          `Average LightGBM: $${Math.round(p.lightgbmPredictedPrice || 0).toLocaleString()}<br>` +
          `Average Lot Sqft: ${Math.round(p.sqftLot || 0).toLocaleString()}<br>` +
          `Zoom in to inspect individual listings.`,
          { autoPan: false }
        );
        return;
      }
      layer.bindPopup(
        `<b>Predicted at: ${p.predictionDate || p.saleDate || 'Unknown'}</b><br>` +
        `<b>${p.address}</b><br>` +
        `<i style="color:#3399ff">Zillow Listing</i><br>` +
        `List: $${Math.round(p.listPrice || 0).toLocaleString()}<br>` +
        (p.zestimate > 0
          ? `Zestimate: $${Math.round(p.zestimate).toLocaleString()}<br>`
          : `Zestimate: Not available<br>`) +
        `<b>Neural:</b> $${Math.round(p.predictedPrice).toLocaleString()} (${p.pctError.toFixed(1)}%)<br>` + modelIntervalRows(p, 'neural') +
        `<b>LightGBM:</b> $${Math.round(p.lightgbmPredictedPrice).toLocaleString()} (${p.lightgbmPctError.toFixed(1)}%)<br>` + modelIntervalRows(p, 'lightgbm') +
        `<b>Spatial GNN:</b> $${Math.round(p.gnnPredictedPrice || 0).toLocaleString()} (${(p.gnnPctError || 0).toFixed(1)}%)<br>` + modelIntervalRows(p, 'gnn') +
        `Beds: ${p.beds} | Baths: ${p.baths} | Sqft: ${Math.round(p.sqft).toLocaleString()}<br>` +
        `Lot Sqft: ${Math.round(p.sqftLot || 0).toLocaleString()}<br>` +
        waterFeatureRows(p) +
        `Type: ${p.homeType}<br>` +
        (p.url ? `<a href="${p.url}" target="_blank">View Listing</a><br>` : '') +
        `<button class="feature-explanation-button" type="button">Show feature contributions</button>` +
        `<button class="property-time-series-button" type="button">Show predicted value history</button>`,
        { autoPan: false }
      );
      attachPointExplanationButton(layer, p);
    }
  }).addTo(zillowLayer);
  // Zillow points also need a visible scale for the selected map colour.
  buildLegend(variable, values, anchors);
}
