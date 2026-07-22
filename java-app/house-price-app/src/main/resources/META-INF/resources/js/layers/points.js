// Detailed historical/RentCast/Zillow points, popups, and on-demand explanations.
// ── Sales points layer (RentCast / historical / both) ────────────────────────
function waterFeatureRows(properties) {
  const distance = Number(properties.distanceToWaterM);
  if (!Number.isFinite(distance)) return '';
  const formattedDistance = distance < 1000
    ? `${Math.round(distance).toLocaleString()} m`
    : `${(distance / 1000).toFixed(2)} km`;
  return `<b>Waterfront (≤50 m):</b> ${properties.isWaterfront ? 'Yes' : 'No'}<br>` +
    `Nearest water boundary: ${formattedDistance}<br>` +
    `Water proximity (τ=100 m): ${Math.exp(-distance / 100).toFixed(4)}<br>`;
}

async function loadRentcast() {
  rentcastLayer.clearLayers();
  const visible = document.getElementById('showRentcast').checked;
  setLayerGroupVisible(rentcastLayer, visible);
  if (!visible) { abortLayerRequest('rentcast'); return; }

  const variable  = document.getElementById('variable').value;
  const {min: minErr, max: maxErr} = getErrorRange();
  const minSqft   = document.getElementById('minSqft').value;
  const maxSqft   = document.getElementById('maxSqft').value;
  const homeType  = document.getElementById('salesType').value;
  const dateFrom  = document.getElementById('dateFrom').value;
  const dateTo    = document.getElementById('dateTo').value;
  const source    = document.getElementById('salesPointSource').value;

  const gj = await fetchLayerJson('rentcast',
    `/api/rentcast?source=${source}&minError=${minErr}&maxError=${maxErr}` +
    `&minSqft=${minSqft}&maxSqft=${maxSqft}&homeType=${homeType}` +
    `&dateFrom=${dateFrom}&dateTo=${dateTo}&${mapViewportQuery()}`
  );
  if (!gj) return;
  reportPointResponse('Sales points', gj);
  if (!gj.features || gj.features.length === 0) return;

  // Compute colour function from the loaded data (same approach as H3 layer)
  const values  = gj.features.map(f => pointVariableValue(f.properties, variable));
  const anchors = variable === 'pct_error' ? computeErrorAnchors(values) : null;
  const colorFn = getColorFn(variable, values, anchors);

  L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius:      f.properties.cluster
        ? Math.min(15, 5 + Math.log1p(f.properties.count || 1) * 1.5) : 6,
      fillColor:   colorFn(pointVariableValue(f.properties, variable)),
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
          `Zoom in to inspect individual sales.`
        );
        return;
      }
      const attnRow = (label, val) =>
        (val != null && val !== undefined) ? `${label}: ${(val*100).toFixed(1)}%<br>` : '';
      const srcLabel = p.source === 'sales' ? 'Historical Sale' : 'RentCast Sale';
      layer.bindPopup(
        `<b>${p.address || srcLabel}</b><br>` +
        `<i style="color:#00cc66">${srcLabel}</i><br>` +
        (p.saleDate ? `Sold: ${p.saleDate}<br>` : '') +
        `Sale Price: $${Math.round(p.salePrice).toLocaleString()}<br>` +
        `<b>Neural:</b> $${Math.round(p.predictedPrice).toLocaleString()} (${p.pctError.toFixed(1)}%)<br>` +
        `<b>LightGBM:</b> $${Math.round(p.lightgbmPredictedPrice).toLocaleString()} (${p.lightgbmPctError.toFixed(1)}%)<br>` +
        `95% CI: ±$${Math.round((p.predictionStdPrice ?? 0) * 1.96).toLocaleString()} (±${((p.predictionCvPct ?? 0) / 2).toFixed(1)}% of price)<br>` +
        `Beds: ${p.beds} | Baths: ${p.baths} | Sqft: ${Math.round(p.sqft).toLocaleString()}<br>` +
        waterFeatureRows(p) +
        `<b>CLS Attention:</b><br>` +
        attnRow('&nbsp; Community', p.attnCommunity) +
        attnRow('&nbsp; Year',      p.attnYear) +
        attnRow('&nbsp; Week',      p.attnWeek) +
        attnRow('&nbsp; Property',  p.attnProperty) +
        attnRow('&nbsp; Time',      p.attnTime) +
        attnRow('&nbsp; Market',    p.attnMarket) +
        `<button class="feature-explanation-button" type="button">Show feature contributions</button>`
      );
      attachPointExplanationButton(layer, p);
    }
  }).addTo(rentcastLayer);
}

// ── Zillow layer ──────────────────────────────────────────────────────────────
async function loadZillow() {
  zillowLayer.clearLayers();
  const visible = document.getElementById('showZillow').checked;
  setLayerGroupVisible(zillowLayer, visible);
  if (!visible) { abortLayerRequest('zillow'); return; }

  const variable  = document.getElementById('variable').value;
  const {min: minErr, max: maxErr} = getErrorRange();
  const minSqft   = document.getElementById('minSqft').value;
  const maxSqft   = document.getElementById('maxSqft').value;
  const homeType  = document.getElementById('zillowType').value;

  const gj = await fetchLayerJson('zillow',
    `/api/zillow?minError=${minErr}&maxError=${maxErr}` +
    `&minSqft=${minSqft}&maxSqft=${maxSqft}&homeType=${homeType}` +
    `&${mapViewportQuery()}`
  );
  if (!gj) return;
  reportPointResponse('Zillow', gj);
  if (!gj.features || !gj.features.length) return;

  const values  = gj.features.map(f => pointVariableValue(f.properties, variable));
  const anchors = variable === 'pct_error' ? computeErrorAnchors(values) : null;
  const colorFn = getColorFn(variable, values, anchors);

  L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius:      f.properties.cluster
        ? Math.min(15, 5 + Math.log1p(f.properties.count || 1) * 1.5) : 7,
      fillColor:   colorFn(pointVariableValue(f.properties, variable)),
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
          `Zoom in to inspect individual listings.`
        );
        return;
      }
      const attnRow = (label, val) =>
        (val != null && val !== undefined) ? `${label}: ${(val*100).toFixed(1)}%<br>` : '';
      layer.bindPopup(
        `<b>Predicted at: ${p.predictionDate || p.saleDate || 'Unknown'}</b><br>` +
        `<b>${p.address}</b><br>` +
        `<i style="color:#3399ff">Zillow Listing</i><br>` +
        `List: $${Math.round(p.listPrice || 0).toLocaleString()}<br>` +
        (p.zestimate > 0
          ? `Zestimate: $${Math.round(p.zestimate).toLocaleString()}<br>`
          : `Zestimate: Not available<br>`) +
        `<b>Neural:</b> $${Math.round(p.predictedPrice).toLocaleString()} (${p.pctError.toFixed(1)}%)<br>` +
        `<b>LightGBM:</b> $${Math.round(p.lightgbmPredictedPrice).toLocaleString()} (${p.lightgbmPctError.toFixed(1)}%)<br>` +
        `95% CI: ±$${Math.round((p.predictionStdPrice ?? 0) * 1.96).toLocaleString()} (±${((p.predictionCvPct ?? 0) / 2).toFixed(1)}% of price)<br>` +
        `Beds: ${p.beds} | Baths: ${p.baths} | Sqft: ${Math.round(p.sqft).toLocaleString()}<br>` +
        waterFeatureRows(p) +
        `Type: ${p.homeType}<br>` +
        `<b>CLS Attention:</b><br>` +
        attnRow('&nbsp; Community', p.attnCommunity) +
        attnRow('&nbsp; Year',      p.attnYear) +
        attnRow('&nbsp; Week',      p.attnWeek) +
        attnRow('&nbsp; Property',  p.attnProperty) +
        attnRow('&nbsp; Time',      p.attnTime) +
        attnRow('&nbsp; Market',    p.attnMarket) +
        (p.url ? `<a href="${p.url}" target="_blank">View Listing</a><br>` : '') +
        `<button class="feature-explanation-button" type="button">Show feature contributions</button>`
      );
      attachPointExplanationButton(layer, p);
    }
  }).addTo(zillowLayer);
}

