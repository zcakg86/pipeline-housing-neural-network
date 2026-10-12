// Detailed historical/RentCast/Zillow/Redfin points, popups, and explanations.
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

function propertyAttributes(p) {
  const values = [
    `Beds: ${escapeHtml(p.beds ?? '—')}`,
    `Baths: ${escapeHtml(p.baths ?? '—')}`,
    `Sqft: ${Number(p.sqft || 0).toLocaleString()}`
  ];
  if (Number(p.sqftLot) > 0) values.push(`Lot: ${Number(p.sqftLot).toLocaleString()} sqft`);
  if (p.homeType) values.push(`Type: ${escapeHtml(p.homeType)}`);
  return values.join(' · ');
}

function propertyEstimateHtml(p) {
  const estimate = (label, model, price, error) =>
    `<section class="property-popup-estimate"><b>${label}</b><br>` +
    `<span class="property-popup-estimate-price">$${Math.round(Number(price) || 0).toLocaleString()}</span> ` +
    `<span>(${(Number(error) || 0).toFixed(1)}%)</span><br>` +
    modelIntervalRows(p, model) + `</section>`;
  return estimate('SpatialGNN', 'gnn', p.gnnPredictedPrice, p.gnnPctError) +
    estimate('Transformer', 'neural', p.predictedPrice ?? p.neuralPredictedPrice, p.pctError) +
    estimate('LightGBM', 'lightgbm', p.lightgbmPredictedPrice, p.lightgbmPctError);
}

function propertyPopupHtml({ p, sourceLabel, accent, priceLabel, price, extraTop = '', details = '', buttons = '' }) {
  const listingUrl = typeof p.url === 'string' && /^https:\/\//.test(p.url)
    ? `<a href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer">View Listing</a>`
    : '<span class="property-popup-muted">Not available</span>';
  return `<div class="property-popup-card">` +
    `<div class="property-popup-summary">` +
      `<h3>${escapeHtml(p.address || sourceLabel)}</h3>` +
      `<div class="property-popup-source" style="color:${accent}">${escapeHtml(sourceLabel)}</div>` +
      `<div><b>Listing Link:</b> ${listingUrl}</div>` +
      `<div><b>${escapeHtml(priceLabel)}:</b> $${Math.round(Number(price) || 0).toLocaleString()}</div>` +
      extraTop +
      `<div><b>Property Attributes:</b> ${propertyAttributes(p)}</div>` +
    `</div>` +
    `<div class="property-popup-estimates-scroll" tabindex="0">` +
      `<div class="property-popup-prediction-date"><b>Predicted at:</b> ` +
        `${escapeHtml(p.predictionDate || p.saleDate || 'Unknown')}</div>` +
      propertyEstimateHtml(p) + details +
    `</div>` +
    `<div class="property-popup-actions">${buttons}</div>` +
  `</div>`;
}

function initializePropertyPopup(popup, latlng) {
  const container = popup.getElement();
  const card = container?.querySelector('.property-popup-card');
  if (!container || !card) return;
  popup._propertyPopupCleanup?.();
  L.DomEvent.disableClickPropagation(container);
  L.DomEvent.disableScrollPropagation(container);

  let userMoved = false;
  let initiallyPlaced = false;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  // Dragging deliberately has no frame-position limits.
  const setPosition = (left, top) => {
    container.style.bottom = 'auto';
    container.style.left = `${left}px`;
    container.style.top = `${top}px`;
  };

  const placeByPoint = () => {
    if (initiallyPlaced || userMoved || !popup.isOpen()) return;

    const size = map.getSize();
    const point = map.latLngToContainerPoint(latlng);
    const rect = container.getBoundingClientRect();
    const margin = 10;
    const gap = 18;

    // Open toward the larger portion of the visible map.
    let left = point.x < size.x / 2
      ? point.x + gap
      : point.x - rect.width - gap;
    let top = point.y < size.y / 2
      ? point.y + gap
      : point.y - rect.height - gap;

    // Only the initial position is kept within the visible map frame.
    left = clamp(left, margin, size.x - rect.width - margin);
    top = clamp(top, margin, size.y - rect.height - margin);

    // Leaflet positions popups with a combination of transforms and offsets.
    // Move from the rendered position to the desired visible position rather
    // than treating style.left/style.top as container coordinates.
    const target = map.containerPointToLayerPoint([left, top]);
    const viewportOrigin = map.containerPointToLayerPoint([0, 0]);
    const currentLeft = parseFloat(container.style.left) || 0;
    const currentTop = parseFloat(container.style.top) || 0;
    const mapElement = map.getContainer();
    const mapRect = mapElement.getBoundingClientRect();
    const actualLeft = rect.left - mapRect.left - mapElement.clientLeft + viewportOrigin.x;
    const actualTop = rect.top - mapRect.top - mapElement.clientTop + viewportOrigin.y;

    setPosition(
      currentLeft + target.x - actualLeft,
      currentTop + target.y - actualTop
    );
    initiallyPlaced = true;

    // Verify the rendered result after switching from Leaflet's bottom offset
    // to top positioning, then remove any residual CSS margin/transform error.
    requestAnimationFrame(() => {
      if (userMoved || !popup.isOpen()) return;
      const placedRect = container.getBoundingClientRect();
      const visibleLeft = placedRect.left - mapRect.left - mapElement.clientLeft;
      const visibleTop = placedRect.top - mapRect.top - mapElement.clientTop;
      const placedLeft = parseFloat(container.style.left) || 0;
      const placedTop = parseFloat(container.style.top) || 0;
      setPosition(
        placedLeft + left - visibleLeft,
        placedTop + top - visibleTop
      );
    });
  };

  // Ensure the rendered card can fit before measuring it for initial placement.
  const initialMapSize = map.getSize();
  const availableWidth = Math.max(180, initialMapSize.x - 20);
  const availableHeight = Math.max(180, initialMapSize.y - 20);
  const renderedCard = card.getBoundingClientRect();
  card.style.minWidth = `${Math.min(300, availableWidth)}px`;
  card.style.minHeight = `${Math.min(280, availableHeight)}px`;
  card.style.maxWidth = `${availableWidth}px`;
  card.style.maxHeight = `${availableHeight}px`;
  card.style.width = `${Math.min(renderedCard.width || 385, availableWidth)}px`;
  card.style.height = `${Math.min(renderedCard.height || 460, availableHeight)}px`;
  popup.update();
  requestAnimationFrame(() => requestAnimationFrame(placeByPoint));

  const beginDrag = event => {
    event.preventDefault();
    event.stopPropagation();
    userMoved = true;
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = parseFloat(container.style.left) || container.offsetLeft;
    const startTop = parseFloat(container.style.top) || container.offsetTop;
    const wasMapDragging = map.dragging.enabled();
    if (wasMapDragging) map.dragging.disable();
    document.body.classList.add('property-popup-interacting');

    const move = moveEvent => {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      setPosition(startLeft + dx, startTop + dy);
    };
    const end = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', end);
      document.removeEventListener('pointercancel', end);
      document.body.classList.remove('property-popup-interacting');
      if (wasMapDragging) map.dragging.enable();
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', end);
    document.addEventListener('pointercancel', end);
  };

  const pointerDown = event => {
    if (event.target.closest('a, button, input, select, textarea, summary, .property-popup-estimates-scroll')) return;
    beginDrag(event);
  };
  container.addEventListener('pointerdown', pointerDown);

  const cleanup = () => {
    container.removeEventListener('pointerdown', pointerDown);
    popup.off('remove', cleanup);
    popup._propertyPopupCleanup = null;
  };
  popup._propertyPopupCleanup = cleanup;
  popup.on('remove', cleanup);
}

function bindPropertyPopup(layer, html, properties) {
  layer.bindPopup(html, {
    autoPan: false,
    keepInView: false,
    className: 'property-map-popup',
    closeButton: true,
    maxWidth: 1000,
    minWidth: 0
  });
  layer.on('popupopen', event => initializePropertyPopup(event.popup, layer.getLatLng()));
  attachPointExplanationButton(layer, properties);
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
  const anchors = mapColorAnchors(variable, values);
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
          `Average SpatialGNN: $${Math.round(p.gnnPredictedPrice || 0).toLocaleString()}<br>` +
          `Average Transformer: $${Math.round(p.predictedPrice || 0).toLocaleString()}<br>` +
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
      bindPropertyPopup(layer, propertyPopupHtml({
        p, sourceLabel: srcLabel, accent: '#00a854', priceLabel: 'Sale Price', price: p.salePrice,
        extraTop: p.saleDate ? `<div><b>Sold:</b> ${escapeHtml(p.saleDate)}</div>` : '',
        details: waterFeatureRows(p) + localMarketWarning,
        buttons: `<button class="feature-explanation-button" type="button">Show feature contributions</button>` +
          `<button class="property-time-series-button" type="button">Show predicted value history</button>`
      }), p);
    }
  }).addTo(rentcastLayer);
  // Detailed sales use the same active map-colour scale as the H3 layer.
  buildLegend(variable, values, anchors);
}

// ── Redfin layer ─────────────────────────────────────────────────────────────
async function loadRedfin() {
  redfinLayer.clearLayers();
  const visible = document.getElementById('showRedfin').checked;
  setLayerGroupVisible(redfinLayer, visible);
  if (!visible) { abortLayerRequest('redfin'); return; }

  const variable = document.getElementById('variable').value;
  const model = displayedModel();
  const {min: minErr, max: maxErr} = getErrorRange();
  const minSqft = document.getElementById('minSqft').value;
  const maxSqft = document.getElementById('maxSqft').value;
  const homeType = document.getElementById('redfinType').value;
  const firstLoad = !loadedLiveSources.has('redfin');
  if (firstLoad) beginMapEvent('redfin-load', 'Loading saved Redfin listings and computing predictions…');

  let gj;
  try {
    gj = await fetchLayerJson('redfin',
      `/api/redfin?minError=${minErr}&maxError=${maxErr}` +
      `&minSqft=${minSqft}&maxSqft=${maxSqft}&homeType=${homeType}&model=${model}` +
      `&${mapViewportQuery()}`
    );
  } finally {
    if (firstLoad) completeMapEvent('redfin-load', gj ? 'Redfin layer ready' : '');
  }
  if (!gj) return;
  if (!loadedLiveSources.has('redfin')) {
    loadedLiveSources.add('redfin');
    loadHomeTypes();
    loadStats();
  }
  reportPointResponse('Redfin', gj);
  if (!gj.features || !gj.features.length) return;

  const values = gj.features.map(f => pointVariableValue(f.properties, variable, model));
  const anchors = mapColorAnchors(variable, values);
  const colorFn = getColorFn(variable, values, anchors);
  L.geoJSON(gj, {
    pointToLayer: (f, latlng) => L.circleMarker(latlng, {
      radius: f.properties.cluster
        ? Math.min(15, 5 + Math.log1p(f.properties.count || 1) * 1.5) : 7,
      fillColor: colorFn(pointVariableValue(f.properties, variable, model)),
      fillOpacity: 0.85,
      color: '#e53935',
      weight: 1.5
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      if (p.cluster) {
        layer.bindPopup(
          `<b>${Number(p.count).toLocaleString()} Redfin listings in this map cell</b><br>` +
          `Average list: $${Math.round(p.listPrice || 0).toLocaleString()}<br>` +
          `Zoom in to inspect individual listings.`,
          { autoPan: false }
        );
        return;
      }
      const facts = Array.isArray(p.atAGlanceFacts) && p.atAGlanceFacts.length
        ? `<b>At a glance</b><ul>${p.atAGlanceFacts.map(fact => `<li>${escapeHtml(fact)}</li>`).join('')}</ul>`
        : '';
      const description = p.description
        ? `<details><summary><b>Description</b></summary>${escapeHtml(p.description)}</details>`
        : '';
      bindPropertyPopup(layer, propertyPopupHtml({
        p, sourceLabel: 'Redfin Listing', accent: '#e53935', priceLabel: 'List Price', price: p.listPrice,
        details: facts + description,
        buttons: `<button class="price-assessment-button" type="button">Assess price with AI</button>` +
          `<button class="feature-explanation-button" type="button">Show feature contributions</button>` +
          `<button class="property-time-series-button" type="button">Show predicted value history</button>`
      }), p);
    }
  }).addTo(redfinLayer);
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
  const anchors = mapColorAnchors(variable, values);
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
          `Average SpatialGNN: $${Math.round(p.gnnPredictedPrice || 0).toLocaleString()}<br>` +
          `Average Transformer: $${Math.round(p.predictedPrice || 0).toLocaleString()}<br>` +
          `Average LightGBM: $${Math.round(p.lightgbmPredictedPrice || 0).toLocaleString()}<br>` +
          `Average Lot Sqft: ${Math.round(p.sqftLot || 0).toLocaleString()}<br>` +
          `Zoom in to inspect individual listings.`,
          { autoPan: false }
        );
        return;
      }
      bindPropertyPopup(layer, propertyPopupHtml({
        p, sourceLabel: 'Zillow Listing', accent: '#247dcc', priceLabel: 'List Price', price: p.listPrice,
        extraTop: `<div><b>Zestimate:</b> ${p.zestimate > 0
          ? `$${Math.round(p.zestimate).toLocaleString()}` : 'Not available'}</div>`,
        details: waterFeatureRows(p),
        buttons: `<button class="feature-explanation-button" type="button">Show feature contributions</button>` +
          `<button class="property-time-series-button" type="button">Show predicted value history</button>`
      }), p);
    }
  }).addTo(zillowLayer);
  // Zillow points also need a visible scale for the selected map colour.
  buildLegend(variable, values, anchors);
}
