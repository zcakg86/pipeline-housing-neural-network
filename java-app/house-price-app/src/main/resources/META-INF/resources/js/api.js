// Shared HTTP boundary: cancellation, filter serialization, and stale-request safety.
const layerAbortControllers = new Map();
async function fetchLayerJson(key, url) {
  const previous = layerAbortControllers.get(key);
  if (previous) previous.abort();
  const controller = new AbortController();
  layerAbortControllers.set(key, controller);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`${key} request failed (${response.status})`);
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') return null;
    throw error;
  } finally {
    if (layerAbortControllers.get(key) === controller) {
      layerAbortControllers.delete(key);
    }
  }
}

function abortLayerRequest(key) {
  const controller = layerAbortControllers.get(key);
  if (controller) controller.abort();
}

function mapViewportQuery() {
  const bounds = map.getBounds();
  const params = new URLSearchParams({
    west: bounds.getWest().toFixed(6),
    south: bounds.getSouth().toFixed(6),
    east: bounds.getEast().toFixed(6),
    north: bounds.getNorth().toFixed(6),
    zoom: String(map.getZoom())
  });
  return params.toString();
}

function reportPointResponse(label, response) {
  if (!response) return;
  if (response.truncated) {
    document.getElementById('fetchStatus').textContent =
      `${label}: showing ${response.returned.toLocaleString()} of ` +
      `${response.matched.toLocaleString()} points; zoom in for all points.`;
  } else if (response.clustered) {
    document.getElementById('fetchStatus').textContent =
      `${label}: ${response.matched.toLocaleString()} records grouped into ` +
      `${response.returned.toLocaleString()} map cells.`;
  }
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  const res = await fetch('/api/stats');
  const s = await res.json();
  let html = '';
  if (s.zillow) html +=
    `<b>Zillow Listings</b><br>Total: ${s.zillow.total}<br>` +
    `Avg List: $${Math.round(s.zillow.avgListPrice).toLocaleString()}<br>` +
    `Avg Neural: $${Math.round(s.zillow.avgNeuralPredicted).toLocaleString()}<br>` +
    `Avg LightGBM: $${Math.round(s.zillow.avgLightgbmPredicted).toLocaleString()}<br><br>`;
  if (s.sales) html +=
    `<b>Historical Sales</b><br>Total: ${s.sales.total}<br>` +
    `Unique Hexes: ${s.sales.uniqueHexes}<br>` +
    `Neural MAE: ${s.sales.neuralAvgAbsError.toFixed(1)}%<br>` +
    `LightGBM MAE: ${s.sales.lightgbmAvgAbsError.toFixed(1)}%<br><br>`;
  if (s.rentcast) html +=
    `<b>Recent Sales</b><br>Total: ${s.rentcast.total}<br>` +
    `Unique Hexes: ${s.rentcast.uniqueHexes}<br>` +
    `Unique Sales: ${s.rentcast.uniqueSales}<br>`+
    `Neural MAE: ${s.rentcast.neuralAvgAbsError.toFixed(1)}%<br>` +
    `LightGBM MAE: ${s.rentcast.lightgbmAvgAbsError.toFixed(1)}%`;
  document.getElementById('stats').innerHTML = html || 'No data loaded yet.';
}

// ── Populate home type dropdowns ──────────────────────────────────────────────
async function loadHomeTypes() {
  const res = await fetch('/api/home-types');
  const d = await res.json();
  const fill = (id, types) => {
    const sel = document.getElementById(id);
    types.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t.replace(/_/g,' ');
      sel.appendChild(opt);
    });
  };
  fill('zillowType', d.zillow || []);
  fill('salesType',  d.sales  || []);
}

// ── API fetch ─────────────────────────────────────────────────────────────────
async function fetchApi(source) {
  const el = document.getElementById('fetchStatus');
  const statusRes = await fetch('/api/fetch/status');
  const status = await statusRes.json();
  const info = status[source];
  if (info.cooldownActive) {
    el.textContent = `⏳ ${source}: cooldown active, ${info.minutesRemaining} min remaining`;
    return;
  }
  el.textContent = `Fetching ${source}... (billed call)`;
  try {
    const res = await fetch(`/api/fetch/${source}?confirm=true`, { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      el.textContent = `✓ ${source}: ${data.fetched} records. Next in ${data.nextAllowedIn}`;
      pointExplanationCache.clear();
      loadHomeTypes(); loadSales(); loadZillow(); loadRentcast(); loadStats();
    } else {
      el.textContent = `✗ ${data.message}`;
    }
  } catch (e) { el.textContent = `✗ ${e.message}`; }
}

