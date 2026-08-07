// Shared HTTP boundary: cancellation, filter serialization, and stale-request safety.
const layerAbortControllers = new Map();
const loadedLiveSources = new Set();
const activeMapEvents = new Map();
let interactionSequence = 0;

function beginMapEvent(key, message) {
  activeMapEvents.set(key, message);
  const progress = document.getElementById('mapProgress');
  progress.style.display = 'block';
  progress.setAttribute('aria-hidden', 'false');
  document.getElementById('mapProgressText').textContent = message;
}

function completeMapEvent(key, message = '') {
  activeMapEvents.delete(key);
  const progress = document.getElementById('mapProgress');
  if (activeMapEvents.size) {
    document.getElementById('mapProgressText').textContent = [...activeMapEvents.values()].at(-1);
    return;
  }
  if (message) document.getElementById('mapProgressText').textContent = message;
  setTimeout(() => {
    if (!activeMapEvents.size) {
      progress.style.display = 'none';
      progress.setAttribute('aria-hidden', 'true');
    }
  }, message ? 900 : 0);
}

/** Show and persist a concise user action without blocking the map interaction. */
function reportMapInteraction(action, detail = '') {
  const message = detail ? `${action}: ${detail}` : action;
  console.info(`[House Price Map] ${message}`);
  fetch('/api/events', {
    method: 'POST', keepalive: true,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, detail })
  }).catch(() => {});
  const key = `interaction-${++interactionSequence}`;
  beginMapEvent(key, message);
  setTimeout(() => completeMapEvent(key), 1100);
}

function elementInteractionDetail(element) {
  if (!element) return 'map';
  const label = element.labels?.[0]?.textContent?.trim()
    || element.getAttribute('aria-label') || element.textContent?.trim() || element.id || element.tagName;
  return label.replace(/\s+/g, ' ').slice(0, 120);
}

// Controls, popup buttons, and map interactions all produce an operational
// event. Slider input is logged on `change`, not on every drag frame.
document.addEventListener('click', event => {
  const target = event.target.closest('button, input[type=checkbox], select, .leaflet-interactive');
  if (target) reportMapInteraction('click', elementInteractionDetail(target));
}, true);
document.addEventListener('change', event => {
  const target = event.target;
  if (target.matches('select, input[type=range], input[type=date], input[type=checkbox]')) {
    reportMapInteraction('change', `${elementInteractionDetail(target)} = ${target.value}`);
  }
}, true);
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
    const existing = new Set([...sel.options].map(option => option.value));
    types.forEach(t => {
      if (existing.has(t)) return;
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
  reportMapInteraction(`${source} API fetch requested`, 'saving response and ingesting changed records');
  const progressKey = `api-fetch-${source}`;
  beginMapEvent(progressKey, `Fetching ${source} API, saving its response, and ingesting records…`);
  try {
    const res = await fetch(`/api/fetch/${source}?confirm=true`, { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      el.textContent = `✓ ${source}: ${data.fetched} records. Next in ${data.nextAllowedIn}`;
      pointExplanationCache.clear();
      loadHomeTypes(); loadSales(); loadZillow(); loadRentcast(); loadStats();
      completeMapEvent(progressKey, `${source} API fetch and ingestion complete`);
    } else {
      el.textContent = `✗ ${data.message}`;
      completeMapEvent(progressKey);
    }
  } catch (e) {
    completeMapEvent(progressKey);
    el.textContent = `✗ ${e.message}`;
  }
}
