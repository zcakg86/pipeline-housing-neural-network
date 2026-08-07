// Leaflet initialization and geographic helpers shared by all map layers.
// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map('map').setView([47.61, -122.33], 11);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OpenStreetMap © CARTO', maxZoom: 19
}).addTo(map);

let salesLayer    = L.layerGroup();
let zillowLayer   = L.layerGroup().addTo(map);
let rentcastLayer = L.layerGroup();
let communityLayer = L.layerGroup();
let syntheticLayer = L.layerGroup();
let waterLayer = L.layerGroup();
let railLayer = L.layerGroup();
let busLayer = L.layerGroup();
let majorRoadLayer = L.layerGroup();
let transportFeatureLayer = L.layerGroup();
let lightgbmExplanationCache = new Map();
let pointExplanationCache = new Map();
const syntheticRenderer = L.canvas({ padding: 0.5 });
const waterRenderer = L.canvas({ padding: 0.5 });
map.createPane('majorRoadPane');
map.getPane('majorRoadPane').style.zIndex = 420;
map.createPane('railStationPane');
map.getPane('railStationPane').style.zIndex = 430;
map.createPane('busStationPane');
map.getPane('busStationPane').style.zIndex = 431;
map.createPane('railRoutePane');
map.getPane('railRoutePane').style.zIndex = 440;
map.createPane('busRoutePane');
map.getPane('busRoutePane').style.zIndex = 441;
const majorRoadRenderer = L.canvas({
  padding: 0.5, pane: 'majorRoadPane', tolerance: 7
});
const railStationRenderer = L.canvas({
  padding: 0.5, pane: 'railStationPane', tolerance: 5
});
const busStationRenderer = L.canvas({
  padding: 0.5, pane: 'busStationPane', tolerance: 5
});

// ── Colormaps ─────────────────────────────────────────────────────────────────
const COLORMAPS = {
  YlOrRd: [[255,255,204],[255,237,160],[254,217,118],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[177,0,38]],
  Blues:  [[239,243,255],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107]]
};

function lerp(a, b, t) { return Math.round(a + (b - a) * t); }

function interpolate(t, palette) {
  const n = palette.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const [r0,g0,b0] = palette[i], [r1,g1,b1] = palette[i+1];
  return `rgb(${lerp(r0,r1,f)},${lerp(g0,g1,f)},${lerp(b0,b1,f)})`;
}

function pct(arr, p) {
  const s = [...arr].sort((a,b)=>a-b);
  return s[Math.floor(p/100*(s.length-1))];
}

// ── Diverging error colour scale ──────────────────────────────────────────────
// Anchors (populated from data each load):
//   p10  → #2166AC  dark blue
//   0    → #e0e0e0  neutral
//   p90  → #B2182B  dark red
// Values outside [p10, p90] are clamped to the anchor colours.
// Sign is preserved: negative = blue side, positive = red side.
const ERROR_PALETTE = {
  darkBlue:    [33,  102, 172],   // #2166AC  ≤ p10
  softBlue:    [67,  162, 202],   // #43a2ca
  neutral:     [224, 224, 224],   // #e0e0e0  zero
  softRed:     [227,  74,  51],   // #e34a33
  darkRed:     [178,  24,  43],   // #B2182B  ≥ p90
};

// Compute percentile anchors from the current set of error values
function computeErrorAnchors(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const p = p => sorted[Math.max(0, Math.floor(p / 100 * (sorted.length - 1)))];
  return { p10: p(10), p90: p(90) };
}

function errorColor(v, anchors) {
  const { p10, p90 } = anchors;
  const { darkBlue, softBlue, neutral, softRed, darkRed } = ERROR_PALETTE;

  if (v <= 0) {
    // Blue side: p10 (or below) → neutral (0)
    const lo = Math.min(p10, 0);   // never let lo > 0
    if (lo >= 0 || v <= lo) return `rgb(${darkBlue.join(',')})`;
    const t = (v - lo) / (0 - lo);   // 0 at lo, 1 at 0
    // t=0 → darkBlue, t=0.5 → softBlue, t=1 → neutral
    if (t < 0.5) {
      const s = t * 2;
      return `rgb(${lerp(darkBlue[0],softBlue[0],s)},${lerp(darkBlue[1],softBlue[1],s)},${lerp(darkBlue[2],softBlue[2],s)})`;
    } else {
      const s = (t - 0.5) * 2;
      return `rgb(${lerp(softBlue[0],neutral[0],s)},${lerp(softBlue[1],neutral[1],s)},${lerp(softBlue[2],neutral[2],s)})`;
    }
  } else {
    // Red side: neutral (0) → p90 (or above)
    const hi = Math.max(p90, 0);
    if (hi <= 0 || v >= hi) return `rgb(${darkRed.join(',')})`;
    const t = v / hi;   // 0 at 0, 1 at hi
    // t=0 → neutral, t=0.5 → softRed, t=1 → darkRed
    if (t < 0.5) {
      const s = t * 2;
      return `rgb(${lerp(neutral[0],softRed[0],s)},${lerp(neutral[1],softRed[1],s)},${lerp(neutral[2],softRed[2],s)})`;
    } else {
      const s = (t - 0.5) * 2;
      return `rgb(${lerp(softRed[0],darkRed[0],s)},${lerp(softRed[1],darkRed[1],s)},${lerp(softRed[2],darkRed[2],s)})`;
    }
  }
}

function getErrorRange() {
  const noMin = document.getElementById('noFilterMin').checked;
  const noMax = document.getElementById('noFilterMax').checked;
  return {
    min: noMin ? -99999 : parseInt(document.getElementById('minError').value),
    max: noMax ?  99999 : parseInt(document.getElementById('maxError').value)
  };
}
function getColorFn(variable, values, anchors) {
  if (variable === 'pct_error') return v => errorColor(v, anchors);
  const lo = pct(values, 5), hi = pct(values, 95), range = hi - lo || 1;
  // Attention weights and uncertainty use sequential palettes
  const palette = (
    variable === 'sqft' || variable === 'sqft_lot' ||
    variable === 'pred_std' || variable === 'pred_cv_pct'
  )
    ? COLORMAPS.Blues
    : COLORMAPS.YlOrRd;
  return v => interpolate(Math.max(0, Math.min(1, (v - lo) / range)), palette);
}

// ── Legend ────────────────────────────────────────────────────────────────────
let legendControl = null;
let communityLegendControl = null;
let waterLegendControl = null;
let transportFeatureLegendControl = null;
let communityColorMap = new Map();

function displayedModel() {
  return document.getElementById('DisplayedModel').value;
}

function synchronizeMapVariableOptions() {
  const model = displayedModel();
  const variable = document.getElementById('variable');
  [...variable.options].forEach(option => {
    const unavailable = option.dataset.model && option.dataset.model !== model;
    option.disabled = unavailable;
    option.hidden = unavailable;
  });
  if (variable.selectedOptions[0]?.disabled) variable.value = 'predicted_price';
}

function ptKey(pt) {
  return `${pt[0].toFixed(6)},${pt[1].toFixed(6)}`;
}

function segmentKey(a, b) {
  const aKey = ptKey(a);
  const bKey = ptKey(b);
  return aKey < bKey ? `${aKey}|${bKey}` : `${bKey}|${aKey}`;
}

function polygonRings(feature) {
  const geom = feature.geometry;
  if (!geom || !geom.coordinates) return [];
  if (geom.type === 'Polygon') return geom.coordinates;
  if (geom.type === 'MultiPolygon') return geom.coordinates.flat();
  return [];
}

function buildCommunityAdjacency(features) {
  const segmentOwners = new Map();
  const adjacency = new Map();

  features.forEach(feature => {
    const id = String(feature.properties.community);
    adjacency.set(id, new Set());
    polygonRings(feature).forEach(ring => {
      for (let i = 0; i + 1 < ring.length; i++) {
        const key = segmentKey(ring[i], ring[i + 1]);
        if (!segmentOwners.has(key)) segmentOwners.set(key, new Set());
        segmentOwners.get(key).add(id);
      }
    });
  });

  segmentOwners.forEach(owners => {
    if (owners.size <= 1) return;
    const ids = [...owners];
    ids.forEach(a => ids.forEach(b => {
      if (a !== b) adjacency.get(a).add(b);
    }));
  });

  return adjacency;
}
