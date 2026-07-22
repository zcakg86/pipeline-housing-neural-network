// Static OSM water geometry and independent category visibility controls.
// ── OSM natural-water and coastline layer ───────────────────────────────────
const WATER_STYLES = {
  natural_water: { label: 'Natural water', color: '#277da1', fill: '#4ea8de', weight: 1.2 },
  coastline:     { label: 'Coastline',     color: '#083d77', fill: '#083d77', weight: 3.0 }
};
const EXCLUDED_WATERWAY_TYPES = new Set([
  'stream', 'drain', 'ditch', 'weir'
]);
let waterGeoJson = null;
let waterLayerLoaded = false;

function waterStyle(feature) {
  const category = feature.properties.category;
  const style = WATER_STYLES[category] || WATER_STYLES.natural_water;
  const geometryType = feature.geometry?.type || '';
  const isArea = geometryType.includes('Polygon');
  return {
    color: style.color,
    opacity: 0.95,
    weight: isArea ? Math.max(1, style.weight) : style.weight,
    fillColor: style.fill,
    fillOpacity: isArea ? 0.42 : 0,
    lineCap: 'round',
    lineJoin: 'round'
  };
}

function isRenderableWaterFeature(feature) {
  const properties = feature.properties || {};
  if (properties.category === 'coastline') return properties.type === 'coastline';
  if (properties.category !== 'natural_water') return false;
  return !properties.waterway &&
    !EXCLUDED_WATERWAY_TYPES.has(String(properties.type || '').toLowerCase()) &&
    String(feature.geometry?.type || '').includes('Polygon');
}

function buildWaterLegend(metadata) {
  if (waterLegendControl) map.removeControl(waterLegendControl);
  waterLegendControl = L.control({ position: 'bottomright' });
  waterLegendControl.onAdd = () => {
    const div = L.DomUtil.create('div');
    div.style.cssText =
      'background:rgba(22,33,62,0.94);padding:10px 12px;border-radius:6px;' +
      'border:1px solid #0f3460;color:#eee;font-size:11px;line-height:1.45;min-width:170px';
    const counts = metadata?.countsByCategory || {};
    div.innerHTML = '<b style="color:#e94560">OSM water features</b>';
    Object.entries(WATER_STYLES).forEach(([category, style]) => {
      const count = Number(counts[category] || 0).toLocaleString();
      div.innerHTML +=
        `<div style="display:flex;align-items:center;gap:7px;margin-top:6px">` +
        `<span style="display:inline-block;width:20px;height:10px;background:${style.fill};` +
        `border:2px solid ${style.color}"></span>` +
        `<span>${style.label} <span style="color:#999">(${count})</span></span></div>`;
    });
    div.innerHTML += '<div style="color:#777;margin-top:7px">© OpenStreetMap contributors</div>';
    return div;
  };
  waterLegendControl.addTo(map);
}

async function loadWaterLayer() {
  const visible = document.getElementById('showWater').checked;
  setLayerGroupVisible(waterLayer, visible);
  if (!visible) {
    if (waterLegendControl) {
      map.removeControl(waterLegendControl);
      waterLegendControl = null;
    }
    return;
  }

  if (!waterGeoJson) {
    const response = await fetch(
      '/data/king_county_water.geojson?v=river-canal-areas-coastline-v5',
      { cache: 'no-store' }
    );
    if (!response.ok) throw new Error(`Water layer failed to load: ${response.status}`);
    waterGeoJson = await response.json();
  }
  if (!waterLayerLoaded) {
    L.geoJSON(waterGeoJson, {
      renderer: waterRenderer,
      filter: isRenderableWaterFeature,
      style: waterStyle,
      onEachFeature: (feature, layer) => {
        const properties = feature.properties || {};
        const category = WATER_STYLES[properties.category]?.label || 'Water feature';
        const name = properties.name ? `<b>${escapeHtml(properties.name)}</b><br>` : '';
        const intermittent = properties.intermittent
          ? `<br>Intermittent: ${escapeHtml(properties.intermittent)}` : '';
        layer.bindTooltip(
          `${name}${category}<br>Type: ${escapeHtml(properties.type || 'unknown')}` +
          `${intermittent}<br><span style="color:#888">OSM ${escapeHtml(properties.osmId)}</span>`,
          { sticky: true }
        );
      }
    }).addTo(waterLayer);
    waterLayerLoaded = true;
  }
  buildWaterLegend(waterGeoJson.metadata);
}

