// Static OSM-derived transport accessibility fields projected onto H3 L8 cells.
let transportFeatureGeoJson = null;
let transportFeatureData = null;

const TRANSPORT_FEATURE_LABELS = {
  light_rail_proximity: 'Light-rail proximity',
  bus_stop_density_per_sq_km: 'Bus-stop density (per km²)',
  nearby_bus_route_count: 'Distinct nearby bus-route count',
  bus_centrality_score: 'Bus centrality score',
  motorway_connector_proximity: 'Motorway-connector proximity',
  major_road_accessibility: 'Major-road accessibility',
  major_road_density_km_per_sq_km: 'Major-road density (km per km²)'
};

function transportFeatureName(feature) {
  return TRANSPORT_FEATURE_LABELS[feature] || feature.replaceAll('_', ' ');
}

function formatTransportFeatureValue(feature, value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'Unavailable';
  if (feature === 'nearby_bus_route_count') return Math.round(numeric).toLocaleString();
  if (feature.endsWith('_density_per_sq_km')) return numeric.toFixed(2);
  return numeric.toFixed(3);
}

function transportFeatureTooltip(properties) {
  const rows = Object.keys(TRANSPORT_FEATURE_LABELS).map(feature =>
    `${escapeHtml(transportFeatureName(feature))}: ` +
    `<b>${escapeHtml(formatTransportFeatureValue(feature, properties[feature]))}</b>`
  ).join('<br>');
  return `<div class="transport-feature-tooltip"><b>Transport accessibility · H3 L8</b><br>` +
    `<span class="shap-value">${escapeHtml(properties.h3Index)}</span><br><br>${rows}</div>`;
}

function buildTransportFeatureLegend(feature, values, metadata) {
  if (transportFeatureLegendControl) map.removeControl(transportFeatureLegendControl);
  const low = pct(values, 5);
  const high = pct(values, 95);
  transportFeatureLegendControl = L.control({ position: 'bottomright' });
  transportFeatureLegendControl.onAdd = () => {
    const div = L.DomUtil.create('div', 'osm-layer-legend');
    const definition = metadata?.definitions?.[feature];
    div.innerHTML = `<b>Transport accessibility</b><br>` +
      `<span>${escapeHtml(transportFeatureName(feature))}</span><br>` +
      `<div class="transport-feature-gradient"></div>` +
      `<div class="transport-feature-range"><span>${escapeHtml(formatTransportFeatureValue(feature, low))}</span>` +
      `<span>${escapeHtml(formatTransportFeatureValue(feature, high))}</span></div>` +
      (definition ? `<div class="osm-attribution">${escapeHtml(definition)}</div>` : '') +
      '<div class="osm-attribution">Static OSM-derived field · not used by current price models</div>';
    return div;
  };
  transportFeatureLegendControl.addTo(map);
}

async function loadTransportFeatureLayer() {
  const visible = document.getElementById('showTransportFeatures').checked;
  setLayerGroupVisible(transportFeatureLayer, visible);
  if (!visible) {
    if (transportFeatureLegendControl) {
      map.removeControl(transportFeatureLegendControl);
      transportFeatureLegendControl = null;
    }
    return;
  }
  if (!transportFeatureData) {
    const response = await fetch('/api/transport-features');
    if (!response.ok) {
      throw new Error(`Transport feature layer failed to load: ${response.status}`);
    }
    transportFeatureData = await response.json();
  }

  const feature = document.getElementById('transportFeature').value;
  const values = transportFeatureData.features
    .map(item => Number(item.properties?.[feature]))
    .filter(Number.isFinite);
  if (!values.length) throw new Error(`Transport feature '${feature}' has no numeric values`);
  const color = getColorFn(feature, values, null);

  transportFeatureLayer.clearLayers();
  transportFeatureGeoJson = L.geoJSON(transportFeatureData, {
    renderer: syntheticRenderer,
    style: item => ({
      color: '#fff', weight: 0.35, opacity: 0.42,
      fillColor: color(Number(item.properties?.[feature])), fillOpacity: 0.68
    }),
    onEachFeature: (item, layer) => {
      layer.bindTooltip(transportFeatureTooltip(item.properties || {}), {
        sticky: true, direction: 'top'
      });
    }
  }).addTo(transportFeatureLayer);
  buildTransportFeatureLegend(feature, values, transportFeatureData.metadata);
}
