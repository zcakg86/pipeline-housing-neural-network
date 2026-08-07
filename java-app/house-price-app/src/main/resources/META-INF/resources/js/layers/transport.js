// Static OSM stations, public-transport routes, and classified major roads.
const RAIL_STYLES = {
  light_rail: { label: 'Light rail station', color: '#e63946' },
  tram:       { label: 'Tram / streetcar stop', color: '#f4a261' },
  monorail:   { label: 'Monorail station', color: '#9b5de5' },
  subway:     { label: 'Subway station', color: '#7209b7' },
  train:      { label: 'Train station', color: '#4361ee' },
  rail_station: { label: 'Rail station', color: '#577590' }
};
const RAIL_ROUTE_STYLES = {
  light_rail: '#b5172c',
  tram: '#d97706',
  monorail: '#7b2cbf',
  train: '#2648b8'
};
const BUS_STYLES = {
  bus_station:    { label: 'Bus station', color: '#e76f51', radius: 6 },
  bus_stop:       { label: 'Bus stop', color: '#2a9d8f', radius: 3 },
  trolleybus_stop:{ label: 'Trolleybus stop', color: '#8338ec', radius: 3.5 }
};
const ROAD_STYLES = {
  motorway: { label: 'Motorway', color: '#d62828', weight: 4.0 },
  trunk:    { label: 'Trunk road', color: '#f77f00', weight: 3.4 },
  primary:  { label: 'Primary road', color: '#fcbf49', weight: 2.8 },
  secondary:{ label: 'Secondary road', color: '#3a86ff', weight: 2.1 }
};

const transportSources = {
  rail: {
    url: '/data/king_county_rail.geojson?v=osm-transport-v3',
    checkbox: 'showRail', layer: railLayer, data: null, loaded: false, legend: null,
    panes: ['railStationPane', 'railRoutePane']
  },
  bus: {
    url: '/data/king_county_bus.geojson?v=osm-transport-v1',
    checkbox: 'showBus', layer: busLayer, data: null, loaded: false, legend: null,
    panes: ['busStationPane', 'busRoutePane']
  },
  roads: {
    url: '/data/king_county_major_roads.geojson?v=osm-transport-v1',
    checkbox: 'showMajorRoads', layer: majorRoadLayer,
    data: null, loaded: false, legend: null, panes: ['majorRoadPane']
  }
};

function showTransportLayerError(error) {
  document.getElementById('stats').textContent = error.message;
}

function transportValueRow(label, value) {
  if (value == null || value === '' || (Array.isArray(value) && !value.length)) return '';
  const display = Array.isArray(value) ? value.join(', ') : value;
  return `${label}: ${escapeHtml(display)}<br>`;
}

function setTransportInteractivity(layer, enabled) {
  if (typeof layer.eachLayer === 'function') {
    layer.eachLayer(child => setTransportInteractivity(child, enabled));
  }
  if (layer.options && 'interactive' in layer.options) {
    layer.options.interactive = enabled;
    if (layer._path) {
      const action = enabled ? 'addClass' : 'removeClass';
      L.DomUtil[action](layer._path, 'leaflet-interactive');
    }
  }
  if (!enabled && typeof layer.closeTooltip === 'function') layer.closeTooltip();
}

function stationTooltip(properties, styles) {
  const style = styles[properties.category] || {
    label: properties.category || 'Transport station'
  };
  return `${properties.name ? `<b>${escapeHtml(properties.name)}</b><br>` : ''}` +
    `${escapeHtml(style.label)}<br>` +
    transportValueRow('Stop reference', properties.ref) +
    transportValueRow('Routes', properties.routes) +
    transportValueRow('Network', properties.network) +
    transportValueRow('Operator', properties.operator) +
    transportValueRow('Wheelchair access', properties.wheelchair) +
    transportValueRow('Shelter', properties.shelter) +
    transportValueRow('Bench', properties.bench) +
    `<span style="color:#888">OSM ${escapeHtml(properties.osmId)}</span>`;
}

function routeTooltip(properties) {
  const routeName = properties.name || properties.ref ||
    `${properties.routeType || 'transport'} route`;
  const journey = properties.from || properties.to
    ? `${escapeHtml(properties.from || 'Unknown origin')} → ` +
      `${escapeHtml(properties.to || 'Unknown destination')}<br>` : '';
  return `<b>${escapeHtml(routeName)}</b><br>` +
    `Category: ${escapeHtml(properties.routeType || 'route')} route<br>` +
    journey +
    transportValueRow('Reference', properties.ref) +
    transportValueRow('Via', properties.via) +
    transportValueRow('Network', properties.network) +
    transportValueRow('Operator', properties.operator) +
    `<span style="color:#888">OSM relation ${escapeHtml(properties.osmId)}</span>`;
}

function roadTooltip(properties) {
  const style = ROAD_STYLES[properties.category] || {
    label: properties.category || 'Major road'
  };
  return `${properties.name ? `<b>${escapeHtml(properties.name)}</b><br>` : ''}` +
    `${escapeHtml(style.label)}${properties.link ? ' link' : ''}<br>` +
    transportValueRow('Route reference', properties.ref) +
    transportValueRow('Lanes', properties.lanes) +
    transportValueRow('Maximum speed', properties.maxspeed) +
    transportValueRow('Surface', properties.surface) +
    transportValueRow('One way', properties.oneway) +
    transportValueRow('Bridge', properties.bridge) +
    transportValueRow('Tunnel', properties.tunnel) +
    transportValueRow('Access', properties.access) +
    transportValueRow('Busway', properties.busway) +
    `<span style="color:#888">OSM way ${escapeHtml(properties.osmId)}</span>`;
}

function stationMarker(feature, latlng, styles, renderer, pane) {
  const properties = feature.properties || {};
  const style = styles[properties.category] || {
    color: '#777', radius: styles === BUS_STYLES ? 3 : 5
  };
  return L.circleMarker(latlng, {
    renderer,
    pane,
    radius: style.radius || 5,
    color: '#fff',
    weight: 1,
    fillColor: style.color,
    fillOpacity: 0.92
  });
}

function addRailGeometry(data, layerGroup) {
  L.geoJSON(data, {
    // Route relations use SVG so every relation owns a reliable native hover
    // target. Station points continue to use the more efficient Canvas renderer.
    pane: 'railRoutePane',
    pointToLayer: (feature, latlng) => stationMarker(
      feature, latlng, RAIL_STYLES, railStationRenderer, 'railStationPane'
    ),
    style: feature => {
      const properties = feature.properties || {};
      if (properties.kind !== 'route') return {};
      return {
        color: RAIL_ROUTE_STYLES[properties.routeType] || '#495057',
        weight: properties.routeType === 'light_rail' ? 7 : 6,
        opacity: 0.62,
        interactive: true,
        bubblingMouseEvents: false,
        className: 'transport-route-path',
        lineCap: 'round',
        lineJoin: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      const properties = feature.properties || {};
      layer.bindTooltip(
        properties.kind === 'route'
          ? routeTooltip(properties) : stationTooltip(properties, RAIL_STYLES),
        {
          sticky: properties.kind === 'route',
          direction: properties.kind === 'route' ? 'top' : 'auto'
        }
      );
      if (properties.kind === 'route') {
        layer.on('mouseover', event => layer.openTooltip(event.latlng));
      }
    }
  }).addTo(layerGroup);
}

function addBusGeometry(data, layerGroup) {
  L.geoJSON(data, {
    // SVG is intentional here: Canvas route hit detection becomes unreliable
    // when hundreds of overlapping relations are removed and re-added.
    pane: 'busRoutePane',
    pointToLayer: (feature, latlng) => stationMarker(
      feature, latlng, BUS_STYLES, busStationRenderer, 'busStationPane'
    ),
    style: feature => {
      const properties = feature.properties || {};
      if (properties.kind !== 'route') return {};
      const trolleybus = properties.routeType === 'trolleybus';
      return {
        color: trolleybus ? '#8338ec' : '#168b78',
        weight: trolleybus ? 7 : 6,
        opacity: trolleybus ? 0.42 : 0.2,
        dashArray: trolleybus ? '5 3' : null,
        interactive: true,
        bubblingMouseEvents: false,
        className: 'transport-route-path',
        lineCap: 'round',
        lineJoin: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      const properties = feature.properties || {};
      layer.bindTooltip(
        properties.kind === 'route'
          ? routeTooltip(properties) : stationTooltip(properties, BUS_STYLES),
        {
          sticky: properties.kind === 'route',
          direction: properties.kind === 'route' ? 'top' : 'auto'
        }
      );
      if (properties.kind === 'route') {
        layer.on('mouseover', event => layer.openTooltip(event.latlng));
      }
    }
  }).addTo(layerGroup);
}

function addRoadGeometry(data, layerGroup) {
  L.geoJSON(data, {
    renderer: majorRoadRenderer,
    pane: 'majorRoadPane',
    style: feature => {
      const properties = feature.properties || {};
      const style = ROAD_STYLES[properties.category] || {
        color: '#777', weight: 1.5
      };
      return {
        color: style.color,
        weight: properties.link ? Math.max(1.2, style.weight - 0.8) : style.weight,
        opacity: properties.link ? 0.68 : 0.86,
        dashArray: properties.link ? '4 3' : null,
        lineCap: 'round',
        lineJoin: 'round'
      };
    },
    onEachFeature: (feature, layer) => {
      layer.bindTooltip(roadTooltip(feature.properties || {}), { sticky: true });
    }
  }).addTo(layerGroup);
}

function buildStaticLayerLegend(source, title, styles, routeCategories = []) {
  if (source.legend) map.removeControl(source.legend);
  source.legend = L.control({ position: 'bottomright' });
  source.legend.onAdd = () => {
    const div = L.DomUtil.create('div', 'osm-layer-legend');
    const counts = source.data?.metadata?.countsByCategory || {};
    div.innerHTML = `<b>${escapeHtml(title)}</b>`;
    Object.entries(styles).forEach(([category, style]) => {
      const count = Number(counts[category] || 0);
      if (!count) return;
      div.innerHTML +=
        `<div class="osm-legend-row"><span style="background:${style.color}"></span>` +
        `${escapeHtml(style.label)} <em>(${count.toLocaleString()})</em></div>`;
    });
    if (routeCategories.length) {
      const routeCount = routeCategories.reduce(
        (total, category) => total + Number(counts[category] || 0), 0
      );
      div.innerHTML +=
        `<div class="osm-legend-row"><span class="route-swatch"></span>` +
        `Route geometry <em>(${routeCount.toLocaleString()})</em></div>`;
    }
    div.innerHTML += '<div class="osm-attribution">© OpenStreetMap contributors</div>';
    return div;
  };
  source.legend.addTo(map);
}

async function loadStaticTransportSource(name, addGeometry, title, styles, routes = []) {
  const source = transportSources[name];
  const visible = document.getElementById(source.checkbox).checked;
  // Keep vector renderers mounted after their first load. Leaflet can lose the
  // Canvas/SVG event surface when a populated group is detached and reattached.
  // Pane visibility is cheap and preserves all bound tooltip handlers.
  if (visible && !map.hasLayer(source.layer)) map.addLayer(source.layer);
  if (source.loaded) {
    source.panes.forEach(paneName => {
      map.getPane(paneName).style.display = visible ? '' : 'none';
    });
    setTransportInteractivity(source.layer, visible);
  }
  if (!visible) {
    if (source.legend) {
      map.removeControl(source.legend);
      source.legend = null;
    }
    return;
  }
  if (!source.data) {
    const response = await fetch(source.url);
    if (!response.ok) {
      throw new Error(`${title} layer failed to load: ${response.status}`);
    }
    source.data = await response.json();
  }
  if (!source.loaded) {
    addGeometry(source.data, source.layer);
    source.loaded = true;
    setTransportInteractivity(source.layer, true);
    source.panes.forEach(paneName => {
      map.getPane(paneName).style.display = '';
    });
  }
  buildStaticLayerLegend(source, title, styles, routes);
}

function loadRailLayer() {
  return loadStaticTransportSource(
    'rail', addRailGeometry, 'Rail and light rail', RAIL_STYLES,
    ['light_rail_route', 'tram_route', 'monorail_route', 'train_route']
  );
}

function loadBusLayer() {
  return loadStaticTransportSource(
    'bus', addBusGeometry, 'Bus stations and routes', BUS_STYLES,
    ['bus_route', 'trolleybus_route']
  );
}

function loadMajorRoadLayer() {
  return loadStaticTransportSource(
    'roads', addRoadGeometry, 'Motorways and major roads', ROAD_STYLES
  );
}
