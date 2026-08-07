// Chart rendering and the shared, stable community color vocabulary.
// ── Community colormap (10 distinct colours shared by map + chart) ────────────
const COMMUNITY_COLORS = [
  '#e94560','#4a9eff','#00cc66','#ffcc00','#ff6b35',
  '#a855f7','#06b6d4','#f97316','#84cc16','#ec4899'
];
let selectedCommunities = new Set(); // explicit chart/map selection; empty = none

function buildCommunityColorMap(features) {
  const adjacency = buildCommunityAdjacency(features);
  const ids = [...adjacency.keys()];
  ids.sort((a, b) => (adjacency.get(b)?.size || 0) - (adjacency.get(a)?.size || 0));

  const map = new Map();
  const usage = new Map(COMMUNITY_COLORS.map(color => [color, 0]));

  ids.forEach((id) => {
    const used = new Set();
    (adjacency.get(id) || []).forEach(neighbor => {
      if (map.has(neighbor)) used.add(map.get(neighbor));
    });

    const available = COMMUNITY_COLORS.filter(color => !used.has(color));
    let assigned;
    if (available.length) {
      assigned = available.reduce((best, color) =>
        (usage.get(color) < usage.get(best) ? color : best), available[0]
      );
    } else {
      assigned = COMMUNITY_COLORS[0];
    }

    map.set(id, assigned);
    usage.set(assigned, usage.get(assigned) + 1);
  });

  return map;
}

let communityGeoJson = null;
let communityGeoJsonPromise = null;

async function ensureCommunityColorMap() {
  if (!communityGeoJsonPromise) {
    beginMapEvent('community-geometry', 'Preparing community map geometry…');
    communityGeoJsonPromise = fetch('/api/sales/community')
      .then(response => response.json())
      .then(value => {
        completeMapEvent('community-geometry', 'Community map geometry ready');
        return value;
      })
      .catch(error => {
        completeMapEvent('community-geometry');
        throw error;
      });
  }
  communityGeoJson = await communityGeoJsonPromise;
  if (!communityColorMap.size) {
    communityColorMap = buildCommunityColorMap(communityGeoJson.features || []);
  }
  return communityColorMap;
}

function buildCommunityLegend(colorMap) {
  if (communityLegendControl) map.removeControl(communityLegendControl);
  communityLegendControl = L.control({ position: 'topright' });
  communityLegendControl.onAdd = () => {
    const div = L.DomUtil.create('div');
    div.style.cssText =
      'background:rgba(22,33,62,0.93);padding:10px 10px;border-radius:6px;' +
      'border:1px solid #0f3460;color:#eee;font-size:11px;line-height:1.4;max-height:300px;overflow:auto;min-width:180px';
    div.innerHTML = `<b style="color:#e94560">Community colors</b><br>`;

    let count = 0;
    for (const [community, color] of colorMap.entries()) {
      if (count >= 18) {
        div.innerHTML += `<div style="color:#888; margin-top:6px">...+${colorMap.size - 18} more</div>`;
        break;
      }
      div.innerHTML +=
        `<div style="display:flex;align-items:center;gap:6px;margin:5px 0">` +
          `<span style="display:inline-block;width:16px;height:12px;background:${color};border-radius:3px;flex-shrink:0"></span>` +
          `<span>Community ${community}</span>` +
        `</div>`;
      count += 1;
    }

    return div;
  };
  communityLegendControl.addTo(map);
}

// ── Community map layer ───────────────────────────────────────────────────────
let perfData = null; // cached performance response
let perfDefaultsInitialized = false;

async function ensurePerformanceData() {
  if (!perfData) {
    beginMapEvent('community-performance', 'Calculating community-level trends…');
    try {
      perfData = await (await fetch('/api/performance')).json();
      completeMapEvent('community-performance', 'Community-level trends ready');
    } catch (error) {
      completeMapEvent('community-performance');
      throw error;
    }
  }
  if (!perfDefaultsInitialized) {
    selectedCommunities = new Set(
      (perfData.defaultCommunities || []).map(String)
    );
    perfDefaultsInitialized = true;
  }
  populateCommunitySelector();
  return perfData;
}

function populateCommunitySelector() {
  if (!perfData) return;
  const select = document.getElementById('chartCommunities');
  const currentOptions = [...select.options].map(option => option.value);
  const communities = (perfData.communities || []).map(item => String(item.id));
  if (currentOptions.join(',') !== communities.join(',')) {
    select.innerHTML = '';
    (perfData.communities || []).forEach(item => {
      const option = document.createElement('option');
      option.value = String(item.id);
      option.textContent = `Community ${item.id} (${item.count.toLocaleString()} sales)`;
      select.appendChild(option);
    });
  }
  [...select.options].forEach(option => {
    option.selected = selectedCommunities.has(option.value);
  });
}

async function loadCommunityLayer() {
  communityLayer.clearLayers();
  const visible = document.getElementById('showCommunity').checked;
  setLayerGroupVisible(communityLayer, visible);
  if (!visible) {
    if (communityLegendControl) {
      map.removeControl(communityLegendControl);
      communityLegendControl = null;
    }
    return;
  }

  const firstLoad = !historicalSalesLoaded;
  if (firstLoad) beginMapEvent('community-sales-load', 'Loading historical sales for community trends…');
  try {
    await ensurePerformanceData();
    await ensureCommunityColorMap();
  } finally {
    if (firstLoad) completeMapEvent('community-sales-load', 'Community layer ready');
  }
  historicalSalesLoaded = true;
  const gj = communityGeoJson;

  L.geoJSON(gj, {
    style: f => {
      const c = String(f.properties.community);
      const active = selectedCommunities.has(c);
      const color = communityColorMap.get(c) || '#888';
      return {
        fillColor:   color,
        fillOpacity: active ? 0.45 : 0.1,
        color:       color,
        weight:      0.8
      };
    },
    onEachFeature: (f, layer) => {
      const p = f.properties || {};
      const c = String(p.community);
      const salesCount = p.salesCount || 0;
      const meanPrice = Math.round((p.meanSalePrice || 0));
      const meanNeuralPred = Math.round((p.meanNeuralPredictedPrice || 0));
      const meanLightgbmPred = Math.round((p.meanLightgbmPredictedPrice || 0));
      const neuralError = (p.avgNeuralPctError ?? 0).toFixed(1);
      const lightgbmError = (p.avgLightgbmPctError ?? 0).toFixed(1);
      const meanSqft   = Math.round((p.meanSqft || 0));
      const meanSqftLot = Math.round((p.meanSqftLot || 0));
      const meanStd    = Math.round(((p.meanPredStd || 0) * 1.96));

      layer.on('click', () => toggleCommunity(c));
      layer.bindTooltip(
        `<b>Community ${c}</b><br>` +
        `Sales: ${salesCount}<br>` +
        `Mean Price: $${meanPrice.toLocaleString()}<br>` +
        `<b>Neural:</b> $${meanNeuralPred.toLocaleString()} (${neuralError}%)<br>` +
        `<b>LightGBM:</b> $${meanLightgbmPred.toLocaleString()} (${lightgbmError}%)<br>` +
        `Avg Sqft: ${meanSqft.toLocaleString()}<br>` +
        `Avg Lot Sqft: ${meanSqftLot.toLocaleString()}<br>` +
        `95% CI: ±$${meanStd}`,
        { sticky: true }
      );
    }
  }).addTo(communityLayer);

  buildCommunityLegend(communityColorMap);
}

function toggleCommunity(id) {
  const key = String(id);
  if (selectedCommunities.has(key)) selectedCommunities.delete(key);
  else selectedCommunities.add(key);
  populateCommunitySelector();
  loadCommunityLayer();
  if (perfChart) updateChartSelection();
}

// ── Performance chart popup ───────────────────────────────────────────────────
let perfChart = null;

async function openPerfPopup() {
  const popup = document.getElementById('perfPopup');
  popup.style.visibility = 'visible';
  popup.style.zIndex = '9999';
  await ensurePerformanceData();
  if (!perfChart) {
    const existing = Chart.getChart('perfChart');
    if (existing) existing.destroy();
    buildChart(perfData);
  }
}

function closePerfPopup() {
  document.getElementById('perfPopup').style.visibility = 'hidden';
}

function closeFeaturePopup() {
  document.getElementById('featurePopup').style.visibility = 'hidden';
}

function communityTrendDefinition() {
  const variable = document.getElementById('chartVariable').value;
  const model = document.getElementById('chartModel').value;
  const modelPrefix = model === 'lightgbm' ? 'lightgbm' : model === 'gnn' ? 'gnn' : 'neural';
  const definitions = {
    mean_error: {
      key: `${modelPrefix}Mean`, title: 'Mean signed error', unit: 'percent',
      minimum: -100, maximum: 100
    },
    mape: {
      key: `${modelPrefix}Mape`, title: 'Mean absolute error', unit: 'percent',
      minimum: 0, maximum: 100
    },
    predicted_price: {
      key: `${modelPrefix}PredictedMean`, title: 'Mean predicted price', unit: 'money',
      minimum: undefined, maximum: undefined
    },
    sale_price: {
      key: 'actualMean', title: 'Mean sale price', unit: 'money',
      minimum: undefined, maximum: undefined
    },
    count: {
      key: 'count', title: 'Sales count', unit: 'count',
      minimum: 0, maximum: undefined
    }
  };
  return { ...definitions[variable], variable, model };
}

function trendValueLabel(value, definition) {
  if (value == null) return '';
  if (definition.unit === 'money') return `$${Math.round(value).toLocaleString()}`;
  if (definition.unit === 'percent') return `${Number(value).toFixed(1)}%`;
  return Math.round(value).toLocaleString();
}

function rebuildCommunityTrendChart() {
  if (!perfData) return;
  const definition = communityTrendDefinition();
  document.getElementById('chartModel').disabled =
    ['sale_price', 'count'].includes(definition.variable);
  if (perfChart) {
    perfChart.destroy();
    perfChart = null;
  }
  buildChart(perfData);
}

function buildChart(data) {
  const definition = communityTrendDefinition();
  const allQuarters = [...new Set(
    data.series.flatMap(s => s.points.map(p => p.quarter))
  )].sort();

  const datasets = [];
  data.series.forEach((s, i) => {
    const color = COMMUNITY_COLORS[i % COMMUNITY_COLORS.length];
    const qMap  = Object.fromEntries(s.points.map(p => [p.quarter, p]));

    datasets.push({
      label: `Community ${s.community}`,
      communityId: String(s.community),
      modelName: definition.model,
      data: allQuarters.map(q => qMap[q]?.[definition.key] ?? null),
      borderColor: color,
      backgroundColor: color + '33',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.25,
      spanGaps: true,
      fill: false
    });
  });

  const ctx = document.getElementById('perfChart').getContext('2d');
  perfChart = new Chart(ctx, {
    type: 'line',
    data: { labels: allQuarters, datasets },    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#eee', font: { size: 11 },
            boxWidth: 18,
            filter: (item, chartData) => {
              const dataset = chartData.datasets[item.datasetIndex];
              return dataset && selectedCommunities.has(String(dataset.communityId));
            }
          },
          onClick: (e, item, legend) => {
            const ds = legend.chart.data.datasets[item.datasetIndex];
            if (ds && ds.communityId) toggleCommunity(ds.communityId);
          }
        },
        tooltip: {
          backgroundColor: '#16213e',
          titleColor: '#e94560',
          bodyColor: '#eee',
          callbacks: {
            label: context =>
              `${context.dataset.label}: ${trendValueLabel(context.parsed.y, definition)}`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#1a3a6e' } },
        y: {
          min: definition.minimum, max: definition.maximum,
          ticks: {
            color: '#aaa',
            callback: value => trendValueLabel(value, definition)
          },
          grid:  { color: '#1a3a6e' },
          title: {
            display: true,
            text: `${definition.title}` +
              (['sale_price', 'count'].includes(definition.variable)
                ? '' : ` — ${definition.model === 'lightgbm' ? 'LightGBM' : 'Neural Network'}`),
            color: '#aaa'
          }
        }
      }
    }
  });

  // Store all quarters and all data on each dataset for axis filtering
  perfChart._allQuarters = allQuarters;
  datasets.forEach(ds => { ds._allData = [...ds.data]; });

  // Populate quarter dropdowns
  const qFrom = document.getElementById('chartQFrom');
  const qTo   = document.getElementById('chartQTo');
  qFrom.innerHTML = ''; qTo.innerHTML = '';
  allQuarters.forEach(q => {
    qFrom.innerHTML += `<option value="${q}">${q}</option>`;
    qTo.innerHTML   += `<option value="${q}">${q}</option>`;
  });
  qTo.value = allQuarters[allQuarters.length - 1];
  document.getElementById('chartYMin').value =
    definition.minimum === undefined ? '' : definition.minimum;
  document.getElementById('chartYMax').value =
    definition.maximum === undefined ? '' : definition.maximum;
  populateCommunitySelector();
  updateChartSelection();
}

function updateChartSelection() {
  if (!perfChart) return;
  perfChart.data.datasets.forEach(ds => {
    const active = selectedCommunities.has(String(ds.communityId));
    ds.hidden = !active;
  });
  perfChart.update();
}

function updateChartAxes() {
  if (!perfChart) return;
  const yMin = parseFloat(document.getElementById('chartYMin').value);
  const yMax = parseFloat(document.getElementById('chartYMax').value);
  const qFrom = document.getElementById('chartQFrom').value;
  const qTo   = document.getElementById('chartQTo').value;

  // Y axis
  perfChart.options.scales.y.min = isNaN(yMin) ? undefined : yMin;
  perfChart.options.scales.y.max = isNaN(yMax) ? undefined : yMax;

  // X axis - filter labels and data to selected quarter range
  const allQ = perfChart._allQuarters;
  const fromIdx = allQ.indexOf(qFrom);
  const toIdx   = allQ.indexOf(qTo);
  const filtered = allQ.slice(
    fromIdx >= 0 ? fromIdx : 0,
    toIdx   >= 0 ? toIdx + 1 : allQ.length
  );
  perfChart.data.labels = filtered;
  perfChart.data.datasets.forEach(ds => {
    ds.data = filtered.map(q => {
      const i = allQ.indexOf(q);
      return ds._allData[i];
    });
  });
  perfChart.update();
}

function resetChartAxes() {
  const definition = communityTrendDefinition();
  document.getElementById('chartYMin').value =
    definition.minimum === undefined ? '' : definition.minimum;
  document.getElementById('chartYMax').value =
    definition.maximum === undefined ? '' : definition.maximum;
  if (perfChart) {
    const allQ = perfChart._allQuarters;
    document.getElementById('chartQFrom').value = allQ[0];
    document.getElementById('chartQTo').value   = allQ[allQ.length - 1];
  }
  updateChartAxes();
}

function clearChartCommunities() {
  selectedCommunities.clear();
  populateCommunitySelector();
  updateChartSelection();
  if (document.getElementById('showCommunity').checked) loadCommunityLayer();
}
