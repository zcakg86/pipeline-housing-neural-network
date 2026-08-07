// Aggregated H3 sales layer and its viewport-aware loading lifecycle.
// ── Sales H3 layer ────────────────────────────────────────────────────────────
let salesRequestId = 0;
let historicalSalesLoaded = false;

function setLayerGroupVisible(layer, visible) {
  if (visible && !map.hasLayer(layer)) map.addLayer(layer);
  if (!visible && map.hasLayer(layer)) map.removeLayer(layer);
}

async function loadSales() {
  const requestId = ++salesRequestId;
  salesLayer.clearLayers();
  if (legendControl) { map.removeControl(legendControl); legendControl = null; }

  const variable  = document.getElementById('variable').value;
  const colorModel = displayedModel();
  const {min: minErr, max: maxErr} = getErrorRange();
  const homeType  = document.getElementById('salesType').value;
  const dateFrom  = document.getElementById('dateFrom').value;
  const dateTo    = document.getElementById('dateTo').value;
  const showSales = document.getElementById('showSales').checked;
  setLayerGroupVisible(salesLayer, showSales);

  // Nothing to show
  if (!showSales) { abortLayerRequest('sales'); return; }

  const firstLoad = !historicalSalesLoaded;
  if (firstLoad) beginMapEvent('historical-sales-load', 'Loading historical sales and computing GNN predictions…');
  let gj;
  try {
    gj = await fetchLayerJson('sales',
      `/api/sales/h3?variable=${variable}&model=${colorModel}&homeType=${homeType}` +
      `&dateFrom=${dateFrom}&dateTo=${dateTo}&minError=${minErr}&maxError=${maxErr}`);
  } finally {
    if (firstLoad) completeMapEvent('historical-sales-load', gj ? 'Historical H3 layer ready' : '');
  }
  if (!gj) return;
  historicalSalesLoaded = true;
  loadHomeTypes();
  loadStats();
  if (requestId !== salesRequestId) return;
  const features = gj.features || [];
  if (!features.length) { if (legendControl) map.removeControl(legendControl); return; }

  const values  = features.map(f => f.properties.displayValue);
  const anchors = variable === 'pct_error' ? computeErrorAnchors(values) : null;
  const colorFn = getColorFn(variable, values, anchors);

  L.geoJSON(gj, {
    style: f => ({ fillColor: colorFn(f.properties.displayValue), fillOpacity: 0.55, color: '#444', weight: 0.5 }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>H3 Hex</b>: ${p.h3Index}<br>` +
        `Sales: ${p.numSales}<br>` +
        `Avg Price: $${Math.round(p.avgSalePrice).toLocaleString()}<br>` +
        `<b>Neural:</b> $${Math.round(p.avgNeuralPredictedPrice).toLocaleString()} (${p.avgNeuralPctError.toFixed(1)}%)<br>` +
        `<b>LightGBM:</b> $${Math.round(p.avgLightgbmPredictedPrice).toLocaleString()} (${p.avgLightgbmPctError.toFixed(1)}%)<br>` +
        `<b>Spatial GNN:</b> $${Math.round(p.avgGnnPredictedPrice || 0).toLocaleString()} (${(p.avgGnnPctError || 0).toFixed(1)}%)<br>` +
        `Color model: ${p.colorModel === 'lightgbm' ? 'LightGBM' : 'Neural Network'}<br>` +
        `Avg Sqft: ${Math.round(p.avgSqft).toLocaleString()}<br>` +
        `Avg Lot Sqft: ${Math.round(p.avgSqftLot || 0).toLocaleString()}<br>` +
        `95% CI: ±$${Math.round((p.avgPredStd ?? 0) * 1.96).toLocaleString()} (±${((p.avgPredCvPct ?? 0) / 2).toFixed(1)}% of price)`,
        { autoPan: false }
      );
    }
  }).addTo(salesLayer);

  buildLegend(variable, values, anchors);
}

function buildLegend(variable, values, anchors) {
  if (legendControl) map.removeControl(legendControl);
  legendControl = L.control({ position: 'bottomright' });
  legendControl.onAdd = () => {
    const div = L.DomUtil.create('div');
    div.style.cssText = 'background:rgba(22,33,62,0.93);padding:10px 14px;border-radius:6px;' +
      'border:1px solid #0f3460;color:#eee;font-size:11px;min-width:175px';
    const titles = {
      pct_error: 'Average Error (%)',
      predicted_price: 'Predicted Price',
      predicted_price_neural: 'Predicted Price — Neural Network',
      predicted_price_lightgbm: 'Predicted Price — LightGBM',
      sale_price: 'Average Sale Price', sqft: 'Average Sqft',
      sqft_lot: 'Average Lot Square Footage', num_sales: 'Sales',
      pred_std: 'Prediction Uncertainty', pred_cv_pct: 'Uncertainty Width'
    };
    const modelSuffix = ['pct_error', 'predicted_price'].includes(variable)
      ? ` — ${document.getElementById('DisplayedModel').selectedOptions[0].textContent}` : '';
    const lo = anchors ? anchors.p10 : pct(values, 5);
    const hi = anchors ? anchors.p90 : pct(values, 95);
    const format = value => [
      'sale_price', 'predicted_price', 'predicted_price_neural',
      'predicted_price_lightgbm', 'pred_std'
    ].includes(variable)
      ? `$${Math.round(value).toLocaleString()}` : Number(value).toFixed(1);
    const palette = variable === 'pct_error'
      ? [ERROR_PALETTE.darkBlue, ERROR_PALETTE.softBlue, ERROR_PALETTE.neutral,
         ERROR_PALETTE.softRed, ERROR_PALETTE.darkRed]
      : (['sqft', 'sqft_lot', 'pred_std', 'pred_cv_pct'].includes(variable)
          ? COLORMAPS.Blues : COLORMAPS.YlOrRd);
    const gradient = palette.map(color => `rgb(${color.join(',')})`).join(',');
    const middleLabel = variable === 'pct_error'
      ? `<span>${format(0)}</span>` : '<span></span>';
    div.innerHTML = `<b style="color:#e94560">${titles[variable] || variable}${modelSuffix}</b>` +
      `<div style="height:14px;margin-top:7px;border-radius:3px;border:1px solid #555;` +
      `background:linear-gradient(to right,${gradient})"></div>` +
      `<div style="display:flex;justify-content:space-between;color:#aaa;margin-top:3px">` +
      `<span>${format(lo)}</span>${middleLabel}<span>${format(hi)}</span></div>`;
    return div;
  };
  legendControl.addTo(map);
}

// ── Point variable accessor ───────────────────────────────────────────────────
// Maps the variable selector value to the correct per-point property field.
// Attention variables use the attn* fields; pred_std and sqft are direct;
// everything else falls back to pctError.
function pointVariableValue(props, variable, model = displayedModel()) {
  switch (variable) {
    case 'sale_price':      return props.salePrice    ?? props.listPrice ?? 0;
    case 'predicted_price': return model === 'lightgbm'
      ? props.lightgbmPredictedPrice ?? 0
      : model === 'gnn' ? props.gnnPredictedPrice ?? 0
      : props.predictedPrice ?? props.neuralPredictedPrice ?? 0;
    case 'sqft':            return props.sqft         ?? 0;
    case 'sqft_lot':        return props.sqftLot      ?? 0;
    case 'pred_std':        return props.predictionStdPrice ?? 0;
    case 'pred_cv_pct':     return props.predictionCvPct   ?? 0;
    case 'attn_community':  return props.attnCommunity ?? 0;
    case 'attn_property':   return props.attnProperty  ?? 0;
    case 'attn_time':       return props.attnTime      ?? 0;
    case 'attn_market':     return props.attnMarket    ?? 0;
    default:                return model === 'lightgbm'
      ? props.lightgbmPctError ?? 0 : model === 'gnn'
      ? props.gnnPctError ?? 0 : props.pctError ?? 0;
  }
}
