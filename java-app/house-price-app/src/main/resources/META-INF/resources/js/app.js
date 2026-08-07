// Application composition root: connect controls to independently owned layers.
// ── Wire controls ─────────────────────────────────────────────────────────────
function updateSqftLabel() {
  document.getElementById('sqftVal').textContent =
    `${document.getElementById('minSqft').value} – ${document.getElementById('maxSqft').value}`;
}
document.getElementById('showWater').addEventListener('change', () => {
  loadWaterLayer().catch(error => {
    document.getElementById('stats').textContent = error.message;
  });
});
document.getElementById('showRail').addEventListener('change', () => {
  loadRailLayer().catch(showTransportLayerError);
});
document.getElementById('showBus').addEventListener('change', () => {
  loadBusLayer().catch(showTransportLayerError);
});
document.getElementById('showMajorRoads').addEventListener('change', () => {
  loadMajorRoadLayer().catch(showTransportLayerError);
});
document.getElementById('showTransportFeatures').addEventListener('change', () => {
  const visible = document.getElementById('showTransportFeatures').checked;
  document.getElementById('transportFeatureControl').style.display = visible ? 'block' : 'none';
  loadTransportFeatureLayer().catch(showTransportLayerError);
});
document.getElementById('transportFeature').addEventListener('change', () => {
  if (document.getElementById('showTransportFeatures').checked) {
    loadTransportFeatureLayer().catch(showTransportLayerError);
  }
});
document.getElementById('showSynthetic').addEventListener('change', () => {
  const visible = document.getElementById('showSynthetic').checked;
  const dateControl = document.getElementById('syntheticDateControl');
  if (dateControl) dateControl.style.display = visible ? 'block' : 'none';
  if (visible) {
    const variable = document.getElementById('variable');
    if (['pct_error', 'sale_price', 'num_sales'].includes(variable.value)) {
      variable.value = 'predicted_price';
      loadSales(); loadZillow(); loadRentcast();
    }
  }
  loadSynthetic();
});
let filterReloadTimer = null;
let viewportReloadTimer = null;
function scheduleFilteredLayerReload() {
  clearTimeout(filterReloadTimer);
  filterReloadTimer = setTimeout(() => {
    loadSales(); loadZillow(); loadRentcast();
  }, 250);
}
function scheduleViewportLayerReload() {
  clearTimeout(viewportReloadTimer);
  viewportReloadTimer = setTimeout(() => {
    const reloaded = [];
    if (document.getElementById('showSales').checked) reloaded.push('H3 sales');
    if (document.getElementById('showZillow').checked) reloaded.push('Zillow');
    if (document.getElementById('showRentcast').checked) reloaded.push('RentCast');
    if (reloaded.length) reportMapInteraction('map viewport changed', `reloading ${reloaded.join(', ')}`);
    if (document.getElementById('showSales').checked) loadSales();
    if (document.getElementById('showZillow').checked) loadZillow();
    if (document.getElementById('showRentcast').checked) loadRentcast();
  }, 250);
}
map.on('moveend', scheduleViewportLayerReload);
map.on('click', event => {
  if (event.originalEvent?.target?.closest?.('.leaflet-interactive')) return;
  reportMapInteraction('map clicked', `${event.latlng.lat.toFixed(5)}, ${event.latlng.lng.toFixed(5)}`);
});

document.getElementById('showZillow').addEventListener('change', loadZillow);
document.getElementById('showRentcast').addEventListener('change', loadRentcast);
document.getElementById('showSales').addEventListener('change', loadSales);
document.getElementById('showCommunity').addEventListener('change', event => {
  document.getElementById('communityTrendsButton').style.display = event.target.checked ? 'block' : 'none';
  loadCommunityLayer();
});
document.getElementById('variable').addEventListener('change', () => {
  if (document.getElementById('showSynthetic').checked &&
      ['pct_error', 'sale_price', 'num_sales'].includes(
        document.getElementById('variable').value
      )) {
    document.getElementById('variable').value = 'predicted_price';
  }
  loadSales(); loadZillow(); loadRentcast(); loadSynthetic();
});
document.getElementById('DisplayedModel').addEventListener('change', () => {
  synchronizeMapVariableOptions();
  loadSales(); loadZillow(); loadRentcast(); loadSynthetic();
});
document.getElementById('salesPointSource').addEventListener('change', loadRentcast);
document.getElementById('zillowType').addEventListener('change', loadZillow);
document.getElementById('salesType').addEventListener('change', () => { loadSales(); loadRentcast(); });
document.getElementById('dateFrom').addEventListener('change', () => { loadSales(); loadRentcast(); });
document.getElementById('dateTo').addEventListener('change', () => { loadSales(); loadRentcast(); });
document.getElementById('minError').addEventListener('input', () => {
  document.getElementById('minErrVal').textContent = document.getElementById('minError').value;
  scheduleFilteredLayerReload();
});
document.getElementById('maxError').addEventListener('input', () => {
  document.getElementById('maxErrVal').textContent = document.getElementById('maxError').value;
  scheduleFilteredLayerReload();
});
document.getElementById('noFilterMin').addEventListener('change', () => {
  const disabled = document.getElementById('noFilterMin').checked;
  document.getElementById('minError').disabled = disabled;
  document.getElementById('minErrVal').textContent = disabled ? '–' : document.getElementById('minError').value;
  loadSales(); loadZillow(); loadRentcast();
});
document.getElementById('noFilterMax').addEventListener('change', () => {
  const disabled = document.getElementById('noFilterMax').checked;
  document.getElementById('maxError').disabled = disabled;
  document.getElementById('maxErrVal').textContent = disabled ? '–' : document.getElementById('maxError').value;
  loadSales(); loadZillow(); loadRentcast();
});
document.getElementById('minSqft').addEventListener('input', () => { updateSqftLabel(); scheduleFilteredLayerReload(); });
document.getElementById('maxSqft').addEventListener('input', () => { updateSqftLabel(); scheduleFilteredLayerReload(); });
document.getElementById('chartCommunities').addEventListener('change', event => {
  selectedCommunities = new Set(
    [...event.target.selectedOptions].map(option => String(option.value))
  );
  updateChartSelection();
  if (document.getElementById('showCommunity').checked) loadCommunityLayer();
});

// ── Init ──────────────────────────────────────────────────────────────────────
synchronizeMapVariableOptions();
loadHomeTypes();
initializeSyntheticDateControl();
loadZillow();
loadStats();
// Community layer loads on demand (checkbox off by default)
