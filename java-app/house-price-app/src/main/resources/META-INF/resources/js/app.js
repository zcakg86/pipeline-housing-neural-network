// Application composition root: connect controls to independently owned layers.
// ── Wire controls ─────────────────────────────────────────────────────────────
function updateSqftLabel() {
  document.getElementById('sqftVal').textContent =
    `${document.getElementById('minSqft').value} – ${document.getElementById('maxSqft').value}`;
}
document.getElementById('showCommunity').addEventListener('change', loadCommunityLayer);
document.getElementById('showWater').addEventListener('change', () => {
  loadWaterLayer().catch(error => {
    document.getElementById('stats').textContent = error.message;
  });
});
document.getElementById('showSynthetic').addEventListener('change', () => {
  const visible = document.getElementById('showSynthetic').checked;
  const dateControl = document.getElementById('syntheticDateControl');
  if (dateControl) dateControl.style.display = visible ? 'block' : 'none';
  if (visible) {
    const variable = document.getElementById('variable');
    if (['pct_error', 'sale_price', 'num_sales'].includes(variable.value)) {
      variable.value = 'predicted_price_neural';
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
    if (document.getElementById('showSales').checked) loadSales();
    if (document.getElementById('showZillow').checked) loadZillow();
    if (document.getElementById('showRentcast').checked) loadRentcast();
  }, 250);
}
map.on('moveend', scheduleViewportLayerReload);

document.getElementById('showZillow').addEventListener('change', loadZillow);
document.getElementById('showRentcast').addEventListener('change', loadRentcast);
document.getElementById('showSales').addEventListener('change', loadSales);
document.getElementById('variable').addEventListener('change', () => {
  loadSales(); loadZillow(); loadRentcast(); loadSynthetic();
});
document.getElementById('h3Model').addEventListener('change', loadSales);
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
loadHomeTypes();
initializeSyntheticDateControl();
loadZillow();
loadStats();
// Community layer loads on demand (checkbox off by default)
