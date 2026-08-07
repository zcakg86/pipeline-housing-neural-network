// On-demand monthly property valuation chart and shared movable popup behavior.
let propertyTimeSeriesChart = null;

function closeTimeSeriesPopup() {
  document.getElementById('timeSeriesPopup').style.visibility = 'hidden';
}

async function openPropertyTimeSeries(source, id, title) {
  const popup = document.getElementById('timeSeriesPopup');
  const heading = document.getElementById('timeSeriesPopupTitle');
  const note = document.getElementById('timeSeriesNote');
  heading.textContent = `Property value history — ${title || id}`;
  note.textContent = 'Calculating monthly Neural, LightGBM, and Spatial GNN predictions…';
  reportMapInteraction('property value history requested', title || id);
  const progressKey = `time-series-${source}-${id}`;
  beginMapEvent(progressKey, 'Calculating monthly property value history…');
  popup.style.visibility = 'visible';
  popup.style.zIndex = '10001';

  try {
    const query = new URLSearchParams({ source, id });
    const response = await fetch(`/api/points/time-series?${query}`);
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    renderPropertyTimeSeries(payload);
    completeMapEvent(progressKey, 'Property value history ready');
  } catch (error) {
    completeMapEvent(progressKey);
    note.textContent = `Prediction history failed: ${error.message}`;
  }
}

function renderPropertyTimeSeries(payload) {
  const points = payload.points || [];
  const labels = points.map(point => point.date);
  const highlightIndex = points.findIndex(point => point.highlight);
  const highlightedRadius = labels.map((_, index) => index === highlightIndex ? 7 : 2);
  const actual = labels.map((_, index) =>
    index === highlightIndex && Number(payload.actualPrice) > 0
      ? Number(payload.actualPrice) : null
  );
  const storedNeural = labels.map((_, index) =>
    index === highlightIndex ? Number(payload.storedNeuralPrediction) : null
  );
  const storedLightgbm = labels.map((_, index) =>
    index === highlightIndex ? Number(payload.storedLightgbmPrediction) : null
  );
  const storedGnn = labels.map((_, index) =>
    index === highlightIndex ? Number(payload.storedGnnPrediction) : null
  );
  const reconstructed = points.some(point => point.historicalSnapshotReconstruction);
  document.getElementById('timeSeriesNote').textContent =
    `${payload.highlightLabel}: ${payload.highlightDate}` +
    (reconstructed
      ? ' · Dates through the snapshot use the deployed local-market state as a historical demonstration.'
      : '');

  if (propertyTimeSeriesChart) propertyTimeSeriesChart.destroy();
  const context = document.getElementById('timeSeriesChart').getContext('2d');
  propertyTimeSeriesChart = new Chart(context, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Neural Network',
          data: points.map(point => point.neuralPredictedPrice),
          borderColor: '#4a9eff',
          backgroundColor: '#4a9eff33',
          pointRadius: highlightedRadius,
          pointBackgroundColor: '#4a9eff',
          borderWidth: 2,
          tension: 0.2,
          spanGaps: true
        },
        {
          label: 'LightGBM',
          data: points.map(point => point.lightgbmPredictedPrice),
          borderColor: '#46d17a',
          backgroundColor: '#46d17a33',
          pointRadius: highlightedRadius,
          pointStyle: 'rectRot',
          borderDash: [6, 3],
          borderWidth: 2,
          tension: 0.2,
          spanGaps: true
        },
        {
          label: 'Spatial GNN',
          data: points.map(point => point.gnnPredictedPrice),
          borderColor: '#f6b73c',
          backgroundColor: '#f6b73c33',
          pointRadius: highlightedRadius,
          pointStyle: 'triangle',
          borderDash: [2, 3],
          borderWidth: 2,
          tension: 0.2,
          spanGaps: true
        },
        {
          label: 'Stored Neural prediction',
          data: storedNeural,
          borderColor: '#b9dcff',
          backgroundColor: '#b9dcff',
          pointRadius: 7,
          pointStyle: 'circle',
          showLine: false,
          spanGaps: false
        },
        {
          label: 'Stored LightGBM prediction',
          data: storedLightgbm,
          borderColor: '#baf5ce',
          backgroundColor: '#baf5ce',
          pointRadius: 7,
          pointStyle: 'rectRot',
          showLine: false,
          spanGaps: false
        },
        {
          label: 'Stored Spatial GNN prediction',
          data: storedGnn,
          borderColor: '#ffe2a4',
          backgroundColor: '#ffe2a4',
          pointRadius: 7,
          pointStyle: 'triangle',
          showLine: false,
          spanGaps: false
        },
        {
          label: 'Actual/list price',
          data: actual,
          borderColor: '#e94560',
          backgroundColor: '#e94560',
          pointRadius: 8,
          pointStyle: 'star',
          showLine: false,
          spanGaps: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#eee' } },
        tooltip: {
          backgroundColor: '#16213e',
          titleColor: '#e94560',
          bodyColor: '#eee',
          callbacks: {
            label: context => context.parsed.y == null ? '' :
              `${context.dataset.label}: $${Math.round(context.parsed.y).toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#aaa', maxTicksLimit: 18 },
          grid: { color: '#1a3a6e' }
        },
        y: {
          ticks: {
            color: '#aaa',
            callback: value => `$${Math.round(value / 1000).toLocaleString()}k`
          },
          grid: { color: '#1a3a6e' },
          title: { display: true, text: 'Predicted value', color: '#aaa' }
        }
      }
    }
  });
}

function makePopupDraggable(popupId, handleId) {
  const popup = document.getElementById(popupId);
  const handle = document.getElementById(handleId);
  let dragging = false;
  let offsetX = 0;
  let offsetY = 0;
  handle.addEventListener('mousedown', event => {
    if (event.target.closest('button, input, select')) return;
    dragging = true;
    const bounds = popup.getBoundingClientRect();
    offsetX = event.clientX - bounds.left;
    offsetY = event.clientY - bounds.top;
    popup.style.transform = 'none';
    event.preventDefault();
  });
  document.addEventListener('mousemove', event => {
    if (!dragging) return;
    const maximumLeft = Math.max(0, window.innerWidth - popup.offsetWidth);
    const maximumTop = Math.max(0, window.innerHeight - popup.offsetHeight);
    popup.style.left = `${Math.max(0, Math.min(maximumLeft, event.clientX - offsetX))}px`;
    popup.style.top = `${Math.max(0, Math.min(maximumTop, event.clientY - offsetY))}px`;
  });
  document.addEventListener('mouseup', () => { dragging = false; });
}

makePopupDraggable('perfPopup', 'perfDragHandle');
makePopupDraggable('featurePopup', 'featureDragHandle');
makePopupDraggable('timeSeriesPopup', 'timeSeriesDragHandle');

new ResizeObserver(() => {
  if (perfChart) perfChart.resize();
  if (propertyTimeSeriesChart) propertyTimeSeriesChart.resize();
}).observe(document.getElementById('perfPopup'));
new ResizeObserver(() => {
  if (propertyTimeSeriesChart) propertyTimeSeriesChart.resize();
}).observe(document.getElementById('timeSeriesPopup'));
