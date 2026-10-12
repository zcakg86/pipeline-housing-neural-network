// Grounded Redfin price assessment and human evaluation feedback.
let currentAgentAssessmentId = null;

async function openPriceAssessment(listingId, title) {
  const popup = document.getElementById('featurePopup');
  const heading = document.getElementById('featurePopupTitle');
  const body = document.getElementById('featurePopupBody');
  currentAgentAssessmentId = null;
  heading.textContent = `AI price assessment — ${title || listingId}`;
  body.innerHTML = '<span class="shap-value">Building evidence…</span>';
  popup.style.visibility = 'visible';
  reportMapInteraction('Redfin AI assessment requested', title || listingId);
  const progressKey = `agent-assessment-${listingId}`;
  beginMapEvent(progressKey, 'Running grounded Redfin price assessment…');

  try {
    const response = await fetch(
      `/api/agent/redfin/${encodeURIComponent(listingId)}/assessment`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }
    );
    if (!response.ok) throw new Error(await agentApiError(response));
    const payload = await response.json();
    currentAgentAssessmentId = payload.assessmentId;
    body.innerHTML = agentAssessmentHtml(payload);
    completeMapEvent(progressKey, 'Grounded price assessment ready');
  } catch (error) {
    completeMapEvent(progressKey);
    body.innerHTML = `<span class="shap-negative">Agent assessment failed: ${escapeHtml(error.message)}</span>`;
  }
}

function agentAssessmentHtml(payload) {
  const evidence = payload.evidence || {};

  const reviewEvidence = {
    listing: evidence.listing,
    predictions: evidence.prediction?.predictions,
    timeEvidence: evidence.prediction?.timeEvidence,
    h3ModelArtifact: evidence.h3Market,
    listingPageToolCallCount: evidence.listingPageToolCallCount,
    retrievedListingPageFacts: evidence.retrievedListingPageFacts
  };

  const thinkingHtml =
    typeof payload.modelThinking === 'string' &&
    payload.modelThinking.trim()
      ? `<details class="agent-evidence">` +
          `<summary>Review model thinking</summary>` +
          `<pre>${escapeHtml(payload.modelThinking)}</pre>` +
        `</details>`
      : '';
  return `<div class="agent-meta">Assessment ${escapeHtml(payload.assessmentId)} · ` +
    `${escapeHtml(payload.model)} · prompt ${escapeHtml(payload.promptVersion)}</div>` +
    `<div class="agent-assessment-text">${sanitizeAgentHtml(payload.assessment)}</div>` +
    thinkingHtml+
    `<details class="agent-evidence"><summary>Review grounded evidence</summary>` +
    `<pre>${escapeHtml(JSON.stringify(reviewEvidence, null, 2))}</pre></details>` +
    `<div class="agent-feedback"><h4>Evaluate this assessment</h4>` +
    `<label>Overall rating` +
    `<select id="agentFeedbackRating"><option value="5">5 — excellent</option>` +
    `<option value="4">4 — good</option><option value="3" selected>3 — mixed</option>` +
    `<option value="2">2 — weak</option><option value="1">1 — incorrect</option></select></label>` +
    `<label>Your expected fair price (optional)` +
    `<input id="agentExpectedPrice" type="number" min="0" step="1000" placeholder="e.g. 825000"/></label>` +
    `<label>Feedback<textarea id="agentFeedbackComments" rows="4" ` +
    `placeholder="What was correct, unsupported, or missing?"></textarea></label>` +
    `<button type="button" onclick="submitAgentFeedback()">Save evaluation feedback</button>` +
    `<div id="agentFeedbackStatus" class="explanation-note"></div></div>`;
}

function sanitizeAgentHtml(value) {
  const allowed = new Set([
    'H2', 'H3', 'H4', 'P', 'UL', 'OL', 'LI', 'STRONG', 'EM', 'B', 'I', 'BR',
    'HR', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'BLOCKQUOTE', 'CODE', 'A'
  ]);
  const discarded = new Set([
    'SCRIPT', 'STYLE', 'IFRAME', 'OBJECT', 'EMBED', 'SVG', 'MATH', 'FORM',
    'INPUT', 'BUTTON', 'TEXTAREA', 'SELECT', 'OPTION', 'LINK', 'META'
  ]);
  let source = String(value || '').trim();
  source = source.replace(/^```(?:html)?\s*/i, '').replace(/\s*```$/i, '');
  if (!source) return '<p>No assessment was returned.</p>';

  const template = document.createElement('template');
  template.innerHTML = source;
  for (const element of Array.from(template.content.querySelectorAll('*'))) {
    const tag = element.tagName.toUpperCase();
    if (discarded.has(tag)) {
      element.remove();
    } else if (!allowed.has(tag)) {
      element.replaceWith(...element.childNodes);
    } else if (tag === 'A') {
      const href = safeAgentListingUrl(element.getAttribute('href'));
      for (const attribute of Array.from(element.attributes)) {
        element.removeAttribute(attribute.name);
      }
      if (!href) {
        element.replaceWith(...element.childNodes);
      } else {
        element.setAttribute('href', href);
        element.setAttribute('target', '_blank');
        element.setAttribute('rel', 'noopener noreferrer');
      }
    } else {
      for (const attribute of Array.from(element.attributes)) {
        element.removeAttribute(attribute.name);
      }
    }
  }
  return template.innerHTML;
}

function safeAgentListingUrl(value) {
  try {
    const url = new URL(String(value || ''));
    const hostname = url.hostname.toLowerCase();
    const isRedfin = hostname === 'redfin.com' || hostname.endsWith('.redfin.com');
    if (url.protocol !== 'https:' || url.port || url.username || url.password || !isRedfin) {
      return null;
    }
    return url.href;
  } catch (_error) {
    return null;
  }
}

async function submitAgentFeedback() {
  const status = document.getElementById('agentFeedbackStatus');
  if (!currentAgentAssessmentId) {
    status.textContent = 'No assessment is selected.';
    return;
  }
  const rawExpected = document.getElementById('agentExpectedPrice').value;
  const body = {
    rating: Number(document.getElementById('agentFeedbackRating').value),
    expectedPrice: rawExpected === '' ? null : Number(rawExpected),
    comments: document.getElementById('agentFeedbackComments').value
  };
  status.textContent = 'Saving feedback…';
  try {
    const response = await fetch(
      `/api/agent/assessments/${encodeURIComponent(currentAgentAssessmentId)}/feedback`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    );
    if (!response.ok) throw new Error(await agentApiError(response));
    status.textContent = '✓ Feedback saved for evaluation.';
  } catch (error) {
    status.textContent = `Feedback failed: ${error.message}`;
  }
}

async function agentApiError(response) {
  const raw = await response.text();
  let detail = raw.trim();
  if (detail) {
    try {
      const payload = JSON.parse(detail);
      detail = payload.detail || payload.error || detail;
    } catch (_) {
      // Preserve a provider or proxy's plain-text response.
    }
  }
  return detail
    ? `HTTP ${response.status}: ${detail}`
    : `HTTP ${response.status}: the server returned no error details; check the Quarkus log`;
}
