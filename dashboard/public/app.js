'use strict';
// Read-only dashboard for the current project; all data is inserted as text.

const $ = id => document.getElementById(id);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const pct = (n, d) => (d ? `${Math.round((100 * n) / d)}%` : '—');
const when = iso => (iso ? new Date(iso).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) : '—');
const bytes = n => (n ? `${(n / 1024 ** 3).toFixed(0)} GB` : '—');
const SET_NAMES = {dev: 'Dev (52 mazes)', holdout: 'Hold-out', course: 'Course seeds', validation: 'Validation', test1: 'Test 1', test2: 'Test 2', reserve: 'Reserve'};
const expanded = new Set();

function row(cells, className) {
  const tr = el('tr', undefined, className);
  for (const cell of cells) {
    const td = el('td');
    if (cell instanceof Node) td.append(cell); else td.textContent = cell ?? '—';
    tr.append(td);
  }
  return tr;
}

function header(names) {
  const tr = el('tr');
  for (const name of names) tr.append(el('th', name));
  const thead = el('thead');
  thead.append(tr);
  return thead;
}

function score(result) {
  if (!result) return el('span', '—', 'muted');
  const span = el('span');
  span.append(el('strong', `${result.success}/${result.completed}`));
  span.append(el('span', ` (${result.timed})`, 'muted'));
  if (result.completed < result.planned) span.append(el('span', ' running', 'tag'));
  return span;
}

function bar(value, total, className) {
  const outer = el('div', undefined, 'bar ' + (className || ''));
  const inner = el('div');
  inner.style.width = `${total ? Math.round((100 * value) / total) : 0}%`;
  outer.append(inner);
  return outer;
}

function renderResources(resources) {
  const target = $('resources');
  target.replaceChildren();
  if (!resources) return;
  const parts = [`CPU load ${resources.load_1m ?? '—'} / ${resources.logical_cpus ?? '—'}`,
    `RAM free ${bytes(resources.memory_available_bytes)}`, `Disk free ${bytes(resources.disk_free_bytes)}`];
  for (const gpu of resources.gpus || []) parts.push(`GPU ${Math.round(gpu.utilization)}% · ${(gpu.memory_used_mib / 1024).toFixed(1)} GB`);
  for (const part of parts) target.append(el('span', part));
}

function renderOverview(overview, timestamp) {
  $('title').textContent = overview.title || 'Pick–Nav–Reach';
  $('phase').textContent = overview.phase || '';
  $('updated').textContent = `Updated ${when(timestamp)}`;
  $('summary').textContent = overview.summary || '';
  const cards = $('headline');
  cards.replaceChildren();
  for (const card of overview.headline || []) {
    const box = el('div', undefined, 'card');
    box.append(el('div', card.label, 'card-label'), el('div', card.value, 'card-value'), el('div', card.detail, 'card-detail'));
    cards.append(box);
  }
  const arms = $('arms');
  arms.replaceChildren(header(['Arm', 'Dev', 'Hold-out', 'Course', 'Rubric / trial']));
  const body = el('tbody');
  for (const arm of overview.arms || []) {
    const name = el('div');
    name.append(el('strong', arm.name), el('div', arm.description, 'muted small'));
    const results = arm.results || {};
    const rubric = Object.values(results).map(r => r.rubric).find(Boolean);
    body.append(row([name, score(results.dev), score(results.holdout), score(results.course),
      rubric ? `${rubric.per_trial} / 40` : '—']));
  }
  arms.append(body);
  const plan = $('plan');
  plan.replaceChildren();
  for (const item of overview.plan || []) {
    const li = el('li', undefined, `plan-${item.status}`);
    li.append(el('span', item.status, 'tag'), el('span', ` ${item.step}`));
    plan.append(li);
  }
  const docs = $('documents');
  docs.replaceChildren();
  for (const doc of overview.documents || []) {
    const a = el('a', doc.label);
    a.href = doc.href;
    const li = el('li');
    li.append(a);
    docs.append(li);
  }
  renderRecordings(overview.recordings || []);
}

let recordingSignature = '';
function renderRecordings(list) {
  // Only rebuild when the list changes, so polling never interrupts playback.
  const signature = JSON.stringify(list.map(r => [r.run, (r.media || []).map(m => m.path)]));
  if (signature === recordingSignature) return;
  recordingSignature = signature;
  const recordings = $('recordings');
  recordings.replaceChildren();
  for (const recording of list) {
    const group = el('div', undefined, 'recording-group');
    group.append(el('h3', recording.label));
    const strip = el('div', undefined, 'videos');
    for (const media of recording.media || []) {
      const figure = el('figure');
      const video = el('video');
      video.controls = true;
      video.preload = 'none';
      video.src = `/artifacts/${encodeURIComponent(recording.run)}/${media.path.split('/').map(encodeURIComponent).join('/')}`;
      figure.append(video, el('figcaption', media.label));
      strip.append(figure);
    }
    if (!(recording.media || []).length) strip.append(el('p', 'No video artifacts found.', 'muted'));
    group.append(strip);
    recordings.append(group);
  }
}

function renderActive(active) {
  $('active-panel').hidden = !active.length;
  const target = $('active');
  target.replaceChildren();
  for (const job of active) {
    const line = el('div', undefined, 'job');
    const label = el('div');
    label.append(el('strong', job.name), el('span', ` · ${job.kind || 'run'} · started ${when(job.started_at)}`, 'muted'));
    line.append(label);
    if (job.progress !== null && job.progress !== undefined) line.append(bar(job.progress, 1, 'progress'));
    line.append(el('div', job.detail, 'muted small'));
    target.append(line);
  }
}

function settingsText(evaluation) {
  const ex = evaluation.execution || {};
  const parts = [`controller ${evaluation.controller}`, `planned speed ${ex.planned_speed_scale}`];
  if (ex.holding_speed_scale) parts.push(`holding speed ${ex.holding_speed_scale}`);
  if (evaluation.controller !== 'reference') parts.push(`carry step ${ex.delta_scale} m`);
  if (ex.carry_max_turn_deg) parts.push(`turn limit ${ex.carry_max_turn_deg}°`);
  if (ex.carry_slip_drift_mm) parts.push(`slow ×${ex.carry_slip_scale} after ${ex.carry_slip_drift_mm} mm slip`);
  return parts.join(' · ');
}

function detail(evaluation) {
  const box = el('div', undefined, 'detail');
  const objects = el('div');
  objects.append(el('h4', 'By object'));
  for (const [name, entry] of Object.entries(evaluation.by_object)) {
    const line = el('div', undefined, 'object-line');
    line.append(el('span', name), bar(entry.success, entry.cases), el('span', `${entry.success}/${entry.cases}`, 'muted'));
    objects.append(line);
  }
  const sets = el('div');
  sets.append(el('h4', 'By set'));
  for (const [name, entry] of Object.entries(evaluation.by_set)) {
    sets.append(el('div', `${SET_NAMES[name] || name}: ${entry.success}/${entry.cases} complete, ${entry.timed} timed`));
  }
  const failures = el('div');
  failures.append(el('h4', 'Failures'));
  const failureList = Object.entries(evaluation.failures);
  if (!failureList.length) failures.append(el('div', 'None', 'muted'));
  for (const [label, count] of failureList) failures.append(el('div', `${count} × ${label}`));
  const settings = el('div');
  settings.append(el('h4', 'Settings'), el('div', settingsText(evaluation)));
  for (const [skill, source] of Object.entries(evaluation.models || {})) settings.append(el('div', `${skill}: ${source}`, 'muted small'));
  if (evaluation.rubric) {
    settings.append(el('h4', 'Rubric estimate (per trial)'));
    for (const [item, points] of Object.entries(evaluation.rubric.items)) settings.append(el('div', `${item.replaceAll('_', ' ')}: ${points}`, 'small'));
  }
  box.append(objects, sets, failures, settings);
  return box;
}

function renderEvaluations(evaluations) {
  const table = $('evaluations');
  table.replaceChildren(header(['Run', 'Set', 'Complete', 'Within 15k', 'Main failure', 'Rubric', 'Started']));
  const body = el('tbody');
  for (const evaluation of evaluations) {
    const name = el('div');
    name.append(el('strong', evaluation.label), el('div', evaluation.id, 'muted small'));
    if (evaluation.status === 'running') name.append(el('span', 'running', 'tag'));
    const main = Object.entries(evaluation.failures)[0];
    const complete = el('div');
    complete.append(el('span', `${evaluation.success}/${evaluation.completed} · ${pct(evaluation.success, evaluation.completed)}`),
      bar(evaluation.success, evaluation.completed));
    const tr = row([name, SET_NAMES[evaluation.set] || evaluation.set, complete,
      `${evaluation.timed} · ${pct(evaluation.timed, evaluation.completed)}`,
      main ? `${main[1]} × ${main[0]}` : '—',
      evaluation.rubric ? `${evaluation.rubric.per_trial}/40` : '—', when(evaluation.started_at)], 'clickable');
    tr.addEventListener('click', () => {
      if (expanded.has(evaluation.id)) expanded.delete(evaluation.id); else expanded.add(evaluation.id);
      renderEvaluations(evaluations);
    });
    body.append(tr);
    if (expanded.has(evaluation.id)) {
      const td = el('td');
      td.colSpan = 7;
      td.append(detail(evaluation));
      const extra = el('tr', undefined, 'detail-row');
      extra.append(td);
      body.append(extra);
    }
  }
  table.append(body);
}

function renderHistory(history) {
  $('history-count').textContent = `(${history.length})`;
  const table = $('history');
  table.replaceChildren(header(['Run', 'Kind', 'Status', 'Summary', 'Updated']));
  const body = el('tbody');
  for (const item of history) {
    body.append(row([item.name && item.name !== item.id ? `${item.name} · ${item.id}` : item.id, item.kind,
      item.status, (item.summary || '').slice(0, 160), when(item.updated_at)]));
  }
  table.append(body);
}

let lastHistoryLength = -1;
async function poll() {
  try {
    const response = await fetch('/api/overview', {cache: 'no-store'});
    const state = await response.json();
    renderOverview(state.overview || {}, state.timestamp);
    renderResources(state.resources);
    renderActive(state.active || []);
    renderEvaluations(state.evaluations || []);
    if ((state.history || []).length !== lastHistoryLength) {
      renderHistory(state.history || []);
      lastHistoryLength = (state.history || []).length;
    }
  } catch (error) {
    $('updated').textContent = `Connection problem: ${error.message}`;
  }
  setTimeout(poll, 10000);
}

poll();
