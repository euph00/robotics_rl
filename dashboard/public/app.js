'use strict';
// Read-only project dashboard. Everything comes from /api/overview: the study plan and its matrix
// results (configs/study.json), live hand-training curves, recordings and running jobs.
// All data is inserted as text; diagrams are drawn as HTML grids or SVG.

const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const svg = (tag, attrs = {}, text) => {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
};
const when = iso => (iso ? new Date(iso).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) : '—');
const bytes = n => (n ? `${(n / 1024 ** 3).toFixed(0)} GB` : '—');
const percent = v => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;

// Grasp controller -> colour slot (validated palette, see style.css): who produces the behaviour.
const SOURCE_OF = {scripted: 'planned', imitation: 'imitation', imitation_rl: 'imitation_rl', rl: 'rl'};
const SOURCE_TEXT = {planned: 'Hand-written', imitation: 'Cloned', imitation_rl: 'Cloned + RL', rl: 'RL'};
const OBJECT_NAMES = {'0': 'chef can', '2': 'soup can', '5': 'banana', '7': 'apple', '10': 'mug', '14': 'foam brick',
                      '102': 'soup can ×1.6', '107': 'apple ×1.45', '110': 'mug ×1.35'};

/* ---------- tooltip ---------- */
const tip = () => $('tip');
function showTip(event, lines) {
  const box = tip();
  box.replaceChildren(...lines.map(([text, cls]) => el('div', text, cls)));
  box.hidden = false;
  const x = Math.min(event.pageX + 14, window.scrollX + window.innerWidth - box.offsetWidth - 12);
  box.style.left = `${x}px`;
  box.style.top = `${event.pageY + 12}px`;
}
const hideTip = () => { tip().hidden = true; };
function hover(node, lines) {
  node.addEventListener('mousemove', event => showTip(event, typeof lines === 'function' ? lines() : lines));
  node.addEventListener('mouseleave', hideTip);
}

/* ---------- study results ---------- */
// One matrix cell: aggregate the seeds of a condition on a task.
function aggregate(runs) {
  const done = runs.filter(r => r.status === 'complete' && r.cases > 0 && r.cases >= (r.planned || r.cases));
  const running = runs.filter(r => !done.includes(r) && r.status !== 'missing');
  const byObject = {};
  for (const r of done) {
    for (const [obj, v] of Object.entries(r.by_object || {})) {
      const e = byObject[obj] = byObject[obj] || {cases: 0, success: 0, timed: 0};
      e.cases += v.cases; e.success += v.success; e.timed += v.timed;
    }
  }
  return {
    done, running, byObject,
    rate: done.length ? mean(done.map(r => r.success / r.cases)) : null,
    timed: done.length ? mean(done.map(r => r.timed / r.cases)) : null,
    picked: done.length ? mean(done.map(r => (r.picked || 0) / r.cases)) : null,
    progress: running.length ? running.map(r => `${r.cases}/${r.planned || '?'}`).join(', ') : null,
  };
}

function cellState(condition, task) {
  if (condition.na) return {kind: 'na', text: condition.na};
  const runs = (condition.results || {})[task.key];
  if (runs === null || runs === undefined) return {kind: 'na', text: condition.note || 'not applicable'};
  const agg = aggregate(runs);
  if (agg.done.length) return {kind: 'done', ...agg};
  if (agg.running.length) return {kind: 'running', ...agg};
  const training = ((condition.training_status || {})[task.key] || []);
  if (training.length) return {kind: 'training', text: `${training.length} seed${training.length > 1 ? 's' : ''} training`};
  const dev = condition.dev && condition.dev.result;
  if (task.key === 'standard' && dev && dev.cases) {
    return {kind: 'dev', rate: dev.success / dev.cases, timed: dev.timed / dev.cases, note: condition.dev.note,
            done: [dev], byObject: dev.by_object || {}};
  }
  return {kind: 'planned', text: condition.note || 'planned'};
}

function cellTooltip(condition, task, state) {
  const lines = [[`${condition.system || ''}`, 'tip-title'], [`${task.label}`, 'muted']];
  if (state.kind === 'dev') lines.push([state.note, 'muted']);
  for (const r of state.done || []) {
    lines.push([`${r.id}: ${r.success}/${r.cases} delivered · ${r.picked ?? '?'} picked · ${r.timed} within deadline`]);
  }
  const objects = Object.entries(state.byObject || {}).map(([k, v]) => `${OBJECT_NAMES[k] || k} ${Math.round(100 * v.success / v.cases)}%`);
  if (objects.length) lines.push([objects.join(' · '), 'muted']);
  if (state.progress) lines.push([`running: ${state.progress} cases`, 'muted']);
  return lines;
}

function resultBar(rate, timed, source) {
  const bar = el('div', undefined, `cbar src-${source}`);
  const all = el('span', undefined, 'cbar-all'); all.style.width = `${Math.round((rate || 0) * 100)}%`;
  const time = el('span', undefined, 'cbar-time'); time.style.width = `${Math.round((timed || 0) * 100)}%`;
  bar.append(all, time);
  return bar;
}

function renderMatrix(study) {
  const target = $('matrix');
  target.replaceChildren();
  const byKey = {};
  for (const c of study.conditions || []) byKey[`${c.effector}/${c.controller}`] = c;
  for (const task of study.tasks || []) {
    const block = el('div', undefined, 'matrix');
    const head = el('div', undefined, 'matrix-title');
    head.append(el('strong', task.label), el('span', task.detail, 'muted small'));
    block.append(head);
    const grid = el('div', undefined, 'mgrid');
    grid.append(el('div'));
    for (const e of study.effectors || []) {
      const h = el('div', undefined, 'mhead');
      h.append(el('div', e.label), el('div', e.detail, 'muted small'));
      grid.append(h);
    }
    for (const ctl of study.controllers || []) {
      const label = el('div', undefined, 'mrow');
      label.append(el('div', ctl.label), el('div', ctl.detail, 'muted small'));
      grid.append(label);
      for (const e of study.effectors || []) {
        const condition = byKey[`${e.key}/${ctl.key}`] || {na: '—'};
        const state = cellState(condition, task);
        const cell = el('div', undefined, `mcell mcell-${state.kind}`);
        if (state.kind === 'na' || state.kind === 'planned' || state.kind === 'training') {
          cell.append(el('span', state.kind === 'na' ? '—' : state.kind, `chip${state.kind === 'training' ? ' chip-run' : ''}`));
          if (state.kind === 'training') cell.append(el('div', `${state.text} · see Training now`, 'muted small'));
          if (state.text && state.kind === 'na' && state.text !== '—') cell.append(el('div', state.text, 'muted small'));
        } else {
          const top = el('div', undefined, 'mtop');
          top.append(el('span', percent(state.rate), 'big'));
          if (state.kind === 'dev') top.append(el('span', 'dev', 'chip chip-dev'));
          if (state.kind === 'running') top.append(el('span', 'running', 'chip chip-run'));
          cell.append(top, resultBar(state.rate, state.timed, SOURCE_OF[ctl.key]));
          const seeds = state.kind === 'done' ? `${state.done.length} seed${state.done.length > 1 ? 's' : ''}` :
                        state.kind === 'dev' ? 'development set' : `${state.progress} cases`;
          cell.append(el('div', `${percent(state.timed)} in time · picked ${percent(state.picked)} · ${seeds}`, 'muted small'));
        }
        hover(cell, () => cellTooltip(condition, task, state));
        grid.append(cell);
      }
    }
    block.append(grid);
    target.append(block);
  }
}

/* ---------- plan ---------- */
const STEP_ICON = {done: '✓', active: '●', next: '', stretch: '☆'};
const STEP_TEXT = {done: 'done', active: 'in progress', next: 'next', stretch: 'stretch goal'};
function renderPlan(study) {
  const target = $('plan');
  target.replaceChildren();
  for (const m of study.milestones || []) {
    const step = el('li', undefined, `step step-${m.status}`);
    const dot = el('span', STEP_ICON[m.status] || '', 'dot');
    dot.setAttribute('aria-label', STEP_TEXT[m.status] || m.status);
    step.append(dot, el('span', m.label, 'step-label'), el('span', `${STEP_TEXT[m.status] || m.status} · ${m.detail || ''}`, 'step-detail'));
    target.append(step);
  }
}

/* ---------- what RL adds (grouped bars) ---------- */
function renderAdded(study) {
  const target = $('added');
  target.replaceChildren();
  const task = (study.tasks || [])[0];
  const controllers = (study.controllers || []).filter(c => c.key !== 'rl');
  const effectors = study.effectors || [];
  const byKey = {};
  for (const c of study.conditions || []) byKey[`${c.effector}/${c.controller}`] = c;
  const width = 460, height = 230, left = 40, right = 10, top = 22, bottom = 40;
  const s = svg('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img',
                        'aria-label': 'Delivered share by grasp controller for each end effector'});
  const y = v => top + (1 - v) * (height - top - bottom);
  for (const tick of [0, .5, 1]) {
    s.append(svg('line', {x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: 'grid'}));
    s.append(svg('text', {x: left - 6, y: y(tick) + 4, class: 'axis', 'text-anchor': 'end'}, `${tick * 100}%`));
  }
  const groupW = (width - left - right) / effectors.length, barW = 34, gap = 8;
  effectors.forEach((e, gi) => {
    const x0 = left + gi * groupW + (groupW - controllers.length * barW - (controllers.length - 1) * gap) / 2;
    s.append(svg('text', {x: left + gi * groupW + groupW / 2, y: height - 14, class: 'label', 'text-anchor': 'middle'}, `${e.label} (${e.detail})`));
    controllers.forEach((ctl, bi) => {
      const condition = byKey[`${e.key}/${ctl.key}`] || {};
      const state = task ? cellState(condition, task) : {kind: 'na'};
      const x = x0 + bi * (barW + gap), source = SOURCE_OF[ctl.key];
      const g = svg('g', {class: `bar src-${source}`});
      if (state.rate === null || state.rate === undefined) {
        g.append(svg('rect', {x, y: y(0) - 2, width: barW, height: 2, class: 'bar-empty'}));
        g.append(svg('text', {x: x + barW / 2, y: y(0) - 8, class: 'axis', 'text-anchor': 'middle'}, '–'));
      } else {
        const dev = state.kind === 'dev';
        g.append(svg('rect', {x, y: y(state.rate), width: barW, height: y(0) - y(state.rate), rx: 4, class: `bar-all${dev ? ' bar-dev' : ''}`}));
        g.append(svg('rect', {x, y: y(state.timed), width: barW, height: y(0) - y(state.timed), rx: 4, class: `bar-time${dev ? ' bar-dev' : ''}`}));
        g.append(svg('text', {x: x + barW / 2, y: y(state.rate) - 6, class: 'label', 'text-anchor': 'middle'}, percent(state.rate) + (dev ? '*' : '')));
      }
      g.append(svg('rect', {x: x - 3, y: top, width: barW + 6, height: y(0) - top, fill: 'transparent'}));   // hit area
      hover(g, () => [[`${e.label} · ${ctl.label}`, 'tip-title'], ...(state.rate === null || state.rate === undefined ? [[state.text || 'no result yet', 'muted']] :
        [[`${percent(state.rate)} delivered · ${percent(state.timed)} within deadline`], ...(state.kind === 'dev' ? [[state.note, 'muted']] : [])])]);
      s.append(g);
    });
  });
  target.append(s);
  const legend = el('div', undefined, 'legend');
  for (const ctl of controllers) {
    const item = el('span', undefined, 'legend-item');
    item.append(el('span', undefined, `swatch src-${SOURCE_OF[ctl.key]}`), el('span', ctl.label));
    legend.append(item);
  }
  legend.append(el('span', '– no protocol result yet (training) · bars: mean over seeds', 'muted'));
  target.append(legend);
}

/* ---------- what each system learns (stage diagram) ---------- */
function renderPipeline(study) {
  const target = $('pipeline');
  target.replaceChildren();
  const stages = study.stages || [];
  const grid = el('div', undefined, 'pgrid');
  grid.style.gridTemplateColumns = `minmax(84px, 1.3fr) repeat(${stages.length}, minmax(0, 1fr))`;
  grid.append(el('div'));
  stages.forEach(name => grid.append(el('div', name, 'phead')));
  const labels = Object.fromEntries((study.controllers || []).map(c => [c.key, c.label]));
  const effectors = Object.fromEntries((study.effectors || []).map(e => [e.key, e.label]));
  for (const c of study.conditions || []) {
    if (!c.pipeline) continue;
    const name = el('div', undefined, 'prow');
    name.append(el('strong', effectors[c.effector] || c.effector), el('span', labels[c.controller] || c.controller, 'muted small'));
    grid.append(name);
    c.pipeline.forEach((source, i) => {
      const cell = el('div', SOURCE_TEXT[source] || source, `pcell src-${source}`);
      hover(cell, [[`${effectors[c.effector]} · ${labels[c.controller]}`, 'tip-title'], [`${stages[i]}: ${(study.sources || []).find(s => s.key === source)?.label || source}`]]);
      grid.append(cell);
    });
  }
  target.append(grid);
  const legend = el('div', undefined, 'legend');
  for (const s of study.sources || []) {
    const item = el('span', undefined, 'legend-item');
    item.append(el('span', undefined, `swatch src-${s.key}`), el('span', s.label));
    legend.append(item);
  }
  target.append(legend);
}

/* ---------- algorithm comparison (learning curves per seed) ---------- */
const ALGO_SOURCE = {ppo: 'imitation_rl', rlpd: 'rl'};
function comparisonChart(algorithms, xKey, xLabel, xFormat) {
  const width = 460, height = 210, left = 40, right = 70, top = 12, bottom = 34;
  const all = algorithms.flatMap(a => a.curves.flatMap(c => c.points.map(p => p[xKey])));
  const xMax = Math.max(1e-9, ...all);
  const x = v => left + (v / xMax) * (width - left - right);
  const y = v => top + (1 - v) * (height - top - bottom);
  const s = svg('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': `Lift and carry against ${xLabel}`});
  for (const tick of [0, .5, 1]) {
    s.append(svg('line', {x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: 'grid'}));
    s.append(svg('text', {x: left - 6, y: y(tick) + 4, class: 'axis', 'text-anchor': 'end'}, `${tick * 100}%`));
  }
  for (const f of [0, .5, 1]) s.append(svg('text', {x: x(xMax * f), y: height - bottom + 15, class: 'axis', 'text-anchor': 'middle'}, xFormat(xMax * f)));
  s.append(svg('text', {x: (left + width - right) / 2, y: height - 3, class: 'axis', 'text-anchor': 'middle'}, xLabel));
  for (const a of algorithms) {
    for (const curve of a.curves) {
      const pts = curve.points.filter(p => Number.isFinite(p[xKey]) && Number.isFinite(p.carried));
      if (!pts.length) continue;
      const g = svg('g', {class: `src-${ALGO_SOURCE[a.key] || 'planned'}`});
      g.append(svg('polyline', {points: pts.map(p => `${x(p[xKey])},${y(p.carried)}`).join(' '), class: 'cmp-line'}));
      const last = pts[pts.length - 1];
      g.append(svg('circle', {cx: x(last[xKey]), cy: y(last.carried), r: 3.5, class: 'cmp-dot'}));
      g.append(svg('text', {x: x(last[xKey]) + 6, y: y(last.carried) + 4, class: 'label'}, `${a.label.split(' ').pop()} s${curve.id.slice(-1)}`));
      hover(g, [[`${a.label} · ${curve.id}`, 'tip-title'], [`latest: ${percent(last.carried)} at ${(last.steps / 1e6).toFixed(1)}M steps, ${last.hours.toFixed(1)} h`],
                [`best: ${percent(Math.max(...pts.map(p => p.carried)))}`, 'muted']]);
      s.append(g);
    }
  }
  return s;
}

function renderComparison(study) {
  const target = $('comparison');
  target.replaceChildren();
  const cmp = study.comparison;
  if (!cmp) return;
  $('comparison-label').textContent = cmp.label || '';
  const algorithms = cmp.algorithms || [];
  target.append(figureBox('Lift and carry vs environment steps', comparisonChart(algorithms, 'steps', 'environment steps', v => `${(v / 1e6).toFixed(1)}M`)),
                figureBox('Lift and carry vs wall-clock time', comparisonChart(algorithms, 'hours', 'hours of training', v => `${v.toFixed(1)} h`)));
  const legend = el('div', undefined, 'legend');
  for (const a of algorithms) {
    const item = el('span', undefined, 'legend-item');
    item.append(el('span', undefined, `swatch src-${ALGO_SOURCE[a.key] || 'planned'}`), el('span', a.label));
    legend.append(item);
  }
  legend.append(el('span', 'one line per training seed · hand-task evaluations (12 episodes per object)', 'muted'));
  target.append(legend);
}
function figureBox(title, content) {
  const box = el('div', undefined, 'figbox');
  box.append(el('div', title, 'small figtitle'), content);
  return box;
}

/* ---------- training curves ---------- */
function trainingChart(run) {
  const width = 460, height = 190, left = 40, right = 96, top = 12, bottom = 32;
  const points = run.points.filter(p => Number.isFinite(p.transitions));
  const xMax = Math.max(run.budget || 0, ...points.map(p => p.transitions), 1);
  const x = v => left + (v / xMax) * (width - left - right);
  const y = v => top + (1 - v) * (height - top - bottom);
  const s = svg('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': `${run.id} evaluation success`});
  for (const tick of [0, .5, 1]) {
    s.append(svg('line', {x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: 'grid'}));
    s.append(svg('text', {x: left - 6, y: y(tick) + 4, class: 'axis', 'text-anchor': 'end'}, `${tick * 100}%`));
  }
  const step = xMax > 30e6 ? 10e6 : xMax > 10e6 ? 5e6 : 2e6;
  for (let v = 0; v <= xMax + 1; v += step) {
    s.append(svg('text', {x: x(v), y: height - bottom + 15, class: 'axis', 'text-anchor': 'middle'}, `${v / 1e6}M`));
  }
  s.append(svg('text', {x: (left + width - right) / 2, y: height - 3, class: 'axis', 'text-anchor': 'middle'}, 'environment steps'));
  const series = [['success_rate', 'train-lift', 'Lift']];
  if (run.carry) series.push(['carry_success_rate', 'train-carry', 'Carry']);
  const labels = [];
  for (const [key, cls, name] of series) {
    const valid = points.filter(p => Number.isFinite(p[key]));
    if (!valid.length) continue;
    s.append(svg('polyline', {points: valid.map(p => `${x(p.transitions)},${y(p[key])}`).join(' '), class: `train-line ${cls}`}));
    const last = valid[valid.length - 1];
    s.append(svg('circle', {cx: x(last.transitions), cy: y(last[key]), r: 4, class: `train-dot ${cls}`}));
    labels.push({y: y(last[key]), text: `${name} ${Math.round(last[key] * 100)}%`, x: x(last.transitions)});
  }
  labels.sort((a, b) => b.y - a.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.min(labels[i].y, labels[i - 1].y - 14);
  for (const label of labels) s.append(svg('text', {x: label.x + 8, y: label.y + 4, class: 'label'}, label.text));
  const cross = svg('line', {y1: top, y2: height - bottom, class: 'crosshair', visibility: 'hidden'});
  const hit = svg('rect', {x: left, y: top, width: width - left - right, height: height - top - bottom, fill: 'transparent'});
  s.append(cross, hit);
  hit.addEventListener('mousemove', event => {
    if (!points.length) return;
    const box = s.getBoundingClientRect();
    const value = ((event.clientX - box.left) / box.width * width - left) / (width - left - right) * xMax;
    const p = points.reduce((a, b) => Math.abs(b.transitions - value) < Math.abs(a.transitions - value) ? b : a);
    cross.setAttribute('x1', x(p.transitions)); cross.setAttribute('x2', x(p.transitions)); cross.setAttribute('visibility', 'visible');
    const objects = Object.entries(p.per_object || {}).map(([k, v]) => `${OBJECT_NAMES[k] || k} ${Math.round(v * 100)}%`).join(' · ');
    showTip(event, [[`${(p.transitions / 1e6).toFixed(1)}M steps`, 'tip-title'],
      [`Lift ${percent(p.success_rate)}` + (run.carry ? ` · carry ${percent(p.carry_success_rate)}` : '')], [objects, 'muted']]);
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  return s;
}

function objectBars(point) {
  const box = el('div', undefined, 'obj-bars');
  for (const [id, value] of Object.entries((point && point.per_object) || {})) {
    const row = el('div', undefined, 'obj-bar');
    const track = el('span', undefined, 'obj-track');
    const fill = el('span', undefined, 'obj-fill');
    fill.style.width = `${Math.round((value || 0) * 100)}%`;
    track.append(fill);
    row.append(el('span', OBJECT_NAMES[id] || id, 'obj-name'), track, el('span', percent(value), 'obj-value'));
    box.append(row);
  }
  return box;
}

let trainingSignature = '';
function renderTraining(curves) {
  const target = $('training');
  const signature = JSON.stringify(curves.map(c => [c.id, c.status, c.points.length, c.transitions]));
  if (signature === trainingSignature) return;
  trainingSignature = signature;
  target.replaceChildren();
  if (!curves.length) { target.append(el('p', 'No hand training runs.', 'muted')); return; }
  for (const run of curves) {
    const card = el('div', undefined, 'train-card');
    const last = run.points[run.points.length - 1];
    const speed = last && last.steps_per_second;
    const remaining = run.status === 'running' && speed && run.budget ? (run.budget - (run.transitions || 0)) / speed / 3600 : null;
    const head = el('div', undefined, 'train-head');
    head.append(el('strong', run.id), el('span', run.status || '—', `chip chip-${run.status === 'running' ? 'run' : 'idle'}`));
    const meta = [run.robot === 'fetch' ? 'on the Fetch' : 'test rig',
                  run.init_from ? `from ${run.init_from}` : run.imitation ? 'imitation, then PPO' : 'from scratch',
                  `${((run.transitions || 0) / 1e6).toFixed(1)}M / ${((run.budget || 0) / 1e6).toFixed(0)}M steps`,
                  remaining !== null ? `~${remaining.toFixed(1)} h left` : null].filter(Boolean);
    card.append(head, el('div', meta.join(' · '), 'muted small'), trainingChart(run), objectBars(last));
    target.append(card);
  }
  const legend = el('div', undefined, 'legend');
  for (const [label, cls] of [['Lift 15 cm (evaluation)', 'swatch-lift'], ['Lift and carry', 'swatch-carry']]) {
    const item = el('span', undefined, 'legend-item');
    item.append(el('span', undefined, `swatch ${cls}`), el('span', label));
    legend.append(item);
  }
  target.append(legend);
}

/* ---------- clips, header, footer ---------- */
let recordingSignature = '';
function renderRecordings(list) {
  const signature = JSON.stringify(list.map(r => [r.run, (r.media || []).map(m => m.path)]));
  if (signature === recordingSignature) return;  // never interrupt playback while polling
  recordingSignature = signature;
  const target = $('recordings');
  target.replaceChildren();
  for (const recording of list) {
    const group = el('div', undefined, 'recording-group');
    group.append(el('h3', recording.label));
    const strip = el('div', undefined, 'videos');
    for (const media of recording.media || []) {
      const fig = el('figure'), video = el('video');
      video.controls = true; video.preload = 'none';
      const url = path => `/artifacts/${encodeURIComponent(recording.run)}/${path.split('/').map(encodeURIComponent).join('/')}`;
      video.src = url(media.path);
      if (media.poster) video.poster = url(media.poster);
      fig.append(video, el('figcaption', media.label));
      strip.append(fig);
    }
    group.append(strip);
    target.append(group);
  }
}

function renderHeader(study, resources, timestamp) {
  $('title').textContent = study.title || 'Project dashboard';
  $('question').textContent = study.claim ? `Claim under test: ${study.claim}` : '';
  $('protocol').textContent = study.protocol ? `Protocol (frozen ${study.protocol.frozen}): ${study.protocol.summary}` : '';
  $('updated').textContent = `Updated ${when(timestamp)}`;
  const r = $('resources');
  r.replaceChildren();
  if (resources) {
    const parts = [`CPU ${resources.load_1m ?? '—'}/${resources.logical_cpus ?? '—'}`, `RAM free ${bytes(resources.memory_available_bytes)}`];
    for (const gpu of resources.gpus || []) parts.push(`GPU ${Math.round(gpu.utilization)}%`);
    for (const p of parts) r.append(el('span', p));
  }
}

function renderFooter(overview, active) {
  const docs = $('documents');
  docs.replaceChildren();
  for (const doc of overview.documents || []) {
    const a = el('a', doc.label); a.href = doc.href;
    docs.append(a);
  }
  const target = $('active');
  target.replaceChildren();
  if (!active.length) return;
  target.append(el('div', 'Running jobs', 'muted small'));
  for (const job of active) {
    const row = el('div', undefined, 'job');
    const bar = el('div', undefined, 'pbar'); const fill = el('div');
    fill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    bar.append(fill);
    row.append(el('span', job.id, 'job-name'), bar, el('span', job.detail || '', 'muted small'));
    target.append(row);
  }
}

let drawnSignature = '';
async function poll() {
  try {
    const response = await fetch('/api/overview', {cache: 'no-store'});
    const state = await response.json();
    const study = state.study || {};
    renderHeader(study, state.resources, state.timestamp);
    const signature = JSON.stringify(study);
    if (signature !== drawnSignature) {
      drawnSignature = signature;
      renderPlan(study);
      renderMatrix(study);
      renderAdded(study);
      renderPipeline(study);
    }
    renderComparison(study);
    renderTraining(state.training || []);
    renderRecordings((state.overview || {}).recordings || []);
    renderFooter(state.overview || {}, state.active || []);
  } catch (error) {
    $('updated').textContent = `Connection problem: ${error.message}`;
  }
  setTimeout(poll, 15000);
}

poll();
