'use strict';
// Read-only dashboard: diagrams are drawn as SVG from /api/overview; all data is inserted as text.

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
const pct = (n, d) => (d ? Math.round((100 * n) / d) : null);
const when = iso => (iso ? new Date(iso).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) : '—');
const bytes = n => (n ? `${(n / 1024 ** 3).toFixed(0)} GB` : '—');

const COLORS = {planned: '#3b6fd8', imitation: '#27a06b', imitation_sac: '#27a06b', imitation_td3: '#27a06b',
                rl: '#e5891b', hold: '#9aa3b2', settle: '#c9ced8'};
const OBJECTS = ['foam', 'banana', 'can', 'apple', 'mug'];
const FAILURES = [
  ['Dropped in carry', '#d64545', l => l === 'carry: object dropped'],
  ['Lost in maze', '#e5891b', l => l === 'carry: ran out of decisions'],
  ['Hit a wall', '#7d5bd6', l => l.startsWith('carry: collision')],
  ['Pickup / posture', '#3b6fd8', l => /^(grasp|hold|planned_carry_posture|pick)/.test(l)],
  ['Reach / final hold', '#2aa3b8', l => /^(reach|final_hold)/.test(l)],
];
const STAGE_PHASES = {  // learned system and classical pipeline stage names -> phase keys
  planned_approach: 'approach', fold: 'approach', approach: 'approach', pregrasp: 'approach',
  grasp: 'grasp', descend: 'grasp', close: 'grasp', lift: 'grasp',
  pick_hold: 'hold', hold: 'hold', planned_carry_posture: 'posture', retreat: 'posture', carry_posture: 'posture',
  carry: 'carry', reach: 'reach', final_hold: 'final'};

let STATE = {evaluations: {}, overview: {}};
const run = id => STATE.evaluations[id];

/* ---------- small chart helpers ---------- */
function frame(width, height) {
  return svg('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img'});
}
function legend(items) {
  const box = el('div', undefined, 'legend');
  for (const [label, color, cls] of items) {
    const item = el('span', undefined, 'legend-item');
    const swatch = el('span', undefined, 'swatch ' + (cls || ''));
    if (color) swatch.style.background = color;
    item.append(swatch, el('span', label));
    box.append(item);
  }
  return box;
}
function figure(title, caption, ...content) {
  const box = el('figure', undefined, 'fig');
  box.append(el('h3', title), ...content);
  if (caption) box.append(el('figcaption', caption));
  return box;
}

function lineChart(series, {xs, xLabel, yMax = 100, yLabel = '%', width = 300, height = 180}) {
  const s = frame(width, height), left = 38, right = 24, top = 12, bottom = 34;
  const x = v => left + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (width - left - right);
  const y = v => top + (1 - v / yMax) * (height - top - bottom);
  for (const tick of [0, yMax / 2, yMax]) {
    s.append(svg('line', {x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: 'grid'}));
    s.append(svg('text', {x: left - 6, y: y(tick) + 4, class: 'axis', 'text-anchor': 'end'}, `${tick}${yLabel}`));
  }
  for (const v of xs) s.append(svg('text', {x: x(v), y: height - bottom + 16, class: 'axis', 'text-anchor': 'middle'}, String(v)));
  s.append(svg('text', {x: (left + width - right) / 2, y: height - 4, class: 'axis', 'text-anchor': 'middle'}, xLabel));
  for (const {points, color, dashed} of series) {
    const valid = points.filter(([, v]) => v !== null && v !== undefined);
    if (!valid.length) continue;
    s.append(svg('polyline', {points: valid.map(([a, b]) => `${x(a)},${y(b)}`).join(' '), fill: 'none',
      stroke: color, 'stroke-width': 2.5, 'stroke-dasharray': dashed ? '5 4' : 'none'}));
    for (const [a, b] of valid) s.append(svg('circle', {cx: x(a), cy: y(b), r: 3.5, fill: color}));
  }
  return s;
}

function heatGrid({rows, cols, values, format, color, width = 300}) {
  const cell = (width - 110) / cols.length, height = 30 + rows.length * 44;
  const s = frame(width, height);
  cols.forEach((c, j) => s.append(svg('text', {x: 110 + cell * (j + 0.5), y: 18, class: 'axis', 'text-anchor': 'middle'}, c)));
  rows.forEach((r, i) => {
    s.append(svg('text', {x: 104, y: 30 + 44 * i + 26, class: 'label', 'text-anchor': 'end'}, r));
    cols.forEach((c, j) => {
      const v = values[i][j];
      s.append(svg('rect', {x: 110 + cell * j + 2, y: 30 + 44 * i + 2, width: cell - 4, height: 40, rx: 6,
        fill: v === null ? '#eceff4' : color(v)}));
      s.append(svg('text', {x: 110 + cell * (j + 0.5), y: 30 + 44 * i + 27, class: 'cell', 'text-anchor': 'middle'},
        v === null ? '—' : format(v)));
    });
  });
  return s;
}
const goodScale = v => `hsl(${Math.max(0, Math.min(120, (v - 50) * 2.4))}, 55%, 58%)`;   // 50% red .. 100% green
const badScale = v => `hsl(${Math.max(0, 120 - v * 4.4)}, 60%, 60%)`;                     // 0% green .. ~27% red

/* ---------- sections ---------- */
function renderHeader(overview, resources, timestamp) {
  $('title').textContent = overview.title || 'Pick–Nav–Reach';
  $('phase').textContent = overview.phase || '';
  $('summary').textContent = overview.summary || '';
  $('updated').textContent = `Updated ${when(timestamp)}`;
  const r = $('resources');
  r.replaceChildren();
  if (resources) {
    const parts = [`CPU ${resources.load_1m ?? '—'}/${resources.logical_cpus ?? '—'}`, `RAM free ${bytes(resources.memory_available_bytes)}`,
      `Disk free ${bytes(resources.disk_free_bytes)}`];
    for (const gpu of resources.gpus || []) parts.push(`GPU ${Math.round(gpu.utilization)}%`);
    for (const p of parts) r.append(el('span', p));
  }
  const cards = $('headline');
  cards.replaceChildren();
  for (const card of overview.headline || []) {
    const box = el('div', undefined, 'card');
    box.append(el('div', card.label, 'card-label'), el('div', card.value, 'card-value'), el('div', card.detail, 'card-detail'));
    cards.append(box);
  }
}

function resultBar(evaluation) {
  const wrap = el('div', undefined, 'result');
  if (!evaluation) { wrap.append(el('span', '—', 'muted')); return wrap; }
  const complete = pct(evaluation.success, evaluation.completed), timed = pct(evaluation.timed, evaluation.completed);
  const bar = el('div', undefined, 'rbar');
  const a = el('div', undefined, 'rbar-complete'); a.style.width = `${complete}%`;
  const b = el('div', undefined, 'rbar-timed'); b.style.width = `${timed}%`;
  bar.append(a, b);
  wrap.append(bar, el('span', `${complete}%`, 'rnum'));
  wrap.title = `${evaluation.success}/${evaluation.completed} complete · ${evaluation.timed} within 15k ticks`;
  return wrap;
}

function renderSystems(overview) {
  const target = $('systems');
  target.replaceChildren();
  const phases = overview.phases || [], sets = overview.sets || [];
  const grid = el('div', undefined, 'sysgrid');
  grid.style.gridTemplateColumns = `150px repeat(${phases.length}, minmax(56px, 1fr)) repeat(${sets.length}, minmax(90px, 1.3fr))`;
  grid.append(el('div'));
  for (const [i, p] of phases.entries()) {
    const head = el('div', undefined, 'phase-head');
    head.append(el('div', String(i + 1), 'phase-num'), el('div', p.label));
    grid.append(head);
  }
  for (const s of sets) {
    const head = el('div', undefined, 'set-head');
    head.append(el('div', s.label), el('div', s.detail, 'muted small'));
    grid.append(head);
  }
  for (const system of overview.systems || []) {
    const name = el('div', undefined, 'sys-name' + (system.best ? ' best' : '') + (system.planned ? ' planned-row' : ''));
    name.append(el('strong', system.name));
    if (system.best) name.append(el('span', 'best', 'tag tag-best'));
    if (system.planned) name.append(el('span', 'next', 'tag'));
    grid.append(name);
    system.cells.forEach(kind => {
      const cell = el('div', (overview.controllers[kind] || {}).label, `ctl ctl-${kind}`);
      cell.title = (overview.controllers[kind] || {}).detail || '';
      grid.append(cell);
    });
    for (const s of sets) grid.append(resultBar(run((system.runs || {})[s.key])));
  }
  target.append(grid);
  target.append(legend([
    ['Planner', COLORS.planned], ['Imitation (cloned from teacher)', COLORS.imitation],
    ['Imitation → SAC / TD3 fine-tuning', null, 'swatch-sac'], ['RL from reward (SAC+HER)', COLORS.rl], ['Fixed hold', COLORS.hold],
    ['Complete', null, 'swatch-complete'], ['Within 15k-tick deadline', null, 'swatch-timed']]));
}

function renderTimeline(overview) {
  const target = $('timeline');
  target.replaceChildren();
  const rows = (overview.figures?.timeline || []).map(([label, id]) => [label, run(id)]).filter(([, e]) => e);
  const systemCells = Object.fromEntries((overview.systems || []).map(s => [s.name, s.cells]));
  const phaseIndex = Object.fromEntries((overview.phases || []).map((p, i) => [p.key, i]));
  const width = 900, left = 130, height = 40 + rows.length * 46, max = 16000;
  const s = frame(width, height);
  const x = v => left + (v / max) * (width - left - 20);
  for (let t = 0; t <= max; t += 2000) {
    s.append(svg('line', {x1: x(t), x2: x(t), y1: 18, y2: height - 6, class: 'grid'}));
    s.append(svg('text', {x: x(t), y: 12, class: 'axis', 'text-anchor': 'middle'}, `${t / 1000}k`));
  }
  rows.forEach(([label, evaluation], i) => {
    const y = 26 + i * 46;
    s.append(svg('text', {x: left - 8, y: y + 21, class: 'label', 'text-anchor': 'end'}, label));
    const totals = {settle: 480};
    for (const [stage, ticks] of Object.entries(evaluation.stage_ticks_median || {})) {
      if (evaluation.controller === 'reference' && stage === 'pick_hold') continue;  // duplicate of 'hold'
      const phase = STAGE_PHASES[stage];
      if (phase) totals[phase] = (totals[phase] || 0) + ticks;
    }
    let at = 0;
    for (const key of ['settle', ...(overview.phases || []).map(p => p.key)]) {
      const ticks = totals[key];
      if (!ticks) continue;
      const kind = key === 'settle' ? 'settle' : (systemCells[label] || [])[phaseIndex[key]];
      s.append(svg('rect', {x: x(at), y, width: Math.max(1, x(at + ticks) - x(at) - 1), height: 32, rx: 3, fill: COLORS[kind] || '#999'}));
      if (x(at + ticks) - x(at) > 48) {
        const name = key === 'settle' ? 'settle' : (overview.phases.find(p => p.key === key) || {}).label;
        s.append(svg('text', {x: x(at) + 5, y: y + 20, class: 'seg'}, name));
      }
      at += ticks;
    }
    s.append(svg('text', {x: x(at) + 4, y: y + 20, class: 'axis'}, `${(at / 1000).toFixed(1)}k`));
  });
  s.append(svg('line', {x1: x(15000), x2: x(15000), y1: 18, y2: height - 4, stroke: '#d64545', 'stroke-width': 2, 'stroke-dasharray': '6 4'}));
  s.append(svg('text', {x: x(15000) - 4, y: height - 8, class: 'axis deadline', 'text-anchor': 'end'}, 'deadline 15k ticks'));
  target.append(s, el('div', 'Left to right: settle → ' + (overview.phases || []).map(p => p.label).join(' → ') + ' · colours as above', 'muted small'));
}

function renderImitation(overview) {
  const target = $('imitation');
  target.replaceChildren();
  const wrap = text => String(text).split(' · ').flatMap(part => {
    const lines = [''];
    for (const word of part.split(' ')) {
      if ((lines[lines.length - 1] + ' ' + word).trim().length > 26) lines.push(word);
      else lines[lines.length - 1] = (lines[lines.length - 1] + ' ' + word).trim();
    }
    return lines;
  }).slice(0, 4);
  const boxH = 70, gap = 34, width = 900, height = 40 + (overview.imitation || []).length * (boxH + gap) - gap + 6;
  const s = frame(width, height);
  const columns = [['Scripted teacher', 20, COLORS.planned], ['Demonstrations', 250, '#6b7486'],
                   ['Behaviour cloning', 480, COLORS.imitation], ['SAC+HER fine-tuning', 690, COLORS.rl]];
  columns.forEach(([label, x]) => s.append(svg('text', {x: x + 95, y: 18, class: 'label', 'text-anchor': 'middle'}, label)));
  (overview.imitation || []).forEach((skill, i) => {
    const y = 40 + i * (boxH + gap);
    const boxes = [[skill.teacher, 20], [skill.data, 250], [skill.policy, 480], ['optional · online practice', 690]];
    boxes.forEach(([text, x], j) => {
      s.append(svg('rect', {x, y, width: 190, height: boxH, rx: 8, fill: 'none', stroke: columns[j][2], 'stroke-width': 2,
        'stroke-dasharray': j === 3 ? '5 4' : 'none'}));
      const lines = wrap(text);
      lines.forEach((line, k) => s.append(svg('text', {x: x + 95, y: y + boxH / 2 + 4 + (k - (lines.length - 1) / 2) * 14,
        class: 'boxtext', 'text-anchor': 'middle'}, line)));
      if (j < 3) s.append(svg('path', {d: `M${x + 194} ${y + boxH / 2} L${x + 222} ${y + boxH / 2}`, class: 'arrow', 'marker-end': 'url(#arrowhead)'}));
    });
    s.append(svg('text', {x: 20, y: y - 8, class: 'skill'}, skill.skill));
  });
  const defs = svg('defs');
  const marker = svg('marker', {id: 'arrowhead', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto'});
  marker.append(svg('path', {d: 'M0,0 L10,5 L0,10 z', fill: '#8a93a6'}));
  defs.append(marker);
  s.prepend(defs);
  target.append(s);
}

/* ---------- animated maze ---------- */
function pathLengths(points) {
  const lengths = [0];
  for (let i = 1; i < points.length; i++) lengths.push(lengths[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]));
  return lengths;
}
function motion(points, times, total, color, radius, endMarker) {
  // Dot moving along points at the given cumulative times (native calls); holds at the end until the loop restarts.
  const lengths = pathLengths(points), L = lengths[lengths.length - 1] || 1;
  const keyPoints = [], keyTimes = [];
  points.forEach((_, i) => { keyPoints.push((lengths[i] / L).toFixed(4)); keyTimes.push((times[i] / total).toFixed(4)); });
  if (times[times.length - 1] < total) { keyPoints.push('1'); keyTimes.push('1'); }
  const dot = svg('circle', {r: radius, fill: color, stroke: '#fff', 'stroke-width': 0.06});
  dot.append(svg('animateMotion', {dur: `${(total / 330).toFixed(1)}s`, repeatCount: 'indefinite', calcMode: 'linear',
    keyPoints: keyPoints.join(';'), keyTimes: keyTimes.join(';'),
    path: 'M' + points.map(p => `${p[0]},${-p[1]}`).join(' L')}));
  return dot;
}

function renderMaze(maze) {
  const target = $('maze');
  target.replaceChildren();
  if (!maze) { target.append(el('p', 'Maze example not generated (scripts/dashboard_maze.py).', 'muted')); return; }
  const s = svg('svg', {viewBox: '-4.2 -6.2 22.4 12.4', class: 'maze'});
  for (const [x, y] of maze.walls) s.append(svg('rect', {x: x - 0.5, y: -y - 0.5, width: 1, height: 1, class: 'wall'}));
  const [[tx0, ty0], [tx1, ty1]] = maze.table;
  s.append(svg('rect', {x: tx0, y: -ty1, width: tx1 - tx0, height: ty1 - ty0, class: 'table'}));
  s.append(svg('text', {x: (tx0 + tx1) / 2, y: 0.18, class: 'maze-label', 'text-anchor': 'middle'}, 'table'));
  s.append(svg('circle', {cx: maze.goal[0], cy: -maze.goal[1], r: 0.35, class: 'goal'}));
  s.append(svg('text', {x: maze.goal[0] - 0.5, y: -maze.goal[1] - 0.5, class: 'maze-label', 'text-anchor': 'end'}, 'goal'));
  const poly = (points, cls) => svg('polyline', {points: points.map(p => `${p[0]},${-p[1]}`).join(' '), class: cls});
  s.append(poly(maze.astar, 'path-astar'));
  s.append(poly(maze.learned_success, 'path-learned'));
  s.append(poly(maze.learned_drop, 'path-drop'));
  for (const [x, y] of maze.astar.slice(1, -1)) s.append(svg('circle', {cx: x, cy: -y, r: 0.09, class: 'waypoint'}));
  const [dx, dy] = maze.drop_point;
  s.append(svg('path', {d: `M${dx - .3},${-dy - .3} L${dx + .3},${-dy + .3} M${dx - .3},${-dy + .3} L${dx + .3},${-dy - .3}`, class: 'drop-x'}));
  // Timing in native calls: learned traces are sampled every 5 calls; A* moves 18 mm per call and pauses 8 calls per waypoint.
  const step = maze.trace_interval_native_calls || 5;
  const learnedTimes = maze.learned_success.map((_, i) => i * step);
  const dropTimes = maze.learned_drop.map((_, i) => i * step);
  const astarTimes = [0];
  for (let i = 1; i < maze.astar.length; i++) {
    const a = maze.astar[i - 1], b = maze.astar[i];
    astarTimes.push(astarTimes[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.018 + 8);
  }
  const total = Math.max(learnedTimes.at(-1), dropTimes.at(-1), astarTimes.at(-1)) + 120;
  s.append(motion(maze.astar, astarTimes, total, COLORS.planned, 0.28));
  s.append(motion(maze.learned_success, learnedTimes, total, COLORS.imitation, 0.3));
  s.append(motion(maze.learned_drop, dropTimes, total, '#d64545', 0.17));  // on top: paths coincide until the drop
  target.append(s);
  target.append(legend([['Planner: A* waypoints, stops at each', COLORS.planned, 'swatch-dash'],
    [maze.captions?.success || 'Learned carry', COLORS.imitation], [maze.captions?.drop || 'Learned carry, dropped', '#d64545']]));
  // In-hand slip over the carry.
  const n = Math.max(maze.slip_success_mm.length, maze.slip_drop_mm.length);
  const xs = [0, Math.round(n * step / 100) * 100];
  const toPoints = arr => arr.map((v, i) => [i * step, Math.max(-60, v)]);
  const chart = slipChart(toPoints(maze.slip_success_mm), toPoints(maze.slip_drop_mm), xs[1]);
  $('slip').replaceChildren(figure('Object height in the hand during carry', 'Delivered object settles ~10 mm; dropped object slides ~30 mm, then falls', chart));
}

function slipChart(success, drop, xMax) {
  const width = 360, height = 200, left = 58, right = 10, top = 10, bottom = 30;
  const s = frame(width, height);
  const x = v => left + (v / xMax) * (width - left - right);
  const y = v => top + (-v / 60) * (height - top - bottom);
  for (const tick of [0, -20, -40, -60]) {
    s.append(svg('line', {x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: 'grid'}));
    s.append(svg('text', {x: left - 6, y: y(tick) + 4, class: 'axis', 'text-anchor': 'end'}, `${tick} mm`));
  }
  s.append(svg('text', {x: (left + width) / 2, y: height - 4, class: 'axis', 'text-anchor': 'middle'}, 'native calls into the carry'));
  for (const [points, color] of [[success, COLORS.imitation], [drop, '#d64545']]) {
    s.append(svg('polyline', {points: points.map(([a, b]) => `${x(a)},${y(b)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2.5}));
  }
  return s;
}

/* ---------- results ---------- */
function renderResults(overview) {
  const systems = (overview.systems || []).filter(s => !s.planned);
  // Per-object heat map on dev.
  const objectValues = systems.map(sys => OBJECTS.map(name => {
    const e = run((sys.runs || {}).dev), entry = e && e.by_object[name];
    return entry ? pct(entry.success, entry.cases) : null;
  }));
  const objects = heatGrid({rows: systems.map(s => s.name), cols: OBJECTS, values: objectValues, format: v => `${v}%`, color: goodScale, width: 460});
  // Failure breakdown on dev, stacked per system.
  const width = 460, rowH = 40, height = 20 + systems.length * rowH;
  const failures = frame(width, height);
  const maxFail = 80;
  systems.forEach((sys, i) => {
    const e = run((sys.runs || {}).dev);
    const y = 10 + i * rowH;
    failures.append(svg('text', {x: 104, y: y + 20, class: 'label', 'text-anchor': 'end'}, sys.name));
    if (!e) return;
    let at = 0;
    for (const [label, color, test] of FAILURES) {
      const count = Object.entries(e.failures).filter(([l]) => test(l)).reduce((a, [, c]) => a + c, 0);
      if (!count) continue;
      const w = (count / maxFail) * (width - 160);
      failures.append(svg('rect', {x: 110 + at, y, width: Math.max(1, w - 1), height: 28, rx: 3, fill: color}));
      if (w > 18) failures.append(svg('text', {x: 110 + at + w / 2, y: y + 19, class: 'seg', 'text-anchor': 'middle'}, String(count)));
      at += w;
    }
  });
  $('results').replaceChildren(
    figure('Success by object · dev', null, objects),
    figure('Why runs fail · dev (count of 260)', null, failures, legend(FAILURES.map(([l, c]) => [l, c]))));
}

function renderFindings(overview) {
  const f = overview.figures || {};
  const series = (pairs, key) => pairs.map(([x, id]) => [x, run(id) ? pct(run(id)[key], run(id).completed) : null]);
  const planned = lineChart([{points: series(f.planned_speed || [], 'success'), color: COLORS.imitation},
                             {points: series(f.planned_speed || [], 'timed'), color: COLORS.planned, dashed: true}],
                            {xs: (f.planned_speed || []).map(p => p[0]), xLabel: 'planned motion speed (×)', yMax: 100});
  const carry = lineChart([{points: series(f.carry_speed || [], 'success'), color: COLORS.imitation},
                           {points: (f.carry_speed || []).map(([x, id]) => [x, run(id) ? Math.round(100 * run(id).drops / run(id).completed) : null]), color: '#d64545'}],
                          {xs: (f.carry_speed || []).map(p => p[0]), xLabel: 'carry step (m per decision)', yMax: 100});
  const grid = spec => spec && spec.runs.map(r => r.map(id => (run(id) ? run(id) : null)));
  const sac = grid(f.sac_ablation), swap = grid(f.swap);
  const sacFig = sac && heatGrid({rows: f.sac_ablation.rows, cols: f.sac_ablation.cols, width: 330,
    values: sac.map(r => r.map(e => (e ? pct(e.success, e.completed) : null))), format: v => `${v}%`, color: goodScale});
  const swapFig = swap && heatGrid({rows: f.swap.rows, cols: f.swap.cols, width: 330,
    values: swap.map(r => r.map(e => (e ? Math.round(100 * e.drops / e.completed) : null))), format: v => `${v}% drop`, color: badScale});
  // Carry rules: drops vs wall collisions.
  const rules = f.carry_rules || [], width = 330, rowH = 30, height = 14 + rules.length * rowH;
  const rulesFig = frame(width, height);
  rules.forEach(([label, id], i) => {
    const e = run(id), y = 6 + i * rowH;
    rulesFig.append(svg('text', {x: 92, y: y + 16, class: 'label', 'text-anchor': 'end'}, label));
    if (!e) return;
    const walls = Object.entries(e.failures).filter(([l]) => l.startsWith('carry: collision')).reduce((a, [, c]) => a + c, 0);
    const scale = v => (v / 170) * (width - 170);
    rulesFig.append(svg('rect', {x: 98, y, width: Math.max(1, scale(e.drops)), height: 11, rx: 2, fill: '#d64545'}));
    rulesFig.append(svg('rect', {x: 98, y: y + 12, width: Math.max(1, scale(walls)), height: 11, rx: 2, fill: '#7d5bd6'}));
    rulesFig.append(svg('text', {x: 102 + scale(Math.max(e.drops, walls)), y: y + 16, class: 'axis'}, `${e.drops} / ${walls}`));
  });
  $('findings').replaceChildren(
    figure('Faster planned motion', 'Approach and posture run faster → more runs meet the deadline',
      planned, legend([['complete', COLORS.imitation], ['within deadline', COLORS.planned, 'swatch-dash']])),
    figure('Carry speed', 'Slowing the carry does not reduce drops',
      carry, legend([['complete', COLORS.imitation], ['dropped in carry', '#d64545']])),
    figure('Does SAC fine-tuning help?', 'Complete · dev · all within noise (±3%)', sacFig || el('p', '—')),
    figure('Which stage causes drops?', 'Drop rate · dev · the carry controller matters, the grasp barely', swapFig || el('p', '—')),
    figure('Hand-written carry rules', 'Drops (red) vs wall hits (purple) · dev', rulesFig));
}

function renderRubric(overview) {
  const rows = (overview.figures?.rubric || []).map(([label, id]) => [label, run(id)]).filter(([, e]) => e && e.rubric);
  const target = $('rubric');
  target.replaceChildren();
  if (!rows.length) return;
  const items = [['lift_within_3000', 'Lift within 3000 steps', 6], ['lift_10cm', 'Lift ≥ 10 cm', 6], ['hold_50_steps', 'Hold ≥ 50 steps', 6],
    ['pose_close', 'Pose near target grasp', 6], ['reach_10cm', 'Reach within 10 cm', 6], ['reach_within_15000', 'Reach within 15k steps', 6],
    ['carry_stability', 'Carry stability', 4]];
  const palette = [COLORS.planned, COLORS.imitation, COLORS.rl];
  const width = 900, rowH = 9 * rows.length + 12, height = 10 + items.length * rowH;
  const s = frame(width, height);
  const x = v => 200 + (v / 6) * 520;
  items.forEach(([key, label, max], i) => {
    const y = 10 + i * rowH;
    s.append(svg('text', {x: 192, y: y + 4 + 4.5 * rows.length, class: 'label', 'text-anchor': 'end'}, `${label} (${max})`));
    s.append(svg('rect', {x: 200, y, width: x(max) - 200, height: 9 * rows.length, rx: 2, class: 'track'}));
    rows.forEach(([, e], j) => s.append(svg('rect', {x: 200, y: y + 9 * j, width: Math.max(1, x(e.rubric.items[key]) - 200), height: 8, rx: 2, fill: palette[j % 3]})));
  });
  rows.forEach(([label, e], j) => s.append(svg('text', {x: 740, y: 20 + j * 22, class: 'label'}, `${label}: ${e.rubric.per_trial} / 40`)));
  target.append(s, legend(rows.map(([label], j) => [label, palette[j % 3]])));
}

/* ---------- tables ---------- */
function row(cells) {
  const tr = el('tr');
  for (const cell of cells) {
    const td = el('td');
    if (cell instanceof Node) td.append(cell); else td.textContent = cell ?? '—';
    tr.append(td);
  }
  return tr;
}
function table(target, headers, rows) {
  const t = $(target);
  t.replaceChildren();
  const head = el('tr');
  for (const h of headers) head.append(el('th', h));
  const thead = el('thead'); thead.append(head);
  const body = el('tbody');
  for (const r of rows) body.append(row(r));
  t.append(thead, body);
}

function renderActive(active) {
  $('active-panel').hidden = !active.length;
  const target = $('active');
  target.replaceChildren();
  for (const job of active) {
    const line = el('div', undefined, 'job');
    line.append(el('strong', job.name), el('span', ` · ${job.detail}`, 'muted'));
    if (job.progress !== null && job.progress !== undefined) {
      const bar = el('div', undefined, 'pbar'), fill = el('div');
      fill.style.width = `${Math.round(100 * job.progress)}%`;
      bar.append(fill);
      line.append(bar);
    }
    target.append(line);
  }
}

function renderTables(evaluations, history) {
  table('evaluations', ['Run', 'Set', 'Complete', 'Timed', 'Drops', 'Rubric', 'Started'],
    evaluations.map(e => [e.label, e.set, `${e.success}/${e.completed}`, String(e.timed), String(e.drops ?? '—'),
      e.rubric ? `${e.rubric.per_trial}/40` : '—', when(e.started_at)]));
  $('history-count').textContent = `(${history.length})`;
  table('history', ['Run', 'Kind', 'Status', 'Updated'], history.map(h => [h.id, h.kind, h.status, when(h.updated_at)]));
}

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

let drawnSignature = '';
let mazeDrawn = false;  // drawn once so the animation is never restarted by polling
async function poll() {
  try {
    const response = await fetch('/api/overview', {cache: 'no-store'});
    const state = await response.json();
    STATE = {overview: state.overview || {}, evaluations: Object.fromEntries((state.evaluations || []).map(e => [e.id, e]))};
    renderHeader(STATE.overview, state.resources, state.timestamp);
    renderActive(state.active || []);
    // Redraw diagrams only when results change, so animations keep running.
    const signature = JSON.stringify([STATE.overview, (state.evaluations || []).map(e => [e.id, e.completed, e.success])]);
    if (signature !== drawnSignature) {
      drawnSignature = signature;
      renderSystems(STATE.overview);
      renderTimeline(STATE.overview);
      renderImitation(STATE.overview);
      if (!mazeDrawn && state.maze) { renderMaze(state.maze); mazeDrawn = true; }
      renderResults(STATE.overview);
      renderFindings(STATE.overview);
      renderRubric(STATE.overview);
      renderTables(state.evaluations || [], state.history || []);
      const docs = $('documents');
      docs.replaceChildren();
      for (const doc of STATE.overview.documents || []) {
        const a = el('a', doc.label); a.href = doc.href;
        docs.append(a);
      }
    }
    renderRecordings(STATE.overview.recordings || []);
  } catch (error) {
    $('updated').textContent = `Connection problem: ${error.message}`;
  }
  setTimeout(poll, 15000);
}

poll();
