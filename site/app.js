/* Charts for the project site, drawn as SVG from window.RESULTS (site/data/results.js, scripts/site_data.py). */
'use strict';

const NS = 'http://www.w3.org/2000/svg';
const R = () => window.RESULTS;
const pct = v => `${Math.round(v * 100)}%`;
const f2 = v => v.toFixed(2);

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}
const scale = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/* A chart card: title, subtitle, HTML legend, SVG. */
function card(id, title, sub, legend, w, h) {
  const host = document.getElementById(id);
  if (!host) return null;
  host.innerHTML = '';
  const t = document.createElement('div'); t.className = 'chart-title'; t.textContent = title; host.appendChild(t);
  if (sub) { const s = document.createElement('div'); s.className = 'chart-sub'; s.textContent = sub; host.appendChild(s); }
  if (legend && legend.length) {
    const l = document.createElement('div'); l.className = 'legend';
    for (const [label, color, kind] of legend) {
      const span = document.createElement('span');
      const mark = kind === 'line' ? `<svg width="18" height="10" style="vertical-align:middle;margin-right:5px"><line x1="1" y1="5" x2="17" y2="5" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/></svg>`
        : `<span class="key" style="background:${color}"></span>`;
      span.innerHTML = mark + label; l.appendChild(span);
    }
    host.appendChild(l);
  }
  const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `${title}. ${sub || ''}` }, host);
  return svg;
}

/* Axes: horizontal gridlines with tick labels on y; category or numeric ticks on x. */
function yGrid(svg, y, x0, x1, ticks, fmt = pct) {
  for (const t of ticks) {
    el('line', { x1: x0, x2: x1, y1: y(t), y2: y(t), class: t === ticks[0] ? 'c-axis' : 'c-grid' }, svg);
    el('text', { x: x0 - 6, y: y(t) + 4, 'text-anchor': 'end', class: 'c-tick', text: fmt(t) }, svg);
  }
}
function xTicks(svg, x, yBase, ticks, fmt) {
  for (const t of ticks) el('text', { x: x(t), y: yBase + 15, 'text-anchor': 'middle', class: 'c-tick', text: fmt(t) }, svg);
}
function tip(node, html) { node.dataset.tip = html; return node; }
function path(points) { return points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(''); }
/* A bar with a 4px rounded data end and a square baseline (vertical bars grow up from y0). */
function vbar(svg, x, w, y0, y1, color, opacity = 1) {
  const h = Math.max(0, y0 - y1), r = Math.min(4, h, w / 2);
  const d = h <= 0 ? '' : `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  return el('path', { d, fill: color, opacity }, svg);
}
function hbar(svg, x0, x1, y, h, color, opacity = 1) {
  const w = Math.max(0, x1 - x0), r = Math.min(4, w, h / 2);
  const d = w <= 0 ? '' : `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
  return el('path', { d, fill: color, opacity }, svg);
}
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = () => ({ gripper: css('--gripper'), hand: css('--hand'), ppo: css('--ppo'), rlpd: css('--rlpd'), nowarm: css('--nowarm'),
  muted: css('--muted'), ink2: css('--ink-2'), surface: css('--surface'), h1: css('--hand-1'), h2: css('--hand-2'), h3: css('--hand-3'),
  good: css('--good') });

/* ---------------------------------------------------------------- history */
function chartAblation() {
  const c = C(), h = R().history.pybullet_skills.sac_ablation_dev, n = h.cases;
  const rows = [['imitation grasp, imitation carry', h['imitation grasp + imitation carry']], ['SAC grasp, imitation carry', h['SAC grasp + imitation carry']],
    ['imitation grasp, SAC carry', h['imitation grasp + SAC carry']], ['SAC grasp, SAC carry', h['SAC grasp + SAC carry']]];
  const svg = card('chart-ablation', 'SAC fine-tuning vs imitation alone', 'PyBullet, full task on 260 development cases; one standard deviation ≈ ±7 cases', null, 460, 170);
  if (!svg) return;
  const x = scale(0, 260, 176, 430);
  for (const t of [0, 100, 200, 260]) { el('line', { x1: x(t), x2: x(t), y1: 8, y2: 148, class: t ? 'c-grid' : 'c-axis' }, svg); el('text', { x: x(t), y: 162, 'text-anchor': 'middle', class: 'c-tick', text: t }, svg); }
  rows.forEach(([label, v], i) => {
    const y = 18 + i * 32;
    el('text', { x: 198, y: y + 13, 'text-anchor': 'end', class: 'c-label', text: label }, svg);
    tip(hbar(svg, x(0), x(v), y, 18, c.gripper), `${label}: ${v} of ${n}`);
    el('text', { x: x(v) + 5, y: y + 13, class: 'c-value', text: v }, svg);
  });
}

function pairChart(id, title, sub, groups, labels, colors, w = 460) {
  const svg = card(id, title, sub, labels.map((l, i) => [l, colors[i]]), w, 190);
  if (!svg) return;
  const y = scale(0, 1, 160, 14), x0 = 48, band = (w - x0 - 10) / groups.length;
  yGrid(svg, y, x0, w - 10, [0, .25, .5, .75, 1]);
  groups.forEach(([name, values], g) => {
    const bw = Math.min(24, band / 4);
    values.forEach(([k, n], i) => {
      const xb = x0 + g * band + band / 2 - bw - 2 + i * (bw + 4);
      tip(vbar(svg, xb, bw, y(0), y(k / n), colors[i]), `${name}, ${labels[i]}: ${k} of ${n}`);
      el('text', { x: xb + bw / 2, y: y(k / n) - 5, 'text-anchor': 'middle', class: 'c-value', text: `${k}/${n}` }, svg);
    });
    el('text', { x: x0 + g * band + band / 2, y: 178, 'text-anchor': 'middle', class: 'c-label', text: name }, svg);
  });
}
function chartSpike() {
  const c = C(), h = R().history.pybullet_hand;
  pairChart('chart-spike', 'Hand vs gripper in PyBullet: objects held through carries', 'Scripted Allegro power grasp vs the gripper planner on a gantry replaying recorded carry motions',
    [['recorded routes', [h.recorded_routes.gripper, h.recorded_routes.allegro]], ['unseen routes, tuned grasp', [h.unseen_routes.gripper, h.unseen_routes.allegro]]],
    ['gripper', 'Allegro hand'], [c.gripper, c.hand]);
}
function chartPort() {
  const m = R().history.mujoco_port;
  pairChart('chart-port', 'Same task, same settings, different simulator', 'Gripper planner, full task: course seeds (100) and development mazes (260)',
    [['course seeds', [[m.planner_course.pybullet, 100], [m.planner_course.mujoco, 100]]], ['development mazes', [[m.planner_dev.pybullet, 260], [m.planner_dev.mujoco, 260]]]],
    ['PyBullet', 'MuJoCo'], [C().muted, C().gripper]);
}

function chartRig() {
  const c = C(), curve = R().rig_feasibility.curve;
  const svg = card('chart-rig', 'Imitation then PPO on the rig', 'Lift-and-carry over training (36 evaluation episodes per point; soup can, apple, mug)', null, 460, 210);
  if (!svg) return;
  const x = scale(0, 9e6, 48, 440), y = scale(0, 1, 176, 12);
  yGrid(svg, y, 48, 440, [0, .25, .5, .75, 1]);
  xTicks(svg, x, 176, [0, 2e6, 4e6, 6e6, 8e6], t => `${t / 1e6}M`);
  el('rect', { x: x(0), y: 12, width: x(1e6) - x(0), height: 164, class: 'c-band', opacity: .8 }, svg);
  el('text', { x: x(.5e6), y: 26, 'text-anchor': 'middle', class: 'c-tick', text: 'warm-up' }, svg);
  el('line', { x1: x(0), x2: x(9e6), y1: y(.72), y2: y(.72), stroke: c.ink2, 'stroke-width': 1 }, svg);
  el('text', { x: x(9e6), y: y(.72) + 14, 'text-anchor': 'end', class: 'c-label', text: 'scripted teacher 72%' }, svg);
  el('path', { d: path(curve.map(p => [x(p[0]), y(p[1])])), fill: 'none', stroke: c.hand, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  for (const p of curve) tip(el('circle', { cx: x(p[0]), cy: y(p[1]), r: 4, fill: c.hand, stroke: c.surface, 'stroke-width': 2 }, svg), `${(p[0] / 1e6).toFixed(1)}M steps: ${pct(p[1])}`);
  const best = curve.reduce((a, b) => (b[1] > a[1] ? b : a));
  el('text', { x: x(best[0]), y: y(best[1]) - 9, 'text-anchor': 'middle', class: 'c-value', text: pct(best[1]) }, svg);
  el('text', { x: x(0) + 8, y: y(curve[0][1]) - 8, class: 'c-value', text: `cloned ${pct(curve[0][1])}` }, svg);
}

/* ---------------------------------------------------------------- first study */
function chartFirst() {
  const c = C(), cells = R().first_study.cells;
  const cols = [['hand-written', 'planner', 'scripted'], ['imitation', 'imitation', 'imitation'], ['imitation + RL', 'imitation_sac', 'imitation_ppo']];
  const svg = card('chart-first', 'First study (hold-out mazes, 3 seeds)', 'Full-task delivery per seed; gripper + RL = SAC, hand + RL = PPO. Development evidence after the redesign.',
    [['gripper', c.gripper], ['hand', c.hand]], 900, 200);
  if (!svg) return;
  const panels = [['standard objects', 'std', 60], ['wide cylinders', 'wide', 500]];
  for (const [title, task, px] of panels) {
    const x0 = px, x1 = px + 380, y = scale(0, 1, 160, 18), band = (x1 - x0) / 3;
    yGrid(svg, y, x0, x1, [0, .5, 1]);
    el('text', { x: (x0 + x1) / 2, y: 10, 'text-anchor': 'middle', class: 'c-label', text: title }, svg);
    cols.forEach(([label, gk, hk], i) => {
      const cx = x0 + band * (i + .5);
      el('text', { x: cx, y: 178, 'text-anchor': 'middle', class: 'c-label', text: label }, svg);
      [['gripper', gk, -12], ['hand', hk, 12]].forEach(([e, k, dx]) => {
        const vals = cells[`${e}.${k}.${task}`];
        if (!vals) return;
        vals.forEach((v, j) => tip(el('circle', { cx: cx + dx + (j - (vals.length - 1) / 2) * 5, cy: y(v), r: 4.5, fill: c[e], stroke: c.surface, 'stroke-width': 2 }, svg), `${e}, ${label}, ${task === 'std' ? 'standard' : 'wide'}: ${pct(v)}`));
      });
    });
  }
}

/* ---------------------------------------------------------------- final study design */
function chartTuning() {
  const c = C(), t = R().tuning;
  const factors = [['learning_rate', 'learning rate 1e-4 (vs 3e-4)'], ['demos_per_object', '300 demos per object (vs 100)'],
    ['bc_log_std', 'exploration log-std −0.7 (vs −1.2)'], ['bc_interleave', 'interleaved cloning (vs none)']];
  const svg = card('chart-tuning', 'Tuning sweep: main effect of each factor', 'Change in score (mean lift-and-carry of the last three evaluations), 2⁴⁻¹ design, 8 runs × 4M steps per effector',
    [['gripper', c.gripper], ['hand', c.hand]], 460, 170);
  if (!svg) return;
  const x = scale(-.5, .25, 212, 450);
  for (const v of [-.5, -.25, 0, .25]) { el('line', { x1: x(v), x2: x(v), y1: 6, y2: 146, class: v === 0 ? 'c-axis' : 'c-grid' }, svg); el('text', { x: x(v), y: 160, 'text-anchor': 'middle', class: 'c-tick', text: (v > 0 ? '+' : '') + Math.round(v * 100) }, svg); }
  factors.forEach(([k, label], i) => {
    const y = 22 + i * 34;
    el('text', { x: 204, y: y + 4, 'text-anchor': 'end', class: 'c-label', text: label }, svg);
    for (const [e, dy] of [['gripper', -5], ['hand', 5]]) {
      const v = t[e].effects[k];
      tip(el('circle', { cx: x(v), cy: y + dy, r: 5, fill: c[e], stroke: c.surface, 'stroke-width': 2 }, svg), `${e}: ${v > 0 ? '+' : ''}${Math.round(v * 100)} points`);
    }
  });
}

/* ---------------------------------------------------------------- results */
const LADDER = [['hand-written', { gripper: 'planner', hand: 'scripted' }], ['imitation', 'imitation'], ['imitation + RL', 'imitation_rl'], ['RL from scratch', 'rl_scratch']];
function chartLadder() {
  const c = C(), res = R().reserve;
  const svg = card('chart-ladder', 'Full-task delivery by grasp controller', 'Sealed reserve mazes; pooled over 5 seeds with 95% Wilson interval; small dots are seeds (hand-written controllers are deterministic)',
    [['gripper', c.gripper], ['hand', c.hand]], 940, 420);
  if (!svg) return;
  const panels = [['gripper', 'std', 60, 20], ['hand', 'std', 530, 20], ['gripper', 'wide', 60, 225], ['hand', 'wide', 530, 225]];
  for (const [e, task, px, py] of panels) {
    const x0 = px, x1 = px + 380, y = scale(0, 1, py + 160, py + 18), band = (x1 - x0) / 4;
    yGrid(svg, y, x0, x1, [0, .25, .5, .75, 1]);
    el('text', { x: x0, y: py + 8, class: 'c-label', 'font-weight': 600, text: `${e === 'gripper' ? 'Gripper' : 'Hand'} · ${task === 'std' ? 'standard objects' : 'wide cylinders'}` }, svg);
    const pts = LADDER.map(([label, key], i) => {
      const k = typeof key === 'string' ? key : key[e], cell = res[`${e}.${k}.${task}`];
      return { label, cell, cx: x0 + band * (i + .5) };
    });
    el('path', { d: path(pts.map(p => [p.cx, y(p.cell.rate)])), fill: 'none', stroke: c[e], 'stroke-width': 2, opacity: .5 }, svg);
    for (const p of pts) {
      const { cell, cx } = p;
      if (cell.per_seed.length > 1) cell.per_seed.forEach((v, j) => el('circle', { cx: cx + 14 + j * 4, cy: y(v), r: 2.5, fill: c[e], opacity: .45 }, svg));
      el('line', { x1: cx, x2: cx, y1: y(cell.wilson95[0]), y2: y(cell.wilson95[1]), stroke: c[e], 'stroke-width': 2 }, svg);
      const dot = el('circle', { cx, cy: y(cell.rate), r: 5.5, fill: c[e], stroke: c.surface, 'stroke-width': 2 }, svg);
      tip(dot, `${e}, ${p.label}, ${task === 'std' ? 'standard' : 'wide'}: ${pct(cell.rate)} delivered (${cell.delivered}/${cell.cases}); 95% CI ${pct(cell.wilson95[0])}–${pct(cell.wilson95[1])}; seeds ${cell.per_seed.map(pct).join(', ')}`);
      el('text', { x: cx - 9, y: y(cell.rate) + 4, 'text-anchor': 'end', class: 'c-value', text: pct(cell.rate) }, svg);
      if (py === 225) el('text', { x: cx, y: py + 178, 'text-anchor': 'middle', class: 'c-label', text: p.label }, svg);
    }
  }
}

function chartObjects() {
  const c = C(), res = R().reserve;
  const objects = ['foam brick', 'banana', 'apple', 'mug', 'soup can'];
  const series = [['scripted', 'hand.scripted.std', c.h1], ['imitation', 'hand.imitation.std', c.h2], ['imitation + RL', 'hand.imitation_rl.std', c.h3]];
  const svg = card('chart-objects', 'Hand: delivery per object', 'Standard objects, reserve mazes, 5 seeds pooled', series.map(s => [s[0], s[2]]), 460, 210);
  if (!svg) return;
  const y = scale(0, 1, 176, 12), x0 = 44, band = (450 - x0) / objects.length, bw = 16;
  yGrid(svg, y, x0, 450, [0, .25, .5, .75, 1]);
  objects.forEach((o, i) => {
    series.forEach(([name, key, color], j) => {
      const v = res[key].per_object[o], xb = x0 + band * i + band / 2 - (bw * 3 + 4) / 2 + j * (bw + 2);
      tip(vbar(svg, xb, bw, y(0), y(v), color), `${o}, ${name}: ${pct(v)}`);
    });
    el('text', { x: x0 + band * (i + .5), y: 192, 'text-anchor': 'middle', class: 'c-label', text: o }, svg);
  });
}

function campaignPanel(svg, px, py, e, task, c) {
  const camp = R().campaign, x0 = px, x1 = px + 180, x = scale(0, 1e7, x0, x1), y = scale(0, 1, py + 120, py + 16);
  yGrid(svg, y, x0, x1, [0, .5, 1]);
  xTicks(svg, x, py + 120, [0, 5e6, 1e7], t => `${t / 1e6}M`);
  el('text', { x: x0, y: py + 8, class: 'c-label', 'font-weight': 600, text: `${e} · ${task === 'std' ? 'standard' : 'wide'}` }, svg);
  for (const [recipe, color, label] of [['ppo', c.muted, 'from scratch'], ['bcppo', c[e], 'imitation + PPO']]) {
    const runs = camp[`${e}.${recipe}.${task}`];
    runs.forEach((r, s) => tip(el('path', { d: path(r.map(p => [x(p[0]), y(p[1])])), fill: 'none', stroke: color, 'stroke-width': 1.5, opacity: .55, 'stroke-linejoin': 'round' }, svg),
      `${e}, ${label}, ${task === 'std' ? 'standard' : 'wide'}, seed ${s}: ${pct(r[r.length - 1][1])} at 10M`));
  }
}
function chartCampaign() {
  const c = C();
  const svg = card('chart-campaign', 'Training curves, all 40 runs', 'Training-time lift-and-carry vs environment steps, one line per seed',
    [['imitation + PPO (effector colour)', c.ink2, 'line'], ['PPO from scratch', c.muted, 'line']], 460, 300);
  if (!svg) return;
  campaignPanel(svg, 40, 6, 'gripper', 'std', c); campaignPanel(svg, 270, 6, 'hand', 'std', c);
  campaignPanel(svg, 40, 152, 'gripper', 'wide', c); campaignPanel(svg, 270, 152, 'hand', 'wide', c);
}

function chartProxy() {
  const c = C(), camp = R().campaign, res = R().reserve;
  const gain = e => {
    const runs = camp[`${e}.bcppo.std`];
    const train = mean(runs.map(r => mean(r.slice(-3).map(p => p[1])) - r[0][1]));
    const task = res[`${e}.imitation_rl.std`].rate - res[`${e}.imitation.std`].rate;
    return [train, task];
  };
  const svg = card('chart-proxy', 'RL’s gain over imitation: training vs full task', 'Standard objects. Training: lift-and-carry at the end vs the cloned start. Full task: reserve delivery, imitation + RL minus imitation.',
    [['gripper', c.gripper], ['hand', c.hand]], 460, 300);
  if (!svg) return;
  const y = scale(-.15, .45, 260, 20), xa = 150, xb = 330;
  yGrid(svg, y, 60, 420, [-.1, 0, .1, .2, .3, .4], v => `${v > 0 ? '+' : ''}${Math.round(v * 100)}`);
  el('line', { x1: 60, x2: 420, y1: y(0), y2: y(0), stroke: c.ink2, 'stroke-width': 1 }, svg);
  el('text', { x: xa, y: 280, 'text-anchor': 'middle', class: 'c-label', text: 'training-time' }, svg);
  el('text', { x: xb, y: 280, 'text-anchor': 'middle', class: 'c-label', text: 'full task (reserve)' }, svg);
  for (const e of ['gripper', 'hand']) {
    const [a, b] = gain(e);
    el('line', { x1: xa, x2: xb, y1: y(a), y2: y(b), stroke: c[e], 'stroke-width': 2 }, svg);
    for (const [xx, v, anchor, dx] of [[xa, a, 'end', -10], [xb, b, 'start', 10]]) {
      tip(el('circle', { cx: xx, cy: y(v), r: 5, fill: c[e], stroke: c.surface, 'stroke-width': 2 }, svg), `${e}: ${v > 0 ? '+' : ''}${Math.round(v * 100)} points`);
      el('text', { x: xx + dx, y: y(v) + 4, 'text-anchor': anchor, class: 'c-value', text: `${v > 0 ? '+' : ''}${Math.round(v * 100)}` }, svg);
    }
  }
}

const WIDE = [['gripper', 'planner', 'planner'], ['gripper', 'imitation', 'imitation'], ['gripper', 'imitation_rl', 'imitation + RL'], ['gripper', 'rl_scratch', 'from scratch'],
  ['hand', 'scripted', 'scripted'], ['hand', 'imitation', 'imitation'], ['hand', 'imitation_rl', 'imitation + RL'], ['hand', 'rl_scratch', 'from scratch']];
function chartWide() {
  const c = C(), res = R().reserve;
  const svg = card('chart-wide', 'Wide cylinders: delivery', 'Reserve mazes, 120 cases per system; bars pooled over 5 seeds with 95% interval, small dots are seeds', [['gripper', c.gripper], ['hand', c.hand]], 460, 300);
  if (!svg) return;
  const x = scale(0, 1, 150, 440);
  for (const t of [0, .25, .5, .75, 1]) {
    el('line', { x1: x(t), x2: x(t), y1: 6, y2: 274, class: t ? 'c-grid' : 'c-axis' }, svg);
    el('text', { x: x(t), y: 290, 'text-anchor': 'middle', class: 'c-tick', text: pct(t) }, svg);
  }
  WIDE.forEach(([e, k, label], i) => {
    const cell = res[`${e}.${k}.wide`], y = 12 + i * 33 + (e === 'hand' ? 8 : 0);
    el('text', { x: 142, y: y + 13, 'text-anchor': 'end', class: 'c-label', text: `${e} · ${label}` }, svg);
    tip(hbar(svg, x(0), x(cell.rate), y, 18, c[e], .85), `${e}, ${label}: ${pct(cell.rate)} (${cell.delivered}/${cell.cases}); 95% CI ${pct(cell.wilson95[0])}–${pct(cell.wilson95[1])}`);
    el('line', { x1: x(cell.wilson95[0]), x2: x(cell.wilson95[1]), y1: y + 9, y2: y + 9, stroke: c.ink2, 'stroke-width': 1.5 }, svg);
    if (cell.per_seed.length > 1) cell.per_seed.forEach(v => el('circle', { cx: x(v), cy: y + 23, r: 2.2, fill: c.ink2, opacity: .7 }, svg));
    el('text', { x: x(Math.max(cell.rate, cell.wilson95[1])) + 6, y: y + 13, class: 'c-value', text: pct(cell.rate) }, svg);
  });
}

function chartStages() {
  const res = R().reserve, c = C();
  const kinds = [['delivered', c.good], ['grasp', '#eda100'], ['hold', '#e34948'], ['carry', '#e87ba4'], ['other', c.muted]];
  const rows = [['gripper planner', 'gripper.planner.wide'], ['gripper, from scratch', 'gripper.rl_scratch.wide'], ['hand, imitation + RL', 'hand.imitation_rl.wide']];
  const svg = card('chart-stages', 'Wide cylinders: where each system fails', 'Share of reserve cases by outcome; “hold” = the object slips out right after the lift', kinds.map(k => [k[0], k[1]]), 460, 150);
  if (!svg) return;
  const x = scale(0, 1, 150, 452);
  rows.forEach(([label, key], i) => {
    const st = res[key].stages, y = 14 + i * 40;
    el('text', { x: 144, y: y + 15, 'text-anchor': 'end', class: 'c-label', text: label }, svg);
    let at = 0;
    for (const [k, color] of kinds) {
      const v = k === 'other' ? Object.entries(st).filter(([s]) => !['delivered', 'grasp', 'hold', 'carry'].includes(s)).reduce((a, [, b]) => a + b, 0) : (st[k] || 0);
      if (v <= 0) continue;
      tip(el('rect', { x: x(at) + (at > 0 ? 1 : 0), y, width: Math.max(0, x(at + v) - x(at) - (at > 0 ? 1 : 0)), height: 22, fill: color }, svg), `${label}: ${k} ${pct(v)}`);
      const ink = ['grasp', 'carry'].includes(k) ? '#0b0b0b' : '#ffffff';
      if (x(at + v) - x(at) > 30) el('text', { x: (x(at) + x(at + v)) / 2, y: y + 15, 'text-anchor': 'middle', class: 'c-value', style: `fill:${ink}`, text: pct(v) }, svg);
      at += v;
    }
  });
  for (const t of [0, .5, 1]) el('text', { x: x(t), y: 140, 'text-anchor': 'middle', class: 'c-tick', text: pct(t) }, svg);
}

function c3Panel(svg, px, e, c, byHours) {
  const alg = R().algorithms, x1 = px + 400, xmax = byHours ? 15 : 4e6;
  const x = scale(0, xmax, px, x1), y = scale(0, 1, 176, 22);
  yGrid(svg, y, px, x1, [0, .25, .5, .75, 1]);
  xTicks(svg, x, 176, byHours ? [0, 2, 4, 6, 8, 10, 12, 14] : [0, 1e6, 2e6, 3e6, 4e6], t => (byHours ? `${t} h` : `${t / 1e6}M`));
  el('text', { x: px, y: 12, class: 'c-label', 'font-weight': 600, text: e === 'hand' ? 'Hand' : 'Gripper' }, svg);
  if (!byHours) {
    el('rect', { x: x(0), y: 22, width: x(1e6) - x(0), height: 154, class: 'c-band', opacity: .8 }, svg);
    el('text', { x: x(.5e6), y: 36, 'text-anchor': 'middle', class: 'c-tick', text: 'actor frozen' }, svg);
  }
  const xv = p => (byHours ? p[2] / 3600 : p[0]);
  for (const [a, color, label] of [['ppo', c.ppo, 'PPO'], ['rlpd', c.rlpd, 'RLPD']]) {
    const runs = byHours ? [alg[`${e}.${a}`][0]] : alg[`${e}.${a}`];
    runs.forEach((r, s) => {
      const pts = r.map(p => [x(xv(p)), y(p[1])]);
      tip(el('path', { d: path(pts), fill: 'none', stroke: color, 'stroke-width': byHours ? 2 : 1.4, opacity: byHours ? 1 : .55, 'stroke-linejoin': 'round' }, svg),
        `${label}, seed ${byHours ? 0 : s}: ${pct(r[r.length - 1][1])} at 4M steps after ${(r[r.length - 1][2] / 3600).toFixed(1)} training hours`);
      if (byHours) { const last = pts[pts.length - 1]; el('circle', { cx: last[0], cy: last[1], r: 4, fill: color, stroke: c.surface, 'stroke-width': 2 }, svg); }
    });
    if (!byHours) {
      const runsAll = alg[`${e}.${a}`], n = Math.min(...runsAll.map(r => r.length));
      const m = Array.from({ length: n }, (_, i) => [x(runsAll[0][i][0]), y(mean(runsAll.map(r => r[i][1])))]);
      el('path', { d: path(m), fill: 'none', stroke: color, 'stroke-width': 3, 'stroke-linejoin': 'round' }, svg);
      const last = m[m.length - 1];
      el('text', { x: last[0] - 4, y: last[1] - 8, 'text-anchor': 'end', class: 'c-value', text: `${label} ${f2(mean(runsAll.map(r => r[r.length - 1][1])))}` }, svg);
    }
  }
  if (!byHours) {
    const nw = R().algorithms_no_warmup[e];
    tip(el('path', { d: path(nw.map(p => [x(p[0]), y(p[1])])), fill: 'none', stroke: c.nowarm, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, svg), `RLPD without warm-up, seed 0: ${pct(nw[nw.length - 1][1])} at 4M`);
  }
}
function chartC3() {
  const c = C();
  const s1 = card('chart-c3-steps', 'Per environment step (5 seeds)', 'Training-time lift-and-carry; thin lines are seeds, thick lines their mean',
    [['PPO', c.ppo, 'line'], ['RLPD', c.rlpd, 'line'], ['RLPD without warm-up (seed 0)', c.nowarm, 'line']], 460, 400);
  if (s1) { const g1 = el('g', {}, s1); c3Panel(g1, 40, 'hand', c, false); const g2 = el('g', { transform: 'translate(0,200)' }, s1); c3Panel(g2, 40, 'gripper', c, false); }
  const s2 = card('chart-c3-hours', 'Per training hour (seed 0, machine to itself)', 'Same runs against wall-clock training time, evaluations excluded',
    [['PPO', c.ppo, 'line'], ['RLPD', c.rlpd, 'line']], 460, 400);
  if (s2) { const g1 = el('g', {}, s2); c3Panel(g1, 40, 'hand', c, true); const g2 = el('g', { transform: 'translate(0,200)' }, s2); c3Panel(g2, 40, 'gripper', c, true); }
}

function chartDevRes() {
  const c = C(), dev = R().dev, res = R().reserve;
  const svg = card('chart-devres', 'Development vs reserve, every cell', 'Delivery on development mazes (x) and on the sealed reserve (y); 16 cells', [['gripper', c.gripper], ['hand', c.hand]], 460, 300);
  if (!svg) return;
  const x = scale(0, 1, 50, 440), y = scale(0, 1, 270, 14);
  yGrid(svg, y, 50, 440, [0, .25, .5, .75, 1]);
  xTicks(svg, x, 270, [0, .25, .5, .75, 1], pct);
  el('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: c.ink2, 'stroke-width': 1 }, svg);
  el('text', { x: x(.92), y: y(.97), 'text-anchor': 'end', class: 'c-tick', text: 'equal' }, svg);
  for (const k of Object.keys(res)) {
    const e = k.split('.')[0];
    tip(el('circle', { cx: x(dev[k].rate), cy: y(res[k].rate), r: 5, fill: c[e], stroke: c.surface, 'stroke-width': 2 }, svg),
      `${k.replace(/\./g, ' · ').replace('_rl', ' + RL').replace('std', 'standard')}: dev ${pct(dev[k].rate)}, reserve ${pct(res[k].rate)}`);
  }
}

/* ---------------------------------------------------------------- page behaviour */
function tooltips() {
  const box = document.getElementById('tip');
  document.addEventListener('pointermove', ev => {
    const t = ev.target.closest && ev.target.closest('[data-tip]');
    if (!t) { box.style.display = 'none'; return; }
    box.textContent = t.dataset.tip; box.style.display = 'block';
    const x = Math.min(ev.clientX + 14, window.innerWidth - box.offsetWidth - 8);
    box.style.left = `${x}px`; box.style.top = `${ev.clientY + 14}px`;
  });
}
function videos() {
  const seen = new IntersectionObserver(entries => {
    for (const { target: v, isIntersecting } of entries) {
      if (isIntersecting) {
        if (!v.src) v.src = v.dataset.src;
        v.play().catch(() => {});
      } else if (!v.paused) v.pause();
    }
  }, { rootMargin: '120px 0px' });
  document.querySelectorAll('video[data-src]').forEach(v => seen.observe(v));
}
function theme() {
  const root = document.documentElement, button = document.getElementById('theme');
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (e) { /* storage unavailable */ }
  if (saved) root.dataset.theme = saved;
  button.addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) { /* ignore */ }
    drawAll();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
}
function drawAll() {
  for (const f of [chartAblation, chartSpike, chartPort, chartRig, chartFirst, chartTuning, chartLadder, chartObjects, chartCampaign, chartProxy, chartWide, chartStages, chartC3, chartDevRes]) {
    try { f(); } catch (err) { console.error(f.name, err); }
  }
}
document.addEventListener('DOMContentLoaded', () => {
  theme();
  drawAll();
  tooltips();
  videos();
  if (window.renderMathInElement) {
    renderMathInElement(document.body, { delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
  }
});
