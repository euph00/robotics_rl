'use strict';
const $ = id => document.getElementById(id);
const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
const number = value => Number.isFinite(Number(value)) && value !== null && value !== '';
const count = value => number(value) ? Math.round(Number(value)).toLocaleString() : '—';
const compact = value => number(value) ? Intl.NumberFormat(undefined, {notation:'compact', maximumFractionDigits:1}).format(value) : '—';
const percent = value => number(value) ? `${Math.round(100 * value)}%` : '—';
const scopeLabel = scope => ({full_task:'Full task', actor_initialization:'Actor initialization', four_start_grasp_diagnostic:'Grasp · four starts', full_pregrasp_diagnostic:'Grasp · full approach', skill:'Skill only', reference:'Physical reference', reference_carry_handoff:'Reach · reference handover', training_data:'Physical training data', upstream:'Upstream proximity', easy_carry_diagnostic:'Carry · short segment', carry_segment_curriculum:'Carry · curriculum', full_carry:'Carry · full route', easy_manipulation_diagnostic:'Manipulation · local start'}[scope] || (scope ? String(scope).replaceAll('_', ' ') : 'Scope unspecified'));
const metricScope = row => row?.scope === 'full_task' ? 'full_task' : (row?.evaluation_scope || row?.scope || null);
const ago = value => { if (!value) return '—'; const seconds = Math.max(0, (Date.now() - Date.parse(value)) / 1000); return seconds < 60 ? `${Math.round(seconds)}s ago` : seconds < 3600 ? `${Math.floor(seconds/60)}m ago` : `${Math.floor(seconds/3600)}h ago`; };
let state = null, selectedId = new URLSearchParams(location.search).get('run'), artifactKey = null, showAllRuns = false;

function renderProject(project) {
  $('phase').textContent = String(project.phase || 'setup').replaceAll('_', ' ').toUpperCase();
  $('summary').textContent = project.summary || 'Original Fetch · SAC + HER · Physical pick–carry–reach';
  $('milestones').replaceChildren(...(project.milestones || []).map((item, i) => {
    const node = el('li', undefined, ['pending','active','complete','blocked'].includes(item.status) ? item.status : 'pending');
    node.title = `${item.label}: ${item.status}`;
    node.append(el('span', item.status === 'complete' ? '✓' : String(i+1), 'marker'), el('span', item.label));
    return node;
  }));
}

const supervisedUpdates = row => row?.bc_updates ?? row?.updates;
const runAccounting = run => run?.metrics?.accounting || run?.accounting;

function chartData(run) {
  const initialization=run?.kind==='initialization'||run?.metrics?.scope==='actor_initialization';
  const fullRouteFit=initialization && (run.series||[]).some(row=>row.bc_training_scope==='full_route_only');
  const trainingFit=fullRouteFit||(run?.series||[]).some(row=>number(row.bc_training_loss));
  const lossKey=fullRouteFit?'bc_full_route_training_loss':trainingFit?'bc_training_loss':'bc_validation_loss';
  const losses=initialization ? (run.series||[]).filter(row=>number(row[lossKey]) && row[lossKey]>=0 && number(supervisedUpdates(row))) : [];
  if (losses.length) return {
    points:losses.map(row=>({x:Number(supervisedUpdates(row)),y:Number(row[lossKey])})),
    title:'Actor initialization',ylabel:fullRouteFit?'Full-route training action loss':trainingFit?'Training action loss':'Validation action loss',
    scope:fullRouteFit?'Training fit · same fixed goal':trainingFit?'Training fit · lower is better':'Demonstration fit · lower is better',
    xCaption:'Supervised updates',xUnit:'supervised updates',percentage:false,
  };
  const rows=(run?.series||[]).filter(row=>number(row.success_rate) && row.success_rate>=0 && row.success_rate<=1);
  const scope=rows.length ? metricScope(rows[rows.length-1]) : null;
  const updates=initialization && rows.some(row=>number(supervisedUpdates(row)));
  const value=row=>updates ? supervisedUpdates(row) : row.transitions;
  const points=rows.filter(row=>metricScope(row)===scope && number(value(row)))
    .map(row=>({x:Number(value(row)),y:Number(row.success_rate)}));
  const scopeName=run?.kind==='reference' ? 'Physical reference' : scopeLabel(scope);
  return {points,title:initialization?'Actor initialization':'Evaluation progress',ylabel:'Evaluation success',
    scope:points.length ? `${scopeName} · ${points.length} measurements` : 'No measurements yet',
    xCaption:initialization ? (updates||!points.length?'Supervised updates':'Physical evaluation calls') :
      run?.kind==='reference'?'Native action calls':run?.kind==='train'?'Online policy decisions · demonstrations excluded':'Environment interactions',
    xUnit:initialization?(updates?'supervised updates':'evaluation calls'):run?.kind==='train'?'online decisions':run?.kind==='reference'?'native calls':'interactions',
    percentage:true};
}

function renderChart(run) {
  const data=chartData(run),points=data.points;
  $('chart-heading').textContent=data.title;
  $('y-caption').textContent=data.ylabel;
  $('axis-caption').textContent=data.xCaption;
  $('chart-scope').textContent=data.scope;
  if (!points.length) { $('chart').replaceChildren(el('p',run?.kind==='initialization'?'Supervised progress and physical checks will appear when recorded.':'Measured success will appear after the first evaluation.','empty')); return; }
  const ns = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attributes, text) => { const node = document.createElementNS(ns, tag); for (const [key,value] of Object.entries(attributes)) node.setAttribute(key,value); if (text !== undefined) node.textContent=text; return node; };
  const svg = svgEl('svg',{viewBox:'0 0 620 220',role:'img','aria-label':`${data.ylabel} versus ${data.xCaption}. ${data.scope}.`});
  const left=40,right=608,top=18,bottom=190,maxX=Math.max(1,...points.map(p=>p.x));
  const maxY=data.percentage?1:Math.max(.0001,...points.map(p=>p.y))*1.08;
  const formatY=value=>data.percentage?percent(value):Number(value.toPrecision(2)).toString();
  const x = v => left + v/maxX*(right-left), y = v => bottom - v/maxY*(bottom-top);
  for (const ratio of [0,.5,1]) { const value=ratio*maxY; svg.append(svgEl('line',{x1:left,y1:y(value),x2:right,y2:y(value),class:'grid'}),svgEl('text',{x:left-9,y:y(value)+3,'text-anchor':'end'},formatY(value))); }
  for (const value of [0,.25,.5,.75,1]) svg.append(svgEl('text',{x:x(value*maxX),y:211,'text-anchor':'middle'},compact(value*maxX)));
  svg.append(svgEl('polyline',{points:points.map(p=>`${x(p.x)},${y(p.y)}`).join(' '),class:'series'}));
  for (const row of points) { const circle=svgEl('circle',{cx:x(row.x),cy:y(row.y),r:points.length>50?2:3.5,class:'point'}); circle.append(svgEl('title',{},`${count(row.x)} ${data.xUnit} · ${formatY(row.y)}`)); svg.append(circle); }
  $('chart').replaceChildren(svg);
}

function renderEvaluation(run) {
  const rows = run?.series || [];
  const measured = [...rows].reverse().find(row => number(row.success_rate) || number(row.full_task_success_rate));
  const values = measured || run?.metrics || {};
  const success = values.full_task_success_rate ?? values.success_rate;
  const scopeName = run?.kind === 'reference' ? 'Physical reference' : scopeLabel(values.full_task_success_rate !== undefined ? 'full_task' : metricScope(values));
  $('eval-scope').textContent = (number(success) || measured) ? scopeName : '';
  if (!number(success)) { $('evaluation').replaceChildren(el('p','Awaiting evaluated episodes.','empty')); return; }
  const episodes = values.eval_episodes ?? values.episodes;
  const hero = el('div'); hero.append(el('div',percent(success),'hero-metric'),el('p', `${scopeName} success${number(episodes) ? ` · ${count(episodes)} episodes` : ''}`,'hero-label'));
  if (measured && run?.kind==='train' && number(measured.transitions)) hero.append(el('p',`Evaluated at ${count(measured.transitions)} online decisions · ${ago(measured.timestamp)}`,'run-summary'));
  const metrics = el('dl',undefined,'metric-grid');
  const candidates = [
    ['Goal error',values.object_goal_distance ?? values.goal_error_m ?? values.mean_goal_distance ?? values.goal_distance,v=>`${(v*100).toFixed(1)} cm`],
    ['Upstream proximity',values.upstream_success_rate,percent],
    ['Drops',values.drops ?? values.drop_events,count],
    ['Physics ticks',values.physics_ticks,compact],
    ['Evaluation episodes',episodes,count],
    ['Elapsed',values.wall_time_seconds ?? values.wall_seconds,v=>`${Math.round(v/60)} min`],
  ];
  for (const [label,value,format] of candidates.filter(item=>number(item[1])).slice(0,4)) { const group=el('div'); group.append(el('dt',label),el('dd',format(Number(value)))); metrics.append(group); }
  hero.append(metrics); $('evaluation').replaceChildren(hero);
}

function progressValue(run) {
  if (run.kind==='initialization'||run.metrics?.scope==='actor_initialization') return {value:supervisedUpdates(run.metrics),unit:'supervised updates',budget:null};
  const collection=run.metrics?.scope==='training_data' || ['demonstration','correction_collection'].includes(run.metrics?.metric_source);
  return {value:run.transitions||0,unit:run.kind==='train'?'online decisions':collection?'data collection decisions':'interactions',budget:run.kind==='train'?run.budget:null};
}

function renderRunProgress(run) {
  if (!run) { $('selected-run').replaceChildren(el('span','Training has not started.')); return; }
  const node=el('div'),copy=el('div',undefined,'progress-copy');
  const progress=progressValue(run),initialization=run.kind==='initialization';
  const progressText=number(progress.value) ? (number(progress.budget) ? `${compact(progress.value)} / ${compact(progress.budget)} ${progress.unit}` : `${count(progress.value)} ${progress.unit}`) : 'Supervised updates pending';
  copy.append(el('span',`${run.algorithm || run.kind || 'Experiment'}${run.skill ? ' · '+run.skill : ''}`),el('span',progressText));node.append(copy);
  if (number(progress.budget) && progress.budget>0) { const track=el('div',undefined,'progress-track'),fill=el('div',undefined,'progress-fill'); fill.style.width=`${Math.min(100,100*progress.value/progress.budget)}%`; track.append(fill); node.append(track); }
  const accounting=runAccounting(run);
  const counts=initialization ? [
    ['new online SAC decisions',accounting?.new_online_transitions],
    ['physical evaluation calls',accounting?.new_evaluation_source_action_calls ?? run.metrics?.source_action_calls],
    ['evaluation physics ticks',accounting?.new_evaluation_physics_ticks ?? run.metrics?.physics_ticks],
  ] : [[run.kind==='train'?'training native calls':'native calls',run.metrics?.source_action_calls],['physics ticks',run.metrics?.physics_ticks],['updates',run.metrics?.updates]];
  const measured=counts.filter(item=>number(item[1]));
  if (measured.length) node.append(el('p',measured.map(([label,value])=>`${count(value)} ${label}`).join(' · '),'run-summary'));
  if (run.kind==='train' && number(run.inherited_transitions)) {
    const inherited=Number(run.inherited_transitions),current=Number(run.transitions||0);
    node.append(el('p',`${count(inherited)} inherited${current>=inherited ? ` + ${count(current-inherited)} new online decisions` : ' online decisions · preparing continuation'}`,'run-summary'));
  }
  if (accounting) {
    const detail=[];
    if (number(accounting.demonstration_transitions_imported) && accounting.demonstration_transitions_imported>0) detail.push(`${count(accounting.demonstration_transitions_imported)} demonstration decisions`);
    if (number(accounting.total_source_action_calls_known)) detail.push(`${count(accounting.total_source_action_calls_known)} total known native calls`);
    if (number(accounting.total_physics_ticks_known)) detail.push(`${count(accounting.total_physics_ticks_known)} total known physics ticks`);
    if (detail.length) node.append(el('p',detail.join(' · '),'run-summary'));
    const bc=accounting.bc_initialization,costs=bc?.incremental_accounting;
    if (bc?.included_in_cumulative_evaluation_totals && costs && number(costs.new_setup_source_action_calls) && number(costs.new_evaluation_source_action_calls)) {
      const bcCalls=Number(costs.new_setup_source_action_calls)+Number(costs.new_evaluation_source_action_calls);
      node.append(el('p',`BC: ${count(bc.total_bc_updates)} supervised updates · ${count(bcCalls)} setup/evaluation calls already included in totals`,'run-summary'));
    }
    if (accounting.complete===false) node.append(el('p','Earlier evaluation or setup costs are incomplete.','run-summary'));
  }
  if (run.kind==='train') node.append(el('p','Runs use different starts and training data; curves are development diagnostics.','run-summary'));
  if (run.summary) node.append(el('p',run.summary,'run-summary'));
  $('selected-run').replaceChildren(node);
}

function visibleRuns(runs, selected=selectedId, expanded=showAllRuns) {
  return expanded ? runs : runs.filter((run,index)=>index<10 || run.status==='running' || run.id===selected);
}

function renderRuns(runs) {
  $('run-count').textContent = runs.length;
  const toggle=$('run-toggle');toggle.hidden=runs.length<=10;
  toggle.textContent=showAllRuns?'Show recent':`Show all ${runs.length}`;
  toggle.setAttribute('aria-expanded',String(showAllRuns));
  const active=runs.filter(run=>run.status==='running').length;
  $('active-runs').textContent=active ? `${active} running` : 'None running';
  if (!runs.length) { const row=el('tr'),cell=el('td','No experiments registered.','empty');cell.colSpan=5;row.append(cell);$('run-table').replaceChildren(row);return; }
  $('run-table').replaceChildren(...visibleRuns(runs).map(run=>{
    const row=el('tr'),name=el('td'),button=el('button',run.name||run.id,'run-name');button.addEventListener('click',()=>{selectedId=run.id;renderSelected();});name.append(button);
    const status=el('td'),badge=el('span',run.status||'pending','status'); if (/^[a-z]+$/.test(run.status||'')) badge.classList.add(run.status);status.append(badge);
    const progress=progressValue(run);
    const value=number(progress.value)?`${count(progress.value)}${number(progress.budget)?' / '+compact(progress.budget):''}${run.kind==='initialization'?' BC updates':progress.unit==='data collection decisions'?' collection decisions':''}`:'—';
    row.append(name,el('td',[run.skill,run.algorithm||(run.kind||'')].filter(Boolean).join(' · ')),status,el('td',value),el('td',ago(run.updated_at)));return row;
  }));
}

function renderSelected() {
  const runs=state?.runs||[];
  if (!runs.some(run=>run.id===selectedId)) selectedId=runs.find(run=>run.kind==='train' && run.status==='running')?.id || runs[0]?.id || null;
  $('run-select').replaceChildren(...(runs.length ? runs.map(run=>{const option=el('option',run.name||run.id);option.value=run.id;return option;}) : [el('option','No experiments yet')]));
  $('run-select').value=selectedId||'';
  if (selectedId && new URLSearchParams(location.search).get('run') !== selectedId) {
    const url=new URL(location.href);url.searchParams.set('run',selectedId);history.replaceState(null,'',url);
  }
  const run=runs.find(item=>item.id===selectedId);renderChart(run);renderEvaluation(run);renderRunProgress(run);renderRuns(runs);renderArtifacts(runs);
}

const artifactUrl = (runId,path) => '/artifacts/'+encodeURIComponent(runId)+'/'+path.split('/').map(encodeURIComponent).join('/');
const isVideoArtifact = item => item.kind==='video' || (!item.kind && /\.(mp4|webm)$/i.test(item.path));
function posterPath(run) {
  const candidates=(run.artifacts||[]).filter(item=>{
    const name=item.path.split('/').pop();
    return /\.(png|jpe?g|gif)$/i.test(name) && /(preview|frame|final)/i.test(name)
      && !/(chart|plot|curve|route|trajectory|segments|loss)/i.test(name);
  });
  const rank=item=>/preview/i.test(item.path)?0:/frame/i.test(item.path)?1:2;
  return candidates.sort((a,b)=>rank(a)-rank(b))[0]?.path;
}

function renderArtifacts(runs) {
  const all=runs.filter(run=>!['stopped','failed','interrupted'].includes(run.status)).flatMap(run=>(run.artifacts||[]).map(item=>({
    ...item,run_id:run.id,run_name:run.name||run.id,poster:posterPath(run),
    mp4:/\.webm$/i.test(item.path) ? (run.artifacts||[]).find(candidate=>candidate.path===item.path.replace(/\.webm$/i,'.mp4'))?.path : null,
  })));
  const rank=item=>item.kind==='video'?(item.run_id===selectedId?0:1):item.kind==='image'?(item.run_id===selectedId?2:3):4;
  const items=all.sort((a,b)=>rank(a)-rank(b)).slice(0,6);
  const key=JSON.stringify(items);if(key===artifactKey)return;artifactKey=key;
  if(!items.length){$('artifacts').replaceChildren(el('p','Reference and learned-policy recordings will appear here.','empty'));return;}
  $('artifacts').replaceChildren(...items.map(item=>{
    const figure=el('figure'),url=artifactUrl(item.run_id,item.path);
    if(isVideoArtifact(item)){const video=el('video');video.src=url;if(item.poster)video.poster=artifactUrl(item.run_id,item.poster);video.controls=true;video.preload='metadata';video.playsInline=true;figure.append(video);}
    else if(item.kind==='image'||/\.(png|jpe?g|gif)$/i.test(item.path)){const image=el('img');image.src=url;image.alt=item.label||'Experiment visualization';image.loading='lazy';figure.append(image);}
    const caption=el('figcaption'),link=el('a',item.label||item.path);link.href=url;link.target='_blank';link.rel='noopener';caption.append(link);
    if(isVideoArtifact(item) && item.mp4){const download=el('a','MP4 download');download.href=artifactUrl(item.run_id,item.mp4);download.download=item.mp4.split('/').pop();caption.append(document.createTextNode(' · '),download);}
    caption.append(el('br'),document.createTextNode(item.run_name));figure.append(caption);return figure;
  }));
}

function renderEvents(events) {
  if(!events.length){$('events').replaceChildren(el('li','No recorded events.','empty'));return;}
  $('events').replaceChildren(...events.map(event=>{const item=el('li'),date=new Date(event.timestamp),time=el('time',Number.isNaN(date.valueOf())?'—':date.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}));time.title=event.timestamp||'';const message=el('div',event.message||'',event.level==='error'?'error':'');message.append(el('span',event.run_id,'event-run'));item.append(time,message);return item;}));
}

function renderResources(resources) {
  const parts=[];const add=(label,value)=>{const node=el('span');node.append(el('strong',label+' '),document.createTextNode(value));parts.push(node);};
  if(resources.gpus?.length){const gpu=resources.gpus[0];add(gpu.name.replace('NVIDIA ',''),`${Math.round(gpu.memory_used_mib/1024)}/${Math.round(gpu.memory_total_mib/1024)} GiB · ${Math.round(gpu.utilization)}%`);}
  if(number(resources.load_1m))add('CPU load',`${resources.load_1m} / ${resources.logical_cpus}`);
  if(number(resources.memory_available_bytes))add('RAM free',`${Math.round(resources.memory_available_bytes/2**30)} GiB`);
  if(number(resources.disk_free_bytes))add('Disk free',`${Math.round(resources.disk_free_bytes/2**30)} GiB`);
  $('resources').replaceChildren(...parts);
}

async function poll() {
  try { const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(30000)});if(!response.ok)throw new Error('HTTP '+response.status);state=await response.json();renderProject(state.project);renderSelected();renderArtifacts(state.runs);renderEvents(state.events);renderResources(state.resources);$('connection').textContent='Live';$('connection-dot').className='dot live';$('refreshed').textContent='Updated '+new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
  catch(error){$('connection').textContent='Disconnected';$('connection-dot').className='dot offline';$('refreshed').textContent=state?'Showing last received data':'Waiting for dashboard server';}
  finally{setTimeout(poll,5000);}
}
$('run-toggle').addEventListener('click',()=>{showAllRuns=!showAllRuns;renderRuns(state?.runs||[]);});
$('run-select').addEventListener('change',event=>{selectedId=event.target.value;renderSelected();});
poll();
