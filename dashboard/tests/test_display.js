/* Metric-semantics checks, runnable with Node or an existing JS evaluator. */
function runDisplayChecks(appSource) {
  const nodes={};
  function node(tag) { return {tag,textContent:'',children:[],attributes:{},style:{},classList:{add(){}},
    append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},
    addEventListener(){},setAttribute(key,value){this.attributes[key]=value}}; }
  const document={getElementById:id=>nodes[id]||(nodes[id]=node('div')),createElement:node,
    createElementNS:(ns,tag)=>node(tag),createTextNode:text=>Object.assign(node('text'),{textContent:text})};
  const source=appSource.replace(/\npoll\(\);\s*$/,'\n');
  const api=new Function('document','location','history',source+'\nreturn {chartData,renderChart,renderEvaluation,renderRunProgress,renderRuns,renderSelected,visibleRuns,progressValue,posterPath,renderArtifacts,setView:(value,selected)=>{state=value;selectedId=selected;}};')(document,{search:'',href:'http://localhost/'},{replaceState(){}});
  const text=item=>item.textContent+' '+item.children.map(text).join(' ');
  const assert=(condition,message)=>{if(!condition)throw new Error(message);};
  const accounting={new_online_transitions:0,new_evaluation_source_action_calls:1025,new_evaluation_physics_ticks:5125,
    demonstration_transitions_imported:3200,total_source_action_calls_known:136674,total_physics_ticks_known:686730};
  const init={kind:'initialization',transitions:1025,metrics:{updates:500,accounting},series:[
    {transitions:0,updates:100,bc_validation_loss:.08,scope:'actor_initialization'},
    {transitions:450,updates:500,bc_validation_loss:.014,scope:'actor_initialization'},
    {transitions:1025,updates:500,success_rate:1,eval_episodes:4,scope:'four_start_grasp_diagnostic'}]};
  let chart=api.chartData(init);
  assert(chart.points.map(point=>point.x).join(',')==='100,500','BC curve must use update counts, never physical calls');
  assert(!chart.percentage && chart.ylabel==='Validation action loss','BC loss must not be labelled success');
  api.renderChart(init);api.renderEvaluation(init);api.renderRunProgress(init);api.renderRuns([init]);
  assert(nodes['axis-caption'].textContent==='Supervised updates','BC axis missing');
  let progress=text(nodes['selected-run']);
  assert(progress.includes('500 supervised updates') && progress.includes('1,025 physical evaluation calls'),'BC update and physical costs must be separate');
  assert(!progress.includes('1,025 online') && progress.includes('0 new online SAC decisions'),'BC physical calls presented as online training');
  assert(text(nodes.evaluation).includes('Grasp · four starts') && text(nodes.evaluation).includes('4 episodes'),'Physical scope/count lost');
  assert(text(nodes['run-table']).includes('500 BC updates') && !text(nodes['run-table']).includes('1,025'),'BC table progress conflates physical calls with updates');
  chart=api.chartData({...init,series:[{transitions:627,bc_updates:500,success_rate:0,scope:'full_carry'}]});
  assert(chart.percentage && chart.points[0].x===500 && chart.xCaption==='Supervised updates','BC success fallback must use supervised updates');
  chart=api.chartData({...init,series:[{transitions:627,success_rate:0,scope:'full_carry'}]});
  assert(chart.points[0].x===627 && chart.xCaption==='Physical evaluation calls','Missing update fallback must retain physical units');
  chart=api.chartData({...init,series:[{updates:1000,bc_training_scope:'full_route_only',bc_full_route_training_loss:.02,bc_validation_loss:.8}]});
  assert(chart.points[0].y===.02 && chart.ylabel==='Full-route training action loss','Focused BC must plot actual full-route training fit, not out-of-distribution segment loss');
  assert(chart.scope==='Training fit · same fixed goal' && chart.xCaption==='Supervised updates','Full-goal training fit must remain clearly in-sample');
  api.renderEvaluation({kind:'evaluation',series:[{scope:'full_task',success_rate:.78,
    success_rate_within_deadline:.67,success_rate_within_deadline_including_initialization:.63,
    eval_episodes:100,physics_ticks:1456190,total_physics_ticks_including_initialization:1504190}]});
  const fullTask=text(nodes.evaluation);
  assert(fullTask.includes('78%') && fullTask.includes('63 / 100') && fullTask.includes('67 / 100'),'Full, inclusive and action-only deadline results must remain distinct');
  assert(fullTask.includes('incl. settling') && fullTask.includes('action only'),'Deadline scope labels missing');
  const partial={id:'confirmation',kind:'evaluation',transitions:3939,planned_cases:100,
    metrics:{scope:'full_task',success_rate:.01,success_rate_within_deadline:.01,
      success_rate_within_deadline_including_initialization:.01,eval_episodes:2,planned_cases:100},
    series:[{scope:'full_task',success_rate:.01,transitions:3939,eval_episodes:2,planned_cases:100}]};
  // Rates use all planned cases, even before every case has finished.
  api.renderEvaluation({...partial,series:[]});
  const partialResult=text(nodes.evaluation);
  assert(partialResult.includes('1 / 100 planned cases') && partialResult.includes('1 / 100') && !partialResult.includes('0 / 2'),'Partial evaluation must retain fixed planned denominator');
  api.renderRunProgress(partial);api.renderRuns([partial]);
  assert(text(nodes['selected-run']).includes('2 / 100 evaluated cases') && text(nodes['selected-run']).includes('3,939 native calls'),'Evaluated cases must remain separate from native calls');
  assert(text(nodes['run-table']).includes('2 / 100 cases'),'Live table must use matching case units');
  chart=api.chartData(partial);
  assert(chart.points[0].x===2 && chart.points[0].y===.01 && chart.xCaption==='Evaluated cases' && chart.scope.includes('fixed 100-case denominator'),'Live plot must preserve supplied rate and explain its denominator');
  api.renderEvaluation({...partial,series:[],metrics:{...partial.metrics,eval_episodes:100,success_rate:.78,success_rate_within_deadline:.67,success_rate_within_deadline_including_initialization:.63}});
  assert(text(nodes.evaluation).includes('78 / 100 planned cases') && text(nodes.evaluation).includes('63 / 100'),'Final planned-case results must retain the same denominator');
  api.setView({project:{run_labels:{pending:'Declared confirmation'}},runs:[{id:'old',kind:'evaluation',metrics:{success_rate:1,episodes:1}}]},'pending');
  api.renderSelected();
  assert(nodes['run-select'].value==='pending' && text(nodes['selected-run']).includes('Awaiting experiment registration') && !text(nodes.evaluation).includes('100%'),'Declared pending permalink must not silently borrow an older result');
  api.setView({project:{run_labels:{pending:'Declared confirmation'}},runs:[{...partial,id:'pending'}]},'pending');
  api.renderSelected();
  assert(nodes['run-select'].value==='pending' && text(nodes['selected-run']).includes('2 / 100 evaluated cases'),'Pending permalink must become live when its run registers');
  const fine={kind:'train',transitions:42000,inherited_transitions:40000,budget:42000,
    accounting:{total_source_action_calls_known:138959},metrics:{accounting:{total_source_action_calls_known:145853,
      total_physics_ticks_known:735025,bc_initialization:{total_bc_updates:500,included_in_cumulative_evaluation_totals:true,
        incremental_accounting:{new_setup_source_action_calls:457,new_evaluation_source_action_calls:1025}}}},series:[
    {transitions:39000,success_rate:1,scope:'easy_manipulation_diagnostic'},
    {transitions:40000,success_rate:1,scope:'skill',evaluation_scope:'full_pregrasp_diagnostic'},
    {transitions:42000,success_rate:1,scope:'skill',evaluation_scope:'full_pregrasp_diagnostic'}]};
  chart=api.chartData(fine);
  assert(chart.points.length===2 && chart.points[0].x===40000 && chart.xCaption.startsWith('Online policy decisions'),'Training axis/scope changed');
  api.renderRunProgress(fine);progress=text(nodes['selected-run']);
  assert(progress.includes('40,000 inherited + 2,000 new online decisions'),'Inherited/new online counters missing');
  assert(progress.includes('145,853 total known native calls') && !progress.includes('138,959'),'Stale accounting chosen');
  assert(progress.includes('1,482 setup/evaluation calls already included') && !progress.includes('147,335'),'BC costs counted twice');
  chart=api.chartData({kind:'reference',series:[{transitions:4417,success_rate:1,scope:'reference'}]});
  assert(chart.xCaption==='Native action calls','Reference axis changed');
  const history=Array.from({length:45},(_,i)=>({id:'run'+i,status:i===20?'running':'complete',kind:'train',transitions:i}));
  const recent=api.visibleRuns(history,'run30',false);
  assert(recent.length===12 && recent.some(run=>run.id==='run20') && recent.some(run=>run.id==='run30'),'Recent table must retain active and selected old runs');
  assert(api.visibleRuns(history,null,true).length===45,'Show all must retain full history');
  api.renderRuns(history);
  assert(nodes['run-table'].children.length===11 && nodes['run-toggle'].textContent==='Show all 45','Default table limit or toggle label wrong');
  const collector={id:'demos',kind:'validation',transitions:5614,budget:28,metrics:{scope:'training_data'}};
  assert(api.progressValue(collector).budget===null,'Episode budget incorrectly compared with collector decisions');
  api.renderRunProgress(collector);api.renderRuns([collector]);
  assert(text(nodes['selected-run']).includes('5,614 data collection decisions') && !text(nodes['selected-run']).includes('/ 28'),'Collector progress mixes units');
  assert(text(nodes['run-table']).includes('5,614 collection decisions') && !text(nodes['run-table']).includes('/ 28'),'Collector table mixes units');
  assert(nodes['run-toggle'].hidden,'Unnecessary toggle for short history');
  const mediaRun={id:'reference',status:'complete',artifacts:[
    {path:'artifacts/rollout.webm',kind:'video'},
    {path:'artifacts/rollout.mp4',kind:'video'},
    {path:'artifacts/final_chart.png',kind:'image'},
    {path:'artifacts/final.png',kind:'image'},
    {path:'artifacts/learned_reach_frame.png',kind:'image'}]};
  assert(api.posterPath(mediaRun)==='artifacts/learned_reach_frame.png','Video poster should use a same-run frame, not a chart');
  assert(api.posterPath({artifacts:[{path:'artifacts/frame001.png'}]})==='artifacts/frame001.png','Numbered frames should be eligible');
  assert(api.posterPath({artifacts:[{path:'artifacts/final_chart.png'}]})===undefined,'Chart must not become video poster');
  api.renderArtifacts([mediaRun]);
  const media=nodes.artifacts.children.flatMap(item=>item.children).filter(item=>item.tag==='video');
  assert(media.length===1 && media[0].poster==='/artifacts/reference/artifacts/learned_reach_frame.png','Video poster URL or explicit MP4 download behavior wrong');
  const captionLinks=nodes.artifacts.children.flatMap(item=>item.children).filter(item=>item.tag==='figcaption').flatMap(item=>item.children).filter(item=>item.tag==='a');
  assert(captionLinks.some(link=>link.textContent==='MP4 download' && link.download==='rollout.mp4'),'WebM caption must retain MP4 download despite capped gallery');
  api.setView({project:{featured_run:'featured'}},'selected-test');
  api.renderArtifacts([
    {id:'selected-test',status:'complete',artifacts:[{path:'artifacts/matrix.png',kind:'image'}]},
    {id:'featured',status:'complete',artifacts:[{path:'artifacts/featured.webm',kind:'video'}]},
    ...Array.from({length:10},(_,i)=>({id:'prior'+i,status:'complete',artifacts:[{path:'artifacts/prior.webm',kind:'video'}]}))]);
  const gallery=nodes.artifacts.children.flatMap(item=>item.children);
  assert(gallery.some(item=>item.tag==='img' && item.src==='/artifacts/selected-test/artifacts/matrix.png'),'Selected test matrix must not disappear behind unrelated videos');
  assert(gallery.some(item=>item.tag==='video' && item.src==='/artifacts/featured/artifacts/featured.webm'),'Featured reproduction video must stay accessible');
  assert(text(nodes.artifacts).includes('Historical recording · featured'),'Featured prior video must be identified as historical');
  return 'Display, fixed-denominator progress, accounting, history, units, and video-poster checks passed.';
}
if (typeof module!=='undefined' && require.main===module) {
  const fs=require('fs'),path=require('path');
  console.log(runDisplayChecks(fs.readFileSync(path.join(__dirname,'../public/app.js'),'utf8')));
}
