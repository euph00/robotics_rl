#!/usr/bin/env python3
"""Exactly three declared original-start reproduction attempts; logging only."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse,ctypes,hashlib,json,multiprocessing as mp,os,pickle,platform,random,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import pybullet as pb
from scripts import phase2_evaluate_system as evaluator
from pnr.env import PickNavReachEnv
from pnr.system import LearnedSystem
from pnr.telemetry import atomic_json

_RECORDER=None
LIBC=ctypes.CDLL(None)
LIBC.fegetenv.argtypes=[ctypes.c_void_p]
LIBC.fegetenv.restype=ctypes.c_int
LIBC.fegetround.restype=ctypes.c_int
CRITICAL=['pnr/env.py','pnr/system.py','pnr/skills.py','pnr/general_manipulation.py',
 'pnr/general_carry.py','pnr/carry.py','pnr/reference.py','pnr/planning.py',
 'pnr/clearance_grasps.py','pnr/local_control.py','pnr/policies.py','pnr/features.py',
 'pnr/spatial_features.py','scripts/phase2_evaluate_system.py']


def digest(path):return evaluator.sha256_file(path)
def hash_bytes(value):return hashlib.sha256(value).hexdigest()
def fp_environment():
    value=ctypes.create_string_buffer(256)
    result=LIBC.fegetenv(value)
    return {'rounding_mode':int(LIBC.fegetround()),'fegetenv_result':result,
            'fenv_first32_hex':value.raw[:32].hex(),
            'x87_control_word':int.from_bytes(value.raw[:2],'little'),
            'mxcsr_glibc_x86_64':int.from_bytes(value.raw[28:32],'little') if platform.machine()=='x86_64' else None}

def rng_state():
    return {'numpy':np.random.get_state(),'python':random.getstate(),
            'torch_cpu':torch.get_rng_state().numpy().copy()}
def rng_digest():return {k:hash_bytes(pickle.dumps(v,protocol=4)) for k,v in rng_state().items()}
def metadata():
    return {'pid':os.getpid(),'ppid':os.getppid(),'python':sys.version,'platform':platform.platform(),
        'machine':platform.machine(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'numpy_version':np.__version__,'torch_version':str(torch.__version__),'pybullet_api':pb.getAPIVersion(),
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'mkldnn_enabled':torch.backends.mkldnn.enabled,'mkldnn_deterministic':torch.backends.mkldnn.deterministic,
        'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),'torch_config':torch.__config__.show(),
        'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MKL_CBWR','PYTHONHASHSEED']},
        'rng_digests':rng_digest(),'floating_point':fp_environment()}

def physical(native):
    states=native.p.getJointStates(native.robot_id,native.joint_indices)
    position,quaternion=native.p.getBasePositionAndOrientation(native.object_id)
    linear,angular=native.p.getBaseVelocity(native.object_id)
    return {'qpos':[r[0] for r in states],'qvel':[r[1] for r in states],
            'object_position':list(position),'object_quaternion':list(quaternion),
            'object_linear_velocity':list(linear),'object_angular_velocity':list(angular),
            'contact_manifold':native.p.getContactPoints(native.robot_id,native.object_id)}

def obs_copy(observation):
    return {k:{'dtype':str(np.asarray(v).dtype),'shape':list(np.asarray(v).shape),'values':np.asarray(v).tolist()}
            for k,v in observation.items()}

class Recorder:
    def __init__(self,directory):
        self.directory=directory;self.native=None;self.system=None;self.decisions={};self.current_policy=None
        self.native_log=(directory/'native_actions.jsonl').open('x')
        self.policy_log=(directory/'policy_predictions.jsonl').open('x')
        self.render_log=(directory/'render_calls.jsonl').open('x')
        self.initial_metadata=metadata();self.native_count=0;self.render_count=0
        (directory/'rng_before.pkl').write_bytes(pickle.dumps(rng_state(),protocol=4))
    def write(self,stream,data):stream.write(json.dumps(data,default=evaluator.plain)+'\n')
    def close(self,result):
        for stream in [self.native_log,self.policy_log,self.render_log]:stream.close()
        (self.directory/'rng_after.pkl').write_bytes(pickle.dumps(rng_state(),protocol=4))
        atomic_json(self.directory/'metadata.json',{'initial':self.initial_metadata,'final':metadata(),
            'logged_native_calls':self.native_count,'render_calls':self.render_count,
            'policy_decisions':self.decisions,'outcome':result,
            'logging_contract':'read-only getters/copies only; no seeds, targets, action outputs, physics parameters or timestep changes'})

class LoggedNative(PickNavReachEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        _RECORDER.native=self
        atomic_json(_RECORDER.directory/'initial_physics.json',{'scene_manifest':self.scene_manifest(),
            'state':physical(self),'physics_parameters':self.p.getPhysicsEngineParameters(),
            'initial_motor_targets':self.last_target.tolist(),
            'initial_observation_hashes':{k:hash_bytes(np.asarray(v).tobytes()) for k,v in self.observe().items()},
            'settling_ticks':self.settle_steps,
            'substeps':self.substeps,'max_force':self.max_force,'position_gains':self.position_gains,
            'real_time':self.real_time,'rng_digests':rng_digest(),'floating_point':fp_environment()})
    def step(self,action):
        r=_RECORDER;s=r.system
        before=physical(self);fp_before=fp_environment()
        requested=np.asarray(action).copy()
        observation,info=super().step(action)
        r.native_count+=1
        r.write(r.native_log,{'source_action':self.step_count,'physics_tick':self.physics_steps,
            'stage':s.display_stage if s else None,'controller':s.control_source if s else None,
            'current_policy':r.current_policy,'requested_action_dtype':str(requested.dtype),
            'requested_absolute_target':requested.tolist(),'executed_absolute_target':self.last_target.tolist(),
            'before':before,'after':physical(self),
            'returned_observation':obs_copy({k:observation[k] for k in ['qpos','qvel','object_pos','object_xyzw','goal_pos']}),
            'bilateral_contact':info['bilateral_contact'],'normal_force':info['normal_force'],
            'floating_point_before':fp_before,'floating_point_after':fp_environment()})
        return observation,info
    def render_rgb(self,*args,**kwargs):
        r=_RECORDER;before=fp_environment();state_before=physical(self);rng_before=rng_digest()
        frame=super().render_rgb(*args,**kwargs)
        r.render_count+=1
        r.write(r.render_log,{'source_action':self.step_count,'args':args,'kwargs':kwargs,
            'floating_point_before':before,'floating_point_after':fp_environment(),
            'rng_before':rng_before,'rng_after':rng_digest(),
            'physical_state_unchanged':state_before==physical(self),
            'frame_shape':list(frame.shape),'frame_sha256':hash_bytes(frame.tobytes())})
        return frame

class LoggedPolicy:
    def __init__(self,name,policy):self.name=name;self.policy=policy
    def __getattr__(self,name):return getattr(self.policy,name)
    def predict(self,observation,*args,**kwargs):
        r=_RECORDER;decision=r.decisions.get(self.name,0)+1
        r.decisions[self.name]=decision;r.current_policy={'skill':self.name,'decision':decision}
        inputs=obs_copy(observation);rng_before=rng_digest();fp_before=fp_environment()
        prediction=self.policy.predict(observation,*args,**kwargs)
        action=prediction[0] if isinstance(prediction,tuple) else prediction
        r.write(r.policy_log,{'skill':self.name,'decision':decision,'source_action_before':r.native.step_count,
            'input':inputs,'action':np.asarray(action).tolist(),'action_dtype':str(np.asarray(action).dtype),
            'deterministic':kwargs.get('deterministic'),'rng_before':rng_before,'rng_after':rng_digest(),
            'floating_point_before':fp_before,'floating_point_after':fp_environment()})
        return prediction

def logged_system(*args,**kwargs):
    system=LearnedSystem(*args,**kwargs);_RECORDER.system=system;return system

def initialize(records,worker_directory=None):
    evaluator.initialize_worker(records,'cpu',1)
    if worker_directory:
        atomic_json(Path(worker_directory)/f'worker_{os.getpid()}.json',metadata())

def noop_setup():
    start=time.time();time.sleep(2)
    return {'pid':os.getpid(),'started_wall_time':start,'completed_wall_time':time.time(),'native_calls':0}

def attempt(label,plan_path):
    global _RECORDER
    plan=json.loads(Path(plan_path).read_text());directory=Path(plan['output'])/'artifacts'/label
    directory.mkdir(exist_ok=False)
    (directory/'attempt_started.json').write_text(json.dumps({'pid':os.getpid(),'label':label,'wall_time':time.time()}))
    _RECORDER=Recorder(directory)
    evaluator.PickNavReachEnv=LoggedNative;evaluator.LearnedSystem=logged_system
    evaluator._WORKER_POLICIES={k:LoggedPolicy(k,v) for k,v in evaluator._WORKER_POLICIES.items()}
    result=None
    try:
        result=evaluator.evaluate_case(plan['case'],plan['execution'],directory,video=label=='single_render')
        assert result['source_action_calls']==_RECORDER.native_count
        return {'label':label,'pid':os.getpid(),'success':result['success'],'error':result.get('error'),
                'source_action_calls':result['source_action_calls'],'physics_ticks':result['physics_ticks'],
                'total_physics_ticks_including_initialization':result['total_physics_ticks_including_initialization']}
    finally:
        _RECORDER.close(result);evaluator.PickNavReachEnv=PickNavReachEnv;evaluator.LearnedSystem=LearnedSystem

def lines(path):
    with Path(path).open() as f:return [json.loads(line) for line in f]
def maxdiff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return None if a.shape!=b.shape else float(np.max(np.abs(a-b),initial=0))
def compare(left,right):
    a,b=lines(left/'native_actions.jsonl'),lines(right/'native_actions.jsonl')
    result={'left':left.name,'right':right.name,'common_calls':min(len(a),len(b))}
    fields=['qpos','qvel','object_position','object_quaternion','object_linear_velocity','object_angular_velocity']
    for key in ['requested_absolute_target','executed_absolute_target']:
        first=next(((x,y) for x,y in zip(a,b) if x[key]!=y[key]),None)
        result['first_'+key+'_difference']=None if first is None else {'source_action':first[0]['source_action'],
            'stage':first[0]['stage'],'max_abs':maxdiff(first[0][key],first[1][key]),
            'before_physical_differences':{k:maxdiff(first[0]['before'][k],first[1]['before'][k]) for k in fields}}
    for phase in ['before','after']:
        first=next(((x,y) for x,y in zip(a,b) if any(x[phase][k]!=y[phase][k] for k in fields)),None)
        result['first_'+phase+'_state_difference']=None if first is None else {'source_action':first[0]['source_action'],
            'stage':first[0]['stage'],'max_abs_by_field':{k:maxdiff(first[0][phase][k],first[1][phase][k]) for k in fields}}
    first=next(((x,y) for x,y in zip(a,b) if x['after']['contact_manifold']!=y['after']['contact_manifold']),None)
    result['first_contact_manifold_difference']=None if first is None else first[0]['source_action']
    p,q=lines(left/'policy_predictions.jsonl'),lines(right/'policy_predictions.jsonl')
    for key in ['input','action']:
        first=next(((x,y) for x,y in zip(p,q) if (x['skill'],x['decision'])==(y['skill'],y['decision']) and x[key]!=y[key]),None)
        result['first_policy_'+key+'_difference']=None if first is None else {'skill':first[0]['skill'],
            'decision':first[0]['decision'],'source_action_before':first[0]['source_action_before'],
            'inputs_equal':first[0]['input']==first[1]['input'],'actions_equal':first[0]['action']==first[1]['action'],
            'action_max_abs':maxdiff(first[0]['action'],first[1]['action']),
            'input_max_abs':{k:maxdiff(first[0]['input'][k]['values'],first[1]['input'][k]['values']) for k in first[0]['input']}}
    return result

def summarize(plan,attempts,workers):
    out=Path(plan['output']);art=out/'artifacts';labels=plan['attempts']
    comparisons=[compare(art/labels[i],art/labels[j]) for i,j in [(0,1),(1,2),(0,2)]]
    historical=[]
    for label in labels:
        current={r['source_action']:r for r in lines(art/label/'native_actions.jsonl')}
        for name,path in plan['historical_traces'].items():
            old=lines(path);first=None;identical=0
            for r in old:
                if r['source_action'] not in current:break
                now=current[r['source_action']]['returned_observation']
                differences={'qpos':maxdiff(r['qpos'],now['qpos']['values']),
                    'object_position':maxdiff(r['object_position'],now['object_pos']['values']),
                    'object_quaternion':maxdiff(r['object_quaternion'],now['object_xyzw']['values'])}
                if any(v!=0 for v in differences.values()):first={'source_action':r['source_action'],'max_abs':differences};break
                identical=r['source_action']
            historical.append({'attempt':label,'historical':name,'last_identical_saved_sample':identical,'first_difference':first})
    observations={}
    for label in labels:
        native=lines(art/label/'native_actions.jsonl');policies=lines(art/label/'policy_predictions.jsonl');render=lines(art/label/'render_calls.jsonl')
        observations[label]={'policy_rng_changed_calls':sum(r['rng_before']!=r['rng_after'] for r in policies),
            'render_rng_changed_calls':sum(r['rng_before']!=r['rng_after'] for r in render),
            'render_physical_state_changed_calls':sum(not r['physical_state_unchanged'] for r in render),
            'mxcsr_control_values':sorted({r[p]['mxcsr_glibc_x86_64'] & ~0x3f for r in native for p in ['floating_point_before','floating_point_after']}),
            'rounding_modes':sorted({r[p]['rounding_mode'] for r in native for p in ['floating_point_before','floating_point_after']}),
            'render_calls':len(render)}
    summary={'scope':'three predeclared development reproducibility diagnostics; not added to headline60-case result',
        'attempts':attempts,'executor':workers,'pairwise_comparisons':comparisons,'historical_comparisons':historical,
        'observations':observations,'total_new_original_start_attempts':3,
        'total_new_physics_ticks_including_initialization':sum(r['total_physics_ticks_including_initialization'] for r in attempts),
        'limits':['One physical case in six-worker context; no other simultaneous physical cases or previous worker episode history.',
                  'Metadata logging adds read-only queries and wall time; no action, seed or simulation parameter change.',
                  'One observation per context is not a statistical reproducibility estimate.']}
    atomic_json(art/'summary.json',summary);print(json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--single');parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if args.single:
        plan=json.loads(args.plan.read_text());assert args.single in ['single_no_render','single_render']
        initialize(plan['checkpoints']);print(json.dumps(attempt(args.single,args.plan)),flush=True);return
    if args.output is None:parser.error('--output is required')
    out=args.output.resolve();out.mkdir(exist_ok=False);art=out/'artifacts';art.mkdir()
    old=ROOT/'runs/phase2_system_learned_balanced_040k01/artifacts';video=ROOT/'runs/phase2_system_learned_balanced_040k_video01/artifacts'
    control=json.loads((old/'evaluation_plan.json').read_text());saved=json.loads((video/'reproduction_trace_audit.json').read_text())
    assert saved['same_case'] and saved['same_execution'] and saved['same_checkpoint_hashes']
    case=control['cases'][0];assert case['case_id']=='validation_000_object_14' and case['partition']=='validation'
    for source in CRITICAL:assert digest(ROOT/source)==control['source_hashes'][source],source
    for record in control['checkpoints'].values():assert digest(record['frozen_path'])==record['sha256']
    records=evaluator.freeze_checkpoints({k:Path(v['frozen_path']) for k,v in control['checkpoints'].items()},art/'checkpoints')
    evaluator.freeze_sources(art/'source')
    plan={'protocol':'three-context-logging-only-reproduction-v1','output':str(out),'case':case,
        'execution':control['execution'],'checkpoints':records,'control_plan_sha256':control['plan_sha256'],
        'reviewed_prior_audit_sha256':digest(video/'reproduction_trace_audit.json'),
        'attempts':['executor6_no_render','single_no_render','single_render'],'repeat_or_replacement_allowed':False,
        'historical_traces':{'full_eval_success':str(old/(case['case_id']+'_trajectory.jsonl')),
            'preselected_video_failure':str(video/(case['case_id']+'_trajectory.jsonl'))},
        'source_sha256':digest(Path(__file__)),'critical_source_hashes':{p:digest(ROOT/p) for p in CRITICAL},
        'thread_settings':{'torch_threads':1,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        'executor_context':'six spawned initialized workers, six2-second no-op setup tasks, one selected physical case; actual PIDs retained',
        'video':'same evaluator callback every4native calls; TinyRenderer640x360 follow camera pitch-78, PIL overlay,12fpsMP4 and existingWebM conversion',
        'logging':'all native absolute input/executed targets, float64 queried state/contact manifolds, policy inputs/outputs, RNG hashes and backend/FPU metadata; read-only'}
    plan_path=art/'predeclared_plan.json';atomic_json(plan_path,plan)
    workers=art/'executor_workers';workers.mkdir();initialize(records)
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(records,str(workers))) as pool:
        jobs=[pool.submit(noop_setup) for _ in range(6)]
        noops=[job.result() for job in jobs]
        first=pool.submit(attempt,plan['attempts'][0],str(plan_path)).result()
    worker_meta={'configured_workers':6,'initialized_worker_pids':[json.loads(p.read_text())['pid'] for p in sorted(workers.glob('*.json'))],
        'noop_tasks':noops,'physical_case_pid':first['pid'],'physical_cases_dispatched':1}
    atomic_json(art/'executor_behavior.json',worker_meta)
    results=[first]
    for label in plan['attempts'][1:]:
        with (art/(label+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--single',label,'--plan',str(plan_path)],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{label} process failed; no retry authorized; see retained log')
        record=json.loads((art/label/'metadata.json').read_text());r=record['outcome']
        results.append({'label':label,'pid':record['initial']['pid'],'success':r['success'],'error':r.get('error'),
            'source_action_calls':r['source_action_calls'],'physics_ticks':r['physics_ticks'],
            'total_physics_ticks_including_initialization':r['total_physics_ticks_including_initialization']})
    summarize(plan,results,worker_meta)

if __name__=='__main__':main()
