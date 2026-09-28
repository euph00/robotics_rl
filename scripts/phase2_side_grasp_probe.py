#!/usr/bin/env python3
"""Declared side-approach feasibility matrix; teacher diagnostic, original physics."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import phase2_carry_retention_probe as retention
from pnr.planning import Grasp, antipodal_grasps
from pnr.reference import ReferenceController
from pnr.scenes import case_specs, load_scene_bank
from pnr.telemetry import RunLogger, atomic_json

CONTEXT = {}

class SideApproachController(ReferenceController):
    def collision_report(self, q):
        report=[]
        with self.planner.query():
            self.planner._put(q)
            self.env.p.performCollisionDetection()
            for name,body in [('table',self.env.table_id),('maze',self.env.maze_id),('object',self.env.object_id)]:
                for c in self.env.p.getClosestPoints(self.env.robot_id,body,distance=.001):
                    report.append({'body':name,'robot_link':int(c[3]),
                        'robot_link_name':self.env.p.getJointInfo(self.env.robot_id,c[3])[12].decode() if c[3]>=0 else 'base',
                        'distance_m':float(c[8]),'robot_point':list(c[5]),'other_point':list(c[6])})
        return report

    def prepare_grasp(self):
        ctx = CONTEXT
        log = {'tilt_from_topdown_deg': ctx['angle'], 'tcp_setback_m': ctx['setback_m'],
               'pregrasp_distance_m': .17, 'candidate_limit': 24,
               'candidate_attempts': [], 'source_action_calls': self.env.step_count,
               'physics_ticks': self.env.physics_steps, 'selected': None}
        def save():
            log.update(source_action_calls=self.env.step_count, physics_ticks=self.env.physics_steps,
                       collision_checks=self.planner.collision_checks, rrt_calls=self.planner.rrt_calls)
            atomic_json(ctx['proposal_log'], log)
        try:
            topdown = Rotation.from_matrix(np.array([[0,0,1],[0,1,0],[-1,0,0]])).as_quat()
            base = self.base_position()
            self.move_ee([base[0]+.35,base[1],1.1], topdown, 'fold', closed=False)
            self.move_base([-.78,0.], stage='approach')
            obs = self.env.observe()
            candidates = antipodal_grasps(obs['object_pc'], obs['object_normals'],
                                          table_height=self.env.table_height)
            log['generated_antipodal_pairs'] = len(candidates)
            angle = np.deg2rad(ctx['angle'])
            for index, original in enumerate(candidates):
                y = Rotation.from_quat(original.quaternion).apply([0.,1.,0.])
                horizontal = np.array([y[1], -y[0], 0.])
                x = np.sin(angle)*horizontal + np.cos(angle)*np.array([0.,0.,-1.])
                z = np.cross(x,y)
                orientation = Rotation.from_matrix(np.column_stack((x,y,z))).as_quat()
                # The original palm ends32mm behind TCP. The declared setback
                # supplies clearance while preserving the measured contact pair.
                position = original.position - ctx['setback_m']*x
                grasp = Grasp(position, orientation, original.width, original.score, original.contacts)
                pre = position - .17*x
                attempt = {'index': index, 'original_proposal': original.as_dict(),
                           'side_proposal': grasp.as_dict(), 'approach_axis_world': x.tolist(),
                           'pregrasp_world': pre.tolist(), 'before_calls': self.env.step_count,
                           'status': 'querying'}
                log['candidate_attempts'].append(attempt)
                started = time.monotonic()
                try:
                    q = self.planner.ik(pre, orientation)
                    q[self.finger_slots] = .05
                    if not self.planner.collision_free(q, include_object=True):
                        attempt['collision_details'] = self.collision_report(q)
                        raise ValueError('Pregrasp collides with table, maze or object')
                    q_grasp = self.planner.ik(position, orientation, seed=q)
                    q_grasp[self.finger_slots] = .05
                    if not self.planner.collision_free(q_grasp, include_object=True):
                        attempt['collision_details'] = self.collision_report(q_grasp)
                        raise ValueError('Open grasp pose collides with table, maze or object')
                    # Verify the complete Cartesian approach geometrically before
                    # physical execution; these save/restore queries use zero ticks.
                    previous = q
                    for t in np.linspace(0,1,10)[1:]:
                        qi = self.planner.ik(pre+t*(position-pre), orientation, seed=previous)
                        qi[self.finger_slots] = .05
                        if not self.planner.collision_free(qi, include_object=True):
                            attempt['collision_details'] = self.collision_report(qi)
                            raise ValueError(f'Open approach collides at fraction{t:.3f}')
                        previous = qi
                    attempt['status'] = 'geometrically_feasible'
                    save()
                    self.move_joint(q, stage='pregrasp', include_object=True)
                    self.grasp = grasp
                    attempt['status'] = 'selected_and_prepared'
                    log['selected'] = index
                    save()
                    return grasp
                except ValueError as error:
                    attempt['status'] = 'rejected'
                    attempt['error'] = str(error)
                finally:
                    attempt['wall_seconds'] = time.monotonic()-started
                    attempt['after_calls'] = self.env.step_count
                    save()
            raise RuntimeError('No collision-free side-approach candidate among declared24pairs')
        except Exception as error:
            log['error'] = f'{type(error).__name__}: {error}'
            save()
            raise

def worker(task):
    case, angle, gate, setback_m, output = task
    name = f"{case['layout_id']}_tilt{int(angle):03d}_gate{int(gate):02d}"
    actual = dict(case, diagnostic_id=name)
    CONTEXT.clear()
    CONTEXT.update(angle=angle, setback_m=setback_m, proposal_log=Path(output)/f'{name}_proposals.json')
    retention.ReferenceController = SideApproachController
    result = retention.worker((actual, 'general075', 'contact', output, 1400, None,
                               'original', 0., False, True, gate))
    proposal_log = json.loads(CONTEXT['proposal_log'].read_text())
    result['side_approach'] = proposal_log
    result['tilt_from_topdown_deg'] = angle
    result['tcp_setback_m'] = setback_m
    atomic_json(Path(output)/f"{result['case_id']}.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--angles-deg', type=int, nargs='+', choices=[45,60,90], default=[45,60,90])
    parser.add_argument('--tcp-setback-mm', type=float, default=8.)
    parser.add_argument('--close-distances-mm', type=float, nargs='+', default=[5.,18.])
    args = parser.parse_args()
    if not 1<=args.workers<=8:parser.error('Use1..8 CPU workers')
    if not 0<=args.tcp_setback_mm<=25:parser.error('TCP setback must be0..25mm')
    if any(not 5<=gate<=18 for gate in args.close_distances_mm):parser.error('Closure gate must be5..18mm')
    if args.output.exists():parser.error('Use a new output directory')
    out=args.output/'artifacts';out.mkdir(parents=True)
    bank=load_scene_bank('configs/phase2_scene_bank.json')
    train=case_specs(bank,'train',object_indices=[2])[0]
    cases=[{'layout_id':'course_legacy42','maze_seed':None,'object_idx':2,'scene_seed':42},
           {'layout_id':train['layout_id'],'maze_seed':train['env_kwargs']['maze_seed'],
            'object_idx':2,'scene_seed':train['env_kwargs']['seed']}]
    tasks=[(case,angle,gate,args.tcp_setback_mm/1000.,str(out)) for case in cases for angle in args.angles_deg for gate in args.close_distances_mm]
    sources={}
    for relative in ['scripts/phase2_side_grasp_probe.py','scripts/phase2_carry_retention_probe.py',
                     'pnr/env.py','pnr/reference.py','pnr/planning.py','pnr/general_manipulation.py',
                     'pnr/general_carry.py','pnr/skills.py','pnr/local_control.py']:
        file=ROOT/relative;sources[relative]=hashlib.sha256(file.read_bytes()).hexdigest()
        shutil.copy2(file,out/(relative.replace('/','_')))
    atomic_json(args.output/'config.json', {'protocol':'side-approach-feasibility-v1',
        'bank_sha256':bank['manifest_sha256'],'cases':cases,'angles_deg':args.angles_deg,
        'close_distances_mm':args.close_distances_mm, 'tcp_setback_m':args.tcp_setback_mm/1000.,'delta_scale':.075,
        'original_physics':True,'teacher_grasp_and_navigation':True,'sources':sources})
    logger=RunLogger(args.output.name,root=args.output.parent,kind='reference',skill='grasp',
                     name='Side grasp feasibility',algorithm='Physical teacher diagnostic',
                     summary='Original gripper/physics; can only; declared side approaches; all attempts counted.')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            summary={'protocol':'side-approach-feasibility-v1','planned_attempts':len(tasks),
                     'completed_attempts':len(results),'successes':sum(r['success'] for r in results),
                     'source_action_calls':sum(r['source_action_calls'] for r in results),
                     'physics_ticks':sum(r['physics_ticks'] for r in results),
                     'initialization_physics_ticks':sum(r['initialization_physics_ticks'] for r in results),
                     'results':results}
            atomic_json(out/'summary.json',summary)
            logger.metric(transitions=summary['source_action_calls'],scope='physical_teacher_diagnostic',
                          eval_episodes=len(results),physics_ticks=summary['physics_ticks'],
                          reference_success_rate=summary['successes']/len(results))
            print(json.dumps({key:result.get(key) for key in ['case_id','success','error','physics_ticks']}),flush=True)
    logger.artifact('artifacts/summary.json','All side-approach attempts')
    logger.finish(summary=f"{summary['successes']}/{len(tasks)} declared side-approach transport attempts passed; teacher diagnostic.")

if __name__=='__main__':main()
