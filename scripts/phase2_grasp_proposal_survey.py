#!/usr/bin/env python3
"""Bounded physical TRAIN pregrasp survey; no policy outcomes guide selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def worker(task):
    seed,out=task
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pnr.env import PickNavReachEnv
    from pnr.reference import ReferenceController
    from pnr.telemetry import atomic_json
    native=None;record={'object_idx':2,'scene_seed':seed,'training_only':True,'success':False}
    with (Path(out)/f'can_{seed}.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            native=PickNavReachEnv(seed=seed,object_idx=2)
            controller=ReferenceController(native,seed=61000,grasp_proposal_mode='clearance-v2')
            grasp=controller.prepare_grasp();axis=Rotation.from_quat(grasp.quaternion).apply([1.,0.,0.])
            record.update(success=True,proposal=grasp.as_dict(),grasp_generation=controller.stage_results['grasp_generation'],
                          tilted=bool(abs(axis[2])<.999),approach_yaw_degrees=float(np.rad2deg(np.arctan2(axis[1],axis[0]))))
        except Exception as error:record['error']=repr(error)
        finally:
            if native is not None:
                record.update(source_action_calls=native.step_count,physics_ticks=native.physics_steps,
                              initialization_physics_ticks=native.settle_steps)
                native.close()
            else:record['partial_setup_cost_unknown']=True
            import pickle
            with (Path(out)/f'can_{seed}.pkl').open('wb') as backup:pickle.dump(record,backup)
            record=json.loads(json.dumps(record,default=lambda value: value.item() if isinstance(value,np.generic) else value.tolist()))
            atomic_json(Path(out)/f'can_{seed}.json',record)
    return record


def main():
    from pnr.telemetry import RunLogger,atomic_json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();seeds=list(range(42024,42056))
    logger=RunLogger(args.run_id,kind='validation',skill='grasp',budget=len(seeds),
        name='TRAIN can proposal coverage survey',algorithm='Physical geometric pregrasp survey',
        summary='Next32 fixed TRAIN point seeds; every setup cost/failure retained, no policy evaluation.')
    out=logger.directory/'artifacts';out.mkdir();(out/'source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(logger.directory/'config.json',{'seeds':seeds,'training_only':True,'workers':6,
        'selection_basis':'Measured proposal orientation only; no learned policy evaluation','grasp_proposal_mode':'clearance-v2'})
    rows=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,(seed,str(out))) for seed in seeds]):
            rows.append(future.result())
            atomic_json(out/'summary.json',{'cases':rows,'policy_evaluations':0,
                'source_action_calls':sum(r.get('source_action_calls',0) for r in rows),
                'physics_ticks_including_settling':sum(r.get('physics_ticks',0)+r.get('initialization_physics_ticks',0) for r in rows)})
            logger.metric(len(rows),scope='training_geometry',cases_completed=len(rows),
                positive_yaw_tilted=sum(r.get('tilted',False) and r['approach_yaw_degrees']>0 for r in rows))
    logger.artifact('artifacts/summary.json','All TRAIN proposal orientations and physical setup costs')
    logger.finish(summary='Completed bounded geometric TRAIN survey; no policy success filtering.')
    print(json.dumps([(r['scene_seed'],r.get('approach_yaw_degrees')) for r in rows if r.get('tilted') and r['approach_yaw_degrees']>0]))

if __name__=='__main__':main()
