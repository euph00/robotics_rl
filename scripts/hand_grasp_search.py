#!/usr/bin/env python3
"""Robustness search over scripted Allegro power-grasp settings (feasibility spike).

Each setting is scored on every object across several recorded carry routes and
point-cloud seeds (placement noise), so a setting must hold reliably, not once.
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts import hand_feasibility as hf

GRID = [dict(reach=r, palm_gap=g, curl=c, thumb=th, along=a, spread=sp)
        for r, g, c, th, a, sp in itertools.product((.14, .16, .18), (-.005, .005, .015), (1.2, 1.4, 1.6),
                                                     (1.3, 1.6), (False, True), (0., .3))]


def run(task):
    obj, params, urdf, case, seed = task
    scene = hf.Scene('allegro', obj, urdf)
    try:
        z0 = scene.object_state()[0][2]
        _, rotation = hf.grasp_allegro(scene, seed, params)
        grasped = scene.object_state()[0][2] - z0 > .10 and len(scene.contact_links()) >= 2
        retained = bool(grasped and hf.carry(scene, hf.route(case), rotation)['retained'])
        return {'object': hf.OBJECTS[obj], **params, 'route': case, 'seed': seed, 'grasped': bool(grasped), 'retained': retained}
    except Exception as exc:
        return {'object': hf.OBJECTS[obj], **params, 'route': case, 'seed': seed, 'error': str(exc)}
    finally:
        scene.close()


if __name__ == '__main__':
    out = hf.ROOT / 'runs/hand_feasibility01'
    urdf = hf.gantry_urdf('allegro', out / 'urdf')
    routes = sorted({json.loads(l)['route'] for l in (out / 'results.jsonl').open()})[:5]
    tasks = [(obj, params, str(urdf), case, 2000 + k) for obj in hf.OBJECTS for params in GRID
             for k, case in enumerate(routes)]
    with ProcessPoolExecutor(44) as pool:
        rows = list(pool.map(run, tasks, chunksize=4))
    (out / 'grasp_search.json').write_text(json.dumps(rows))
    key = lambda r: (r['reach'], r['palm_gap'], r['curl'], r['thumb'], r['along'], r['spread'])
    for name in hf.OBJECTS.values():
        scores = {}
        for r in rows:
            if r['object'] == name:
                scores.setdefault(key(r), []).append(r.get('retained', False))
        ranked = sorted(scores.items(), key=lambda kv: -np.mean(kv[1]))
        top = ranked[0]
        print(f'{name}: best setting retains {sum(top[1])}/{len(top[1])} {top[0]}; '
              f'settings with >=4/5: {sum(np.sum(v) >= 4 for v in scores.values())}; with 5/5: {sum(np.sum(v) == 5 for v in scores.values())}')
    totals = {}
    for r in rows:
        totals.setdefault(key(r), []).append(r.get('retained', False))
    best = max(totals.items(), key=lambda kv: np.mean(kv[1]))
    print(f'best single setting across all objects: {sum(best[1])}/{len(best[1])} {best[0]}')
