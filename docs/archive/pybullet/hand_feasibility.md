# Dexterous-hand feasibility spike

Question: would a multi-finger hand hold objects better than Fetch's parallel
gripper in this simulator, so that grasp & lift becomes a real job for RL?

## Method

`scripts/hand_feasibility.py` mounts a hand on a 6-DoF gantry (x, y, z slides;
yaw, pitch, roll) driven exactly like the Fetch base and arm: position control,
2e6 N force cap, gain 0.3, five physics ticks per command. At that force cap the
controller closes 30% of the remaining error every tick regardless of mass, which
reproduces the measured base pulse (1.08 m/s on the first tick of a 15 mm step).
Fingers use the environment's finger gain (0.03). Same table, objects, gravity,
timestep and 480-tick settling as `pnr/env.py`.

Each trial: settle the object, scripted grasp, lift 20 cm, hold 60 commands, then
carry along a route recorded from a real learned carry (the carry adapter's motion:
0.075 m per decision in five 15 mm commands, steering 0.15 m ahead on the route).
Lost = fewer than two hand links in contact for five commands, or object near the
table.

## Hands surveyed (`robot_descriptions`, loaded in PyBullet)

| Hand | Joints | Collision shapes | Verdict |
|---|---:|---|---|
| Allegro | 16 | boxes + spheres | Used: primitive contacts, loads as-is |
| LEAP | 16 | meshes | Possible; mesh contacts less reliable |
| Barrett | 8 (coupling not modelled) | meshes behind ROS package paths | Needs path fixes and manual coupling |
| Ability | 10 (+4 mimic) | meshes | Broken inertial data (36 kg) |

## Results

Fetch gripper: reference controller's antipodal grasp (17 cm pregrasp, descend,
close, lift). Allegro: scripted top-down power grasp (palm down, fingers across the
object's long axis, thumb folded during descent then opposed), parameters chosen by
grid search on one route (`scripts/hand_grasp_search.py` and earlier grids).

10 recorded routes × 5 objects per hand, placement noise from point-cloud seeds:

| Retained through carry | Fetch gripper | Allegro (scripted) |
|---|---:|---:|
| Foam | 9/10 | 1/10 |
| Banana | 10/10 | 0/10 |
| Can | 5/10 | 5/10 |
| Apple | 6/10 | 8/10 |
| Mug | 10/10 | 6/10 |
| **Total** | **40/50** | **20/50** |

- The rig reproduces the real system's weak objects for Fetch (can, apple).
- When an Allegro wrap is right the hold is very secure (0.1–3 mm slip over a full
  route), but settings tuned on one route did not transfer: the open-loop scripted
  grasp is fragile and holds worse than the parallel gripper overall.
- Throughput: ~109 commands/s per process with Allegro (~17M commands/hour on 44
  workers) versus ~169 for Fetch; adequate for RL.
- Every object starts in the same pose on the table, so grasping is a per-object
  problem in this task; there is no spatial generalisation to learn.

## Robustness search and validation

`scripts/hand_grasp_search.py` scored 216 grasp settings (reach along the fingers,
palm gap, finger curl, thumb strength, finger direction, spread) on all five objects
across 5 routes and placement seeds each (5,400 trials). Robust settings exist on
those routes for apple (21 settings hold 5/5), mug (17), banana (4) and can (1);
none for foam (best 3/5). Choosing the best setting per object holds 23/25 there,
but that is measured on the routes used for selection.

On the 5 routes the search never saw, with new placement seeds:

| Unseen routes | Allegro (tuned per object) | Fetch gripper |
|---|---:|---:|
| Foam | 3/5 | 4/5 |
| Banana | 4/5 | 5/5 |
| Can | 1/5 | 1/5 |
| Apple | 3/5 | 3/5 |
| Mug | 3/5 | 5/5 |
| **Total** | **14/25** | **18/25** |

## Verdict: no-go for the dexterous hand in this simulator

The go criterion was "the hand holds clearly better than the parallel gripper". It
does not: even after a robustness search the scripted Allegro grasp retains fewer
objects than Fetch's gripper on unseen routes. Outcomes flip with millimetre
placement changes and route differences, the same partly random behaviour seen in
the parallel gripper's drops, so many-contact grasps in PyBullet are not more
reliable. Mounting on the Fetch wrist was not pursued.

