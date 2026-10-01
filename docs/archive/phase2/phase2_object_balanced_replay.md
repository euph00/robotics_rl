# Object-balanced HER replay

The merged192-maze/five-object source has503,901 real transitions. The existing eight-worker seeder imports502,432, dropping1,469 tail transitions from uneven worker streams and closing seven cropped prefixes as collector timeouts. All18 real physical failure endpoints remain. This audit reproduced the current greedy worker assignment and cropping without running physics or training.

| Object | Imported transitions | Ordinary HER exposure | Opt-in object balance |
| --- | ---: | ---: | ---: |
| Foam14 | 483,061 | 96.145% | 20% |
| Banana5 | 5,903 | 1.175% | 20% |
| Can2 | 4,706 | 0.937% | 20% |
| Apple7 | 4,013 | 0.799% | 20% |
| Mug10 | 4,749 | 0.945% | 20% |

Under ordinary transition-uniform HER,95.210% of updates use reference-prepared foam and4.790% use learned handovers. The18 physical loss endpoints occupy0.00358% of the buffer: a256-sample batch includes any endpoint with probability0.913%, or about18 endpoint draws over2,000 updates. Object-balanced sampling raises learned-handover exposure to80.195% and endpoint-batch probability to10.085% (about213 endpoint draws over2,000 updates), without directly prioritizing failures.

Learned-foam remains underrepresented: reference and learned foam share one stratum, leaving learned foam only0.195% of all balanced draws. The other four objects come entirely from learned handovers. Source balancing or failure prioritization is deliberately absent and would require a separate declared change. This sampler also changes the replay-state distribution used by SAC's actor loss, not only its critics.

`pnr.object_balanced_her.ObjectBalancedHerReplayBuffer` is an opt-in subclass pinned to installed `stable-baselines3==2.7.0`. Constructor arguments are the ordinary HER arguments plus required `object_indices=[14,5,2,7,10]`; `copy_info_dict=True` is required. Every real transition's `info` must contain integer `object_idx` from its source episode or online case. Optional `replay_source` metadata may remain in info but does not influence selection. Labels never enter policy observations and do not duplicate transitions or inflate physical accounting.

The sampler draws one declared object uniformly, then one of that object's eligible completed-episode transitions uniformly. It preserves the declaration's order. It refuses missing eligible objects, unlabelled/undeclared transitions and an object switch inside one episode. Native SB3 still handles episode/ring validity, inclusive future-goal selection from the same episode, the unchanged HER ratio, reward recomputation with copied physical info, true terminal flags and timeout masking. With `n_sampled_goal=1`, half the selected transitions receive hindsight goals; relabeling does not change object exposure or erase physical failures.

`sampling_report()` returns eligible counts, declared probabilities, missing objects and cumulative actual original-transition draws per object, plus protocol/version/HER ratio. Labels and draw counters persist with the ordinary replay pickle; loading requires the pinned SB3 version and intact sampling metadata. The existing `HerReplayBuffer` and all default runners remain unchanged unless explicitly switched by the caller.

Ten focused tests cover transition imbalance, incomplete/missing objects, invalid labels, vector-worker isolation, ring overwrite, save/load, future-goal boundaries and exact equality to stock HER rewards/dones/goals when both receive identical original indices. Audit numbers are in `runs/phase2_replay_exposure_audit01/artifacts/exposure.json`. The source dataset SHA256 is `42eb7f4673ca3bfc86882d25e868222c78b36e0b158d124f340280c14d2654b3`; projected probabilities describe the seeded buffer before subsequent online additions or overwrites.
