# Fixed preparation-source HER replay

`pnr.preparation_balanced_her.PreparationBalancedHerReplayBuffer` is a separate
opt-in subclass of the existing object-balanced buffer. It changes only original
transition-index sampling. The old sampler, source datasets, replay checkpoints,
policy observations and physical accounting remain unchanged.

The new branch uses a fresh replay seeded from the same real dataset, a zero-online
BC actor and fresh critics. There is no migration of old replay. Its declaration is:

```python
replay_buffer_class = PreparationBalancedHerReplayBuffer
replay_buffer_kwargs = {
    "object_indices": [14, 5, 2, 7, 10],
    "preparation_sources_by_object": {
        14: ["reference", "learned"],
        5: ["learned"],
        2: ["learned"],
        7: ["learned"],
        10: ["learned"],
    },
    "copy_info_dict": True,
    "n_sampled_goal": 1,
    "goal_selection_strategy": "future",
}
```

Every real transition must supply integer `info['object_idx']` and canonical
`info['preparation_source']`. These are replay metadata, never added policy
inputs. Both labels must remain fixed within an episode. Unknown, absent,
undeclared or changing labels fail before buffer mutation. The caller supplies
labels from frozen source provenance; the sampler does not infer them from
collection-mode strings, object geometry, outcomes or missing values.

Each draw selects one of five objects uniformly, then one of its **fixed declared
families** uniformly, then one eligible real transition uniformly. Thus reference
foam and learned foam each receive10% of draws, and each other learned object
receives20%. Every declared pair needs completed-episode transitions. Missing
pairs—including those emptied by ring eviction—stop sampling. This deliberately
supersedes the audit's tentative dynamic-family fallback and replay-migration
suggestion. A new branch must preserve its declared source distribution.

The parent `sample()` and SB3 2.7.0 future-goal/reward/terminal/timeout methods are
inherited unchanged. Goals stay within the selected real episode; half the batch
uses HER with the declared `n_sampled_goal=1`. Failures receive no separate
priority, and no artificial episode or physical attempt is introduced. The change
affects SAC actor replay states as well as critic states; continuing BC action
regularization has its own unchanged sampler.

`sampling_report()` adds nested `eligible_counts_by_object_and_source`,
`joint_sampling_probabilities`, `cumulative_draw_counts_by_object_and_source`,
and `missing_preparation_strata`. Object totals/counters remain available.
Reporting is read-only and does not consume RNG. Nominal joint probabilities are
reported even when sampling is blocked; `sampling_ready` makes that distinction
explicit. Declaration order is preserved independently of input dictionary order.

The extension identifier is
`preparation_sampling_protocol='uniform-declared-object-preparation-her-v1'`.
It is also the report's `protocol`; `base_sampling_protocol` identifies the
inherited object protocol. The parent's internal `sampling_protocol` remains
unchanged for its serialization validation. Ordinary replay pickles preserve
source arrays and actual draw counters; loads reject missing/corrupt source
metadata, inconsistent episode labels and mismatched object/joint counters.
This remains pinned to installed `stable-baselines3==2.7.0`.

Ten new tests plus the ten existing object-buffer tests pass. Checks include
sampling probabilities under severe imbalance, actual counters, missing/incomplete
strata, atomic validation, vector-worker isolation, ring overwrite, future goals
within same-object source episodes, exact stock-HER parity for identical indices,
and save/load with reproducible index draws. No physics or training was run for
implementation. Exposure rationale and the coverage tradeoff are in
[the source audit](phase2_replay_source_audit.md): initial learned-foam exposure
rises from0.195% to10%, while broad reference-foam exposure falls from19.805% to10%.
Physical improvement requires the separately declared experiment.
