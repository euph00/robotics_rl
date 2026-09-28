"""TRAIN caches of independently prepared native-SAC grasp variants.

The original cache and deterministic protocol remain unchanged. Internal cache
slots identify preparations; environment infos retain the canonical scene case.
Every slot starts a fresh native world, including deterministic variants.
"""
from copy import deepcopy

from .grasp_sampling import (SAMPLING_PROTOCOL, fixed_variants, validate_variant,
                             predictor_for_variant)
from .learned_carry import LearnedMultiCaseCarry, CACHE_PROTOCOL as BASE_CACHE_PROTOCOL

CACHE_PROTOCOL = 'fixed-train-native-sac-handover-variants-v1'


def expand_preparation_cases(cases, seed=9401):
    """Freeze three variants per canonical case before worker slicing."""
    result = []
    for index, case in enumerate(cases):
        if 'preparation_variant' in case or 'original_case_id' in case:
            raise ValueError('Expand canonical TRAIN cases exactly once')
        for variant in fixed_variants(index, seed=seed):
            result.append(dict(deepcopy(case), preparation_variant=variant))
    return result


class DiverseLearnedMultiCaseCarry(LearnedMultiCaseCarry):
    def __init__(self, cases, grasp_checkpoint, grasp_sha256, **kwargs):
        slots = []
        for case in cases:
            slot = deepcopy(case)
            variant = slot.get('preparation_variant')
            validate_variant(variant)
            if 'original_case_id' in slot:
                raise ValueError('Expected canonical case ID, not an internal preparation slot')
            slot['original_case_id'] = slot['case_id']
            slot['case_id'] = f"{slot['case_id']}__{variant['variant_id']}"
            slots.append(slot)
        super().__init__(slots, grasp_checkpoint, grasp_sha256, **kwargs)

    def _prepare_case(self, index):
        key = self.case_keys[index]
        if key in self.cache or key in self.failed_cases:
            return
        original_grasp = self.grasp
        variant = self.cases[index]['preparation_variant']
        predictor = predictor_for_variant(original_grasp, variant)
        before = len(self.preparation_attempts)
        self.grasp = predictor
        try:
            super()._prepare_case(index)
        finally:
            self.grasp = original_grasp
            if len(self.preparation_attempts) > before:
                record = self.preparation_attempts[-1]
                record['original_case_id'] = self.cases[index]['original_case_id']
                record['preparation_id'] = self.cases[index]['case_id']
                record['preparation_variant'] = deepcopy(variant)
                record['sampling_report'] = predictor.sampling_report()
                # Base failure records are copies; preserve sampling provenance too.
                if key in self.failed_cases:
                    self.failed_cases[key] = deepcopy(record)
                    self.failed_setup_attempts[-1] = deepcopy(record)

    def _restore_entry(self, key, *, reason):
        result = super()._restore_entry(key, reason=reason)
        if result['success']:
            observation = self.cache[key]['carry']._observation()
            result['post_refresh_observation'] = {name: value.tolist()
                                                 for name, value in observation.items()}
        return result

    def _canonical_info(self, info):
        info = dict(info)
        case = self._active_entry['case']
        info.update(case_id=case['original_case_id'], preparation_id=case['case_id'],
                    preparation_variant=deepcopy(case['preparation_variant']))
        return info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        return observation, self._canonical_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, self._canonical_info(info)

    def cache_report(self):
        report = super().cache_report()
        report.update(protocol=CACHE_PROTOCOL, sampling_protocol=SAMPLING_PROTOCOL,
            case_record_unit='independent original-start preparation slot',
            canonical_case_ids=list(dict.fromkeys(c['original_case_id'] for c in self.cases)),
            preparation_variants=[{'case_id': c['original_case_id'],
                'preparation_id': c['case_id'], 'variant': deepcopy(c['preparation_variant'])}
                for c in self.cases],
            restore_state_note='Five physical hold calls refresh contact; this is a measured post-refresh state, not the untouched continuous handover.')
        return report

    def get_training_state(self):
        state = super().get_training_state()
        state['protocol'] = CACHE_PROTOCOL
        state['sampling_protocol'] = SAMPLING_PROTOCOL
        return state

    def set_training_state(self, state):
        if (state.get('protocol') != CACHE_PROTOCOL or
                state.get('sampling_protocol') != SAMPLING_PROTOCOL):
            raise ValueError('Mismatched native-SAC preparation variant protocol')
        inherited = deepcopy(state)
        inherited['protocol'] = BASE_CACHE_PROTOCOL
        super().set_training_state(inherited)
