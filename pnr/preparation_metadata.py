"""Explicit replay provenance for the opt-in preparation-balanced carry branch.

Collection modes are frozen by the audited TRAIN dataset lineage. The legacy
absent collection mode uses the seeder's existing 'demonstration' name; it is an
explicit reference-preparation declaration, not a fallback for unknown labels.
Online labels come from a wrapper applied only to LearnedMultiCaseCarry.
"""
from __future__ import annotations
import gymnasium as gym

PREPARATION_SOURCES_BY_MODE={
    'demonstration':'reference','standard':'reference','recovery':'reference',
    'geometric-recovery':'reference','learned-handover':'learned',
}
PREPARATION_SOURCES_BY_OBJECT={
    14:['reference','learned'],5:['learned'],2:['learned'],7:['learned'],10:['learned'],
}


class LearnedPreparationMetadata(gym.Wrapper):
    """Add a source label to genuine learned-handover transitions only."""
    def step(self,action):
        observation,reward,terminated,truncated,info=self.env.step(action)
        if info.get('conditional_on_prepared_handover') is not True:
            raise ValueError('Learned source metadata requires a declared learned handover')
        if info.get('preparation_source','learned')!='learned':
            raise ValueError('Conflicting learned-handover source metadata')
        info=dict(info,preparation_source='learned')
        return observation,reward,terminated,truncated,info
