"""Load saved SAC policies for inference without allocating a HER training buffer."""
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer


def load_policy_checkpoint(path,device='cpu'):
    # HER replay needs a live reward environment even though prediction does not.
    # A tiny unused ordinary buffer lets SB3 restore the exact saved policy. Never
    # use this inference-only loader to resume training/replay; train.py does that.
    return SAC.load(path,device=device,replay_buffer_class=DictReplayBuffer,
                    replay_buffer_kwargs={},buffer_size=1)
