# Deferred gated carry encoder prototype

`pnr/gated_spatial_features.py` provides the opt-in `GatedCarrySpatialExtractor`. No runner selects it, and no learned-policy experiment or physical evaluation was run. **Keep it deferred:** the expanded192-maze CNN initializer already achieved12/12 isolated validation successes.

The module preserves the580-state observation schema, five measured occupancy/goal planes, world/grid conventions, compact grip/velocity/contact inputs, raw-pose mask, optional navigation-only mask,128-dimensional fused features, and ordinary continuous SAC heads. It inherits the existing geometry/fusion code; only the map-processing core changes. Actor and critic extractors remain independent.

The core learns initial hidden/cell maps from the supplied observation, then applies a shared spatial convolution and per-cell LSTM update. States are local variables created afresh each forward call. There is no temporal history, map cache, path input, oracle search, or analytic transition model. HER therefore reconstructs the goal planes and all gated features from each sample's current `desired_goal`.

This is **GPPN-inspired**, not an exact reproduction of [Lee et al. (ICML2018)](https://proceedings.mlr.press/v80/lee18c/lee18c.pdf). Differences include rectangular12×22 maps, five input planes, a SiLU input stem, a continuous-control feature readout, and exactly20 recurrent updates by default. The authors' [reference code](https://github.com/RLAgent/gated-path-planning-networks/blob/master/models/GPPN.py) uses discrete action logits and counts its initialization toward its iteration parameter. Our latent channels are not calibrated values or certified paths.

With two3×3 initialization convolutions andK recurrent spatial kernels of odd widthF, a learned map cell has theoretical occupancy receptive-field side `5 + K*(F-1)` before pooling. The defaultK20/F5 gives85 cells. This describes potential input dependence, not guaranteed information preservation, obstacle-aware propagation, or a solved path length. The inherited global pooling already exposes the whole map. IncreasingK adds computation and unrolled gradient depth without adding parameters; optimization may still lose or distort useful information.

Opt-in SB3 policy configuration:

```python
from pnr.gated_spatial_features import GatedCarrySpatialExtractor

policy_kwargs = {
    'net_arch': [128, 128],
    'features_extractor_class': GatedCarrySpatialExtractor,
    'features_extractor_kwargs': {
        'features_dim': 128, 'channels': 32,
        'iterations': 20, 'kernel_size': 5,
    },
}
```

The existing caller still explicitly applies the actor's carry-pose mask. This prototype does not change actor/critic objectives, rewards, action scaling, HER, or native control.

## Bounded throughput measurements

Artifact: `runs/phase2_gated_encoder_benchmark01/artifacts/result.json`; reproduction script: `scripts/phase2_gated_encoder_benchmark.py`. Inputs are a fixed batch from existing physical TRAIN demonstrations. Timing models are disposable, unselected random initializations. There were78 timing-only BC optimizer updates, zero physics, zero validation rollouts, and no saved policy checkpoints. The full benchmark took32.9 seconds.

| Model | Encoder parameters | CPU predict1 | CPU BC256 | A100 predict1 | A100 BC256 | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|---:|
| Current CNN32 | 88,064 | 2.28ms | 121.29ms | 2.20ms | 5.57ms | 156MiB |
| Gated32,K20,F5 | 56,353 | 7.58ms | 1,139.36ms | 5.27ms | 17.72ms | 2,537MiB |
| Gated48,K20,F5 | 94,065 | 10.62ms | 1,827.57ms | 5.11ms | 23.14ms | 3,715MiB |

The48-channel option is within6.8% of the CNN's encoder parameter count. With32 channels, gated BC is3.18× slower on the A100 despite fewer parameters; with48 it is4.15× slower. Recurrence also materially increases training memory. Parameter count alone is a poor runtime estimate.

CPU timings use four PyTorch threads. GPU: NVIDIA A10080GB PCIe. These are median synchronized wall times on a shared machine with other project jobs; full samples and90th percentiles are in JSON. SB3 `predict` includes observation conversion, transfers and continuous-action output. BC256 measures the actual actor forward, action-MSE backward, gradient clipping and Adam step. It **does not** measure complete SAC updates with critics/HER/temperature. CUDA uses high float32 matmul precision and cuDNN benchmarking, with deterministic kernels disabled as in the current runner; numerical execution is not promised bitwise reproducible.

Verification:19 tests passed across gated/spatial/navigation tests. They cover goal relabeling through the recurrent core, A/B/A call independence, batch permutation, unchanged observations/buffers, rectangular map orientation, finite propagation radius, both masks, compact physical inputs, finite gradients, shared-parameter iteration depth, independent actor/critics, bounded continuous actions, and ordinary SAC+HER save/load. No production files were edited for this prototype.
