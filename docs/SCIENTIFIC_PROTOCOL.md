# Scientific protocol

The research question is how well compact skeleton representations reconstruct deliberately hidden TrackID3x3 keypoints within and across the Indoor, Outdoor, and Drone capture domains. This is pose-representation robustness research. It is neither action recognition nor an assessment of basketball technique.

## Primary task

Given the observed portion of a 10-joint pose, reconstruct a deterministic random subset of originally valid joints. Mask ratios are 10%, 20%, 30%, and 40%. Originally absent annotations and experimentally masked joints remain separate masks; loss and metrics are computed only on the latter. The primary metric is normalized mean joint error (NMJE) using the source bounding-box diagonal. PCK at 0.05, 0.10, and 0.20 and per-joint normalized error are secondary metrics.

## Controlled comparisons

Representations are raw image coordinates, bounding-box normalized coordinates, root-centered coordinates, scale-normalized coordinates, and normalized coordinates plus supported joint angles. The ten source joints are never expanded with invented hips, knees, or other landmarks.

Models are a deterministic geometric/interpolation baseline, compact MLP, lightweight TCN when a reliable same-video/same-player continuous window exists, and a graph model with fewer than one million parameters. Per-frame (`T=1`) evaluation is the mandatory common protocol. Temporal `T=8` and `T=16` results are supplemental and must never bridge tracks, videos, or discontinuous frame ranges.

## Domain and leakage controls

- Report within-domain evaluation for Indoor, Outdoor, and Drone.
- Report each ordered cross-domain train-to-test transfer when executed.
- Group train/validation/test by source video; use track as the fallback group only when video identity is unavailable.
- Never split frames from one group across partitions.
- Fit any data-derived normalization on training data only.
- Data-efficiency experiments use deterministic 25%, 50%, and 100% group subsets.
- The canonical sampling seed is `42`, with frame stride `3` and targets of 1,000 Indoor, 1,000 Outdoor, and 500 Drone pose instances.

No value is inserted into a results table without an executed artifact. Smoke-test outputs are labeled synthetic/software validation and excluded from empirical claims.
