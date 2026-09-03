# TrackID3x3 split audit

## Status: passed

- Seed: `42`
- Frame stride before static instance selection: `3`
- Splitting is performed independently within every domain.
- Synthetic: `false`
- Legacy Henu subject/action split: not used
- Duplicate canonical sample IDs: `0`

## Static pose-instance subset

| Domain | Grouping level | Groups train/val/test | Instances train/val/test | Selected instances | Per-group range | Unique selected frames |
|---|---|---:|---:|---:|---:|---:|
| Indoor | video | 9/2/2 | 693/154/153 | 1000 | 76-77 | 604 |
| Outdoor | player | 4/1/1 | 666/167/167 | 1000 | 166-167 | 719 |
| Drone | player | 4/1/1 | 334/83/83 | 500 | 83-84 | 164 |

## Pairwise split-group overlap

- Indoor: `{'train/val': 0, 'train/test': 0, 'val/test': 0}`
- Outdoor: `{'train/val': 0, 'train/test': 0, 'val/test': 0}`
- Drone: `{'train/val': 0, 'train/test': 0, 'val/test': 0}`
- All pairwise overlap counts are zero: `true`

## Temporal dataset audit (T=16)

Temporal windows are built from full assigned source streams before random static-instance thinning. Frames are regular at step 3, windows have hop 16 observations (48 source frames), and therefore do not overlap within a track.

| Domain | Available windows | Selected windows | Selected pose observations | Rule |
|---|---:|---:|---:|---|
| Indoor | 300 | 62 | 992 | reliable video/player track, regular step 3, non-overlapping |
| Outdoor | 447 | 62 | 992 | reliable video/player track, regular step 3, non-overlapping |
| Drone | 0 | 0 | 0 | pose JSON has no reliable video identity |

Drone is deliberately excluded from temporal windows because its pose JSON has no reliable video identity. This is not interpreted as an empty or failed track.

Static bbox provenance: `{'mot_xywh': 2000, 'pose_tight_fallback': 500}`.
