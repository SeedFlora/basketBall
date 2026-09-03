# Compact TrackID3x3 processed format

The canonical NPZ stores pose instances or leakage-safe track windows. `T=1` is always supported; larger `T` is valid only for consecutive observations from one player in one source video.

| Field | Type | Shape | Meaning |
|---|---|---:|---|
| `coords` / `keypoints` | `float32` | `[N,T,10,2]` | Pixel-space source coordinates; invalid values are zeroed |
| `valid_mask` | `bool` | `[N,T,10]` | Originally annotated finite joints |
| `bbox_xywh` / `bboxes` | `float32` | `[N,T,4]` | Matched source bounding box in pixel `x,y,w,h` convention |
| `domains` | Unicode | `[N]` | `Indoor`, `Outdoor`, or `Drone` |
| `video_ids` | Unicode | `[N]` | Source video or explicit unavailable-video surrogate |
| `track_ids` | Unicode | `[N]` | Source player/track identifier, scoped by video |
| `frame_ids` | `int64` | `[N,T]` | Source frame identifiers |
| `sequence_mask` | `bool` | `[N,T]` | Real rather than padded temporal steps |
| `sample_ids` | Unicode | `[N]` | Stable unique instance/window identifiers |
| `splits` | Unicode | `[N]` | Group-disjoint `train`, `val`, or `test` |
| `joint_names` | Unicode | `[10]` | Exact TrackID3x3 order from `config/joint_mapping.yaml` |
| `edges` | `int64` | `[E,2]` | Graph edges over only the ten supported joints |

Representations are derived explicitly at model input time or saved in separately named arrays; image-space and normalized coordinates are never silently mixed. `metadata.json` records source commit, schema variant, bbox association provenance, grouping/split decisions, representation convention, seed, stride, sampling counts, and whether an artifact is synthetic.

The experiment mask is generated deterministically from the run seed and applies only where `valid_mask=true`. Training targets never treat source-missing joints as ground truth.
