# TrackID3x3 schema inspection

## Status: source annotations validated

- Checkout: `G:\basket\basketball-pose-research\external\TrackID3x3`
- Git commit inspected: `9822a3dde59fc80fbb5eaacd443e79e8b059d94e`
- Canonical schema: `TrackID3x3.pose_instance.v1`
- Artifact type: annotation-only; RGB frames/videos are not required or loaded
- Synthetic: `false`
- Legacy MultiSubjects_SA/Henu action-label assumptions: disabled
- Official repository: <https://github.com/open-starlab/TrackID3x3>
- Official paper: <https://arxiv.org/html/2503.18282v2>

## Exact source pose order

The JSON files contain arrays of ten x/y pairs but no embedded joint-name field. The mapping below follows the paper's ordered ten-point protocol: it enumerates head, bilateral shoulders, elbows, wrists, center, and ankles; defines center as the hip midpoint; and states that the corresponding COCO17 baseline points use nose for head. No extra hip, knee, or depth coordinate is introduced.

| Source index | Canonical name | COCO17 evidence |
|---:|---|---|
| 0 | `head` | 0 |
| 1 | `left_shoulder` | 5 |
| 2 | `right_shoulder` | 6 |
| 3 | `left_elbow` | 7 |
| 4 | `right_elbow` | 8 |
| 5 | `left_wrist` | 9 |
| 6 | `right_wrist` | 10 |
| 7 | `center` | midpoint of COCO17 left/right hips (11, 12) |
| 8 | `left_ankle` | 15 |
| 9 | `right_ankle` | 16 |

Coordinates are source-image pixels in `[x, y]` order. Every inspected pose is exactly `[10, 2]`. Exact `[0, 0]` and non-finite pairs are treated as missing and represented by an explicit mask.

## Empirical annotation counts

| Domain | Pose JSON files | Annotated frames | Pose instances | Players | Resolution | Missing points | Bbox source |
|---|---:|---:|---:|---:|---|---:|---|
| Indoor | 13 | 2601 | 15606 | 6 | 1280x720 | 0 | mot_xywh: 15606 |
| Outdoor | 1 | 3600 | 21528 | 6 | 3840x2160 | 1 | mot_xywh: 21528 |
| Drone | 500 | 500 | 3000 | 6 | 3840x2160 | 0 | pose_tight_fallback: 3000 |
| **Total** | **514** | **6701** | **40134** | - | - | **1** | - |

These are pose-annotation counts in the checked-out JSON, not claims about all RGB frames in the full benchmark. Unique-frame and pose-instance counts are recorded separately because each annotated frame usually contains six players.

## Observed JSON layouts

- Indoor: 13 video JSON objects with top-level `video` and `annotations`; each `frame_N` maps six jersey identities to poses.
- Outdoor: one video JSON object with 3,600 frame maps; five or six pose instances per frame. `Purple_10` is absent in 72 frames.
- Drone: 500 image JSON objects with top-level `image` and `annotations`; each contains six identities. The pose JSON does not provide a reliable video identifier or frame rate.

Actual inspected examples are:

- `external/TrackID3x3/ground_truth/Indoor/pose/basket_S4T4_post.json`
- `external/TrackID3x3/ground_truth/Outdoor/pose/IMG_0104_.json`
- `external/TrackID3x3/ground_truth/Drone/pose/frame_1.json`

Indoor and Outdoor use this exact field nesting (one complete ten-pair Indoor pose is shown):

```json
{
  "video": {
    "file_name": "basket_S4T4_post.mp4",
    "resolution": [
      1280,
      720
    ],
    "frame_rate": 20,
    "total_frames": 235
  },
  "annotations": {
    "frame_0": {
      "Pink_Jersey": [
        [
          1010.64,
          392.28
        ],
        [
          1019.4202499389648,
          410.3005767822266
        ],
        [
          1037.3383255004883,
          408.0331665039063
        ],
        [
          1025.9375,
          437.0432692307692
        ],
        [
          1045.4086538461538,
          431.7548076923077
        ],
        [
          1036.5067901611328,
          449.58233032226565
        ],
        [
          1042.6968841552734,
          449.76824340820315
        ],
        [
          1032.9086538461538,
          456.27403846153845
        ],
        [
          1020.1682692307693,
          517.0913461538462
        ],
        [
          1036.514423076923,
          513.7259615384615
        ]
      ]
    }
  }
}
```

Outdoor has the same keys and nesting; for example its `video` object is `{"file_name": "IMG_0104_.mp4", "resolution": [3840, 2160], "frame_rate": 30, "total_frames": 3600}`, and an observed identity key in `annotations.frame_0` is `Green_6`.

Drone uses image metadata and has no `frame_N` wrapper inside `annotations` (one complete ten-pair pose is shown):

```json
{
  "image": {
    "file_name": "frame_1",
    "resolution": [
      3840,
      2160
    ]
  },
  "annotations": {
    "Black_1": [
      [
        1886.552642211914,
        1217.9712408447265
      ],
      [
        1920.9084924316405,
        1227.3658483886718
      ],
      [
        1875.6755212402343,
        1236.9091833496093
      ],
      [
        1934.761870727539,
        1262.5986364746093
      ],
      [
        1865.440726623535,
        1271.4283178710937
      ],
      [
        1936.0398400878905,
        1297.17349609375
      ],
      [
        1864.3644021606444,
        1306.1723974609374
      ],
      [
        1905.9240182495116,
        1304.3544653320312
      ],
      [
        1916.8582452392577,
        1385.2256811523437
      ],
      [
        1891.7508386230468,
        1406.2062719726562
      ]
    ]
  }
}
```

The identity is the object key (`Pink_Jersey`, `Green_6`, or `Black_1` in these examples). A pose value is directly a 10-element list of two-number `[x, y]` lists; there is no nested keypoint-name, confidence, visibility, bbox, action-label, or biomechanics field.

## Bbox linkage and provenance

- Indoor and Outdoor bboxes are stored separately from pose JSON. Inspected files include `external/TrackID3x3/ground_truth/Indoor/MOT/basket_S4T4_post.txt` and `external/TrackID3x3/ground_truth/Outdoor/MOT/IMG_0104.txt`.
- MOT text is headerless, comma-separated, and positional. The adapter consumes exactly columns 1-6 as `[frame, track, x, y, width, height]`. An Indoor row is `1,1,258.3,369.0,43.65,114.3,-1,-1,-1,-1` (10 observed columns); an Outdoor row is `1,1,2033.10,1363.50,137.70,391.50,1.00,-1,-1` (9 observed columns). Columns 7 onward are preserved upstream but are not consumed or assigned new semantics by this project. Pose `frame_0` aligns with MOT frame 1.
- Player-to-MOT identities are established as a dominant one-to-one mapping over full videos after same-frame spatial assignment. Weak mappings are dropped rather than guessed.
- Outdoor annotation suffixes remain canonical player identities; they are not silently equated to raw MOT track IDs.
- Numeric suffixes are parsed separately as `player_number`; the full color/team plus suffix remains `track_id` because a number alone is not globally unique (for example Drone has two number 10 players).
- Drone pose JSON has no bbox field and cannot be linked unambiguously to a particular MOT video. Its derived box is `[min_x, min_y, max_x - min_x, max_y - min_y]` over valid pose points; each instance is marked `pose_tight_fallback`, never `mot_xywh`.
- Leakage guard: a stored Drone pose-tight box uses the full annotated pose and is retained only as evaluation scale/provenance. After synthetic masking, bbox normalization recomputes a tight transform from observed joints only. The stored box is not an independent model-input transform.

- Indoor: 13 pose video(s), 0 dropped instances, mapping dominance range `0.940594 to 1.000000`.
- Outdoor: 1 pose video(s), 0 dropped instances, mapping dominance range `0.997778 to 1.000000`.

Loaded bbox provenance totals: `{'mot_xywh': 37134, 'pose_tight_fallback': 3000}`.

## Split and temporal reliability rules

- Every domain is split independently 70/15/15 with seed 42.
- Indoor grouping unit: video (13 groups -> 9/2/2 train/val/test).
- Outdoor grouping unit: player/track (6 groups -> 4/1/1) because only one pose video is annotated.
- Drone grouping unit: player/track (6 groups -> 4/1/1) because pose files lack reliable video identity.
- `split_group` and `grouping_level` are retained in manifests and arrays; group disjointness is asserted.
- Temporal windows are permitted only for Indoor/Outdoor video-player tracks. Drone is excluded from temporal datasets.

## Storage evidence

- `ground_truth` bytes: 139,071,282
- Checkout `.git` overhead bytes (separate): 38,252,860
- Pose JSON bytes: 48,808,262 bytes
- MOT text: 89,278,024 bytes

Generated from the checked-out source by `scripts/inspect_dataset.py`; no performance metric or research conclusion is reported here.
