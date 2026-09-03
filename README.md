# basketBall

## TrackID3x3 Masked-Keypoint Reconstruction

An end-to-end, storage-conscious research pipeline for reconstructing deliberately hidden landmarks in the official [TrackID3x3](https://github.com/open-starlab/TrackID3x3) 10-joint basketball skeleton. The controlled study compares lightweight pose representations within and across the Indoor, Outdoor, and Drone capture domains.

New to computer vision or skeleton research? Start with the comprehensive Indonesian [beginner guide](docs/PANDUAN_PROYEK.md), or read the illustrated [PDF edition](output/pdf/panduan_proyek_basketball_pose.pdf). Both explain the data, method, metrics, experiments, results, limitations, and terminology from first principles.

This project is **not action classification**, does not infer dribble/lay-up/shoot labels, and does not assess biomechanical correctness, coaching quality, or “good” versus “bad” basketball technique. It reconstructs held-out 2D annotations only.

## Research question and evidence boundary

> How robustly can compact skeleton representations reconstruct masked TrackID3x3 keypoints when camera geometry and viewpoint change across domains?

The mandatory common protocol is per-pose reconstruction (`T=1`). Reliable-track `T=16` experiments are supplemental. A deterministic mask hides 10%, 20%, 30%, or 40% of originally valid joints; naturally absent source annotations are never used as reconstruction targets.

The previous Henu MultiSubjects source was gated and returned HTTP 401, so no Henu archive was downloaded. The executable workflow now uses TrackID3x3 annotations at upstream commit `9822a3dde59fc80fbb5eaacd443e79e8b059d94e`. The rationale and retained infrastructure are recorded in [dataset_migration.md](outputs/reports/dataset_migration.md).

## Dataset audit

The checked-out annotation schema was inspected rather than assumed. Indoor and Outdoor store video-level JSON; Drone stores one JSON per annotated image. Every pose is `[10, 2]` in source-image pixel `[x, y]` order. Exact `[0, 0]` pairs and non-finite coordinates are marked invalid with an explicit mask.

| Domain | Pose JSON files | Annotated frames | Pose instances | Identities | Resolution | Natural missing points | Bbox provenance |
|---|---:|---:|---:|---:|---|---:|---|
| Indoor | 13 | 2,601 | 15,606 | 6 | 1280×720 | 0 | 15,606 official MOT `xywh` |
| Outdoor | 1 | 3,600 | 21,528 | 6 | 3840×2160 | 1 | 21,528 official MOT `xywh` |
| Drone | 500 | 500 | 3,000 | 6 | 3840×2160 | 0 | 3,000 `pose_tight_fallback` |
| **Total** | **514** | **6,701** | **40,134** | — | — | **1** | — |

These are pose-annotation counts, not a claim about every frame in the optional RGB videos. The authoritative audit is [trackid3x3_schema.md](outputs/reports/trackid3x3_schema.md).

### Canonical 10-joint mapping

The source arrays contain no embedded joint names. The mapping below follows the ordered 10-point protocol documented by the TrackID3x3 paper and is fixed in [joint_mapping.yaml](config/joint_mapping.yaml).

| Index | Canonical joint | COCO17 evidence |
|---:|---|---|
| 0 | `head` | nose, index 0 |
| 1 | `left_shoulder` | index 5 |
| 2 | `right_shoulder` | index 6 |
| 3 | `left_elbow` | index 7 |
| 4 | `right_elbow` | index 8 |
| 5 | `left_wrist` | index 9 |
| 6 | `right_wrist` | index 10 |
| 7 | `center` | midpoint of left/right hips, indices 11/12 |
| 8 | `left_ankle` | index 15 |
| 9 | `right_ankle` | index 16 |

Indoor and Outdoor poses were linked to same-frame MOT boxes by spatial assignment followed by a dominant one-to-one identity check; all 37,134 instances were retained. Numeric Outdoor annotation suffixes are **not** assumed to be MOT track IDs. The validated mapping is `Green_6→3`, `Green_9→6`, `Green_11→8`, `Purple_4→4`, `Purple_10→1`, and `Purple_12→5`. Indoor maps `Red→1`, `Blue→2`, `Pink→3`, `Black→4`, `Orange→5`, and `Green→6`, but is still verified spatially.

Drone pose JSON does not provide a reliable video link. Its stored tight-pose box is labeled `pose_tight_fallback`, used only as the NMJE/PCK scale denominator, and never presented as an official independent bbox.

## Annotation-only acquisition and storage policy

The downloader performs a shallow, blob-filtered sparse clone of only `ground_truth`, `court_images`, and `script`. It uses `gh` when available and an equivalent safe `git` fallback otherwise. It never follows the upstream Google Drive link and never downloads the video bundle, tracker outputs, pretrained weights, or bulk RGB frames.

The sparse checkout was 169.83 MiB immediately after acquisition; the current manifest separately accounts for 139.07 MB of `ground_truth` annotations and 38.25 MB of Git metadata. Generated artifacts are guarded by a preferred 500 MB target, a 600 MB hard project/data limit, a per-artifact preflight, and a minimum free-space reserve. The current `outputs/figures` category is 63.01 MB, above its preferred 30 MB subtarget, because old and repeated generated figures have been preserved rather than silently deleted; this soft category overage does not breach the hard project/data limit. Use [storage_manifest.csv](outputs/reports/storage_manifest.csv), rather than an estimate in this README, for the current footprint.

Upstream original code is Apache-2.0, while the TrackID3x3 dataset is CC BY 4.0 unless otherwise stated; third-party upstream directories retain their own licenses. This project uses the annotations and does not redistribute the optional media bundle.

## Deterministic subsets and leakage controls

Static selection uses seed 42, frame stride 3, and exact targets of 1,000 Indoor, 1,000 Outdoor, and 500 Drone pose instances. Splitting is performed independently inside each domain before selection.

| Domain | Split group | Groups train/val/test | Instances train/val/test | Selected | Unique frames |
|---|---|---:|---:|---:|---:|
| Indoor | video | 9/2/2 | 693/154/153 | 1,000 | 604 |
| Outdoor | player/track | 4/1/1 | 666/167/167 | 1,000 | 719 |
| Drone | player/track | 4/1/1 | 334/83/83 | 500 | 164 |

All pairwise split-group overlaps are zero. Outdoor uses player grouping because only one pose video is available; Drone uses player grouping because reliable video identity is absent. See [split_audit.md](outputs/reports/split_audit.md).

For `T=16`, windows are formed from full assigned source streams **before** random thinning. Frames advance by source-frame step 3; windows use a 16-observation hop, do not overlap inside a track, and never cross a video, player, split, or discontinuity.

| Domain | Available windows | Selected windows | Pose observations | Train/val/test windows |
|---|---:|---:|---:|---:|
| Indoor | 300 | 62 | 992 | 43/10/9 |
| Outdoor | 447 | 62 | 992 | 41/10/11 |
| Drone | 0 | 0 | 0 | excluded |

Drone is deliberately excluded from temporal experiments because continuity cannot be established from its pose JSON. The resulting temporal artifact contains 124 windows and 1,984 observations.

### Mask and transform order

The anti-leakage order is enforced at runtime:

1. Start from source-pixel coordinates and the original-valid mask.
2. Generate a deterministic synthetic target mask only over originally valid joints.
3. Remove those target coordinates **before** computing any representation.
4. Estimate centers, scales, fallback boxes, and angle features from observed joints only.
5. Compute loss only on the synthetic target mask.
6. Invert predictions to source pixels, then compute NMJE/PCK using the separately stored source bbox diagonal.

Official Indoor/Outdoor MOT boxes are independent of the pose and may normalize inputs. A Drone input transform is recomputed from post-mask observed joints; its full-pose fallback box remains metric-only. This distinction prevents a hidden target from leaking through the normalization transform.

## Representations, models, and metrics

| CLI value | Input representation |
|---|---|
| `raw` | source-pixel coordinates |
| `bbox_normalized` | independent MOT normalization, or observed-only Drone fallback transform |
| `root_centered` | observed-only anatomical/fallback center |
| `scale_normalized` | observed-only center and torso/robust scale |
| `normalized_angles` | scale-normalized coordinates plus five supported observed-joint angles |

The learned models are a compact MLP, a graph network constrained below one million parameters, and a lightweight TCN restricted to reliable temporal windows. Fit-free `geometric`, `temporal`, and `hybrid` baselines are available through `evaluate.py`; temporal interpolation is meaningful only for `T>1`.

The primary metric is normalized mean joint error (NMJE): source-pixel Euclidean error divided by the source bbox diagonal, averaged over synthetically masked valid joints. Lower is better. Supporting metrics are PCK@0.05, PCK@0.10, PCK@0.20, and per-joint NMJE. No metric scores naturally absent joints.

## Installation

Python 3.11 or newer is required.

```bash
cd basketball-pose-research
python -m pip install -r requirements.txt
```

The current runs used Python 3.13.5, PyTorch 2.11.0+cu128, and an NVIDIA RTX 3070 Ti Laptop GPU. CUDA is optional; replace `--device cuda` with `--device cpu` on a CPU-only machine.

## Reproduce the data artifacts

All commands below run from the project root and make every non-default scientific choice explicit.

### 1. Acquire and inspect annotations

```bash
python scripts/download_dataset.py --dry-run
python scripts/download_dataset.py --destination external/TrackID3x3
python scripts/inspect_dataset.py --input external/TrackID3x3 --output outputs/reports/trackid3x3_schema.md --minimum-association-dominance 0.80
```

An already validated official clone is reused without downloading again.

### 2. Build the static `T=1` subset

```bash
python scripts/build_subset.py --input external/TrackID3x3 --output data/processed/trackid3x3/subset_manifest.csv --metadata-output data/processed/trackid3x3/subset_metadata.json --audit-output outputs/reports/split_audit.md --seed 42 --stride 3 --indoor-target 1000 --outdoor-target 1000 --drone-target 500 --minimum-association-dominance 0.80
python scripts/data_budget.py --samples 2500 --sequence-length 1 --output-dir data/processed/trackid3x3 --max-artifact-mb 600 --max-dataset-mb 600 --minimum-free-mb 1024 --json-output outputs/reports/data_budget_t1.json
python scripts/preprocess.py --input external/TrackID3x3 --manifest data/processed/trackid3x3/subset_manifest.csv --output-dir data/processed/trackid3x3 --sequence-length 1 --frame-step 3 --mask-rate 0.30 --seed 42 --indoor-observation-cap 1000 --outdoor-observation-cap 1000 --drone-observation-cap 500 --max-artifact-mb 600 --max-dataset-mb 600 --minimum-free-mb 1024 --minimum-association-dominance 0.80
```

The canonical output is `data/processed/trackid3x3/dataset.npz`; its array contract is documented in [PROCESSED_FORMAT.md](docs/PROCESSED_FORMAT.md).

### 3. Build the reliable `T=16` supplement

```bash
python scripts/data_budget.py --samples 124 --sequence-length 16 --output-dir data/processed/trackid3x3_t16 --max-artifact-mb 600 --max-dataset-mb 600 --minimum-free-mb 1024 --json-output outputs/reports/data_budget_t16.json
python scripts/preprocess.py --input external/TrackID3x3 --output-dir data/processed/trackid3x3_t16 --sequence-length 16 --frame-step 3 --mask-rate 0.20 --seed 42 --indoor-observation-cap 1000 --outdoor-observation-cap 1000 --drone-observation-cap 0 --max-artifact-mb 600 --max-dataset-mb 600 --minimum-free-mb 1024 --minimum-association-dominance 0.80
```

## Reproduce training and evaluation

### Static core

These are the executed primary model commands at 20% masking:

```bash
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation raw --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_raw_all_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation bbox_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_bbox_all_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation root_centered --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_root_all_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation scale_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_scale_all_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation normalized_angles --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_angles_all_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model graph --representation bbox_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --device cuda --experiment-name core_graph_bbox_all_mask20_seed42
```

Every run writes one `best.pt`, its resolved `config.yaml`, environment, history, and metrics.

Evaluate the checkpoint and fit-free baseline at all required missingness levels:

```bash
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_bbox_all_mask20_seed42/best.pt --mask-ratios 0.10 0.20 0.30 0.40 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_mlp_bbox_all_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --baseline geometric --representation bbox_normalized --mask-ratios 0.10 0.20 0.30 0.40 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_geometric_all
```

These executed 20% evaluations complete the representation and model comparisons:

```bash
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_raw_all_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_mlp_raw_all_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_root_all_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_mlp_root_all_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_scale_all_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_mlp_scale_all_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_angles_all_mask20_seed42/best.pt --mask-ratios 0.10 0.20 0.30 0.40 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_mlp_angles_all_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_graph_bbox_all_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/core --run-name core_graph_bbox_all_mask20_seed42_eval
```

### Temporal supplement

These commands reproduce all four `T=16` rows (fit-free geometric baseline plus trained MLP, graph, and TCN) in the canonical model table:

```bash
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3_t16/dataset.npz --model mlp --representation bbox_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 32 --num-workers 0 --seed 42 --device cuda --experiment-name core_mlp_bbox_all_t16_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3_t16/dataset.npz --model graph --representation bbox_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 32 --num-workers 0 --seed 42 --device cuda --experiment-name core_graph_bbox_all_t16_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3_t16/dataset.npz --model tcn --representation bbox_normalized --mask-ratio 0.20 --domain all --train-fraction 1.0 --epochs 30 --batch-size 32 --num-workers 0 --seed 42 --device cuda --experiment-name core_tcn_bbox_all_t16_mask20_seed42
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --baseline geometric --representation bbox_normalized --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 64 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/supplemental --run-name core_geometric_all_t16
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --checkpoint outputs/checkpoints/core_mlp_bbox_all_t16_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 64 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/supplemental --run-name core_mlp_bbox_all_t16_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --checkpoint outputs/checkpoints/core_graph_bbox_all_t16_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 64 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/supplemental --run-name core_graph_bbox_all_t16_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --checkpoint outputs/checkpoints/core_tcn_bbox_all_t16_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 64 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/supplemental --run-name core_tcn_bbox_all_t16_mask20_seed42_eval
```

### Data efficiency

The 25% and 50% training subsets select whole split groups deterministically; the 100% point is `core_mlp_angles_all_mask20_seed42` above. Validation and test partitions stay unchanged.

```bash
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation normalized_angles --mask-ratio 0.20 --domain all --train-fraction 0.25 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name dataeff_mlp_angles_all_data25_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation normalized_angles --mask-ratio 0.20 --domain all --train-fraction 0.50 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name dataeff_mlp_angles_all_data50_mask20_seed42
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/dataeff_mlp_angles_all_data25_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/data_efficiency --run-name dataeff_mlp_angles_all_data25_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/dataeff_mlp_angles_all_data50_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/data_efficiency --run-name dataeff_mlp_angles_all_data50_mask20_seed42_eval
```

### Within-domain and required cross-domain transfer

Train one static bbox-normalized MLP per domain:

```bash
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation bbox_normalized --mask-ratio 0.20 --domain Indoor --train-fraction 1.0 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name domain_mlp_bbox_indoor_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation bbox_normalized --mask-ratio 0.20 --domain Outdoor --train-fraction 1.0 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name domain_mlp_bbox_outdoor_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation bbox_normalized --mask-ratio 0.20 --domain Drone --train-fraction 1.0 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name domain_mlp_bbox_drone_mask20_seed42
```

Evaluate each checkpoint only on its matching held-out domain:

```bash
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/domain_mlp_bbox_indoor_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain Indoor --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/domain --run-name domain_mlp_bbox_indoor_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/domain_mlp_bbox_outdoor_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain Outdoor --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/domain --run-name domain_mlp_bbox_outdoor_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/domain_mlp_bbox_drone_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain Drone --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/domain --run-name domain_mlp_bbox_drone_mask20_seed42_eval
```

The first two commands below execute the four required transfers: Indoor→Outdoor, Indoor→Drone, Outdoor→Indoor, and Outdoor→Drone. The third reproduces the two supplementary Drone-source cells included in the canonical six-cell matrix. Separate output roots prevent one source-domain evidence index from overwriting another.

```bash
python scripts/run_cross_domain.py --processed data/processed/trackid3x3/dataset.npz --checkpoints-root outputs/checkpoints/domain_mlp_bbox_indoor_mask20_seed42 --domains Outdoor Drone --models mlp --representations bbox_normalized --mask-ratios 0.20 --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/cross_domain/indoor_source --figures-dir outputs/figures/domain/indoor_source --tables-dir outputs/tables/cross_domain/indoor_source --no-save-predictions
python scripts/run_cross_domain.py --processed data/processed/trackid3x3/dataset.npz --checkpoints-root outputs/checkpoints/domain_mlp_bbox_outdoor_mask20_seed42 --domains Indoor Drone --models mlp --representations bbox_normalized --mask-ratios 0.20 --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/cross_domain/outdoor_source --figures-dir outputs/figures/domain/outdoor_source --tables-dir outputs/tables/cross_domain/outdoor_source --no-save-predictions
python scripts/run_cross_domain.py --processed data/processed/trackid3x3/dataset.npz --checkpoints-root outputs/checkpoints/domain_mlp_bbox_drone_mask20_seed42 --domains Indoor Outdoor --models mlp --representations bbox_normalized --mask-ratios 0.20 --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/cross_domain/drone_source --figures-dir outputs/figures/domain/drone_source --tables-dir outputs/tables/cross_domain/drone_source --no-save-predictions
```

### Indoor-to-Outdoor normalization shift

This five-command slice trains the two additional Indoor MLPs, evaluates them on Outdoor, and reuses the existing Indoor bbox-normalized checkpoint/evaluation bundle for the third row. The bbox command intentionally evaluates `all` because that executed bundle also supplies its Indoor within-domain row; `normalization_shift.csv` selects only its Outdoor metric.

```bash
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation raw --mask-ratio 0.20 --domain Indoor --train-fraction 1.0 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name shift_mlp_raw_indoor_mask20_seed42
python scripts/train.py --config config/trackid3x3.yaml --processed data/processed/trackid3x3/dataset.npz --model mlp --representation scale_normalized --mask-ratio 0.20 --domain Indoor --train-fraction 1.0 --epochs 30 --batch-size 64 --num-workers 0 --seed 42 --device cuda --experiment-name shift_mlp_scale_indoor_mask20_seed42
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/shift_mlp_raw_indoor_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain Outdoor --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/cross_normalization --run-name shift_mlp_raw_indoor_to_outdoor_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/domain_mlp_bbox_indoor_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain all --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/domain --run-name domain_mlp_bbox_indoor_mask20_seed42_eval
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/shift_mlp_scale_indoor_mask20_seed42/best.pt --mask-ratios 0.20 --primary-mask-ratio 0.20 --split test --test-domain Outdoor --seed 42 --batch-size 256 --device cuda --output-dir outputs/logs/evaluations --figures-dir outputs/figures/cross_normalization --run-name shift_mlp_scale_indoor_to_outdoor_mask20_seed42_eval
```

### Aggregate, report, and audit

Aggregation is evidence-only: it discovers executed leakage-audited reconstruction metrics and never inserts placeholders or zeros for missing experiments.

```bash
python scripts/run_ablation.py --logs-root outputs/logs --data data/processed/trackid3x3/dataset.npz --tables-dir outputs/tables/reconstruction --figures-dir outputs/figures/reconstruction
python scripts/generate_report.py --project-root . --output outputs/reports/final_report.md
python scripts/generate_project_guide_pdf.py --source docs/PANDUAN_PROYEK.md --output output/pdf/panduan_proyek_basketball_pose.pdf
python scripts/report_storage.py --project-root . --output outputs/reports/storage_manifest.csv --json-output outputs/reports/storage_manifest.json
python scripts/cleanup.py --project-root .
python scripts/quality_gate.py --require-real-results
```

`cleanup.py` is a dry run unless `--confirm` is supplied. Pre-pivot software fixtures and reusable legacy code are isolated from the TrackID3x3 public workflow and excluded from empirical aggregation; they are not silently deleted.

## Executed exploratory results

The values below come from real TrackID3x3 annotations, not smoke fixtures. They are **single-run, seed-42 exploratory results** (`Runs=1` per cell), so they are not confidence intervals and should not be treated as established model rankings. Lower NMJE is better; higher PCK@0.10 is better.

The primary controlled static MLP summary below is **joint-weighted pooled (micro)**: target-joint errors from all three held-out domain partitions are combined, so domains contribute in proportion to their scored joints. At 20% masking this pool contains 806 actually scored target joints.

| Representation | NMJE | PCK@0.10 |
|---|---:|---:|
| `raw` | 8.0215 | 0.0000 |
| `bbox_normalized` | 0.0733 | 0.7779 |
| `root_centered` | 0.1370 | 0.3623 |
| `scale_normalized` | 0.0673 | 0.8065 |
| `normalized_angles` | **0.0672** | **0.8052** |

Under that primary joint-weighted pooling, `normalized_angles` has the lowest executed NMJE so far (0.06720597 over 806 target joints; PCK@0.10 0.80521092), but its near-tie with `scale_normalized` and single-seed scope preclude a universal-superiority claim.

The canonical [representation.csv](outputs/tables/reconstruction/representation.csv) uses a different, **equal-domain macro** aggregation: it computes one score per held-out domain and then gives Indoor, Outdoor, and Drone equal weight. Its `normalized_angles` cell is therefore NMJE 0.06818125 (PCK@0.10 0.79743851), not 0.06720597. Both numbers are correct for their stated aggregation; neither is a multi-seed estimate.

For the analogous pooled static bbox-normalized model comparison, the geometric baseline scored 0.1558 NMJE / 0.4243 PCK@0.10, the 82,226-parameter graph model scored 0.0818 / 0.7345, and the 26,996-parameter MLP scored 0.0733 / 0.7779.

Pooled missingness results for the executed `normalized_angles` MLP were:

| Masked valid joints | Scored joints | NMJE | PCK@0.10 |
|---:|---:|---:|---:|
| 10% | 403 | 0.0689 | 0.8040 |
| 20% | 806 | 0.0672 | 0.8052 |
| 30% | 1,209 | 0.0765 | 0.7519 |
| 40% | 1,612 | 0.0948 | 0.6538 |

Static within-domain bbox-normalized MLP results at 20% masking were:

| Train/test domain | NMJE | PCK@0.10 |
|---|---:|---:|
| Indoor→Indoor | 0.0761 | 0.7451 |
| Outdoor→Outdoor | 0.0677 | 0.8114 |
| Drone→Drone | 0.0962 | 0.6566 |

The four required cross-domain cells were:

| Train→test | NMJE | PCK@0.10 |
|---|---:|---:|
| Indoor→Outdoor | 0.0688 | 0.8114 |
| Indoor→Drone | 0.1399 | 0.4096 |
| Outdoor→Indoor | 0.0799 | 0.7320 |
| Outdoor→Drone | 0.1347 | 0.4518 |

The `T=16` bbox-normalized TCN scored 0.0751 / 0.7674 on Indoor and 0.0723 / 0.7699 on Outdoor at 20% masking. These temporal numbers use only 20 held-out windows in total and are supplemental, not evidence of a general temporal advantage.

Canonical machine-readable and rendered results live in [outputs/tables/reconstruction](outputs/tables/reconstruction) and [final_report.md](outputs/reports/final_report.md). The report also audits all expected figures. RGB overlay is explicitly **not generated** because no source RGB frames were acquired.

## Repository layout

```text
config/                         TrackID3x3 protocol and joint mapping
data/processed/trackid3x3/      compact static NPZ and manifests
data/processed/trackid3x3_t16/  reliable temporal windows
docs/                           scientific protocol and array contract
external/TrackID3x3/            ignored official sparse checkout
scripts/                        acquisition, preprocessing, training, evaluation, reports
src/data/                       strict JSON/MOT adapter, grouped splitting, storage guards
src/features/                   observed-only reversible representations
src/models/                     compact reconstruction models and baselines
src/training/                   deterministic masking, losses, trainer, pixel-space metrics
src/evaluation/                 reconstruction metrics and error analysis
src/visualization/              evidence-backed figures
outputs/                        checkpoints, logs, figures, tables, and reports
tests/                          schema, leakage, model, evaluation, and safety tests
```

## Tests and quality gates

```bash
python -m pytest -q
python -m compileall -q src scripts
python scripts/quality_gate.py --require-real-results
```

The gate verifies the official upstream checkout, exact 10-joint mapping, group-disjoint manifest, absence of forbidden Henu archives/dependencies, sub-600 MiB experiment/data footprint, and at least one executed non-synthetic reconstruction metric.

## Limitations

- Results use a deterministic subset and one seed; repeated seeds are required before uncertainty or rankings are interpreted.
- Outdoor and Drone have only six split groups, limiting independent validation/test groups.
- Drone uses a pose-tight metric denominator rather than an independent MOT box, so direct domain comparisons require that provenance caveat.
- Only 124 reliable `T=16` windows are available; Drone has no admissible temporal windows.
- The local acquisition is annotation-only. It cannot support RGB overlays, appearance-conditioned conclusions, ball analysis, 3D pose claims, action recognition, or biomechanical evaluation.

## Citation

If you use the data, cite the [TrackID3x3 paper](https://arxiv.org/abs/2503.18282) and follow the upstream dataset terms. Full authorship and DOI metadata are in [CITATION.cff](CITATION.cff).
