# Cross-Domain Robustness of Lightweight Skeleton Representations for 3x3 Basketball Player Pose Analysis

## Scope and evidence status

This report covers deterministic masked reconstruction of the documented TrackID3x3 10-joint skeleton. It does not infer missing source annotations, create unsupported joints, or assess basketball technique.

Included 77 leakage-audited non-synthetic, non-smoke domain/mask evaluation artifact(s).

All empirical comparisons reported below use seed 42 only and are single-seed exploratory evidence, not confidence intervals or confirmatory estimates. Drone rows are additionally exploratory because their metric denominator uses documented pose-tight fallback provenance rather than an independent MOT box.

Detected 0 synthetic reconstruction artifact(s); these are software validation only / NOT research results and are excluded from tables.

Detected 13 real-data software-smoke artifact(s); these are NOT primary research results and are excluded by default.

The source/task migration is documented in [outputs/reports/dataset_migration.md](dataset_migration.md). 18 task-incompatible legacy metric artifact(s) were detected and excluded.

0 pre-audit Drone evaluation artifact(s) were excluded because pose-tight full-pose bboxes had not yet been proven metric-only. Drone inputs now require an observed-only post-mask transform; the stored pose-tight bbox is retained only as the NMJE/PCK denominator.

## Dataset

TrackID3x3 annotations are locally available and their 10-joint schema, domain counts, bbox provenance, and grouping rules were validated in [outputs/reports/trackid3x3_schema.md](trackid3x3_schema.md).

Processed artifact: `data/processed/trackid3x3/dataset.npz` with 2500 sequences, T=1, 10 joints, domains Drone, Indoor, Outdoor.

The three capture domains are Indoor fixed camera, Outdoor fixed camera, and outdoor Drone. Bboxes are independent MOT `xywh` boxes for Indoor/Outdoor; Drone uses the explicitly recorded tight-pose fallback. The local acquisition is annotation-only. No RGB frame, video bundle, or model weight was downloaded because remote sizes were not established.

### Dataset/subset statistics

| Domain | Videos | Frames | Player instances | Player tracks | Pose instances | Storage |
| --- | --- | --- | --- | --- | --- | --- |
| Indoor | 13 | 604 | 1000 | 78 | 1000 | 26.88 MiB |
| Outdoor | 1 | 719 | 1000 | 6 | 1000 | 84.62 MiB |
| Drone | unavailable / not linkable | 164 | 500 | 6 | 500 | 21.13 MiB |

Storage is the complete on-disk `ground_truth/<Domain>` annotation tree, not only files contributing selected poses. An instance means one player pose at one annotated frame, so `Player instances` and valid `Pose instances` coincide in this one-pose-per-player-frame artifact. `Player tracks` is the additional count of unique scoped video-track identities (or the documented track fallback). Drone video identity is unavailable/not linkable. Drone MOT files are included in that storage footprint but are not linked to, or used for, the sampled Drone pose records.

## Method

For each valid pose, a deterministic seed-42 mask holds out 10%, 20%, 30%, or 40% of originally annotated joints. The input mask and source-valid mask remain separate. Training and scoring use only the synthetically held-out subset of originally valid joints. Video is the preferred split group; track is the documented fallback where reliable video identity is unavailable. Any overlap across train, validation, and test groups invalidates an artifact.

The primary metric is normalized mean joint error (NMJE): source-pixel Euclidean error divided by the recorded source bbox diagonal. Lower is better. Indoor/Outdoor denominators are independent MOT boxes; Drone's explicitly labeled full-pose tight box is metric-only and never normalizes a masked model input. PCK@0.05, PCK@0.10, PCK@0.20, and per-joint NMJE are supporting measures. Predictions from every normalized representation are inverse-transformed to source pixels before metrics; already-normalized distances are never divided by a pixel bbox scale. The `PCK` column in the cross-domain table denotes PCK@0.10.

### Representation definitions

| Representation | Definition | Inverse for metrics |
| --- | --- | --- |
| raw | Source pixel x/y coordinates; missing inputs are zero-filled after masking. | Identity; metrics are computed in source pixels. |
| bbox_normalized | Per-frame (x-x0)/w and (y-y0)/h; official MOT boxes are used directly, while pose-tight fallback frames use a post-mask observed-joint box. | Multiply by the exact leakage-safe input-transform width/height and add its origin; use the separately stored source bbox only for metric normalization. |
| root_centered | Coordinates centred using an observed-only anatomical/bbox fallback centre. | Add the stored observed-only centre. |
| scale_normalized | Observed-only centred coordinates divided by an observed-only scale. | Multiply by the stored scale and add the stored centre. |
| normalized_angles | Scale-normalized coordinates with supported observed-joint angles as auxiliary context. | Invert only predicted x/y coordinates; angles are context, not targets. |

Compared model families, when actually executed, are a fit-free geometric baseline, compact MLP, reliable-window TCN, and a graph model constrained below one million parameters. Static representation, missingness, and cross-domain tests use T=1. The controlled model-family table uses T=16 because all four reported families were executed on the same reliable Indoor/Outdoor track windows; TCN entries require an explicit reliable-window audit, and temporal windows are valid only within continuous same-video/same-player tracks.

## Results

Included 77 leakage-audited non-synthetic, non-smoke domain/mask evaluation artifact(s).

Within the declared Table 2 comparison (pooled-domain MLP, T=1, 20% masking, 100% data, seed 42; equal-domain macro mean over Indoor, Outdoor, and Drone), the lowest NMJE was 0.0682 and PCK@0.10 0.7974 for normalized_angles. This is single-seed exploratory evidence, not a universal-superiority claim.

### Representation table

Policy: Pooled-domain MLP comparison of all five declared representations at T=1, 20% masking, 100% data, and seed 42; macro-average the same Indoor, Outdoor, and Drone rows. Aggregation: equal-domain macro mean after within-domain run averaging. Selected controls: data_percent=100.0, mask_ratio=0.2, model=mlp, seed=42, sequence_length=1, smoke=False, synthetic=False, train_domain=all. Test domains: [Drone, Indoor, Outdoor].

| Representation | NMJE | PCK@.10 | Domains averaged | Result scope |
| --- | --- | --- | --- | --- |
| bbox_normalized | 0.0786 | 0.7507 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| normalized_angles | 0.0682 | 0.7974 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| raw | 8.5826 | 0.0000 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| root_centered | 0.1400 | 0.3561 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| scale_normalized | 0.0687 | 0.7974 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |

Table 2 macro-averages the three per-domain metrics, giving Indoor, Outdoor, and Drone equal weight regardless of sample count. For the same normalized+angles slice, the executed pooled joint-level summary is NMJE 0.0672 and PCK@0.10 0.8052 over 806 scored joints; this micro-style estimate weights domains in proportion to scored joints. The two estimands answer different questions and are not interchangeable.

### Normalization shift hypothesis

Policy: Single-seed Indoor-trained to Outdoor-tested MLP normalization hypothesis; raw, bbox-normalized, and scale-normalized representations at T=1, 20% masking, 100% training data, and seed 42. Aggregation: equal-domain macro mean after within-domain run averaging. Selected controls: data_percent=100.0, mask_ratio=0.2, model=mlp, seed=42, sequence_length=1, smoke=False, synthetic=False, train_domain=Indoor. Test domains: [Outdoor].

This is a single-seed normalization hypothesis test. The raw-coordinate gap also reflects optimization difficulty and coordinate scale; it is not causal proof that normalization alone resolves camera shift.

| Representation | NMJE | PCK@.10 | Domains averaged | Result scope |
| --- | --- | --- | --- | --- |
| bbox_normalized | 0.0688 | 0.8114 | 1 | single-seed exploratory research result |
| raw | 8.1361 | 0.0000 | 1 | single-seed exploratory research result |
| scale_normalized | 0.0683 | 0.8323 | 1 | single-seed exploratory research result |

### Model/compute table

Policy: T=16 bbox-normalized comparison of the geometric baseline, MLP, reliable-window TCN, and sub-1M-parameter graph model at 20% masking, 100% data, and seed 42; macro-average the matched Indoor and Outdoor domains. Aggregation: equal-domain macro mean after within-domain run averaging. Selected controls: data_percent=100.0, mask_ratio=0.2, representation=bbox_normalized, seed=42, sequence_length=16, smoke=False, synthetic=False, train_domain=[all, not_applicable]. Test domains: [Indoor, Outdoor].

`train_domain=not_applicable` denotes the fit-free geometric baseline; its outputs require no fitted training domain.

| Model | Parameters | Checkpoint MB | NMJE | PCK@.10 | Domains averaged | Result scope |
| --- | --- | --- | --- | --- | --- | --- |
| geometric_baseline | 0.0000 | 0.0000 | 0.1562 | 0.4159 | 2 | single-seed exploratory research result |
| graph | 82226.0000 | 0.3447 | 0.0802 | 0.7251 | 2 | single-seed exploratory research result |
| mlp | 26996.0000 | 0.1174 | 0.0766 | 0.7555 | 2 | single-seed exploratory research result |
| tcn | 117572.0000 | 0.4872 | 0.0737 | 0.7686 | 2 | single-seed exploratory research result |

### Missing-keypoint robustness

Policy: Pooled-domain normalized+angles MLP at T=1, 100% data, and seed 42; vary only 10%, 20%, 30%, and 40% masking and macro-average all three test domains. Aggregation: equal-domain macro mean after within-domain run averaging. Selected controls: data_percent=100.0, model=mlp, representation=normalized_angles, seed=42, sequence_length=1, smoke=False, synthetic=False, train_domain=all. Test domains: [Drone, Indoor, Outdoor].

| Missing % | NMJE | PCK@.10 | Domains averaged | Result scope |
| --- | --- | --- | --- | --- |
| 10.0000 | 0.0703 | 0.7986 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| 20.0000 | 0.0682 | 0.7974 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| 30.0000 | 0.0788 | 0.7373 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| 40.0000 | 0.0980 | 0.6367 | 3 | single-seed exploratory; Drone uses distinct metric-bbox provenance |

### Within-domain metrics

| Domain | NMJE | PCK@.10 | Runs | Result scope |
| --- | --- | --- | --- | --- |
| Drone | 0.0962 | 0.6566 | 1 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| Indoor | 0.0761 | 0.7451 | 1 | single-seed exploratory research result |
| Outdoor | 0.0677 | 0.8114 | 1 | single-seed exploratory research result |

### Cross-domain metrics

| Train domain | Test domain | NMJE | PCK | Runs | Result scope |
| --- | --- | --- | --- | --- | --- |
| Drone | Indoor | 0.1729 | 0.3562 | 1 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| Drone | Outdoor | 0.1492 | 0.4311 | 1 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| Indoor | Drone | 0.1399 | 0.4096 | 1 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| Indoor | Outdoor | 0.0688 | 0.8114 | 1 | single-seed exploratory research result |
| Outdoor | Drone | 0.1347 | 0.4518 | 1 | single-seed exploratory; Drone uses distinct metric-bbox provenance |
| Outdoor | Indoor | 0.0799 | 0.7320 | 1 | single-seed exploratory research result |

Only controlled slices are aggregated: representation comparisons fix model/domain/mask/data fraction/sequence length; model comparisons fix representation/domain/mask/data fraction/sequence length; missingness comparisons fix model/representation/domain/data fraction/sequence length; data-efficiency curves require all of 25%, 50%, and 100% at a fixed sequence length. Missing configurations remain blank rather than being imputed.

## Error analysis

Sequence errors below, when present, come only from the declared fixed primary slice. Naturally absent joints are excluded.

**Fixed primary slice:** pooled-domain normalized+angles MLP, T=1, 20% experimental masking, 100% training data, seed 42; Indoor, Outdoor, and Drone.

These are directly observable descriptive strata only; differences are not causal effects and naturally absent joints are never scored.

Same-frame bbox overlap was evaluated directly for Indoor. It was unavailable for Drone, Outdoor.

#### Bbox-scale strata

| Player scale bin | Sequences | Mean NMJE |
| --- | --- | --- |
| small | 100 | 0.0844 |
| medium | 202 | 0.0653 |
| large | 101 | 0.0539 |

#### Distal-target strata

| Masked distal target | Sequences | Mean NMJE |
| --- | --- | --- |
| yes | 276 | 0.0766 |
| no | 127 | 0.0468 |

#### Pose-spread strata

| Pose-spread quartile | Sequences | Mean NMJE |
| --- | --- | --- |
| Q1 | 101 | 0.0594 |
| Q2 | 101 | 0.0658 |
| Q3 | 100 | 0.0620 |
| Q4 | 101 | 0.0816 |

#### Same-frame overlap strata

| Overlap stratum | Sequences | Mean NMJE |
| --- | --- | --- |
| low (<=0.05) | 91 | 0.0735 |
| moderate (0.05-0.20] | 6 | 0.0563 |
| high (>0.20) | 22 | 0.1049 |

#### Highest-error sequences within the fixed slice

| sample_id | evaluation_model | representation | train_domain | sequence_length | domain | video_id | track_id | mask_ratio | n_scored_joints | mean_bbox_diagonal | player_scale_bin | pose_spread_bbox_fraction | masked_distal_joints | distal_nmje | non_distal_nmje | same_frame_max_bbox_iou | nmje | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Indoor:basket_S6T7_post:frame_132:Orange_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T7_post | Orange_Jersey | 0.2000 | 2 | 122.8856 | medium | 0.2492 | 2 | 0.4496 |  | 0.2783 | 0.4496 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T7_post:frame_114:Orange_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T7_post | Orange_Jersey | 0.2000 | 2 | 77.1030 | small | 0.2656 | 1 | 0.4005 | 0.2442 | 0.0000 | 0.3223 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Outdoor:IMG_0104:frame_1287:Purple_4 | mlp | normalized_angles | all | 1 | Outdoor | IMG_0104 | Purple_4 | 0.2000 | 2 | 243.0692 | medium | 0.1994 | 2 | 0.2324 |  |  | 0.2324 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Drone:video_unavailable:frame_1094:Black_1 | mlp | normalized_angles | all | 1 | Drone |  | Black_1 | 0.2000 | 2 | 151.8356 | medium | 0.2489 | 2 | 0.2258 |  |  | 0.2258 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T7_post:frame_27:Orange_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T7_post | Orange_Jersey | 0.2000 | 2 | 84.3670 | small | 0.2102 | 1 | 0.3059 | 0.1432 | 0.0000 | 0.2246 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T3_post:frame_42:Red_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T3_post | Red_Jersey | 0.2000 | 2 | 102.8033 | small | 0.2030 | 1 | 0.3651 | 0.0740 | 0.0000 | 0.2195 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T7_post:frame_183:Blue_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T7_post | Blue_Jersey | 0.2000 | 2 | 112.5714 | small | 0.2103 | 1 | 0.3269 | 0.1104 | 0.0000 | 0.2186 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T3_post:frame_0:Orange_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T3_post | Orange_Jersey | 0.2000 | 2 | 84.1295 | small | 0.2208 | 1 | 0.2620 | 0.1579 | 0.0000 | 0.2100 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T7_post:frame_69:Blue_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T7_post | Blue_Jersey | 0.2000 | 2 | 93.5653 | small | 0.1871 | 2 | 0.1966 |  |  | 0.1966 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |
| Indoor:basket_S6T3_post:frame_81:Blue_Jersey | mlp | normalized_angles | all | 1 | Indoor | basket_S6T3_post | Blue_Jersey | 0.2000 | 2 | 99.7101 | small | 0.2023 | 1 | 0.1805 | 0.2112 | 0.2677 | 0.1958 | outputs/logs/evaluations/core_mlp_angles_all_mask20_seed42_eval/mask_20/sequence_errors.csv |

## Figures and tables audit

| Figure | Status | Files |
| --- | --- | --- |
| 1. Example 10-joint skeleton | generated from available evidence | outputs/figures/reconstruction/example_10_joint_skeleton.png (+30 other executed variant(s)) |
| 2. Normalized skeleton | generated from available evidence | outputs/figures/reconstruction/normalized_skeleton.png (+30 other executed variant(s)) |
| 3. Masked skeleton | generated from available evidence | outputs/figures/domain_mlp_bbox_drone_mask20_seed42__train_drone__test_drone/masked_skeleton.png (+29 other executed variant(s)) |
| 4. Reconstructed skeleton | generated from available evidence | outputs/figures/domain_mlp_bbox_drone_mask20_seed42__train_drone__test_drone/reconstructed_skeleton.png (+29 other executed variant(s)) |
| 5. Ground truth vs reconstruction | generated from available evidence | outputs/figures/domain_mlp_bbox_drone_mask20_seed42__train_drone__test_drone/ground_truth_vs_reconstruction.png (+29 other executed variant(s)) |
| 6. Per-joint reconstruction error | generated from available evidence | outputs/figures/domain_mlp_bbox_drone_mask20_seed42__train_drone__test_drone/per_joint_reconstruction_error.png (+29 other executed variant(s)) |
| 7. Error vs missing percentage | generated from available evidence | outputs/figures/reconstruction/error_vs_missing_percentage.png (+14 other executed variant(s)) |
| 8. Indoor vs Outdoor vs Drone | generated from available evidence | outputs/figures/reconstruction/indoor_outdoor_drone_performance.png (+19 other executed variant(s)) |
| 9. Cross-domain matrix | generated from available evidence | outputs/figures/reconstruction/cross_domain_matrix.png (+1 other executed variant(s)) |
| 10. Normalization ablation | generated from available evidence | outputs/figures/reconstruction/normalization_ablation.png (+1 other executed variant(s)) |
| 11. Data-efficiency curve | generated from available evidence | outputs/figures/reconstruction/data_efficiency_curve.png (+1 other executed variant(s)) |
| 12. RGB overlay | NOT GENERATED: source RGB frames are unavailable locally; remote media size was not assumed |  |
| Dataset domain summary | generated from available evidence | outputs/figures/reconstruction/dataset_domain_summary.png (+30 other executed variant(s)) |
| Pose-instance counts | generated from available evidence | outputs/figures/reconstruction/pose_instance_counts.png (+30 other executed variant(s)) |
| Natural missingness | generated from available evidence | outputs/figures/reconstruction/missingness_by_domain_joint.png (+30 other executed variant(s)) |
| Raw vs normalized coordinates | generated from available evidence | outputs/figures/reconstruction/coordinate_distributions_raw_vs_normalized.png (+30 other executed variant(s)) |
| Domain skeleton examples | generated from available evidence | outputs/figures/reconstruction/skeleton_examples_by_domain.png (+30 other executed variant(s)) |
| Training curves | generated from available evidence | outputs/figures/domain_mlp_bbox_drone_mask20_seed42__train_drone__test_drone/training_curves.png (+27 other executed variant(s)) |
| NMJE by model | generated from available evidence | outputs/figures/reconstruction/nmje_by_model.png (+1 other executed variant(s)) |
| PCK curves | generated from available evidence | outputs/figures/reconstruction/pck_curves.png (+14 other executed variant(s)) |

CSV and Markdown tables are under `outputs/tables/reconstruction`. RGB overlay figure 12 is explicitly skipped unless actual source frames are available; none are present in this acquisition.

## Limitations

- Annotation sparsity and pose-tight Drone bbox fallback limit direct comparability with independent MOT bboxes; provenance remains explicit.
- Outdoor and Drone use player/track-level fallback grouping where video identity is not sufficiently informative. This is leakage-safe but yields fewer independent groups.
- TCN evidence is admissible only for continuous reliable windows. T=1 cannot demonstrate a temporal advantage.
- PCK/NMJE measure held-out annotation reconstruction, not correctness of naturally absent joints and not real-world 3-D pose accuracy.
- A single seed or controlled slice is not a confidence interval. Repeated executed seeds must be present before variability is interpreted.
- Annotation-only evaluation cannot support RGB overlay or appearance-conditioned conclusions.
- Generated figure files occupy 63.0 MB (decimal). This exceeds the preferred 30 MB figure subtarget because repeated and legacy variants were preserved under the no-delete constraint; the project remains below the separate hard 600 MB ceiling.

## Reproducibility

From the project root:

```bash
python scripts/evaluate.py --processed data/processed/trackid3x3/dataset.npz --checkpoint outputs/checkpoints/core_mlp_angles_all_mask20_seed42/best.pt --test-domain all --mask-ratios 0.10 0.20 0.30 0.40 --primary-mask-ratio 0.20 --seed 42 --run-name reproduce_core_mlp_angles_seed42
python scripts/evaluate.py --processed data/processed/trackid3x3_t16/dataset.npz --checkpoint outputs/checkpoints/core_tcn_bbox_all_t16_mask20_seed42/best.pt --test-domain all --mask-ratios 0.20 --primary-mask-ratio 0.20 --seed 42 --run-name reproduce_core_tcn_t16_seed42
python scripts/run_cross_domain.py --processed data/processed/trackid3x3/dataset.npz --checkpoints-root outputs/checkpoints --domains Indoor Outdoor Drone --models mlp --representations bbox_normalized --mask-ratios 0.20 --seed 42 --continue-on-error
python scripts/run_ablation.py --logs-root outputs/logs --data data/processed/trackid3x3/dataset.npz --tables-dir outputs/tables/reconstruction --figures-dir outputs/figures/reconstruction
python scripts/generate_report.py
python scripts/cleanup.py
```

`cleanup.py` is a dry run unless `--confirm` is supplied. Synthetic fixtures additionally require explicit `--include-synthetic`, and their outputs remain labeled software validation only / NOT research results.

## Research narrative

The experiment asks whether lightweight skeleton representations retain enough structure to recover deliberately hidden TrackID3x3 landmarks as camera geometry and viewpoint change. The most defensible comparison is a controlled NMJE/PCK contrast across Indoor, Outdoor, and Drone, accompanied by per-joint errors and missingness curves. Within the declared Table 2 comparison (pooled-domain MLP, T=1, 20% masking, 100% data, seed 42; equal-domain macro mean over Indoor, Outdoor, and Drone), the lowest NMJE was 0.0682 and PCK@0.10 0.7974 for normalized_angles. This is single-seed exploratory evidence, not a universal-superiority claim.

No conclusion is substituted for an unexecuted cell. All requested evidence-backed tables and non-RGB figures are present in this run; RGB overlay remains explicitly unavailable because no source frames are present locally.
