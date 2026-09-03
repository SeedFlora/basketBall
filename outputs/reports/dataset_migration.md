# Dataset migration: Henu MultiSubjects to TrackID3x3

Date: 2026-09-03
Status: completed dataset-source pivot; TrackID3x3 annotations are locally available.

## Why the source changed

The original plan depended on the gated `Henu-Software/Henu-MultiSubjects` Hugging Face dataset. The attempted skeleton archive access returned HTTP 401 because this environment is not on the repository's authorized list. No Henu archive was downloaded. Continuing to design experiments around undocumented Henu pickle fields, action codes, or subject filenames would therefore be neither executable nor evidence-based.

The project now uses the official [`open-starlab/TrackID3x3`](https://github.com/open-starlab/TrackID3x3) repository at commit `9822a3dde59fc80fbb5eaacd443e79e8b059d94e`.

## Acquisition decision and storage control

GitHub reported approximately 289.39 MiB for the repository and 238.54 MiB in the current tree. Before cloning, the tree was inspected through the GitHub API: `ground_truth/` accounted for about 130.01 MiB, third-party baseline code for 66.48 MiB, and demonstration media for 39.89 MiB.

The requested `gh repo clone` command could not run because GitHub CLI is not installed. A functionally equivalent partial Git clone was used:

```text
git clone --depth 1 --filter=blob:none --sparse https://github.com/open-starlab/TrackID3x3.git external/TrackID3x3
git -C external/TrackID3x3 sparse-checkout set ground_truth court_images script
```

The sparse checkout occupies 169.83 MiB and the whole project occupied 204.05 MiB immediately after cloning. No Google Drive videos, model weights, or bulk RGB frames were downloaded because their remote sizes were not established. This follows the 500 MiB target and 600 MiB hard cap.

## Scientific reformulation

TrackID3x3 contains multi-player tracking, bounding boxes, identities, and 10-keypoint pose annotations in Indoor, Outdoor, and Drone domains. It does not contain the Henu action labels. The research task is therefore:

> Cross-Domain Robustness of Lightweight Skeleton Representations for 3x3 Basketball Player Pose Analysis

The primary experiment is masked-keypoint reconstruction with 10%, 20%, 30%, and 40% masking. It compares compact geometric, MLP, temporal (only for reliable tracks), and graph approaches using NMJE, PCK, and per-joint error. It is not action classification and makes no claim about correct or incorrect basketball posture.

## Reused and retired assumptions

Reusable infrastructure retained includes deterministic seeding, configuration loading, compact NPZ artifacts, PyTorch training loops, lightweight model utilities, checkpoint/storage accounting, plotting helpers, safe cleanup, and test/quality-gate structure.

Modified public modules include dataset inspection and sampling, preprocessing and representation construction, model/trainer entry points, reconstruction metrics, evaluation/ablation/cross-domain runners, visualization/report generation, storage budgeting, cleanup, configuration, tests, and README/reproduction documentation. Their TrackID3x3-specific implementations consume JSON/MOT annotations directly and use domain/group metadata instead of action or subject labels.

The public workflow no longer relies on Hugging Face authentication, Henu archive/pickle discovery, `d/p/s` action codes, BlazePose/MediaPipe-33 layouts, action-class labels, or subject IDs parsed from filenames. Legacy modules may remain as isolated compatibility code for the pre-pivot tests, but they are not imported by the TrackID3x3 workflow and cannot trigger a Henu download.

## Legacy fixture and configuration quarantine

The historical `scripts/create_synthetic_fixture.py` generator is retained for code history and archived loader tests, but it is disabled by default. It creates fabricated COCO17/MMAAction-style `d/p/s` action clips—not TrackID3x3 data—and now requires the explicit `--allow-legacy-henu-fixture` acknowledgement. Generated payloads and marker files state `synthetic=true`, `research_results=false`, and `trackid3x3_compatible=false`.

The historical `sample_50.yaml`, `sample_100.yaml`, `sample_200.yaml`, `sample_500.yaml`, `full_skeleton.yaml`, `smoke_synthetic.yaml`, `smoke_synthetic_aug.yaml`, and `smoke_ablation.yaml` paths remain present for provenance. They now resolve to explicit `legacy_workflow.status=disabled` metadata, a non-public sentinel processed-data path, and an invalid legacy representation token so ordinary reconstruction training fails before a legacy configuration can be mistaken for a TrackID3x3 experiment. Public runs use `config/default.yaml` or `config/trackid3x3.yaml`.

## Licensing

The upstream repository states that original TrackID3x3 code is Apache-2.0 and that the dataset annotations/videos are CC BY 4.0 unless otherwise stated. Third-party baseline directories retain their own licenses. This project uses the annotation files only and does not redistribute the optional video bundle.
