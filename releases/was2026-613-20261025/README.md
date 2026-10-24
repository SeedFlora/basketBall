# Basketball pose reconstruction revision artifacts

This release adds the executed multi-seed revision artifacts to the existing TrackID3x3 reconstruction code. The original single-seed files remain unchanged.

The package contains three orchestration and aggregation scripts and 32 CSV tables with 32 Markdown counterparts under outputs/tables/reconstruction/multiseed_*. The paper uses the *_common.csv tables: all domains use the ground-truth pose-extent diagonal for the evaluation denominator. This denominator is used only to score predictions. Files without _common preserve the native source-box denominator.

Seeds 42-46 are complete replicates. multiseed_runs_common.csv supplies the per-run/per-domain values; the representation, model, missingness, transfer, training-budget and paired-test tables provide the reported means, sample standard deviations and paired comparisons. Group-rotation controls are separate from the five-seed controls.

## Reproduce

Use the root repository's environment, original data access instructions, configuration and source code. Obtain TrackID3x3 from its original source; this release redistributes no raw footage, athlete images, keypoints, model checkpoints, or per-pose predictions.

    python scripts/run_multiseed.py --families core temporal dataeff domain shift sizematch --jobs 2
    python scripts/run_multiseed.py --families domain sizematch --transfer-representation scale_normalized --jobs 2
    python scripts/build_rotation_folds.py
    python scripts/run_multiseed.py --seeds 42 --families rotation --jobs 2
    python scripts/run_multiseed.py --seeds 42 --families rotation --transfer-representation scale_normalized --jobs 2
    python scripts/aggregate_multiseed.py --denominator common
    python scripts/aggregate_multiseed.py --denominator native

The stored CSVs can be inspected without downloading the source dataset or running training. Re-running the complete experiments requires the original data and compute. This release contains code and bounded aggregate/per-seed result tables; it does not make every underlying training artifact public.

ARTIFACT_MANIFEST.json records the SHA-256 and size of each copied source artifact. All 67 artifacts were copied byte-for-byte from the executed local research project on 25 October 2026. A manuscript audit traced all 153 four-decimal values in the current paper to existing result artifacts. That check establishes provenance, not an independent replication of training.

No new reuse license is asserted by this release.
