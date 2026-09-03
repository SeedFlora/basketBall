# Archived Henu dataset-access record (superseded)

**Status: historical evidence only; no access action is required.** The executable research workflow has pivoted to the public TrackID3x3 annotations. The gated Henu repository is not an active dependency, and the project must not wait for, retry, or request Henu access.

## Archived pre-pivot check

- Checked: 2026-09-02 (Asia/Bangkok)
- Repository checked: `Henu-Software/Henu-MultiSubjects`
- Requested file: `data/MultiSubjects_SA.zip`
- Hub metadata size reported: 36,433,517 bytes (34.75 MiB)
- Authentication state at the check: not logged in
- Result: HTTP 401; gated content was not downloaded
- `data/Subjects1000.rar`: 20,043,257,461 bytes; not downloaded
- `data/MultiSubjects_train_val_test.rar`: 20,039,289,206 bytes; not downloaded
- `data/key_points_csv.rar`: 372,674,684 bytes; not downloaded

The archive contents and pickle schema were never inspected. Any old synthetic fixture run was software validation only and is excluded from empirical TrackID3x3 aggregation.

## Active public-data workflow

Use the TrackID3x3 acquisition and reconstruction commands in the [project README](../../README.md). The pivot rationale is recorded in [dataset_migration.md](dataset_migration.md), and the inspected source contract is in [trackid3x3_schema.md](trackid3x3_schema.md). The current `scripts/download_dataset.py` acquires the annotation-only TrackID3x3 sparse checkout; obsolete Henu authentication and skeleton-archive flags are not part of the active CLI.
