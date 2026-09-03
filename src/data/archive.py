"""Archived, local-only archive handling retained for compatibility tests.

The public TrackID3x3 downloader does not import this module and has no Henu or
Hugging Face acquisition path.
"""

from __future__ import annotations

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_ARCHIVE_NAME = "MultiSubjects_SA.zip"
FORBIDDEN_ARCHIVE_NAMES = {"Subjects1000.rar", "MultiSubjects_train_val_test.rar"}


class ArchiveSafetyError(RuntimeError):
    """Raised when an archive violates the project's storage/safety policy."""


def discover_allowed_archive(search_root: str | Path) -> Path | None:
    """Recursively find the one permitted archive, rejecting ambiguous copies."""

    root = Path(search_root)
    if root.is_file():
        if root.name != ALLOWED_ARCHIVE_NAME:
            raise ArchiveSafetyError(
                f"Refusing archive {root.name!r}; only {ALLOWED_ARCHIVE_NAME!r} is permitted."
            )
        return root.resolve()
    if not root.exists():
        return None
    candidates = sorted(
        (path.resolve() for path in root.rglob(ALLOWED_ARCHIVE_NAME) if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    if len(candidates) > 1:
        raise ArchiveSafetyError(
            "Multiple permitted archives were found; pass an explicit path to avoid ambiguity: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0] if candidates else None


def discover_pickle_files(search_root: str | Path) -> list[Path]:
    """Find train/validation pickle files recursively without assuming nesting."""

    root = Path(search_root)
    if not root.exists():
        return []
    if root.is_file():
        return [root.resolve()] if root.suffix.lower() in {".pkl", ".pickle"} else []
    paths = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".pickle"}
    ]
    # Prefer documented train/val names, but keep any additional pickle visible
    # to the schema inspector instead of silently ignoring it.
    return sorted(paths, key=lambda path: (path.name.lower() not in {"train.pkl", "val.pkl"}, str(path).lower()))


def _validated_member_target(destination: Path, member: zipfile.ZipInfo) -> Path:
    name = member.filename.replace("\\", "/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ArchiveSafetyError(f"Unsafe archive member path: {member.filename!r}.")
    if pure.parts[0].endswith(":"):
        raise ArchiveSafetyError(f"Windows drive path is forbidden in archive: {member.filename!r}.")
    mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise ArchiveSafetyError(f"Symbolic links are forbidden in archive: {member.filename!r}.")
    target = (destination / Path(*pure.parts)).resolve()
    root = destination.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ArchiveSafetyError(f"Archive member escapes destination: {member.filename!r}.") from exc
    return target


def safe_extract_allowed_archive(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_uncompressed_bytes: int = 1_000_000_000,
) -> list[Path]:
    """Safely extract only ``MultiSubjects_SA.zip`` under a size ceiling.

    Paths, symlinks, encrypted members and extreme expansion ratios are rejected
    before any member is written.  Existing files must match the advertised size
    or extraction stops rather than overwriting unexplained local data.
    """

    archive = Path(archive_path).resolve()
    if archive.name != ALLOWED_ARCHIVE_NAME:
        raise ArchiveSafetyError(
            f"Refusing {archive.name!r}; only {ALLOWED_ARCHIVE_NAME!r} may be extracted."
        )
    if not archive.is_file():
        raise FileNotFoundError(f"Permitted archive not found: {archive}")
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        bundle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveSafetyError(f"Invalid ZIP archive {archive}: {exc}") from exc
    with bundle:
        members = bundle.infolist()
        total = sum(member.file_size for member in members if not member.is_dir())
        if total > max_uncompressed_bytes:
            raise ArchiveSafetyError(
                f"Archive expands to {total:,} bytes, above the {max_uncompressed_bytes:,}-byte limit."
            )
        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            target = _validated_member_target(destination_path, member)
            if member.flag_bits & 0x1:
                raise ArchiveSafetyError(f"Encrypted archive member is unsupported: {member.filename!r}.")
            if member.compress_size and member.file_size / member.compress_size > 1000:
                raise ArchiveSafetyError(f"Suspicious expansion ratio for {member.filename!r}.")
            validated.append((member, target))
        written: list[Path] = []
        for member, target in validated:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ArchiveSafetyError(
                        f"Existing extraction target has unexpected size; refusing overwrite: {target}"
                    )
                written.append(target)
                continue
            with bundle.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            written.append(target)
    return written


def resolve_pickle_source(
    *,
    input_path: str | Path | None = None,
    raw_root: str | Path = "data/raw",
    extract_root: str | Path = "data/interim/MultiSubjects_SA",
) -> tuple[list[Path], Path | None]:
    """Resolve direct pickles or safely extract the permitted archive.

    Returns ``(pickle_paths, archive_path)``.  An empty list means no source was
    available and lets inspection generate an honest unavailable report.
    """

    if input_path is not None:
        supplied = Path(input_path)
        if supplied.is_file() and supplied.suffix.lower() == ".zip":
            archive = discover_allowed_archive(supplied)
            assert archive is not None
            safe_extract_allowed_archive(archive, extract_root)
            return discover_pickle_files(extract_root), archive
        pickles = discover_pickle_files(supplied)
        if pickles:
            return pickles, None
        if supplied.exists():
            archive = discover_allowed_archive(supplied)
            if archive is not None:
                safe_extract_allowed_archive(archive, extract_root)
                return discover_pickle_files(extract_root), archive
        return [], None

    existing = discover_pickle_files(extract_root)
    if existing:
        return existing, None
    archive = discover_allowed_archive(raw_root)
    if archive is None:
        return [], None
    safe_extract_allowed_archive(archive, extract_root)
    return discover_pickle_files(extract_root), archive
