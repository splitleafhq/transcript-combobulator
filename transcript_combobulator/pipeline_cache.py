"""Config-aware completion records for resumable per-file processing.

One manifest per input file holds a record per stage (conversion, vad,
inference, vtt). Each stage's fingerprint chains from the previous stage's,
so changing a setting invalidates that stage and everything after it. A
record is only trusted when its fingerprint matches and every artifact it
names still exists. Transcription additionally checkpoints per chunk in a
progress file so an interrupted run resumes at the first missing chunk.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from transcript_combobulator.config import get_stage_fingerprint_settings
from transcript_combobulator.logging_config import get_logger

logger = get_logger(__name__)

# 3: per-stage records replace the single completion fingerprint.
# 2: per-speaker VTT dedup became consecutive-only and cue offsets now use the
#    padded clip start, so VTTs written by version 1 are lossy and shifted.
PIPELINE_CACHE_VERSION = 3

STAGES: tuple[str, ...] = ("conversion", "vad", "inference", "vtt")
FINAL_STAGE = STAGES[-1]


def get_manifest_path(output_dir: Path, audio_stem: str) -> Path:
    """Return the completion-manifest path for an input audio file."""
    return output_dir / f"{audio_stem}_pipeline_manifest.json"


def get_progress_path(output_dir: Path, audio_stem: str) -> Path:
    """Return the per-chunk transcription checkpoint path for an input file."""
    return output_dir / f"{audio_stem}_progress.jsonl"


def _digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_stage_fingerprints(input_path: Path) -> dict[str, str]:
    """Chained fingerprints: source identity, then each stage's settings."""
    stat = input_path.stat()
    parent = _digest(
        {
            "cache_version": PIPELINE_CACHE_VERSION,
            "source": {
                "path": str(input_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
        }
    )
    settings = get_stage_fingerprint_settings()
    fingerprints: dict[str, str] = {}
    for stage in STAGES:
        parent = _digest(
            {"parent": parent, "stage": stage, "settings": settings[stage]}
        )
        fingerprints[stage] = parent
    return fingerprints


def build_pipeline_fingerprint(input_path: Path) -> str:
    """Fingerprint of the whole pipeline (the final stage's chained hash)."""
    return build_stage_fingerprints(input_path)[FINAL_STAGE]


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Return the manifest's stage records, or {} if missing, invalid, or stale."""
    if not manifest_path.is_file():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict[str, Any] = json.load(f)
        if manifest.get("cache_version") != PIPELINE_CACHE_VERSION:
            return {}
        stages = manifest.get("stages")
        return stages if isinstance(stages, dict) else {}
    except (OSError, ValueError, TypeError):
        logger.warning(f"Ignoring invalid pipeline manifest: {manifest_path}")
        return {}


def stage_is_complete(stages: dict[str, Any], stage: str, fingerprint: str) -> bool:
    """A stage counts only if its fingerprint matches and its artifacts exist."""
    record = stages.get(stage)
    if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
        return False
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    return all(isinstance(path, str) and Path(path).is_file() for path in artifacts)


def _write_manifest(manifest_path: Path, stages: dict[str, Any]) -> None:
    """Atomically replace the manifest. If the write fails, remove the manifest.

    A missing manifest is always safe (everything reruns); a stale one is not,
    because a record that was meant to be dropped could later pass
    ``is_pipeline_complete``. So a failed write never leaves the old file.
    """
    payload = {"cache_version": PIPELINE_CACHE_VERSION, "stages": stages}
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temporary_path.replace(manifest_path)
    except OSError as e:
        temporary_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        logger.warning(
            f"Could not write pipeline manifest {manifest_path}: {e}; removed it"
        )


def record_stage(
    manifest_path: Path,
    stage: str,
    fingerprint: str,
    artifacts: Iterable[Path],
) -> None:
    """Atomically record a completed stage; drops records of every later stage.

    Later stages were computed from this stage's previous output, so they are
    no longer valid once it has been redone.
    """
    stages = load_manifest(manifest_path)
    for later in STAGES[STAGES.index(stage) + 1 :]:
        stages.pop(later, None)
    stages[stage] = {
        "fingerprint": fingerprint,
        "artifacts": [str(path.resolve()) for path in artifacts],
    }
    _write_manifest(manifest_path, stages)


def invalidate_stage(manifest_path: Path, stage: str) -> None:
    """Drop a stage record and every later one, keeping earlier stages."""
    stages = load_manifest(manifest_path)
    changed = False
    for name in STAGES[STAGES.index(stage) :]:
        if stages.pop(name, None) is not None:
            changed = True
    if changed:
        _write_manifest(manifest_path, stages)


def is_pipeline_complete(input_path: Path, manifest_path: Path) -> bool:
    """Whether the whole pipeline is recorded complete for the current config."""
    fingerprints = build_stage_fingerprints(input_path)
    return stage_is_complete(
        load_manifest(manifest_path), FINAL_STAGE, fingerprints[FINAL_STAGE]
    )


def write_pipeline_manifest(
    input_path: Path,
    manifest_path: Path,
    artifacts: Iterable[Path],
) -> None:
    """Record the final stage with every artifact the run requires."""
    fingerprints = build_stage_fingerprints(input_path)
    record_stage(manifest_path, FINAL_STAGE, fingerprints[FINAL_STAGE], artifacts)


# ── per-chunk transcription checkpoint ──


def load_chunk_checkpoint(progress_path: Path, key: str) -> dict[int, dict[str, Any]]:
    """Completed chunks {index: record} from a progress file written under ``key``.

    A file written under a different key (settings changed) or with a bad
    header is discarded. A truncated last line (interrupted mid-write) is
    ignored; that chunk is simply redone.
    """
    if not progress_path.is_file():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    try:
        with open(progress_path, encoding="utf-8") as f:
            header_line = f.readline()
            header = json.loads(header_line) if header_line.strip() else {}
            if header.get("key") != key:
                progress_path.unlink(missing_ok=True)
                return {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break
                index = record.get("index")
                if isinstance(index, int) and isinstance(record.get("segments"), list):
                    completed[index] = record
    except (OSError, ValueError, TypeError):
        logger.warning(f"Ignoring invalid progress file: {progress_path}")
        progress_path.unlink(missing_ok=True)
        return {}
    return completed


def _drop_partial_line(progress_path: Path) -> None:
    """Cut an interrupted last write, so the next record starts its own line.

    Appending after the fragment would glue the next record onto it, and the
    reader stops at the first line it cannot parse.
    """
    if not progress_path.is_file():
        return
    data = progress_path.read_bytes()
    if data and not data.endswith(b"\n"):
        with open(progress_path, "r+b") as f:
            f.truncate(data.rfind(b"\n") + 1)


class ChunkCheckpoint:
    """Append-only per-chunk progress. Each completed chunk is one JSON line."""

    def __init__(self, progress_path: Path, key: str) -> None:
        self.path = progress_path
        self.key = key
        self.completed = load_chunk_checkpoint(progress_path, key)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        _drop_partial_line(progress_path)
        new_file = not progress_path.is_file() or progress_path.stat().st_size == 0
        self._handle = open(progress_path, "a", encoding="utf-8")
        if new_file:
            self._handle.write(json.dumps({"key": key}) + "\n")
            self._handle.flush()

    def record(
        self, index: int, segments: list[dict[str, Any]], timings: dict[str, Any]
    ) -> None:
        line = json.dumps({"index": index, "segments": segments, "timings": timings})
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self, *, remove: bool) -> None:
        self._handle.close()
        if remove:
            self.path.unlink(missing_ok=True)
