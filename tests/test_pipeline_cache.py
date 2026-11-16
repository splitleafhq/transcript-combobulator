"""Tests for config-aware per-file pipeline completion manifests."""

import json
from pathlib import Path
from unittest.mock import patch

from transcript_combobulator.pipeline_cache import (
    ChunkCheckpoint,
    build_pipeline_fingerprint,
    build_stage_fingerprints,
    invalidate_stage,
    is_pipeline_complete,
    load_chunk_checkpoint,
    load_manifest,
    record_stage,
    stage_is_complete,
    write_pipeline_manifest,
)


def test_fingerprint_changes_when_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"first")
    original = build_pipeline_fingerprint(source)

    source.write_bytes(b"changed contents")

    assert build_pipeline_fingerprint(source) != original


def test_completed_manifest_requires_every_artifact(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")
    first_artifact = tmp_path / "speaker.vtt"
    second_artifact = tmp_path / "speaker.json"
    first_artifact.touch()
    second_artifact.touch()
    manifest = tmp_path / "manifest.json"

    write_pipeline_manifest(source, manifest, [first_artifact, second_artifact])
    assert is_pipeline_complete(source, manifest)

    second_artifact.unlink()
    assert not is_pipeline_complete(source, manifest)


def test_invalid_manifest_is_not_complete(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{invalid", encoding="utf-8")

    assert not is_pipeline_complete(source, manifest)


def test_manifest_does_not_match_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"original")
    artifact = tmp_path / "speaker.vtt"
    artifact.touch()
    manifest = tmp_path / "manifest.json"
    write_pipeline_manifest(source, manifest, [artifact])

    source.write_bytes(b"updated source")

    assert not is_pipeline_complete(source, manifest)


def test_manifest_contains_resolved_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.touch()
    artifact = tmp_path / "speaker.vtt"
    artifact.touch()
    manifest = tmp_path / "manifest.json"

    write_pipeline_manifest(source, manifest, [artifact])

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["stages"]["vtt"]["artifacts"] == [str(artifact.resolve())]


def test_stage_fingerprints_chain_from_the_changed_stage_onward(tmp_path: Path) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")
    base = build_stage_fingerprints(source)

    with patch("transcript_combobulator.config.DEDUPE_WINDOW_SECONDS", 9.0):
        dedupe_changed = build_stage_fingerprints(source)
    assert dedupe_changed["conversion"] == base["conversion"]
    assert dedupe_changed["vad"] == base["vad"]
    assert dedupe_changed["inference"] == base["inference"]
    assert dedupe_changed["vtt"] != base["vtt"]

    with patch("transcript_combobulator.config.VAD_THRESHOLD", 0.9):
        vad_changed = build_stage_fingerprints(source)
    assert vad_changed["conversion"] == base["conversion"]
    assert vad_changed["vad"] != base["vad"]
    assert vad_changed["inference"] != base["inference"]
    assert vad_changed["vtt"] != base["vtt"]


def test_record_stage_drops_later_stages(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    artifact = tmp_path / "a"
    artifact.touch()
    record_stage(manifest, "conversion", "c1", [artifact])
    record_stage(manifest, "vad", "v1", [artifact])
    record_stage(manifest, "inference", "i1", [artifact])
    stages = load_manifest(manifest)
    assert set(stages) == {"conversion", "vad", "inference"}

    record_stage(manifest, "vad", "v2", [artifact])
    stages = load_manifest(manifest)
    assert set(stages) == {"conversion", "vad"}
    assert stage_is_complete(stages, "conversion", "c1")
    assert stage_is_complete(stages, "vad", "v2")
    assert not stage_is_complete(stages, "vad", "v1")


def test_invalidate_stage_keeps_earlier_stages(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    artifact = tmp_path / "a"
    artifact.touch()
    for stage, fp in (
        ("conversion", "c"),
        ("vad", "v"),
        ("inference", "i"),
        ("vtt", "t"),
    ):
        record_stage(manifest, stage, fp, [artifact])
    invalidate_stage(manifest, "vtt")
    assert set(load_manifest(manifest)) == {"conversion", "vad", "inference"}
    invalidate_stage(manifest, "vad")
    assert set(load_manifest(manifest)) == {"conversion"}


def test_stage_requires_artifacts_and_old_cache_versions_are_ignored(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "m.json"
    artifact = tmp_path / "a"
    artifact.touch()
    record_stage(manifest, "conversion", "c", [artifact])
    assert stage_is_complete(load_manifest(manifest), "conversion", "c")
    artifact.unlink()
    assert not stage_is_complete(load_manifest(manifest), "conversion", "c")

    manifest.write_text(
        '{"cache_version": 2, "stages": {"conversion": {"fingerprint": "c", "artifacts": []}}}'
    )
    assert load_manifest(manifest) == {}


def test_chunk_checkpoint_roundtrip_and_truncation(tmp_path: Path) -> None:
    progress = tmp_path / "p.jsonl"
    cp = ChunkCheckpoint(progress, key="k1")
    assert cp.completed == {}
    cp.record(1, [{"start": 0.0, "end": 1.0, "text": "a"}], {"audio_seconds": 1.0})
    cp.record(2, [], {"audio_seconds": 0.5})
    cp.close(remove=False)
    # simulate an interrupted write of chunk 3
    with open(progress, "a", encoding="utf-8") as f:
        f.write('{"index": 3, "segments": [{"sta')

    loaded = load_chunk_checkpoint(progress, "k1")
    assert set(loaded) == {1, 2}
    assert loaded[1]["segments"][0]["text"] == "a"

    # the next run's chunks are not lost behind the fragment
    cp = ChunkCheckpoint(progress, key="k1")
    cp.record(3, [], {"audio_seconds": 0.5})
    cp.record(4, [], {"audio_seconds": 0.5})
    cp.close(remove=False)
    assert set(load_chunk_checkpoint(progress, "k1")) == {1, 2, 3, 4}

    # a different key (settings changed) discards the file
    assert load_chunk_checkpoint(progress, "k2") == {}
    assert not progress.exists()


def test_chunk_checkpoint_removed_on_close(tmp_path: Path) -> None:
    progress = tmp_path / "p.jsonl"
    cp = ChunkCheckpoint(progress, key="k")
    cp.record(1, [], {})
    cp.close(remove=True)
    assert not progress.exists()


def test_failed_manifest_write_never_leaves_a_stale_record(tmp_path: Path) -> None:
    """Found by a Codex read of the module: a swallowed write error used to
    keep the old completion record on disk."""
    import builtins

    manifest = tmp_path / "m.json"
    artifact = tmp_path / "a"
    artifact.touch()
    record_stage(manifest, "vtt", "done", [artifact])
    assert stage_is_complete(load_manifest(manifest), "vtt", "done")

    real_open = builtins.open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith(".tmp") and "w" in (
            args[0] if args else kwargs.get("mode", "r")
        ):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=failing_open):
        invalidate_stage(manifest, "vtt")

    assert not manifest.exists()
    assert not stage_is_complete(load_manifest(manifest), "vtt", "done")
