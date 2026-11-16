#!/usr/bin/env python3
"""Test batch processing utilities."""

import json
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console

from transcript_combobulator.batch import (
    _build_file_metrics_table,
    _build_run_metrics_table,
    _build_table,
    _calculate_torch_threads,
    _estimate_worker_bytes,
    _format_duration,
    _initialize_worker,
    _memory_capped_workers,
    _publish_metrics_report,
    _resolve_log_file,
    _status_display,
    find_audio_files,
)


class TestFindAudioFiles:
    """Tests for audio file discovery."""

    def test_finds_supported_formats(self, tmp_path):
        """Finds all supported audio file extensions."""
        extensions = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"]
        for ext in extensions:
            (tmp_path / f"speaker{ext}").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == len(extensions)

    def test_ignores_non_audio_files(self, tmp_path):
        """Skips non-audio files like .txt or .json."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.json").touch()
        (tmp_path / "speaker.wav").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "speaker.wav"

    def test_returns_sorted(self, tmp_path):
        """Returns files in sorted order."""
        (tmp_path / "3-charlie.flac").touch()
        (tmp_path / "1-alice.flac").touch()
        (tmp_path / "2-bob.flac").touch()

        files = find_audio_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["1-alice.flac", "2-bob.flac", "3-charlie.flac"]

    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no audio files."""
        files = find_audio_files(tmp_path)
        assert files == []

    def test_case_insensitive_extensions(self, tmp_path):
        """Finds files regardless of extension case."""
        (tmp_path / "speaker.WAV").touch()
        (tmp_path / "speaker2.Flac").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 2

    def test_does_not_recurse(self, tmp_path):
        """Only finds files in the target directory, not subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "top.wav").touch()
        (subdir / "nested.wav").touch()

        files = find_audio_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "top.wav"


class TestDuplicateStems:
    """Two inputs naming one output folder must not both reach the workers."""

    def test_run_rejects_shared_stem(self, tmp_path, capsys):
        from transcript_combobulator.batch import main

        (tmp_path / "jfk_1.flac").touch()
        (tmp_path / "JFK_1.wav").touch()

        # Stop the run where workers would start, in case the check is missing.
        with (
            patch(
                "multiprocessing.Manager", side_effect=AssertionError("workers started")
            ),
            pytest.raises(SystemExit) as exit_info,
        ):
            main([str(tmp_path)])

        assert exit_info.value.code == 1
        out = capsys.readouterr().out
        assert "jfk_1.flac" in out and "JFK_1.wav" in out


class TestBuildTable:
    """Tests for the rich progress table builder."""

    def test_all_waiting(self):
        """Table renders correctly when all files are waiting."""
        names = ["a.flac", "b.flac"]
        status = {"a.flac": "waiting", "b.flac": "waiting"}
        table = _build_table(names, status, 2)
        assert table.title == "Transcription Progress (2 workers)"
        assert table.row_count == 2

    def test_mixed_statuses(self):
        """Table renders with a mix of statuses."""
        names = ["a.flac", "b.flac", "c.flac"]
        status = {"a.flac": "done", "b.flac": "transcribing 3/10", "c.flac": "waiting"}
        table = _build_table(names, status, 2)
        assert table.row_count == 3

    def test_error_status(self):
        """Table handles error status."""
        names = ["a.flac"]
        status = {"a.flac": "error"}
        table = _build_table(names, status, 1)
        assert table.row_count == 1

    def test_unknown_status_fallback(self):
        """Table handles unexpected status values gracefully."""
        names = ["a.flac"]
        status = {"a.flac": "unknown_state"}
        table = _build_table(names, status, 1)
        assert table.row_count == 1

    def test_missing_key_defaults_to_waiting(self):
        """Files not in status dict default to waiting."""
        names = ["a.flac"]
        status = {}
        table = _build_table(names, status, 1)
        assert table.row_count == 1


class TestStatusDisplay:
    """Tests for status label/style mapping."""

    def test_waiting(self):
        assert _status_display("waiting") == ("waiting", "dim")

    def test_converting(self):
        assert _status_display("converting") == ("converting", "yellow")

    def test_splitting(self):
        assert _status_display("splitting") == ("splitting", "yellow")

    def test_loading_model(self):
        assert _status_display("loading model") == ("loading model", "blue")

    def test_done(self):
        label, style = _status_display("done")
        assert "done" in label
        assert style == "green"

    def test_cached(self):
        label, style = _status_display("cached")
        assert "cached" in label
        assert style == "green"

    def test_transcribing_with_progress(self):
        label, style = _status_display("transcribing 3/15")
        assert label == "transcribing 3/15"
        assert style == "magenta"

    def test_error(self):
        label, style = _status_display("error")
        assert "error" in label
        assert "red" in style

    def test_unknown_fallback(self):
        assert _status_display("something_else") == ("something_else", "")


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds_only(self):
        assert _format_duration(45) == "0m 45s"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2m 05s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 01m 01s"

    def test_zero(self):
        assert _format_duration(0) == "0m 00s"

    def test_exact_hour(self):
        assert _format_duration(3600) == "1h 00m 00s"


class TestConfigSettings:
    """Tests for the parallel processing config values."""

    def test_parallel_jobs_default(self):
        """PARALLEL_JOBS defaults to 2."""
        from transcript_combobulator.config import PARALLEL_JOBS

        # The default is 2 unless overridden by env
        assert isinstance(PARALLEL_JOBS, int)
        assert PARALLEL_JOBS >= 1

    def test_torch_threads_default(self):
        """TORCH_THREADS defaults to 0 (auto-detect)."""
        from transcript_combobulator.config import TORCH_THREADS

        assert isinstance(TORCH_THREADS, int)
        assert TORCH_THREADS >= 0

    def test_whisper_pipeline_rejects_non_16khz_audio(self):
        """Array inputs have a fixed 16 kHz interpretation in Whisper."""
        from transcript_combobulator.config import _validate_sample_rate

        assert _validate_sample_rate(16000) == 16000
        with pytest.raises(ValueError, match="SAMPLE_RATE must be 16000"):
            _validate_sample_rate(8000)


class TestProcessSingleFile:
    """Tests for process_single_file status updates."""

    def test_status_dict_updates(self):
        """Verify _update_status writes to shared dict when provided."""
        # We can't easily run the full pipeline without audio files,
        # but we can verify the function signature accepts status_dict/status_key
        import inspect

        from transcript_combobulator.pipeline import process_file as main

        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        assert "status_dict" in params
        assert "status_key" in params

    def test_main_still_works_without_status_dict(self):
        """main() should accept being called without status_dict (backward compat)."""
        import inspect

        from transcript_combobulator.pipeline import process_file as main

        sig = inspect.signature(main)
        # Both params should have defaults (None)
        assert sig.parameters["status_dict"].default is None
        assert sig.parameters["status_key"].default is None

    def test_main_skips_completed_pipeline(self, tmp_path):
        """Completed outputs return without running conversion, VAD, or Whisper."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        statuses = {}
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=tmp_path,
            ),
            patch(
                "transcript_combobulator.pipeline.get_manifest_path",
                return_value=tmp_path / "manifest.json",
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=True,
            ),
            patch("transcript_combobulator.pipeline.process_audio") as process_audio,
        ):
            metrics = main(str(input_file), statuses, "speaker.wav")

        assert statuses["speaker.wav"] == "cached"
        assert metrics["status"] == "cached"
        assert metrics["cache_hit"] is True
        assert metrics["total_seconds"] >= 0
        assert "cache_check_seconds" in metrics["stages"]
        process_audio.assert_not_called()

    @pytest.mark.parametrize("suffix", [".m4a", ".aac"])
    def test_input_seconds_recorded_for_formats_libsndfile_cannot_read(
        self, tmp_path, suffix
    ):
        """The timing report reads .m4a/.aac duration too, so speed and totals count it."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / f"1-cadderly{suffix}"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:a",
                "aac",
                str(input_file),
            ],
            check=True,
        )
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=tmp_path,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=True,
            ),
        ):
            metrics = main(str(input_file))

        assert metrics["input_audio_seconds"] == pytest.approx(2.0, abs=0.1)

    def test_force_ignores_completed_pipeline(self, tmp_path):
        """Force mode runs processing even when a completion manifest exists."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        output_file = tmp_path / "speaker.wav"
        transcription = {
            "vtt_file": str(tmp_path / "speaker.vtt"),
            "json_file": str(tmp_path / "speaker.json"),
            "mapping_file": str(tmp_path / "speaker_mapping.json"),
        }
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=tmp_path,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=True,
            ),
            patch(
                "transcript_combobulator.pipeline.convert_to_wav",
                side_effect=lambda source, destination: destination.touch(),
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(tmp_path, []),
            ) as process_audio,
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ),
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            main(str(input_file), force=True)

        assert output_file.exists()
        process_audio.assert_called_once()

    def test_reprocessing_invalidates_manifest_before_pipeline_work(self, tmp_path):
        """An interrupted forced run cannot leave an old completion record."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "speaker_pipeline_manifest.json"
        manifest.write_text(
            '{"cache_version": 3, "stages": {"vtt": {"fingerprint": "old", "artifacts": ["x"]}}}',
            encoding="utf-8",
        )

        def fail_after_manifest_invalidation(*args, **kwargs):
            # A forced run wipes the old record before any work; whatever the
            # conversion stage wrote since must not be a completion record.
            from transcript_combobulator.pipeline_cache import load_manifest

            stages = load_manifest(manifest)
            assert "vtt" not in stages
            assert all(record.get("fingerprint") != "old" for record in stages.values())
            raise RuntimeError("interrupted")

        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=True,
            ),
            patch(
                "transcript_combobulator.pipeline.convert_to_wav",
                side_effect=lambda source, destination: destination.touch(),
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                side_effect=fail_after_manifest_invalidation,
            ),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            main(str(input_file), force=True)

        from transcript_combobulator.pipeline_cache import load_manifest

        assert "vtt" not in load_manifest(manifest)

    def test_quiet_16k_mono_wav_is_peak_normalised(self, tmp_path):
        """A 16 kHz mono WAV reaches VAD at the same level as any other input."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "input" / "2-danica.wav"
        input_file.parent.mkdir()
        t = np.arange(16000) / 16000
        quiet = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(input_file), quiet, 16000, subtype="PCM_16")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_file = output_dir / "2-danica.wav"
        output_file.write_bytes(b"old audio")
        transcription = {
            "vtt_file": str(output_dir / "2-danica.vtt"),
            "json_file": str(output_dir / "2-danica.json"),
            "mapping_file": str(output_dir / "2-danica_mapping.json"),
        }

        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=False,
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(output_dir, []),
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ),
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            main(str(input_file))

        data, sample_rate = sf.read(str(output_file), dtype="float32")
        assert sample_rate == 16000
        assert 0.98 <= float(np.abs(data).max()) <= 1.0

    def test_cache_miss_removes_existing_audio_before_conversion(self, tmp_path):
        """Conversion cannot silently reuse a valid but stale derived WAV."""
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "speaker.flac"
        input_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_file = output_dir / "speaker.wav"
        output_file.write_bytes(b"old audio")
        transcription = {
            "vtt_file": str(output_dir / "speaker.vtt"),
            "json_file": str(output_dir / "speaker.json"),
            "mapping_file": str(output_dir / "speaker_mapping.json"),
        }

        def convert_after_removal(source, destination):
            assert source == input_file
            assert destination == output_file
            assert not destination.exists()
            destination.touch()

        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=False,
            ),
            patch(
                "transcript_combobulator.pipeline.convert_to_wav",
                side_effect=convert_after_removal,
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(output_dir, []),
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ),
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            main(str(input_file))

        assert output_file.exists()


class TestWorkerInitialization:
    """Tests for process-wide worker setup."""

    def test_applies_priority_and_torch_limits_once(self):
        with (
            patch("transcript_combobulator.batch.os.nice") as nice,
            patch("torch.set_num_threads") as set_threads,
            patch("torch.set_num_interop_threads") as set_interop,
        ):
            _initialize_worker(torch_threads=3, worker_nice=10)

        nice.assert_called_once_with(10)
        set_threads.assert_called_once_with(3)
        set_interop.assert_called_once_with(1)

    def test_zero_values_leave_process_defaults(self):
        with (
            patch("transcript_combobulator.batch.os.nice") as nice,
            patch("torch.set_num_threads") as set_threads,
        ):
            _initialize_worker(torch_threads=0, worker_nice=0)

        nice.assert_not_called()
        set_threads.assert_not_called()


class TestTorchThreadAllocation:
    """Tests for automatic intra-op thread allocation."""

    def test_auto_splits_cpus_across_active_workers(self):
        assert _calculate_torch_threads(2, 0, cpu_count=14) == 7
        assert _calculate_torch_threads(4, 0, cpu_count=14) == 3

    def test_explicit_setting_wins(self):
        assert _calculate_torch_threads(2, 5, cpu_count=14) == 5

    def test_never_returns_less_than_one_thread(self):
        assert _calculate_torch_threads(8, 0, cpu_count=4) == 1

    def test_missing_cpu_count_uses_safe_fallback(self):
        with patch("transcript_combobulator.batch.os.cpu_count", return_value=None):
            assert _calculate_torch_threads(2, 0) == 2


class TestMetricsTables:
    def test_file_metrics_table_includes_each_file(self, tmp_path):
        metrics = [
            {
                "file": "speaker.flac",
                "status": "processed",
                "input_audio_seconds": 60.0,
                "total_seconds": 10.0,
                "stages": {"conversion_seconds": 1.0, "vad_seconds": 2.0},
                "vad": {"chunk_count": 3},
                "transcription": {
                    "model_load_seconds": 0.5,
                    "chunks": [
                        {"timings": {"inference_seconds": 4.0}},
                    ],
                },
            }
        ]

        table = _build_file_metrics_table(metrics)

        assert table.row_count == 1

    def test_run_metrics_table_has_report_path(self, tmp_path):
        summary = {
            "files_processed": 1,
            "files_cached": 0,
            "files_failed": 0,
            "input_audio_seconds": 60.0,
            "processing_seconds": 10.0,
            "audio_x_realtime": 6.0,
            "real_time_factor": 0.1667,
            "vad_chunks": 3,
            "transcript_segments": 5,
            "inference_x_realtime": 8.0,
        }

        table = _build_run_metrics_table(summary, tmp_path / "metrics.json")

        assert table.row_count == 9

    def test_run_metrics_table_does_not_claim_failed_report(self, tmp_path):
        summary = {
            "files_processed": 1,
            "files_cached": 0,
            "files_failed": 0,
            "input_audio_seconds": 60.0,
            "processing_seconds": 10.0,
            "audio_x_realtime": 6.0,
            "real_time_factor": 0.1667,
            "vad_chunks": 3,
            "transcript_segments": 5,
            "inference_x_realtime": 8.0,
        }
        console = Console(record=True, width=120)

        console.print(_build_run_metrics_table(summary, None))
        rendered = console.export_text()

        assert "Metrics report" in rendered
        assert "not written" in rendered


class TestMetricsPublication:
    def test_returns_report_path_after_success(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"

        with patch("transcript_combobulator.batch.write_metrics_report") as write:
            published_path, error = _publish_metrics_report(metrics_path, {})

        write.assert_called_once_with(metrics_path, {})
        assert published_path == metrics_path
        assert error is None

    def test_returns_not_written_after_oserror(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"

        with patch(
            "transcript_combobulator.batch.write_metrics_report",
            side_effect=OSError("disk full"),
        ):
            published_path, error = _publish_metrics_report(metrics_path, {})

        assert published_path is None
        assert error == "disk full"


GIB = 1 << 30


class TestMemoryGuard:
    """Worker cap derived from model size and physical RAM."""

    def test_unknown_ram_or_model_leaves_request_alone(self):
        assert _memory_capped_workers(2, 0, 8 * GIB, 0.85) == (2, None)
        assert _memory_capped_workers(2, GIB, None, 0.85) == (2, None)

    def test_enough_ram_keeps_requested_workers(self):
        # large-v3-turbo (~1.6 GB) x2 on a 32 GB machine fits with room to spare.
        workers, warning = _memory_capped_workers(2, int(1.6 * GIB), 32 * GIB, 0.85)
        assert workers == 2
        assert warning is None

    def test_eight_gb_pi_is_capped_to_one_worker(self):
        workers, warning = _memory_capped_workers(2, int(1.6 * GIB), 8 * GIB, 0.85)
        assert workers == 1
        assert warning is not None
        assert "running 1" in warning
        assert "WHISPER_MODEL" in warning or "running 1" in warning

    def test_single_worker_that_cannot_fit_still_runs_with_warning(self):
        workers, warning = _memory_capped_workers(1, 4 * GIB, 8 * GIB, 0.85)
        assert workers == 1
        assert warning is not None
        assert "one worker" in warning

    def test_never_raises_the_requested_count(self):
        workers, _ = _memory_capped_workers(1, int(0.1 * GIB), 64 * GIB, 0.85)
        assert workers == 1

    def test_estimate_scales_with_model_size(self):
        small = _estimate_worker_bytes(100 * 1024 * 1024)
        large = _estimate_worker_bytes(1600 * 1024 * 1024)
        assert large > small
        assert small > 100 * 1024 * 1024


class TestResolveLogFile:
    """LOG_FILE mapping for batch runs."""

    def test_empty_defaults_to_session_log_under_output(self, tmp_path):
        assert _resolve_log_file("", tmp_path, "night-one") == (
            tmp_path / "night-one" / "night-one.log"
        )

    def test_absolute_path_is_used_as_is(self, tmp_path):
        target = tmp_path / "custom.log"
        assert _resolve_log_file(str(target), tmp_path, "s") == target

    def test_relative_path_resolves_against_project_root(self, tmp_path):
        from transcript_combobulator.config import ROOT_DIR

        assert (
            _resolve_log_file("logs/run.log", tmp_path, "s")
            == ROOT_DIR / "logs" / "run.log"
        )


class TestPartialTranscriptionFailsFile:
    """A file with failed chunks must not be recorded as complete."""

    def _run(self, tmp_path, failed):
        from transcript_combobulator.pipeline import process_file as main

        input_file = tmp_path / "speaker.wav"
        input_file.touch()
        transcription = {
            "vtt_file": str(tmp_path / "speaker.vtt"),
            "json_file": str(tmp_path / "speaker.json"),
            "mapping_file": str(tmp_path / "speaker_mapping.json"),
            "metrics": {"chunk_count": 200, "failed_chunk_count": failed, "chunks": []},
        }
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=tmp_path,
            ),
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=False,
            ),
            patch(
                "transcript_combobulator.pipeline.convert_to_wav",
                side_effect=lambda source, destination: destination.touch(),
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(tmp_path, []),
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ),
            patch(
                "transcript_combobulator.pipeline.write_pipeline_manifest"
            ) as manifest,
        ):
            metrics = main(str(input_file))
        return metrics, manifest

    def test_failed_chunks_raise_and_skip_manifest(self, tmp_path):
        from transcript_combobulator.transcribe import TranscriptionError

        with pytest.raises(TranscriptionError, match="30 of 200 segments failed"):
            self._run(tmp_path, failed=30)

    def test_clean_file_still_completes(self, tmp_path):
        metrics, manifest = self._run(tmp_path, failed=0)
        assert metrics["status"] == "processed"
        manifest.assert_called_once()


class TestOutputPlacement:
    """process_file places output by the given path and session (issue #22)."""

    def _run(self, tmp_path, input_file, session=None):
        from transcript_combobulator.pipeline import process_file as main

        transcription = {
            "vtt_file": str(tmp_path / "out.vtt"),
            "json_file": str(tmp_path / "out.json"),
            "mapping_file": str(tmp_path / "out_mapping.json"),
            "metrics": {"chunk_count": 0, "failed_chunk_count": 0, "chunks": []},
        }
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=tmp_path / "out",
            ) as out,
            patch(
                "transcript_combobulator.pipeline.is_pipeline_complete",
                return_value=False,
            ),
            patch(
                "transcript_combobulator.pipeline.convert_to_wav",
                side_effect=lambda source, destination: destination.touch(),
            ),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(tmp_path, []),
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ) as ts,
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            main(str(input_file), session=session)
        return out, ts

    def test_symlink_is_placed_by_the_link_not_its_target(self, tmp_path):
        real = tmp_path / "archive" / "4-pikel.wav"
        real.parent.mkdir()
        real.touch()
        link = tmp_path / "s11" / "4-pikel.wav"
        link.parent.mkdir()
        link.symlink_to(real)
        out, ts = self._run(tmp_path, link)
        assert out.call_args.args == (link, None)
        assert ts.call_args.args[1] == tmp_path / "out"

    def test_link_named_unlike_its_target_writes_the_vtt_combine_reads(
        self, tmp_path, monkeypatch
    ):
        from transcript_combobulator import config
        from transcript_combobulator.pipeline import process_file

        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
        real = tmp_path / "archive" / "4-pikel_123.wav"
        real.parent.mkdir()
        real.write_bytes(b"audio")
        link = tmp_path / "s11" / "4-pikel.wav"
        link.parent.mkdir()
        link.symlink_to(real)

        def fake_transcribe(audio_path, output_dir, **kwargs):
            # Names its files from the audio it is handed, as the real one does.
            vtt = output_dir / config.vtt_name_for_stem(audio_path.stem)
            vtt.write_text("WEBVTT\n")
            return {
                "vtt_file": str(vtt),
                "json_file": str(output_dir / f"{audio_path.stem}_transcription.json"),
                "mapping_file": str(output_dir / f"{audio_path.stem}_mapping.json"),
                "metrics": {"chunk_count": 0, "failed_chunk_count": 0, "chunks": []},
            }

        with (
            patch("transcript_combobulator.pipeline.convert_to_wav"),
            patch(
                "transcript_combobulator.pipeline.process_audio",
                return_value=(tmp_path, []),
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                side_effect=fake_transcribe,
            ),
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            process_file(str(link), session="s11")
        assert config.vtt_path_for_input(link, "s11").exists()

    def test_session_is_passed_through(self, tmp_path):
        track = tmp_path / "1-cadderly.wav"
        track.touch()
        out, _ = self._run(tmp_path, track, session="s11")
        assert out.call_args.args == (track, "s11")


class TestStageReuse:
    """Completed stages are reused; only the changed stage and later rerun."""

    def _setup(self, tmp_path):
        from transcript_combobulator.pipeline_cache import (
            build_stage_fingerprints,
            record_stage,
        )

        input_file = tmp_path / "speaker.flac"
        input_file.write_bytes(b"audio")
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        wav = output_dir / "speaker.wav"
        wav.touch()
        mapping = output_dir / "speaker_mapping.json"
        seg = output_dir / "speaker_segment_000.wav"
        seg.touch()
        mapping.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start_seconds": 1.0,
                            "end_seconds": 2.0,
                            "clip_start_seconds": 0.7,
                            "clip_end_seconds": 2.3,
                            "segment_file": str(seg),
                        }
                    ]
                }
            )
        )
        manifest = output_dir / "speaker_pipeline_manifest.json"
        fps = build_stage_fingerprints(input_file)
        record_stage(manifest, "conversion", fps["conversion"], [wav])
        record_stage(manifest, "vad", fps["vad"], [mapping, seg])
        return input_file, output_dir, manifest, fps

    def test_conversion_and_vad_are_skipped_when_recorded(self, tmp_path):
        from transcript_combobulator.pipeline import process_file as main

        input_file, output_dir, manifest, fps = self._setup(tmp_path)
        transcription = {
            "vtt_file": str(output_dir / "speaker.vtt"),
            "json_file": str(output_dir / "speaker_transcription.json"),
            "mapping_file": str(output_dir / "speaker_mapping.json"),
            "metrics": {"chunk_count": 1, "failed_chunk_count": 0, "chunks": []},
        }
        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch("transcript_combobulator.pipeline.convert_to_wav") as convert,
            patch("transcript_combobulator.pipeline.process_audio") as vad,
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                return_value=transcription,
            ) as ts,
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            metrics = main(str(input_file))

        convert.assert_not_called()
        vad.assert_not_called()
        assert ts.call_args.kwargs["checkpoint_key"] == fps["inference"]
        assert metrics["conversion"]["action"] == "cached"
        assert metrics["vad_cached"] is True
        assert metrics["vad"]["chunk_count"] == 1

    def test_presentation_only_change_rewrites_vtt_without_inference(self, tmp_path):
        from transcript_combobulator.pipeline import process_file as main
        from transcript_combobulator.pipeline_cache import record_stage

        input_file, output_dir, manifest, fps = self._setup(tmp_path)
        json_path = output_dir / "speaker_transcription.json"
        json_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": 1.0, "end": 2.0, "text": "Hello", "confidence": 90.0}
                    ]
                }
            )
        )
        vtt = output_dir / "speaker.vtt"
        vtt.touch()
        record_stage(manifest, "inference", fps["inference"], [json_path, vtt])

        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch("transcript_combobulator.pipeline.process_audio") as vad,
            patch("transcript_combobulator.pipeline.transcribe_segments") as ts,
            patch("transcript_combobulator.pipeline.write_pipeline_manifest"),
        ):
            metrics = main(str(input_file))

        vad.assert_not_called()
        ts.assert_not_called()
        assert metrics["transcription"]["inference_cached"] is True
        assert "Hello" in vtt.read_text(encoding="utf-8")
        assert "vtt_rewrite_seconds" in metrics["stages"]

    def test_missing_vtt_is_rewritten_from_json_without_inference(self, tmp_path):
        from transcript_combobulator.pipeline import process_file as main

        input_file, output_dir, manifest, fps = self._setup(tmp_path)
        json_path = output_dir / "speaker_transcription.json"
        vtt = output_dir / "speaker.vtt"

        def transcribe(*args, **kwargs):
            json_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "start": 1.0,
                                "end": 2.0,
                                "text": "Hello",
                                "confidence": 90.0,
                            }
                        ]
                    }
                )
            )
            vtt.write_text("WEBVTT\n", encoding="utf-8")
            return {
                "vtt_file": str(vtt),
                "json_file": str(json_path),
                "mapping_file": str(output_dir / "speaker_mapping.json"),
                "metrics": {"chunk_count": 1, "failed_chunk_count": 0, "chunks": []},
            }

        with (
            patch(
                "transcript_combobulator.pipeline.get_output_path_for_input",
                return_value=output_dir,
            ),
            patch(
                "transcript_combobulator.pipeline.transcribe_segments",
                side_effect=transcribe,
            ) as ts,
        ):
            main(str(input_file))
            vtt.unlink()
            metrics = main(str(input_file))

        ts.assert_called_once()
        assert metrics["transcription"]["inference_cached"] is True
        assert "Hello" in vtt.read_text(encoding="utf-8")
