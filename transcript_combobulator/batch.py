"""Parallel batch processor with live progress display."""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import FrameType
from typing import Any, MutableMapping, NoReturn, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from transcript_combobulator.logging_config import add_file_handler, remove_file_handler
from transcript_combobulator.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    build_run_summary,
    elapsed_seconds,
    utc_now_iso,
    write_metrics_report,
)

# Ensure spawn method for macOS torch compatibility
multiprocessing.set_start_method("spawn", force=True)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


def _calculate_torch_threads(
    max_workers: int,
    configured_threads: int,
    cpu_count: int | None = None,
) -> int:
    """Resolve threads per worker, splitting detected CPUs across active workers."""
    if configured_threads > 0:
        return configured_threads
    detected_cpus = cpu_count if cpu_count is not None else os.cpu_count()
    return max(1, (detected_cpus or 4) // max(1, max_workers))


# whisper checkpoints are fp16; load_model promotes them to fp32 while the fp16
# copy is still alive, so peak per worker is roughly 3x the file plus runtime.
_MODEL_MEMORY_MULTIPLIER = 3.0
_WORKER_BASELINE_BYTES = 1 << 30
_GIB = float(1 << 30)


def _total_memory_bytes() -> int | None:
    """Physical RAM via sysconf (Linux and macOS); None if unavailable."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _estimate_worker_bytes(model_bytes: int) -> int:
    return int(model_bytes * _MODEL_MEMORY_MULTIPLIER) + _WORKER_BASELINE_BYTES


def _memory_capped_workers(
    requested: int,
    model_bytes: int,
    total_bytes: int | None,
    fraction: float,
) -> tuple[int, str | None]:
    """Lower the worker count until the estimated footprint fits in RAM.

    Returns (workers, warning). Never returns fewer than 1 and never raises
    the count; with unknown RAM or model size it returns ``requested``.
    """
    if total_bytes is None or total_bytes <= 0 or model_bytes <= 0:
        return requested, None
    per_worker = _estimate_worker_bytes(model_bytes)
    budget = total_bytes * fraction
    fits = int(budget // per_worker)
    if fits < 1:
        return 1, (
            f"Memory guard: one worker needs ~{per_worker / _GIB:.1f} GiB but "
            f"{fraction:.0%} of {total_bytes / _GIB:.1f} GiB RAM is "
            f"{budget / _GIB:.1f} GiB. Expect the OOM killer; use a smaller "
            "WHISPER_MODEL."
        )
    if fits >= requested:
        return requested, None
    return fits, (
        f"Memory guard: {requested} workers need ~{requested * per_worker / _GIB:.1f} GiB "
        f"but {fraction:.0%} of {total_bytes / _GIB:.1f} GiB RAM is "
        f"{budget / _GIB:.1f} GiB; running {fits}."
    )


def _resolve_log_file(configured: str, output_dir: Path, session_name: str) -> Path:
    """Map LOG_FILE to a path; empty means the session's own log file."""
    if not configured:
        return output_dir / session_name / f"{session_name}.log"
    path = Path(configured)
    if not path.is_absolute():
        from transcript_combobulator.config import ROOT_DIR

        path = ROOT_DIR / path
    return path


def find_audio_files(target_dir: Path) -> list[Path]:
    """Find audio files in target directory, excluding converted files."""
    files = []
    for f in sorted(target_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(f)
    return files


def _worker(
    file_path: str,
    status_dict: "MutableMapping[str, str]",
    status_key: str,
    force: bool,
    log_file: str | None = None,
    session: str | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    """Worker function that runs in a subprocess.

    Returns:
        (filename, status, error_message_or_empty, per_file_metrics)
    """
    # The rich table owns the terminal. Logs go to the session log file when
    # one is configured; otherwise they are dropped as before. Attaching the
    # handler first also makes the later setup_logging() basicConfig a no-op,
    # so no stream handler is ever added in a worker.
    log_handler = None
    if log_file:
        log_handler = add_file_handler(
            Path(log_file), context={"audio_file": status_key}
        )
    else:
        logging.disable(logging.CRITICAL)

    # Also suppress stdout/stderr from submodules (tqdm, whisper prints)
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    file_metrics: dict[str, Any] = {}
    try:
        from transcript_combobulator.pipeline import process_file

        process_file(
            file_path,
            status_dict=status_dict,
            status_key=status_key,
            force=force,
            metrics=file_metrics,
            session=session,
        )
        result_status = "cached" if file_metrics.get("cache_hit") else "done"
        return (Path(file_path).name, result_status, "", file_metrics)
    except Exception as e:
        file_metrics.setdefault("file", Path(file_path).name)
        file_metrics.setdefault("status", "error")
        file_metrics.setdefault("error_type", type(e).__name__)
        file_metrics.setdefault("error", str(e))
        return (Path(file_path).name, "error", str(e), file_metrics)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        if log_handler is not None:
            remove_file_handler(log_handler)
        else:
            logging.disable(logging.NOTSET)


def _initialize_worker(torch_threads: int, worker_nice: int) -> None:
    """Apply process-wide scheduling and Torch settings once per worker."""
    if worker_nice:
        try:
            os.nice(worker_nice)
        except OSError:
            pass

    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
        except (RuntimeError, ValueError):
            pass


def _build_table(
    file_names: list[str],
    status_dict: "MutableMapping[str, str]",
    max_workers: int,
) -> Table:
    """Build a rich Table showing current progress."""
    table = Table(title=f"Transcription Progress ({max_workers} workers)")
    table.add_column("Speaker", style="cyan", min_width=24)
    table.add_column("Status", min_width=20)

    for name in file_names:
        raw_status = status_dict.get(name, "waiting")
        label, style = _status_display(raw_status)
        table.add_row(name, Text(label, style=style))

    return table


def _status_display(raw_status: str) -> tuple[str, str]:
    """Map a raw status string to a display label and style."""
    static = {
        "waiting": ("waiting", "dim"),
        "converting": ("converting", "yellow"),
        "splitting": ("splitting", "yellow"),
        "loading model": ("loading model", "blue"),
        "cached": ("\u2713 cached", "green"),
        "done": ("\u2713 done", "green"),
    }
    if raw_status in static:
        return static[raw_status]
    if raw_status.startswith("transcribing"):
        return (raw_status, "magenta")
    if raw_status.startswith("error"):
        return ("\u2717 error", "red bold")
    return (raw_status, "")


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _format_metric_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def _build_file_metrics_table(file_metrics: list[dict[str, Any]]) -> Table:
    """Build the post-run per-file timing summary."""
    table = Table(title="Per-file Pipeline Metrics")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Audio", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Convert", justify="right")
    table.add_column("VAD", justify="right")
    table.add_column("Model", justify="right")
    table.add_column("Inference", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("xRT", justify="right")

    for metrics in sorted(file_metrics, key=lambda item: str(item.get("file", ""))):
        stages = metrics.get("stages", {})
        transcription = metrics.get("transcription", {})
        inference_seconds = sum(
            float(chunk.get("timings", {}).get("inference_seconds", 0.0))
            for chunk in transcription.get("chunks", [])
        )
        audio_seconds = metrics.get("input_audio_seconds")
        total_seconds = metrics.get("total_seconds")
        x_realtime = (
            audio_seconds / total_seconds
            if isinstance(audio_seconds, (int, float))
            and isinstance(total_seconds, (int, float))
            and total_seconds
            else None
        )
        status = str(metrics.get("status", "unknown"))
        status_style = {
            "processed": "green",
            "cached": "green",
            "error": "red bold",
        }.get(status, "")
        table.add_row(
            str(metrics.get("file", "-")),
            Text(status, style=status_style),
            _format_duration(float(audio_seconds))
            if isinstance(audio_seconds, (int, float))
            else "-",
            _format_metric_seconds(total_seconds),
            _format_metric_seconds(stages.get("conversion_seconds")),
            _format_metric_seconds(stages.get("vad_seconds")),
            _format_metric_seconds(transcription.get("model_load_seconds")),
            _format_metric_seconds(inference_seconds),
            str(metrics.get("vad", {}).get("chunk_count", "-")),
            f"{x_realtime:.2f}x" if x_realtime is not None else "-",
        )
    return table


def _build_run_metrics_table(
    summary: dict[str, Any], metrics_path: Optional[Path]
) -> Table:
    """Build a compact aggregate summary for the terminal."""
    table = Table(title="Session Telemetry")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    file_status = (
        f"{summary['files_processed']} processed, "
        f"{summary['files_cached']} cached, {summary['files_failed']} failed"
    )
    table.add_row("Files", file_status)
    table.add_row("Input audio", _format_duration(summary["input_audio_seconds"]))
    table.add_row(
        "Processing wall time", _format_duration(summary["processing_seconds"])
    )
    table.add_row(
        "Audio throughput",
        f"{summary['audio_x_realtime']:.2f}x realtime"
        if summary["audio_x_realtime"] is not None
        else "-",
    )
    table.add_row(
        "Real-time factor",
        f"{summary['real_time_factor']:.4f}"
        if summary["real_time_factor"] is not None
        else "-",
    )
    table.add_row("VAD chunks", str(summary["vad_chunks"]))
    table.add_row("Transcript segments", str(summary["transcript_segments"]))
    table.add_row(
        "Inference throughput",
        f"{summary['inference_x_realtime']:.2f}x realtime"
        if summary["inference_x_realtime"] is not None
        else "-",
    )
    table.add_row(
        "Metrics report",
        str(metrics_path) if metrics_path is not None else "not written",
    )
    return table


def _publish_metrics_report(
    metrics_path: Path, report: dict[str, Any]
) -> tuple[Optional[Path], Optional[str]]:
    """Write telemetry and return the path only when publication succeeds."""
    try:
        write_metrics_report(metrics_path, report)
    except OSError as e:
        return None, str(e)
    return metrics_path, None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="combobulator run",
        description="Batch-process a session folder with live progress",
    )
    parser.add_argument("target_dir", type=str, help="Directory containing audio files")
    parser.add_argument(
        "--session", type=str, default=None, help="Session name for combine step"
    )
    parser.add_argument("--force", action="store_true", help="Ignore completed output")
    args = parser.parse_args(argv)

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    files = find_audio_files(target_dir)
    if not files:
        print(f"No audio files found in {target_dir}")
        sys.exit(1)

    # The stem names the output folder, so two inputs sharing one would race on it.
    stems: dict[str, Path] = {}
    for f in files:
        if (other := stems.setdefault(f.stem.casefold(), f)) is not f:
            print(
                f"{other.name} and {f.name} share a file stem (ignoring case); "
                "keep one per speaker"
            )
            sys.exit(1)

    # Load config (imports dotenv and captures the run profile once).
    from transcript_combobulator.config import (
        LOG_FILE,
        MEMORY_GUARD_FRACTION,
        OUTPUT_DIR,
        PADDING_SECONDS,
        PARALLEL_JOBS,
        SAMPLE_RATE,
        TORCH_THREADS,
        VAD_MIN_SILENCE_DURATION,
        VAD_MIN_SPEECH_DURATION,
        VAD_THRESHOLD,
        WHISPER_BEAM_SIZE,
        WHISPER_COMPRESSION_RATIO_THRESHOLD,
        WHISPER_CONDITION_ON_PREVIOUS,
        WHISPER_DEVICE,
        WHISPER_LANGUAGE,
        WHISPER_LOGPROB_THRESHOLD,
        WHISPER_MODEL,
        WHISPER_NO_SPEECH_THRESHOLD,
        WHISPER_PROMPT,
        WHISPER_RUNTIME,
        WHISPER_TEMPERATURE,
        WHISPER_WORD_TIMESTAMPS,
        WORKER_NICE,
        get_model_weights_path,
        get_whisper_options,
    )

    session_name = args.session or target_dir.name
    log_file = _resolve_log_file(LOG_FILE, OUTPUT_DIR, session_name)
    parent_log_handler = add_file_handler(log_file)
    run_logger = logging.getLogger("process_batch")

    # Fail on a speaker-mapping typo now, not after hours of transcription.
    from transcript_combobulator.combine import CombineError, validate_speaker_mapping

    try:
        validate_speaker_mapping(f.stem for f in files)
    except CombineError as e:
        print(f"Speaker mapping error: {e}")
        run_logger.error("Speaker mapping error: %s", e)
        sys.exit(1)

    max_workers = min(max(1, PARALLEL_JOBS), len(files))
    whisper_options = get_whisper_options()
    model_file = get_model_weights_path()
    model_bytes = model_file.stat().st_size if model_file.exists() else 0
    max_workers, memory_warning = _memory_capped_workers(
        max_workers, model_bytes, _total_memory_bytes(), MEMORY_GUARD_FRACTION
    )
    if memory_warning:
        print(memory_warning)
        run_logger.warning(memory_warning)
    torch_threads = _calculate_torch_threads(max_workers, TORCH_THREADS)

    file_names = [f.name for f in files]

    # Shared dict for cross-process status updates
    manager = multiprocessing.Manager()
    status_dict = manager.dict({name: "waiting" for name in file_names})

    print(f"Processing {len(files)} audio files in {target_dir}")
    print(f"Log file: {log_file}")
    run_logger.info(
        "Run start: %d files in %s, %d workers, %d torch threads each",
        len(files),
        target_dir,
        max_workers,
        torch_threads,
    )
    started_at = utc_now_iso()
    run_started = time.perf_counter()
    processing_started = time.perf_counter()

    errors: list[tuple[str, str]] = []
    completed_file_metrics: list[dict[str, Any]] = []

    # Install SIGINT/SIGTERM handlers that forcefully kill child processes.
    # Without this, ProcessPoolExecutor and Manager ignore the first Ctrl+C,
    # and a systemd/cron SIGTERM would orphan the spawned workers.
    executor_ref: ProcessPoolExecutor | None = None

    def _shutdown_handler(signum: int, frame: FrameType | None) -> NoReturn:
        label = (
            "Cancelled."
            if signum == signal.SIGINT
            else f"Terminated (signal {signum})."
        )
        print(f"\n{label}")
        run_logger.warning(label)
        if executor_ref is not None:
            for pid in executor_ref._processes:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        manager.shutdown()
        sys.exit(128 + signum)

    old_handlers = {
        sig: signal.signal(sig, _shutdown_handler)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        with Live(
            _build_table(file_names, status_dict, max_workers), refresh_per_second=4
        ) as live:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_worker,
                initargs=(torch_threads, WORKER_NICE),
            ) as executor:
                executor_ref = executor
                futures = {}
                for f in files:
                    fut = executor.submit(
                        _worker,
                        str(f),
                        status_dict,
                        f.name,
                        args.force,
                        str(log_file),
                        session_name,
                    )
                    futures[fut] = f.name

                # Poll for status updates instead of blocking on as_completed
                while True:
                    live.update(_build_table(file_names, status_dict, max_workers))

                    done_futures = [f for f in futures if f.done()]
                    for fut in done_futures:
                        if fut in futures:
                            try:
                                name, status, err_msg, file_metrics = fut.result()
                            except BrokenProcessPool as e:
                                # A worker died outside Python (the OOM killer
                                # on a Pi). Record every unfinished file and
                                # still write the metrics report and log.
                                for pending in list(futures.values()):
                                    completed_file_metrics.append(
                                        {
                                            "file": pending,
                                            "status": "error",
                                            "error_type": type(e).__name__,
                                            "error": str(e),
                                        }
                                    )
                                    status_dict[pending] = "error"
                                    errors.append((pending, f"worker died: {e}"))
                                    run_logger.error(
                                        "%s failed: worker died: %s", pending, e
                                    )
                                futures.clear()
                                break
                            completed_file_metrics.append(file_metrics)
                            if status == "error":
                                status_dict[name] = "error"
                                errors.append((name, err_msg))
                                run_logger.error("%s failed: %s", name, err_msg)
                            else:
                                run_logger.info("%s %s", name, status)
                            del futures[fut]

                    if not futures:
                        break

                    time.sleep(0.25)

                # Final refresh
                live.update(_build_table(file_names, status_dict, max_workers))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    processing_seconds = elapsed_seconds(processing_started, time.perf_counter())
    print(f"Processed {len(files)} files in {_format_duration(processing_seconds)}")

    metrics_path = OUTPUT_DIR / session_name / f"{session_name}-metrics.json"
    combine_metrics: dict[str, Any] = {
        "status": "skipped" if errors else "running",
        "seconds": 0.0,
        "output_files": [],
    }
    combine_error: Optional[str] = None

    if not errors:
        print("Combining transcripts...")
        from transcript_combobulator.combine import combine_transcripts_from_env
        from transcript_combobulator.config import vtt_path_for_input

        combine_started = time.perf_counter()
        try:
            # Only the transcripts this run produced (or reused from cache),
            # never whatever else is lying under the session directory.
            output_files = combine_transcripts_from_env(
                OUTPUT_DIR,
                session_name,
                vtt_files=[vtt_path_for_input(f, session_name) for f in files],
            )
            combine_metrics["status"] = "completed"
            combine_metrics["output_files"] = [str(path) for path in output_files]
            for path in output_files:
                print(f"Combined transcript: {path}")
        except Exception as e:
            combine_error = str(e)
            combine_metrics["status"] = "error"
            combine_metrics["error_type"] = type(e).__name__
            combine_metrics["error"] = combine_error
            run_logger.error("Combine failed: %s", combine_error)
        finally:
            combine_metrics["seconds"] = elapsed_seconds(
                combine_started, time.perf_counter()
            )

    wall_seconds = elapsed_seconds(run_started, time.perf_counter())
    completed_file_metrics.sort(key=lambda item: str(item.get("file", "")))
    summary = build_run_summary(
        completed_file_metrics,
        processing_seconds=processing_seconds,
        wall_seconds=wall_seconds,
        max_workers=max_workers,
    )
    run_status = "failed" if errors or combine_error else "completed"
    report = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "run": {
            "session": session_name,
            "status": run_status,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "target_dir": str(target_dir),
            "environment_file": os.environ.get("ENV_FILE") or ".env",
            "force": args.force,
        },
        "runtime": {
            "parallel_jobs_configured": PARALLEL_JOBS,
            "active_workers": max_workers,
            "torch_threads_per_worker": torch_threads,
            "worker_nice": WORKER_NICE,
        },
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "whisper": {
                "runtime": WHISPER_RUNTIME,
                "model": WHISPER_MODEL,
                "device": "metal" if WHISPER_RUNTIME == "mlx" else WHISPER_DEVICE,
                # The effective values: mlx always runs fp16 and has no prompt carry.
                "fp16": whisper_options["fp16"],
                "language": WHISPER_LANGUAGE,
                "temperature": WHISPER_TEMPERATURE,
                "beam_size": WHISPER_BEAM_SIZE,
                "word_timestamps": WHISPER_WORD_TIMESTAMPS,
                "condition_on_previous_text": WHISPER_CONDITION_ON_PREVIOUS,
                "carry_initial_prompt": bool(
                    whisper_options.get("carry_initial_prompt", False)
                ),
                "prompt_configured": bool(WHISPER_PROMPT),
                "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
                "logprob_threshold": WHISPER_LOGPROB_THRESHOLD,
                "compression_ratio_threshold": (WHISPER_COMPRESSION_RATIO_THRESHOLD),
            },
            "vad": {
                "threshold": VAD_THRESHOLD,
                "min_speech_duration": VAD_MIN_SPEECH_DURATION,
                "min_silence_duration": VAD_MIN_SILENCE_DURATION,
                "padding_seconds": PADDING_SECONDS,
            },
        },
        "summary": summary,
        "files": completed_file_metrics,
        "combine": combine_metrics,
    }
    published_metrics_path, metrics_error = _publish_metrics_report(
        metrics_path, report
    )
    if metrics_error:
        print(f"Metrics report error: {metrics_error}")

    console = Console()
    console.print(_build_file_metrics_table(completed_file_metrics))
    console.print(_build_run_metrics_table(summary, published_metrics_path))

    run_logger.info(
        "Run end: status=%s wall=%.1fs metrics=%s",
        run_status,
        wall_seconds,
        published_metrics_path or "not written",
    )
    remove_file_handler(parent_log_handler)

    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)
    if combine_error:
        print(f"Combine error: {combine_error}")
        sys.exit(1)
