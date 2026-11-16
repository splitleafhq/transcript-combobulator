"""Audio format validation and conversion to 16kHz mono WAV."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from transcript_combobulator.logging_config import get_logger

logger = get_logger(__name__)

# Streaming block size for the two-pass peak normalisation (samples).
_NORMALIZE_BLOCK = 16000 * 60

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""


def validate_audio_file(file_path: Path) -> dict[str, Any]:
    """Return {sample_rate, channels, duration, format, frames} for an audio file."""
    if not file_path.exists():
        raise AudioValidationError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format: {file_path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        with sf.SoundFile(str(file_path)) as f:
            return {
                "sample_rate": f.samplerate,
                "channels": f.channels,
                "duration": f.frames / f.samplerate,
                "format": file_path.suffix.lower(),
                "frames": f.frames,
            }
    except Exception as e:
        raise AudioValidationError(f"Failed to read audio file {file_path}: {e}") from e


def convert_to_wav(
    input_path: Path, output_path: Path, target_sample_rate: int = 16000
) -> None:
    """Convert any supported format to 16kHz mono WAV, normalized to [-1, 1].

    Skips the work if output already exists with the right format.
    """
    if output_path.exists():
        try:
            info = validate_audio_file(output_path)
            if (
                info["sample_rate"] == target_sample_rate
                and info["format"] == ".wav"
                and info["channels"] == 1
            ):
                logger.info(f"Using existing converted file: {output_path}")
                return
        except AudioValidationError:
            logger.info(f"Existing file {output_path} is invalid, reconverting...")

    try:
        duration = input_duration(input_path)
        logger.info(f"Converting {input_path} ({duration:.2f}s) to WAV")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _convert_with_ffmpeg(
            input_path,
            output_path,
            target_sample_rate,
            timeout_seconds=max(600.0, 10.0 * duration),
        )

        info = validate_audio_file(output_path)
        logger.info(
            f"Conversion successful: {output_path} "
            f"(SR: {info['sample_rate']}, Duration: {info['duration']:.2f}s)"
        )
    except Exception as e:
        raise AudioValidationError(f"Failed to convert {input_path} to WAV: {e}") from e


def input_duration(input_path: Path) -> float:
    """Duration of a supported input in seconds, read by ffprobe if libsndfile can't.

    libsndfile has no AAC/MP4 decoder, so .m4a and .aac only open in ffmpeg.
    """
    if not input_path.exists() or input_path.suffix.lower() not in SUPPORTED_FORMATS:
        return float(validate_audio_file(input_path)["duration"])
    try:
        with sf.SoundFile(str(input_path)) as f:
            return float(f.frames / f.samplerate)
    except sf.LibsndfileError:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise AudioValidationError(
            f"ffprobe not found on PATH; it is needed to read {input_path.suffix} "
            "(it ships with ffmpeg)"
        )
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as e:
        raise AudioValidationError(f"ffprobe could not read {input_path}: {e}") from e


def _convert_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    timeout_seconds: float = 600.0,
) -> None:
    """Downmix + resample with ffmpeg, then peak-normalise in streaming blocks.

    ffmpeg decodes and resamples in a bounded buffer instead of loading the
    whole file (41 MB peak for a 3 h track versus 8.8 GB in memory). The
    normalisation divides by the post-downmix, post-resample absolute peak
    (plus 1e-8), which is the project's "normalized once" contract.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioValidationError(
            "ffmpeg not found on PATH (Debian: apt install ffmpeg; macOS: brew install ffmpeg)"
        )
    raw_path = output_path.with_name(f"{output_path.stem}.unnormalized.wav")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                "-c:a",
                "pcm_f32le",
                str(raw_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as e:
        raw_path.unlink(missing_ok=True)
        raise AudioValidationError(f"ffmpeg failed: {e.stderr.strip() or e}") from e
    except subprocess.TimeoutExpired as e:
        # A corrupt Craig file can wedge the decoder; never hang the worker.
        raw_path.unlink(missing_ok=True)
        raise AudioValidationError(
            f"ffmpeg exceeded {timeout_seconds:.0f}s converting {input_path.name}"
        ) from e

    try:
        peak = 0.0
        for block in sf.blocks(
            str(raw_path), blocksize=_NORMALIZE_BLOCK, dtype="float32"
        ):
            if block.size:
                peak = max(peak, float(np.abs(block).max()))
        scale = 1.0 / (peak + 1e-8)
        with sf.SoundFile(
            str(output_path),
            "w",
            samplerate=target_sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as out:
            for block in sf.blocks(
                str(raw_path), blocksize=_NORMALIZE_BLOCK, dtype="float32"
            ):
                out.write(block * scale)
    finally:
        raw_path.unlink(missing_ok=True)
    logger.info(
        f"Converted to mono {target_sample_rate}Hz, peak {peak:.4f} normalised to 1.0"
    )


def get_audio_info_summary(file_path: Path) -> str:
    """Human-readable one-liner with format, sample rate, channels, duration."""
    try:
        info = validate_audio_file(file_path)
        return (
            f"{file_path.name}: {info['format'].upper()}, "
            f"{info['sample_rate']}Hz, {info['channels']} channel(s), "
            f"{info['duration']:.2f}s"
        )
    except AudioValidationError as e:
        return f"{file_path.name}: ERROR - {e}"
