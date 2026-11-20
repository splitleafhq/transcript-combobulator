"""Streaming VAD must equal silero's whole-file result exactly."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps

from transcript_combobulator.config import (
    SAMPLE_RATE,
    VAD_MIN_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_THRESHOLD,
)
from transcript_combobulator.vad import (
    _stream_speech_probs,
    detect_speech,
    load_vad_model,
    process_audio,
)


def _composite(tmp_path: Path) -> Path:
    """Speech islands with gaps around every threshold the region logic has."""
    speech, sr = sf.read("samples/jfk.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    def z(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

    parts = [
        speech,
        z(0.1),
        speech[: int(0.15 * SAMPLE_RATE)],
        z(0.2),
        speech,
        z(1.0),
        speech,
        z(VAD_MIN_SILENCE_DURATION + 0.5),
        speech,
        z(0.05),
        speech[: int(2.0 * SAMPLE_RATE)],
        z(7.3),
        speech,
        z(0.03),
        z(4.0),
        speech[: int(1.5 * SAMPLE_RATE)],
        z(3.2),
        speech[: int(0.5 * SAMPLE_RATE)],
    ]
    audio = np.concatenate(parts)
    # make the total not a multiple of 512 so the padded last frame is exercised
    audio = audio[: len(audio) - (len(audio) % 512) - 137]
    path = tmp_path / "composite.wav"
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def _reference(path: Path) -> list[dict]:
    audio, _ = sf.read(str(path), dtype="float32")
    return get_speech_timestamps(
        torch.from_numpy(audio),
        load_vad_model(),
        return_seconds=True,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
    )


def test_streamed_probabilities_match_whole_file(tmp_path: Path) -> None:
    path = _composite(tmp_path)
    model = load_vad_model()
    audio, _ = sf.read(str(path), dtype="float32")
    model.reset_states()
    expected = []
    for start in range(0, len(audio), 512):
        frame = torch.from_numpy(audio[start : start + 512])
        if len(frame) < 512:
            frame = torch.nn.functional.pad(frame, (0, 512 - len(frame)))
        expected.append(float(model(frame, SAMPLE_RATE).item()))

    for block in (10_000, 60 * SAMPLE_RATE):  # non-multiple and multiple of 512
        probs, total = _stream_speech_probs(path, model, block_samples=block)
        assert total == len(audio)
        assert probs == expected


def test_detect_speech_matches_silero(tmp_path: Path) -> None:
    path = _composite(tmp_path)
    expected = _reference(path)
    assert len(expected) >= 4  # the composite must actually exercise the logic
    for block in (10_000, 60 * SAMPLE_RATE):
        assert detect_speech(path, load_vad_model(), block_samples=block) == expected


def test_process_audio_segments_match_silero_regions(tmp_path: Path) -> None:
    path = _composite(tmp_path)
    expected = _reference(path)
    _, segments = process_audio(path)
    assert [(s["start_seconds"], s["end_seconds"]) for s in segments] == [
        (r["start"], r["end"]) for r in expected
    ]
    total = sf.info(str(path)).frames
    for seg in segments:
        clip_start = int(round(seg["clip_start_seconds"] * SAMPLE_RATE))
        clip_end = int(round(seg["clip_end_seconds"] * SAMPLE_RATE))
        assert 0 <= clip_start < clip_end <= total
        assert sf.info(seg["segment_file"]).frames == clip_end - clip_start
    mapping = json.loads((tmp_path / "composite_mapping.json").read_text())
    assert len(mapping["segments"]) == len(expected)
