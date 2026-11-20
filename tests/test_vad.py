#!/usr/bin/env python3
"""Test Voice Activity Detection."""

import json
from pathlib import Path
from unittest.mock import patch

from transcript_combobulator.config import INPUT_DIR
from transcript_combobulator.vad import load_vad_model, process_audio


def test_vad_model_loading():
    """Test that the VAD model loads successfully."""
    model = load_vad_model()
    assert model is not None, "VAD model is None"


def test_vad_model_is_cached_per_process():
    """Long-lived batch workers reuse one Silero model across files."""
    expected_model = object()
    load_vad_model.cache_clear()

    try:
        with patch(
            "transcript_combobulator.vad.load_silero_vad", return_value=expected_model
        ) as load:
            first = load_vad_model()
            second = load_vad_model()

        assert first is expected_model
        assert second is expected_model
        load.assert_called_once()
    finally:
        load_vad_model.cache_clear()


def test_vad_detection():
    """Test that VAD detects speech segments in the test JFK file."""
    input_file = INPUT_DIR / "test_jfk.wav"
    assert input_file.exists(), "Test JFK file not found"

    # Process the audio
    output_path, segments = process_audio(input_file)

    # Verify output files
    assert output_path.exists(), "Output directory not created"
    mapping_file = output_path / f"{input_file.stem}_mapping.json"
    assert mapping_file.exists(), "Mapping file not created"

    # Verify mapping content
    with open(mapping_file) as f:
        mapping = json.load(f)
        assert "segments" in mapping, "No segments in mapping"
        assert len(mapping["segments"]) > 0, "Empty segments in mapping"

        # Verify segment structure
        for segment in mapping["segments"]:
            assert "start_seconds" in segment, "Missing start time"
            assert "end_seconds" in segment, "Missing end time"
            assert "segment_file" in segment, "Missing segment file"
            assert segment["end_seconds"] > segment["start_seconds"], (
                "Invalid segment timing"
            )
            assert Path(segment["segment_file"]).exists(), "Segment file not created"
