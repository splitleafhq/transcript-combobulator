#!/usr/bin/env python3
"""Smoke test: the real whisper path transcribes speech with the tiny model."""

import pytest

from transcript_combobulator.config import INPUT_DIR, WHISPER_MODELS_DIR
from transcript_combobulator.whisper import load_whisper_model, transcribe_segment

TINY_MODEL = WHISPER_MODELS_DIR / "tiny.pt"


def test_tiny_model_transcribes_jfk_speech():
    """Fails when a torch, torchaudio or openai-whisper update breaks inference.

    Regression: after the update, transcription no longer runs, or it returns
    empty or garbage text. The tiny model keeps this cheap enough for CI; it
    checks keywords, not exact text, because tiny's wording and numerics drift
    across torch versions. The large-model tests cover accuracy locally.
    """
    if not TINY_MODEL.exists():
        pytest.fail(
            f"{TINY_MODEL} is missing. Download it first: "
            "WHISPER_MODEL=tiny make setup-whisper"
        )

    model = load_whisper_model("tiny")
    segments = transcribe_segment(INPUT_DIR / "test_jfk.wav", model=model)

    text = " ".join(segment["text"] for segment in segments).lower()
    assert text.strip(), "tiny transcribed no text from the JFK clip"
    for keyword in ("americans", "country"):
        assert keyword in text, f"{keyword!r} not in transcript: {text!r}"
