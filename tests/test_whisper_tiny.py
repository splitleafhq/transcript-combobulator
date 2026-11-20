#!/usr/bin/env python3
"""Smoke test: the real whisper path transcribes speech with the tiny model."""

from unittest.mock import patch

import pytest

from transcript_combobulator import config
from transcript_combobulator.config import INPUT_DIR, WHISPER_MODELS_DIR
from transcript_combobulator.whisper import (
    _load_whisper_model,
    load_whisper_model,
    transcribe_segment,
)

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
            "combobulator setup-model tiny (make setup-tiny)"
        )

    # Stock openai runtime whatever .env says (the loader and the options each
    # read it from their own module).
    _load_whisper_model.cache_clear()
    try:
        with (
            patch.object(config, "WHISPER_RUNTIME", "openai"),
            patch("transcript_combobulator.whisper.WHISPER_RUNTIME", "openai"),
        ):
            model = load_whisper_model("tiny")
            segments = transcribe_segment(INPUT_DIR / "test_jfk.wav", model=model)
    finally:
        _load_whisper_model.cache_clear()

    # tiny has 4 encoder layers; large-v3-turbo has 32. A loader that ignored
    # its argument and loaded the default model would fail here.
    assert model.dims.n_audio_layer == 4, "the loaded model is not tiny"

    text = " ".join(segment["text"] for segment in segments).lower()
    assert text.strip(), "tiny transcribed no text from the JFK clip"
    for keyword in ("americans", "country"):
        assert keyword in text, f"{keyword!r} not in transcript: {text!r}"
