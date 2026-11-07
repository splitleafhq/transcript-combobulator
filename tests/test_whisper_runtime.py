"""The WHISPER_RUNTIME seam: config, fingerprint, loading, and the mlx adapter.

Fast: mlx-whisper is faked, so none of this needs Apple Silicon or a model.
"""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcript_combobulator import config
from transcript_combobulator.pipeline_cache import (
    ChunkCheckpoint,
    build_stage_fingerprints,
    load_manifest,
    record_stage,
    stage_is_complete,
)
from transcript_combobulator.whisper import (
    WhisperError,
    _load_whisper_model,
    transcribe_segment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config_in_subprocess(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    full = {**os.environ, "PYTHONPATH": str(REPO_ROOT), **env}
    full.pop("PROJECT_ROOT", None)
    full.pop("ENV_FILE", None)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import transcript_combobulator.config as c; print(c.WHISPER_RUNTIME)",
        ],
        cwd=REPO_ROOT,
        env=full,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def mlx_runtime(tmp_path: Path):
    """Active mlx runtime with a fake checkpoint dir under tmp_path."""
    models = tmp_path / "models"
    ckpt = models / "mlx" / config.WHISPER_MODEL
    ckpt.mkdir(parents=True)
    (ckpt / config.MLX_WEIGHTS_FILENAME).write_bytes(b"weights")
    (ckpt / "config.json").write_text("{}")
    with (
        patch.object(config, "WHISPER_RUNTIME", "mlx"),
        patch.object(config, "WHISPER_MODELS_DIR", models),
        patch("transcript_combobulator.whisper.WHISPER_RUNTIME", "mlx"),
    ):
        _load_whisper_model.cache_clear()
        yield ckpt
    _load_whisper_model.cache_clear()


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch):
    """A stand-in for mlx.core and mlx_whisper that records how it is called."""
    holder = MagicMock(name="ModelHolder")
    mx = types.ModuleType("mlx.core")
    mx.float16 = "float16"  # type: ignore[attr-defined]
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx  # type: ignore[attr-defined]
    mw = types.ModuleType("mlx_whisper")
    mw.transcribe = MagicMock(  # type: ignore[attr-defined]
        return_value={
            "segments": [{"start": 0.0, "end": 1.0, "text": " hi", "avg_logprob": -0.1}]
        }
    )
    mwt = types.ModuleType("mlx_whisper.transcribe")
    mwt.ModelHolder = holder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_whisper", mw)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", mwt)
    return types.SimpleNamespace(holder=holder, transcribe=mw.transcribe)


# ── config ──


def test_default_runtime_is_openai() -> None:
    result = _config_in_subprocess({})
    assert result.returncode == 0
    assert result.stdout.strip() == "openai"


def test_unknown_runtime_is_an_error() -> None:
    result = _config_in_subprocess({"WHISPER_RUNTIME": "cuda"})
    assert result.returncode != 0
    assert "WHISPER_RUNTIME must be one of openai, mlx" in result.stderr


def test_mlx_rejects_beam_search() -> None:
    result = _config_in_subprocess({"WHISPER_RUNTIME": "mlx", "WHISPER_BEAM_SIZE": "5"})
    assert result.returncode != 0
    assert "no beam search" in result.stderr


def test_mlx_options_carry_only_what_mlx_whisper_honours(mlx_runtime: Path) -> None:
    options = config.get_whisper_options()
    assert "beam_size" not in options
    assert "carry_initial_prompt" not in options
    assert options["fp16"] is True
    assert options["condition_on_previous_text"] is config.WHISPER_CONDITION_ON_PREVIOUS
    assert options["initial_prompt"] == (config.WHISPER_PROMPT or None)


def test_openai_options_keep_their_original_keys() -> None:
    assert set(config.get_whisper_options()) == {
        "language",
        "temperature",
        "beam_size",
        "condition_on_previous_text",
        "carry_initial_prompt",
        "no_speech_threshold",
        "logprob_threshold",
        "compression_ratio_threshold",
        "word_timestamps",
        "initial_prompt",
        "fp16",
    }


# ── fingerprint ──


def test_openai_inference_fingerprint_keys_are_unchanged() -> None:
    """Sessions completed before the setting existed must stay valid."""
    inference = config.get_stage_fingerprint_settings()["inference"]
    assert set(inference) == {
        "whisper_model",
        "whisper_model_file",
        "whisper_device",
        "whisper_options",
    }


def test_runtime_is_part_of_the_inference_fingerprint(
    tmp_path: Path, mlx_runtime: Path
) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")

    mlx_fp = build_stage_fingerprints(source)
    with (
        patch.object(config, "WHISPER_RUNTIME", "openai"),
        patch("transcript_combobulator.whisper.WHISPER_RUNTIME", "openai"),
    ):
        openai_fp = build_stage_fingerprints(source)

    # Audio conversion and VAD do not depend on the runtime; inference and
    # everything after it do, so output cached by one runtime is never reused.
    assert mlx_fp["conversion"] == openai_fp["conversion"]
    assert mlx_fp["vad"] == openai_fp["vad"]
    assert mlx_fp["inference"] != openai_fp["inference"]
    assert mlx_fp["vtt"] != openai_fp["vtt"]


def test_another_runtimes_cached_output_is_never_reused(
    tmp_path: Path, mlx_runtime: Path
) -> None:
    """The user-visible guarantee: switch WHISPER_RUNTIME and both the finished
    inference record and the per-chunk checkpoint are ignored, not reused."""
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")
    artifact = tmp_path / "speaker_transcription.json"
    artifact.touch()
    manifest = tmp_path / "manifest.json"
    progress = tmp_path / "progress.jsonl"

    with patch.object(config, "WHISPER_RUNTIME", "openai"):
        openai_inference = build_stage_fingerprints(source)["inference"]
    mlx_inference = build_stage_fingerprints(source)["inference"]

    record_stage(manifest, "inference", openai_inference, [artifact])
    assert stage_is_complete(load_manifest(manifest), "inference", openai_inference)
    assert not stage_is_complete(load_manifest(manifest), "inference", mlx_inference)

    done = ChunkCheckpoint(progress, openai_inference)
    done.record(1, [{"start": 0.0, "end": 1.0, "text": "kept", "confidence": 90.0}], {})
    done.close(remove=False)
    assert ChunkCheckpoint(progress, openai_inference).completed.keys() == {1}
    assert ChunkCheckpoint(progress, mlx_inference).completed == {}


def test_mlx_fingerprint_follows_the_checkpoint_and_pin(
    tmp_path: Path, mlx_runtime: Path
) -> None:
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")
    base = build_stage_fingerprints(source)["inference"]

    (mlx_runtime / config.MLX_WEIGHTS_FILENAME).write_bytes(b"different weights")
    assert build_stage_fingerprints(source)["inference"] != base

    settings = config.get_stage_fingerprint_settings()["inference"]
    repo, revision = config.MLX_WHISPER_MODELS[config.WHISPER_MODEL]
    assert settings["whisper_runtime"] == "mlx"
    assert settings["whisper_model_repo"] == repo
    assert settings["whisper_model_revision"] == revision


def test_mlx_fingerprint_ignores_device_and_fp16_settings(
    tmp_path: Path, mlx_runtime: Path
) -> None:
    """WHISPER_DEVICE and WHISPER_FP16 do nothing under mlx; changing them
    must not throw away hours of cached inference."""
    source = tmp_path / "speaker.flac"
    source.write_bytes(b"audio")
    base = build_stage_fingerprints(source)["inference"]
    with (
        patch.object(config, "WHISPER_DEVICE", "mps"),
        patch.object(config, "WHISPER_FP16", True),
    ):
        assert build_stage_fingerprints(source)["inference"] == base


def test_memory_guard_sizes_the_runtimes_own_weights(mlx_runtime: Path) -> None:
    assert config.get_model_weights_path() == mlx_runtime / config.MLX_WEIGHTS_FILENAME
    with patch.object(config, "WHISPER_RUNTIME", "openai"):
        assert config.get_model_weights_path().name == f"{config.WHISPER_MODEL}.pt"


# ── loading and the adapter ──


def test_openai_runtime_still_loads_the_pt_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "test-model.pt").touch()
    sentinel = object()
    _load_whisper_model.cache_clear()
    try:
        with (
            patch("transcript_combobulator.whisper.WHISPER_MODELS_DIR", tmp_path),
            patch(
                "transcript_combobulator.whisper.get_whisper_device", return_value="cpu"
            ),
            patch(
                "transcript_combobulator.whisper.whisper.load_model",
                return_value=sentinel,
            ),
        ):
            assert _load_whisper_model("test-model") is sentinel
    finally:
        _load_whisper_model.cache_clear()


def test_mlx_runtime_loads_the_adapter_once_per_process(
    mlx_runtime: Path, fake_mlx
) -> None:
    from transcript_combobulator.whisper import load_whisper_model
    from transcript_combobulator.whisper_mlx import MlxWhisperModel

    first = load_whisper_model()
    second = load_whisper_model()
    assert isinstance(first, MlxWhisperModel)
    assert first is second
    fake_mlx.holder.get_model.assert_called_once_with(str(mlx_runtime), "float16")


def test_mlx_adapter_passes_options_through_and_returns_whisper_shape(
    mlx_runtime: Path, fake_mlx
) -> None:
    from transcript_combobulator.whisper_mlx import load_mlx_whisper_model

    model = load_mlx_whisper_model(mlx_runtime)
    audio = np.zeros(16000, dtype=np.float32)
    result = model.transcribe(audio, language="en", temperature=0.0, fp16=True)

    args, kwargs = fake_mlx.transcribe.call_args
    assert args[0] is audio
    assert kwargs["path_or_hf_repo"] == str(mlx_runtime)
    assert kwargs["verbose"] is None
    assert kwargs["language"] == "en" and kwargs["fp16"] is True
    assert result["segments"][0]["text"] == " hi"


def test_segment_transcription_runs_unchanged_on_the_mlx_adapter(
    mlx_runtime: Path, fake_mlx, tmp_path: Path
) -> None:
    from transcript_combobulator.whisper_mlx import load_mlx_whisper_model

    audio_path = tmp_path / "segment.wav"
    audio_path.touch()
    waveform = np.zeros(1600, dtype=np.float32)
    model = load_mlx_whisper_model(mlx_runtime)
    with patch(
        "transcript_combobulator.whisper.sf.read", return_value=(waveform, 16000)
    ):
        segments = transcribe_segment(audio_path, offset=10.0, model=model)

    assert segments[0]["start"] == 10.0 and segments[0]["end"] == 11.0
    assert segments[0]["text"] == "hi"
    assert segments[0]["confidence"] == pytest.approx(90.0)
    # the options handed to mlx-whisper are exactly the runtime's own set
    kwargs = fake_mlx.transcribe.call_args.kwargs
    assert "beam_size" not in kwargs and "carry_initial_prompt" not in kwargs


def test_mlx_failure_is_a_failure_not_an_empty_segment(
    mlx_runtime: Path, fake_mlx, tmp_path: Path
) -> None:
    """A chunk that raises must surface as WhisperError so the file fails and
    the cache cannot record a transcript with gaps."""
    from transcript_combobulator.whisper_mlx import load_mlx_whisper_model

    audio_path = tmp_path / "segment.wav"
    audio_path.touch()
    fake_mlx.transcribe.side_effect = RuntimeError("metal command buffer failed")
    model = load_mlx_whisper_model(mlx_runtime)
    with (
        patch(
            "transcript_combobulator.whisper.sf.read",
            return_value=(np.zeros(1600, dtype=np.float32), 16000),
        ),
        pytest.raises(WhisperError, match="metal command buffer failed"),
    ):
        transcribe_segment(audio_path, model=model)


def test_missing_mlx_checkpoint_says_how_to_get_it(tmp_path: Path) -> None:
    from transcript_combobulator.whisper_mlx import load_mlx_whisper_model

    with pytest.raises(WhisperError, match="make setup-whisper"):
        load_mlx_whisper_model(tmp_path / "nothing-here")


def test_missing_mlx_package_says_how_to_install_it(
    mlx_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcript_combobulator.whisper_mlx import load_mlx_whisper_model

    monkeypatch.setitem(sys.modules, "mlx", None)  # import raises ImportError
    with pytest.raises(WhisperError, match="EXTRAS=dev,mac"):
        load_mlx_whisper_model(mlx_runtime)


# ── model setup ──


def test_setup_mlx_downloads_the_pinned_revision(tmp_path: Path) -> None:
    from transcript_combobulator import model_setup

    target = tmp_path / "mlx" / "large-v3-turbo"
    (target).mkdir(parents=True)
    (target / config.MLX_WEIGHTS_FILENAME).write_bytes(b"w" * 10)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = MagicMock()  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        assert model_setup.setup_mlx_whisper("large-v3-turbo", target) is True
    repo, revision = config.MLX_WHISPER_MODELS["large-v3-turbo"]
    fake_hub.snapshot_download.assert_called_once_with(  # type: ignore[attr-defined]
        repo_id=repo, revision=revision, local_dir=str(target)
    )


def test_setup_mlx_refuses_a_model_with_no_pinned_checkpoint(tmp_path: Path) -> None:
    from transcript_combobulator import model_setup

    assert model_setup.setup_mlx_whisper("small.en", tmp_path) is False


# ── real inference (slow) ──


@pytest.mark.mlx
def test_mlx_runtime_transcribes_the_jfk_sample() -> None:
    """End to end on the real checkpoint and the Apple GPU. Opt-in: make test-mlx."""
    if importlib.util.find_spec("mlx_whisper") is None:
        pytest.fail("mlx_whisper is not installed: make install EXTRAS=dev,mac")
    checkpoint = config.WHISPER_MODELS_DIR / "mlx" / config.WHISPER_MODEL
    if not (checkpoint / config.MLX_WEIGHTS_FILENAME).is_file():
        pytest.fail("run `WHISPER_RUNTIME=mlx combobulator setup-model` first")
    code = (
        "from pathlib import Path\n"
        "from transcript_combobulator.whisper import transcribe_segment\n"
        "from transcript_combobulator.config import INPUT_DIR\n"
        "segs = transcribe_segment(INPUT_DIR / 'test_jfk.wav')\n"
        "print(' '.join(s['text'] for s in segs).lower())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "WHISPER_RUNTIME": "mlx"}
    env.pop("PROJECT_ROOT", None)
    env.pop("ENV_FILE", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ask not what your country can do for you" in result.stdout
