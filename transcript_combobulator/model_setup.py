#!/usr/bin/env python3
"""Download the configured whisper model into models/ without loading it.
A model name argument fetches that openai checkpoint instead (tests use tiny).

whisper.load_model would also instantiate the fp32 model (about 3x the file
size in RAM); on a Pi that is most of the memory for a step that only needs
the bytes on disk. This uses whisper's own downloader, which also verifies
the checksum and skips the download when the file already matches.

With WHISPER_RUNTIME=mlx it fetches the pinned mlx checkpoint from Hugging
Face into models/mlx/<WHISPER_MODEL>/ instead (huggingface_hub verifies the
files against the repo's hashes).
"""

import argparse
import sys
from pathlib import Path

import whisper

from transcript_combobulator.config import (
    MLX_WEIGHTS_FILENAME,
    MLX_WHISPER_MODELS,
    WHISPER_MODEL,
    WHISPER_MODELS_DIR,
    WHISPER_RUNTIME,
    mlx_model_dir,
)


def setup_whisper(model_name: str, models_dir: Path) -> bool:
    """Ensure models_dir/<model_name>.pt exists and matches whisper's checksum."""
    url = whisper._MODELS.get(model_name)
    if url is None:
        print(
            f"Unknown whisper model {model_name!r}. Available: {', '.join(whisper.available_models())}"
        )
        return False
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.pt"
        if model_path.exists():
            print(f"Verifying {model_path}...")
        else:
            print(f"Downloading {model_name} to {models_dir}...")
        whisper._download(url, str(models_dir), in_memory=False)
        print(
            f"Model ready: {model_path} ({model_path.stat().st_size / 2**20:.0f} MiB)"
        )
        return True
    except Exception as e:
        print(f"Error setting up whisper: {e}")
        return False


def setup_mlx_whisper(model_name: str, target_dir: Path) -> bool:
    """Ensure the pinned mlx checkpoint for ``model_name`` is in ``target_dir``."""
    pinned = MLX_WHISPER_MODELS.get(model_name)
    if pinned is None:
        print(
            f"No mlx checkpoint is pinned for {model_name!r}. "
            f"Available: {', '.join(sorted(MLX_WHISPER_MODELS))}"
        )
        return False
    repo, revision = pinned
    try:
        from huggingface_hub import snapshot_download

        print(f"Fetching {repo}@{revision[:10]} into {target_dir}...")
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target_dir))
        weights = target_dir / MLX_WEIGHTS_FILENAME
        print(f"Model ready: {weights} ({weights.stat().st_size / 2**20:.0f} MiB)")
        return True
    except Exception as e:
        print(f"Error setting up mlx whisper: {e}")
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="combobulator setup-model",
        description="Download WHISPER_MODEL into models/ without loading it (checksum-verified).",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model to fetch instead of WHISPER_MODEL; always the openai (torch) checkpoint",
    )
    opts = parser.parse_args(argv)
    if opts.model:
        # A named model ignores WHISPER_MODEL and WHISPER_RUNTIME (and .env's).
        if not setup_whisper(opts.model, WHISPER_MODELS_DIR):
            sys.exit(1)
        return
    if not WHISPER_MODEL:
        print("Error: WHISPER_MODEL is not set")
        sys.exit(1)
    if WHISPER_RUNTIME == "mlx":
        ok = setup_mlx_whisper(WHISPER_MODEL, mlx_model_dir())
    else:
        ok = setup_whisper(WHISPER_MODEL, WHISPER_MODELS_DIR)
    if not ok:
        sys.exit(1)
