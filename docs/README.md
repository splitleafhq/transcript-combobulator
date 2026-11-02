# Transcript Combobulator

Audio transcription tool for multi-speaker recordings with separate audio files per speaker.

## What It Does

Processes separate audio files and creates organized transcripts:
- **Individual transcripts** - One VTT file per audio file/speaker
- **Combined session transcripts** - Chronological conversation flow across all speakers
- **Speaker mapping** - You configure which audio files map to which speakers

**Requires**: Separate audio files per speaker (tested with Craig Discord bot output)

## Setup

### Prerequisites

| Requirement | Debian / Raspberry Pi OS | macOS |
|---|---|---|
| Python 3.11+ with `venv` | `sudo apt install python3 python3-venv` | `brew install python` (or pyenv, python.org) |
| ffmpeg (audio decoding) | `sudo apt install ffmpeg` | `brew install ffmpeg` |

Debian bookworm ships Python 3.11 and trixie ships 3.13; both work. The Mac
uses 3.14 through pyenv (`.python-version`). No version manager is required. `make check-deps` reports what is missing.

### Install

```sh
git clone https://github.com/seriouslysean/transcript-combobulator.git
cd transcript-combobulator
make setup
```

`make setup` verifies prerequisites, creates `.venv` from `python3` on your
`PATH`, installs pinned dependencies, and downloads the Whisper model
(`WHISPER_MODEL`, ~1.6 GB for the default `large-v3-turbo`). To use a specific
interpreter:

```sh
make setup PYTHON=/usr/bin/python3.12
```

pyenv users: the committed `.python-version` selects 3.14 through the
pyenv shim (`pyenv install 3.14` once).

Apple Silicon can run Whisper on the GPU instead of the CPU. `WHISPER_DEVICE=mps`
needs nothing extra. `WHISPER_RUNTIME=mlx` is faster still: `make install
EXTRAS=dev,mac`, then `WHISPER_RUNTIME=mlx make setup-whisper` (downloads the
pinned mlx checkpoint, about 1.5 GB, to `models/mlx/`), and set
`WHISPER_RUNTIME=mlx` in the campaign env. See AGENTS.md for the measurements and
the differences from stock whisper. The default stays `openai` on the CPU.

Low-memory hosts (for example an 8 GB Raspberry Pi) should set
`PARALLEL_JOBS=1` and a smaller `WHISPER_MODEL` in `.env`; each worker holds a
full copy of the model in RAM. `models/` can be pre-seeded by copying the `.pt`
file from another machine instead of downloading.

## Quick Start

1. **Configure speaker mappings**:
   ```sh
   cp .env.example .env
   # Edit .env to map your audio files to speakers
   ```

2. **Add audio files** (one per speaker):
   ```sh
   mkdir -p tmp/input/my-session
   cp /path/to/craig-output/*.flac tmp/input/my-session/
   ```

3. **Run transcription**:
   ```sh
   make run folder=tmp/input/my-session
   ```

4. **Find results**:
   - Individual transcripts: `tmp/output/my-session/{speaker}/{speaker}.vtt`
   - Combined transcripts: `tmp/output/my-session/my-session-combined-*.txt`

## Configuration

Speaker mapping in `.env`:
```sh
# Map audio files to speakers
TRANSCRIPT_1_USERNAME=dm              # From filename 1-dm.flac
TRANSCRIPT_1_NAME="DM"               # Display name
TRANSCRIPT_1_LABEL="DM"              # Speaker label in transcript
TRANSCRIPT_1_DESCRIPTION="Dungeon Master"

TRANSCRIPT_2_USERNAME=barbarian       # From filename 2-barbarian.flac
TRANSCRIPT_2_NAME="Player 1"
TRANSCRIPT_2_LABEL="Barbarian"
TRANSCRIPT_2_DESCRIPTION="Goliath Barbarian"
```

Put domain vocabulary and proper nouns in the prompt. The default setup
reapplies it to every internal Whisper window, including speech islands longer
than 30 seconds:

```sh
WHISPER_PROMPT="Conversation mentioning LOCATION_NAME and CHARACTER_NAME."
WHISPER_CARRY_INITIAL_PROMPT=true
VAD_MIN_SPEECH_DURATION=0.25
```

## Supported Audio Formats

- **FLAC** (Craig Discord bot default)
- **WAV, MP3, M4A, OGG, AAC, OPUS** (auto-converted to 16kHz WAV)

## Commands

```sh
make help                               # Every target with a one-line description

# Setup
make check-deps                         # Verify Python 3.11+, venv, and ffmpeg
make setup                              # Create .venv, install, download the Whisper model
make install EXTRAS=dev,mac             # Reinstall with extras (mac adds mlx-whisper)

# Processing
make run folder=tmp/input/session-name  # Full pipeline for one session
make run folder=path/to/session force=1 # Reprocess completed files
make run-single file=path/to/file.flac  # One file, no combine

# Post-processing
make combine-transcripts session=name   # Re-merge per-speaker VTTs
make filter-vtt file=path/to/file.flac threshold=60  # Re-emit a VTT above a confidence, no inference
make compare-transcripts ref="old-1.txt old-2.txt" hyp="new-1.txt new-2.txt"  # Per-speaker differences between two runs

# Dev
make test-fast                          # Suite without real inference (seconds)
make test                               # Whole suite (minutes)
make check-sample                       # Transcribe the 4-speaker AMI sample; print WER against AMI's human transcript; fail if any word changed (samples/ami/NOTICE.md)
make lint                               # ruff check and format check
make format                             # ruff format
make typecheck                          # mypy --strict
lefthook install                        # Run lint, typecheck and test-fast before every push
make clean-output                       # Delete every session's outputs under tmp/output
```

### Running unattended

Batch runs validate the speaker mapping before transcribing, cap the worker
count to what fits in RAM, write a session log, and exit non-zero on any
failure. A cron job or systemd unit calls the installed console script from any
working directory, no `make`, no `PYTHONPATH`:

```sh
ENV_FILE=/path/to/.env.campaign /path/to/repo/.venv/bin/combobulator run /srv/craig/2026-09-20 --session 2026-09-20
```

`combobulator --help` lists the other commands (`file`, `combine`,
`filter-vtt`, `session-env`, `compare`, `setup-model`, `samples`); every Makefile target wraps one.
`session-env` writes a session's own env from a recording bundle
([recording-bundle.md](recording-bundle.md)) and a speakers file.
Defaults in `.env`:

```sh
MEMORY_GUARD_FRACTION=0.85   # share of physical RAM the workers may use
LOG_FILE=                    # empty = tmp/output/<session>/<session>.log
```

Always on, no switch: the speaker mapping is validated before any inference,
the worker count is capped to what fits in RAM, a file with any failed chunk
fails so a rerun retries it, and a track with no speech yields an empty
transcript instead of failing the session.

Whisper's own fallback re-decode is off by default (`WHISPER_TEMPERATURE=0.0`).
`WHISPER_TEMPERATURE=0.0,0.2,0.4` turns it on. Measured on a real session it
re-decoded 6% of segments at 1.29x the inference cost and changed short
ambiguous fragments to different guesses rather than better ones, so the
default stays scalar. Each saved segment records the temperature whisper used.

Transcript fidelity knobs, also with defaults shown:

```sh
DEDUPE_WINDOW_SECONDS=2.0            # a repeat within this gap is a whisper hallucination
```

Transcripts produced before this rule existed were deduplicated across
the whole session, so every repeated short line from a speaker was dropped,
and every cue was 0.3 s late. The resume cache version was bumped, so the next
`make run` on an old session reprocesses it.

VAD streams the audio from disk (256 MB peak on a 2.9 h track) using Silero's
ONNX build.

Interrupted runs resume. Conversion, VAD, and transcription each record their
own completion, and transcription checkpoints every chunk, so a run killed at
chunk 300 of 346 continues from chunk 301 on the next `make run`. Changing a
setting re-runs only the stages it affects; changing the dedup rule rewrites
the VTTs without any inference. `force=1` ignores all of it.

Paths are anchored to the repository, not the working directory, so the tools
behave the same when invoked from elsewhere. Set `PROJECT_ROOT` in the shell
environment to override.

### Run telemetry

Every batch run ends with per-file and session timing tables and writes a
machine-readable report beside the combined transcript:

```text
tmp/output/<session>/<session>-metrics.json
```

The report includes total audio and wall time, real-time factor, worker
utilization, cache hits, conversion/VAD/transcription stage durations, model
load reuse, and every VAD chunk's audio decode, Whisper inference, and result
processing time. Persisted files use opaque file IDs and exclude source names,
paths, environment filenames, raw error messages, transcript text, and prompt
content.

## Example Output

Combined transcript format:
```
Summary:
DM - DM - Dungeon Master
Player 1 - Barbarian - Goliath Barbarian
Player 2 - Druid - Human Druid

TRANSCRIPT:
DM: The wind howls through the ruined village.
Barbarian: That's a 16 on my save.
Druid: I cast Detect Magic, just in case.
...
```

## Troubleshooting

**"No mapping found for directories"**: Update `TRANSCRIPT_*_USERNAME` in your .env file to match your audio filenames.

**Need different settings for different sessions?**: Use `ENV_FILE=.env.session2 make run`
