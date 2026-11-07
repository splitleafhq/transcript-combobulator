# Agent Development Guidelines

Concrete rules for working on this codebase. If you're Claude Code or another
coding agent, read this first.

## Module Layout

```
transcript_combobulator/
├── cli.py              # `combobulator <command>`: run, file, combine, filter-vtt, compare, setup-model, samples
├── __main__.py         # `python -m transcript_combobulator` → cli.main
├── batch.py            # Parallel session runner with rich progress UI (`run`)
├── pipeline.py         # Per-file pipeline: convert → VAD → transcribe, stage-level resume (`file`)
├── config.py           # ENV_FILE-aware settings and path anchoring; imported by everything
├── audio_utils.py      # Validate + convert any audio to 16kHz mono WAV (ffmpeg, streaming)
├── vad.py              # Silero VAD (streamed, ONNX) → segment WAVs + mapping JSON
├── whisper.py          # Whisper model wrapping; VTT writing; dedup; repetition filter
├── whisper_mlx.py      # WHISPER_RUNTIME=mlx: mlx-whisper (Apple GPU) behind whisper's transcribe()
├── transcribe.py       # Mapping JSON → per-segment whisper → VTT + JSON
├── pipeline_cache.py   # Per-stage completion records and per-chunk checkpoints
├── telemetry.py        # Per-run, per-file, and per-chunk timing reports
├── combine.py          # Merge per-speaker VTTs → one session transcript
├── vtt_filter.py       # Re-emit a speaker VTT from saved JSON above a confidence
├── compare.py          # Two combined transcripts of one session, per speaker: words different, lost spans, artefacts; `--wer` scores against a human reference
├── bundle.py           # Read + verify a recording bundle (docs/recording-bundle.md)
├── session_env.py      # Bundle + speakers file → one session's env (`session-env`)
├── samples.py          # Build sample/test audio from samples/; `--ami-human` rebuilds the AMI human reference
├── model_setup.py      # Download whisper model into models/ (no load)
└── logging_config.py   # setup_logging / file handlers
```

`pip install -e .` installs the `combobulator` console script; the Makefile
targets are thin wrappers around it. Automation calls the script directly:
`ENV_FILE=.env.<campaign> combobulator run /path/to/session --session <name>`,
from any working directory, with no `make` and no `PYTHONPATH`.

The standard pipeline: `transcript_combobulator.batch.main` →
`transcript_combobulator.pipeline.process_file` → `transcript_combobulator.audio_utils.convert_to_wav` →
`transcript_combobulator.vad.process_audio` (writes segment WAVs + `<stem>_mapping.json`) →
`transcript_combobulator.transcribe.transcribe_segments` → `transcript_combobulator.whisper.transcribe_audio_segments`
→ `transcript_combobulator.combine.combine_transcripts_from_env`.

## Invariants

- **One place for env loading.** `transcript_combobulator/config.py` calls `load_dotenv` at import.
  Do NOT add `load_dotenv` anywhere else.
- **Audio is normalized once.** `transcript_combobulator.audio_utils.convert_to_wav` normalizes to
  `[-1, 1]`. Don't re-normalize downstream.
- **Mapping JSON is written once.** `transcript_combobulator.vad.process_audio` is the only writer.
  `transcript_combobulator.transcribe.transcribe_audio` and the VAD-cached path in
  `transcript_combobulator/pipeline.py` trust it and read it back.
- **Only one `transcribe_audio` public function.** It lives in
  `transcript_combobulator.transcribe`. `transcript_combobulator.whisper.transcribe_file_direct` is the one-shot
  (no-VAD) variant and is internal to the regenerate-VTT flow.
- **Combine preserves original text.** `_normalize_for_dedup` is used ONLY for
  dedup keys, never for the text written to the output file.
- **Dedup is consecutive only.** A cue is dropped when it repeats the previous
  kept cue for that speaker within `DEDUPE_WINDOW_SECONDS`. That is whisper's
  repeated-line hallucination; a genuine "Yeah." ten minutes later must
  survive. The same rule runs in `transcript_combobulator.whisper.dedupe_segments`
  (per-speaker VTT) and `transcript_combobulator.combine._dedupe_entries`
  (session). There is no whole-session mode: it removed real speech.
- **Cue offsets use the padded clip start.** VAD writes `start_seconds` (speech)
  and `clip_start_seconds` (what the WAV actually contains); whisper's
  timestamps are relative to the clip, so `clip_start_seconds` is the offset.
- **Partial transcription is a failure.** Any failed chunk raises before the
  manifest is written, so the cache cannot
  hide a transcript with gaps. A track with no speech at all is an empty
  transcript, not an error.
- **Usernames match as whole tokens** in per-speaker dir names, so `wulf`
  never claims `5-wulfgar`. Batch runs hand combine the exact VTTs they
  produced; stale sibling dirs are ignored.
- **Skip filters match raw text**, not normalized text. So `[BLANK_AUDIO]` in a
  VTT line is filtered by the literal `[BLANK_AUDIO]` filter.

## Makefile Usage

Always use the Makefile. `make help` lists every target; this is the full set:

| Command | Purpose |
|---------|---------|
| `make check-deps` | Verify Python 3.11+, `venv` module, and `ffmpeg` on the host |
| `make setup` | Check deps, create venv from `$(PYTHON)`, `make install`, download whisper model |
| `make install [EXTRAS=dev,mac]` | `pip install -e ".[EXTRAS]"` into `.venv` |
| `make setup-whisper` | Download `WHISPER_MODEL` into `models/` (checksum-verified, no model load); with `WHISPER_RUNTIME=mlx`, the pinned mlx checkpoint into `models/mlx/<model>/` |
| `make run folder=path/ [force=1]` | Full parallel pipeline for one session; the only thing automation calls |
| `make run-single file=path [force=1]` | Convert → VAD → transcribe one file, no combine |
| `make combine-transcripts session=name` | Re-merge a session's per-speaker VTTs |
| `make filter-vtt file=path [threshold=50]` | Re-emit a speaker VTT from its saved JSON above a confidence; no inference |
| `make compare-transcripts ref="…" hyp="…" [json=out]` | Compare two combined transcripts of one session per speaker; numbers only, `WHISPER_PROMPT` from `ENV_FILE` |
| `make session-env bundle=dir output=env [speakers=json] [base=env] [dm=id] [tz=zone] [force=1]` | Write a recording bundle's session env, outside the bundle; no inference |
| `make create-sample-files` | Populate `tmp/input/jfk-sample/` |
| `make sample-ami` | Fetch the 4-speaker AMI sample into `tmp/input/ami-en2002b/` (network, checksum-verified) |
| `make check-sample` | Transcribe the AMI sample; print WER against `samples/ami/EN2002b-human.txt` (no gate); fail if any word differs from `samples/ami/EN2002b-reference.txt` |
| `make test-fast` | pytest without the `slow` (real inference) tests |
| `make test` | The suite `scripts/check` runs, real inference included; never touches `tmp/output` |
| `make test-mlx` | Opt-in MLX inference test; needs `mlx_whisper` and the mlx checkpoint |
| `make build` | `python -m build` into `dist/`, as CI does |
| `make lint` | `ruff check` and `ruff format --check` |
| `make format` | `ruff format` |
| `make typecheck` | `mypy --strict` |
| `make lint-pyrefly` | pyrefly, on trial beside mypy; not a gate yet |
| `make clean-output` | Delete every session's outputs under `tmp/output` |
| `make clean-tmp` | Also delete `tmp/input` and local caches |

Dropped on purpose: `convert-audio` (wrote a `_16khz` layout nothing consumed),
`regenerate-vtt` (re-transcribed the whole file without VAD), `test-segment`,
`create-test-files` (conftest does it), `process-vad`, `transcribe-segments`,
and the old `clean`, which `make test` used to call and which deleted every
session's transcripts.

## Resume Semantics

`transcript_combobulator/pipeline_cache.py` keeps one manifest per input file with a record per
stage: `conversion`, `vad`, `inference`, `vtt`. Each stage's fingerprint is
chained from the previous one over that stage's own settings
(`transcript_combobulator.config.get_stage_fingerprint_settings`), so a change re-runs that stage
and everything after it, nothing before it:

| Changed | Re-runs |
|---|---|
| source file, `SAMPLE_RATE` | everything |
| `VAD_*`, `PADDING_SECONDS` | VAD, inference, VTT |
| `WHISPER_*` (including `WHISPER_RUNTIME`), model file size/mtime | inference, VTT |
| `DEDUPE_*` | VTT only, rewritten from the saved JSON with no inference |

A record is trusted only if its fingerprint matches and every artifact it
names exists. Recording a stage drops every later record; a non-forced run
drops the `vtt` (completion) record before any work so an interrupted run
never looks complete. `force=1` deletes the manifest and the checkpoint.

Transcription checkpoints per chunk to `<stem>_progress.jsonl` (header line
with the inference fingerprint, then one JSON line per completed chunk). A
rerun with the same fingerprint reuses those chunks (`status: resumed`) and
transcribes only the missing ones; a truncated last line is ignored and that
chunk redone. The file is removed when every chunk succeeded and kept when
any failed, so a rerun retries only the failed
chunks. Verified by killing a real 2.9 h track mid-transcription and
resuming: output identical to the uninterrupted run.

## Batch Run Behaviour

`combobulator run` (`transcript_combobulator/batch.py`) is what automation
calls, so it fails fast and leaves a trail:

- Before any worker starts, `transcript_combobulator.combine.validate_speaker_mapping` runs
  against the input stems before any worker starts. Same errors as the combine
  step, minutes earlier.
- Two inputs whose stems match ignoring case (`1-cadderly.flac` and
  `1-cadderly.wav`) exit 1 before any work: the stem names the output folder,
  so both workers would write the same files. Bundles reject the same pair.
- A missing `ENV_FILE` raises at import instead of silently loading nothing.
- `LOG_FILE` (default empty) persists worker and parent logs to
  `tmp/output/<session>/<session>.log`; workers tag lines with their audio
  file. The rich
  table still owns the terminal either way.
- SIGINT and SIGTERM both kill the workers and the Manager and exit
  `128 + signal`, so a systemd stop does not orphan spawned processes.
- All paths derive from `transcript_combobulator.config.ROOT_DIR`, which is the repo (from
  `__file__`, or `PROJECT_ROOT` in the shell env), not the cwd. A relative
  `ENV_FILE` resolves against the cwd first, then the repo.

## Environment Files

- `.env` — production config (default).
- `.env.example` — committed template.
- `.env.jfk-sample` — committed, used by tests and `make create-sample-files`.
- `.env.<campaign>` — per-session overrides (e.g. `.env.annihilation`). Invoke
  with `ENV_FILE=.env.annihilation make run ...`.
- Never modify `.env` directly; override with `ENV_FILE=...`.

### Speaker Mapping

Per-user env vars (indexed starting at 1):

```sh
TRANSCRIPT_1_USERNAME="craig_discord_name"   # required, matched as substring in dir name
TRANSCRIPT_1_NAME="Display Name"             # required (falls back to TRANSCRIPT_1_PLAYER)
TRANSCRIPT_1_LABEL="Speaker Tag"             # required (falls back to TRANSCRIPT_1_CHARACTER)
TRANSCRIPT_1_DESCRIPTION="Short bio"         # required
```

If any of `NAME`, `LABEL`, `DESCRIPTION` is empty the mapping is skipped with a
warning. If the username is ambiguous across directories, the combine step
fails loudly.

## File Naming Conventions

- **Input audio**: `tmp/input/<session>/3-cadderly.flac`. The file stem names the
  speaker; any source works (Craig's stems are `<number>-<username>`)
- **Per-speaker output dir**: `tmp/output/<session>/3-cadderly/` (input stem)
- **Converted**: `3-cadderly.wav` inside it (16 kHz mono PCM_16, peak-normalised)
- **Per-speaker VTT + JSON**: `3-cadderly.vtt`, `3-cadderly_transcription.json`,
  `3-cadderly_mapping.json`, `3-cadderly_segment_NNN.wav`
  (`transcript_combobulator.config.vtt_path_for_input` is the one source of truth for the VTT path)
- **Session outputs**: `<session>-combined.txt` (or `-1.txt`, `-2.txt` when
  `CHUNKS>1`), `<session>-metrics.json`, `<session>.log`

## Performance Notes

- **Conversion streams through ffmpeg.** Measured on a 2.9 h Craig track:
  5.4 s at 41 MB peak RSS versus 13.7 s at 8.8 GB for the old in-memory
  torchaudio path (removed), with sample-identical length and a mean absolute
  difference of 2e-5. Peak normalisation is a two-pass streaming scan, so the
  "normalized once" invariant still holds. ffmpeg gets a timeout of
  max(600 s, 10x the audio duration) so a corrupt file cannot wedge a worker.
- **Silero VAD runs on one thread** (`VAD_THREADS=1`). It processes 512-sample
  frames one at a time; 1 thread measured 2.2x faster than 4 and 3.7x faster
  than 8 on Apple Silicon. The worker's whisper thread count is restored after.
- **VAD streams the WAV and uses Silero's ONNX build.** `transcript_combobulator.vad.detect_speech`
  collects frame probabilities over 60 s blocks with the detector's state
  carried across boundaries, then applies a verbatim port of silero-vad
  6.2.1's region logic; `tests/test_vad_streaming.py` asserts identity with
  the installed `get_speech_timestamps`. Segments are written by seeking
  into the WAV. Measured on a 2.9 h track, whole `process_audio` call:
  256 MB peak RSS, regions identical to the whole-file result. The
  TorchScript build gave the same regions but its peak
  swung between 0.9 and 3.9 GB run to run, which is why only the ONNX build is used.
- **The package `__init__.py` imports nothing.** The batch parent only needs config,
  combine, and telemetry; importing the package must not load torch.
- **The engine is chosen by measured correctness.** Stock openai-whisper
  `large-v3-turbo` is the default because it has beaten every alternative
  measured so far, not because other engines are ruled out. A candidate
  (another model, runtime or ASR engine) replaces it only by doing better on
  real sessions: fewer inserted lines (prompt echoes, repeat loops, filler on
  silence), less dropped speech, more names heard right, and cue times that
  land on the speech. Speed counts only once correctness is at least equal.
- **Apple's SpeechAnalyzer is not a replacement, by measurement.**
  `SpeechTranscriber` (en_US, `.transcription` preset, macOS 27) on the
  1,864 VAD islands of one session (13,859 s of speech) ran at 83x
  realtime on islands only, with no GPU use (0% in 148 of 149 samples).
  The 11.4x for MPS whisper is the whole pipeline, so the speeds are not
  like for like. With no ground truth, the 10.4% of words that differ is
  disagreement with whisper, not Apple's error rate. Apple left 199 islands
  (10.7%) empty and hit campaign names 65 times against 95. Part of the
  gap is whisper's own filler (Apple 5 bare filler lines against 57 for
  MPS whisper; 177 of the empty islands were 1 or 2 words in MPS), but
  not the names. Untested: `contextualStrings` vocabulary hints, other
  presets, a second session, CPU and Neural Engine use inside Apple's
  speech processes. Worth reopening only for work where the GPU must stay
  free.
- **One encoder pass per VAD island is the floor on stock whisper.** Whisper
  encodes a fixed 30 s window, so short islands pay full
  price (utilisation ~24% on a real session). Packing islands into one clip
  was measured at 2.45x on a 2.9 h track with identical words, but whisper
  places segment boundaries from what it hears, never from splices: 56 of
  128 islands (26 with a 2 s gap) were attributed to the previous island's
  time, up to 509 s early. Stock `word_timestamps` doubles decode cost and
  still misassigned 9 of 128. WhisperX and faster-whisper pack the same way
  and fix time with a separate wav2vec2 forced-alignment model; whisper.cpp
  shrinks the encoder per clip (`audio_ctx`, ~3x on short clips). Those are
  speed changes. They were not adopted because they bring no correctness
  gain, and packing without a separate aligner misplaces cues.
- **`WHISPER_RUNTIME=mlx` runs the same model on the Apple GPU, for speed.**
  It runs `large-v3-turbo` through mlx-whisper on the Apple GPU with the same islands
  and the same decoding options, still one encoder pass per island, and
  `WHISPER_DEVICE=mps` runs stock whisper on the GPU. Neither patches the
  model, adds a second one, packs, or retries. The default stays stock
  openai-whisper on the CPU; a campaign opts in with `WHISPER_RUNTIME=mlx`
  or `WHISPER_DEVICE=mps` in its env file.
  Measured on the Mac Studio (M5 Max, 128 GiB, macOS 27.0.1) on 5 x 30 min of
  the 2026-10-04 session (379 islands) against the CPU run (14 m 09 s wall,
  inference about 12 m 40 s): `mps` took 3 m 08 s with identical text (0 words
  differ); `mlx` took 57 s of inference with one worker and 39 s with two
  (4 workers: 38 s, no gain), 1.5% of words different (the loudness
  normalisation noise floor was 1.7%), no prompt echoes, no lost islands, and
  the same nine looping islands the CPU run has. The whole 13 h 53 m session
  on `mlx` took 4 m 12 s with `PARALLEL_JOBS=2` (3 m 41 s with 4) against
  1 h 06 m on the CPU: 0.9% of words different (WER 1.45%), 3 of 1813
  islands lost (1 to 5 words each), bare "Thank you" 35 to 42, single-threaded
  VAD now 118 s of the 458 s of work. At the level of session notes (notes
  written cold from the whole transcript, graded against a 28-bullet human
  reference): CPU transcript 22 of 28, mlx transcript
  22 of 28, inside the 0.77 to 0.80 spread between runs on one transcript;
  the proper-noun variants in the two transcripts are the same, name by name
  and count by count. One run each, so a difference smaller than that
  spread is not shown either way: this is a speed change, not a correctness
  gain. What mlx-whisper does not have:
  `carry_initial_prompt` (the prompt reaches only the first 30 s window of an
  island; 59 of 1813 islands, 3.3%, are longer), beam search, and anything
  but fp16. Both GPU runtimes share the GPU: with a 90 GB llama.cpp model
  loaded, its generation fell from 72 to 34 tokens/s while one mlx worker
  ran, and to 26 with two. A Neural Engine runtime (WhisperKit) left it
  alone (72 to 71) but ran 4.7x slower than mlx and differed in 6.5% of
  words with 19 islands lost; faster-whisper has no Metal backend. Output
  cached by one runtime is never reused by the other: the runtime, the
  pinned mlx checkpoint and its file identity are in the inference
  fingerprint, while the default runtime's fingerprint is unchanged, so
  finished sessions stay valid. Setup: `make install EXTRAS=dev,mac`, then
  `WHISPER_RUNTIME=mlx make setup-whisper`. Not measured here: any other
  model size on mlx.
- **`WHISPER_MODEL` stays `large-v3-turbo`, by measurement.** small.en on the
  same 11.5 h session, judged at the level of session notes against an
  80-item skeleton built from turbo: 56 intact, 21 degraded, 3 lost,
  including the session's biggest outcome (the DM's narration of the boss
  falling and the rest fleeing replaced by the transcription prompt echoed
  sixteen times). small.en substitutes rather than omits: 9 of its 17
  prompt echoes landed on the DM's track mid-narration, and the two PC
  names never appeared in their canonical spelling. Whisper's own guards
  (temperature fallback, no prompt carry) reduced echoes from 17 to 9 and
  did not close the gap (20 to 28% of words still differ). medium.en was
  worse still: 83 prompt echoes, 42 canned phrases, 56% of words different,
  and 1.5x slower than turbo because its 24-layer decoder costs more per
  chunk than turbo's 4. Loudness normalisation (EBU R128, -21 to -16 LUFS)
  on the quietest track changed 1.7% of words and nothing else. Smaller
  models and audio pre-treatment are not options for this audio; the
  pipeline is built for machines that can run turbo. Turbo's own known
  artefact is the bare "Thank you." silence filler (37 lines per session),
  which displaces nothing. large-v3 was worse on a full real session (five
  speakers, mps, production prompt). It was compared with two turbo runs
  whose only difference was 3 words: 51 prompt echoes against turbo's 5,
  299 s of speech dropped against 121 s, 108 repeat loops against 59, and
  1.44x the wall time. It also rewrote 19% of turbo's words, including 1005
  deleted on one player's track.
- **Temperature fallback stays off, by measurement.** `WHISPER_TEMPERATURE=
  0.0,0.2,0.4` enables whisper's re-decode when its compression-ratio or
  logprob guard trips; with the default scalar those guards never fire.
  A/B on a real 11.5 h session (five speakers): the fallback re-decoded 131
  of 2075 segments (6.3%), cost 1.29x inference on the track timed in both
  runs, left text similarity at 0.97 to 0.99 per speaker and the count of
  suspicious cues unchanged, and the segments it changed were short
  ambiguous fragments where the temperature-0.4 sample was a different
  guess, not a better one ("laughs," became "laughs, laughs, laughs,
  laughs."). Sampling also makes output non-deterministic across runs.
  Every segment now records the temperature whisper settled on, and
  telemetry carries `fallback_segment_count`, so a future A/B is one env
  change.

- Whisper's `word_timestamps=True` hangs on some segments. Keep the default
  `WHISPER_WORD_TIMESTAMPS=false`.
- `beam_size=1` and `condition_on_previous_text=false` are intentional for
  VAD-segment transcription (each segment is already a speech island).
- Each parallel worker loads its own whisper model (~3x the checkpoint size at
  load: fp16 file plus fp32 params). The memory guard lowers the
  active worker count so `PARALLEL_JOBS` workers fit in
  `MEMORY_GUARD_FRACTION` of physical RAM; it never raises the count. An 8 GB
  Pi with `large-v3-turbo` lands on 1 worker. `TORCH_THREADS=0` auto-splits
  threads across the active workers.
- mlx-whisper is pinned in the `mac` extra (`mlx-whisper==0.4.3`; `pip check`
  is clean next to `torch==2.11.0`). `whisper_mlx.py` relies on its
  `ModelHolder`, so a bump must keep `tests/test_whisper_runtime.py` green and
  rerun the slow mlx test.
- `transcript_combobulator.whisper.load_whisper_model` loads by file path on purpose. Loading by
  name makes whisper sha256 the whole checkpoint per worker per run.
- On macOS, `multiprocessing.set_start_method("spawn")` is mandatory for torch.
  `transcript_combobulator/batch.py` handles this at import time.

## Known Whisper Failure Modes

- **Repetition hallucination** on laughs/silence: whisper emits one word
  hundreds of times (e.g. `"laughs laughs laughs…"`). `transcript_combobulator.whisper.collapse_repetition`
  collapses these to a single occurrence before they reach the VTT.
- **Confidence drift** on quiet or ambiguous audio: `make filter-vtt
  file=<input> threshold=50` re-emits a filtered VTT from the saved
  `_transcription.json`. No inference; seconds, not hours.

## Testing

- Before writing or changing a test, read https://github.com/splitleafhq/.github/blob/main/docs/testing.md.
- `tests/conftest.py` session fixture creates `tmp/input/test_jfk*.wav` from
  the committed `samples/jfk.wav`. It is autouse, so every test needs working
  audio decoding (ffmpeg) even the pure-logic ones.
- `tests/test_batch.py` is fast (mocks + dir fixtures). Safe to run on every
  change.
- `tests/test_combine.py` is fast (pure Python over synthetic VTTs).
- `tests/test_vad.py`, `tests/test_transcription.py`, `tests/test_whisper.py`
  are marked `slow` (real inference, about 20 s). `scripts/check` and `make test`
  run them; CI and `make test-fast` skip them (no models there). The MLX test is
  marked `mlx` and runs only with `make test-mlx`.
- `make test` never cleans `tmp/output`. Slow tests write under
  `tmp/output/test_jfk*/`; `make clean-output` removes everything.
- Whisper segment count is nondeterministic — use range assertions, not exact
  counts.
- For similarity checks, use `difflib.SequenceMatcher`, not `set` intersection
  (repeated words break set-based similarity).

## Dependencies (gotchas)

- `torch` and `torchaudio` are an exact pair; torchaudio's wheel declares no
  dependencies so pip will not enforce it. torchaudio 2.11.0 is the final
  maintenance line and caps torch. `torchaudio` must stay importable for
  `silero-vad`, but nothing calls its I/O: audio goes through `soundfile` and
  `ffmpeg`, so `torchcodec` (ABI-locked to torch, no aarch64 wheel before
  0.11) is deliberately not a dependency. Do not reintroduce
  `torchaudio.load`/`save`.
- `silero-vad` bundles both its models inside the wheel; `load_silero_vad()`
  needs no network access. It does not declare `onnxruntime`, which its
  ONNX build needs, so `pyproject.toml` pins it.
- `transcript_combobulator.vad._speech_regions_from_probs` is a verbatim port of silero's region
  logic because silero only accepts a whole-file tensor. Bumping `silero-vad`
  must keep `tests/test_vad_streaming.py` green; if silero changes the
  algorithm, port the change, do not paper over the diff.
- `ffmpeg` must be on `PATH`, and its `ffprobe`: libsndfile cannot read
  `.m4a` or `.aac`, so conversion and the timing report take their duration
  from ffprobe. Makefile recipes run under `/bin/sh` (dash on Debian), so keep
  them POSIX: `>/dev/null 2>&1`, never `&>`.

## Error Handling Style

- Custom exceptions: `AudioValidationError`, `VADError`, `WhisperError`,
  `TranscriptionError`, `CombineError`, `BundleError`, `SessionEnvError`. Wrap underlying errors via `raise ... from`.
- Log warnings for skippable problems (missing segment file). Raise for
  structural problems (missing mapping, missing model, unmapped speaker).
- A track with no detected speech is an empty transcript, not an error.
- Individual chunk failures are logged and skipped inside whisper, but a file
  with any failed chunk raises before its manifest is written so the cache
  cannot hide gaps.

## Adding a New Feature

1. Check this file for existing patterns first.
2. Use a Makefile target. Add one if the operation should be reproducible.
3. Put logic in a `transcript_combobulator` module; expose it as a `cli.py`
   subcommand if a person or automation should call it.
4. Load config only through `transcript_combobulator.config`. Never call `load_dotenv` or
   `os.getenv` for a setting anywhere else; add the constant to config.
5. Add a test in `tests/test_<module>.py` that fails if the feature breaks, or
   extend one that already covers it. Use synthetic fixtures where possible, and
   the slow whisper tests only when testing whisper behavior.
6. Run `make lint`, `make typecheck` and `make test` before declaring done.
   `scripts/check` runs CI's steps (ruff, mypy, tests, build) plus the slow real-inference tests CI can't run (it has no models) and fails when `.venv` is missing or off the pyproject pins; `.githooks/pre-push` runs it before every push. Enable the hook once per clone with `git config core.hooksPath .githooks`. A new worktree has no `.venv`: `python3.14 -m venv .venv && make install` (the Python in `.python-version`).

## Common Pitfalls

- ❌ Editing `.env` (use `ENV_FILE=` instead)
- ❌ Calling `load_dotenv` anywhere other than `transcript_combobulator/config.py`
- ❌ Hardcoding paths; use `transcript_combobulator.config.OUTPUT_DIR` / `INPUT_DIR` / `get_output_path_for_input`
- ❌ Hardcoding whisper kwargs; use `transcript_combobulator.config.get_whisper_options()`
- ❌ Normalizing audio twice (audio_utils does it)
- ❌ Normalizing transcript text for display (normalize is dedup-only)
- ❌ Re-reading a file in the same process just to verify it; trust the write
