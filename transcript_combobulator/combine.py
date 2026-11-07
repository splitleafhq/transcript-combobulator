"""Transcript combination utilities.

Reads per-user VTT files, tags each line with a speaker label, sorts by
timestamp, deduplicates, and writes a combined session transcript.
"""

import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from transcript_combobulator.config import (
    CHUNKS,
    DEDUPE_WINDOW_SECONDS,
    INCLUDE_TIMESTAMPS,
    SKIP_FILTERS,
)
from transcript_combobulator.logging_config import get_logger
from transcript_combobulator.pipeline_cache import is_vtt_unfinished

logger = get_logger(__name__)

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)
_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")


class CombineError(Exception):
    """Base exception for transcript combination errors."""


@dataclass
class TranscriptEntry:
    """A single timestamped line spoken by one speaker."""

    timestamp: str
    start_time: float
    end_time: float
    speaker: str
    content: str  # original text, for display
    dedup_key: str  # lowercased/punctuation-stripped, for comparison only


@dataclass
class TranscriptConfig:
    """Mapping from a per-user VTT file to its display metadata."""

    name: str
    label: str
    description: str
    transcript_path: Path


def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used only for dedup comparison."""
    return re.sub(r"\s+", " ", text.strip().lower().translate(_PUNCT_TRANS))


def should_skip_content(content: str, skip_filters: list[str]) -> bool:
    """True if content matches any literal or /regex/ filter."""
    for pattern in skip_filters:
        if pattern.startswith("/") and pattern.endswith("/"):
            if re.search(pattern[1:-1], content):
                return True
        elif pattern in content:
            return True
    return False


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """HH:MM:SS.mmm -> seconds."""
    h, m, sec_ms = timestamp.split(":")
    s, _, ms = sec_ms.partition(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000 if ms else 0)


def parse_vtt_file(
    file_path: Path,
    speaker_label: str,
    skip_filters: Optional[list[str]] = None,
) -> list[TranscriptEntry]:
    """Parse a VTT file into TranscriptEntry objects, preserving original text."""
    skip_filters = skip_filters or []

    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CombineError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise CombineError(f"Failed to read transcript file {file_path}: {e}") from e

    entries: list[TranscriptEntry] = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        match = _TIME_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue

        start_str, end_str = match.group(1), match.group(2)
        i += 1
        content_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            content_lines.append(lines[i].strip())
            i += 1

        if not content_lines:
            continue

        text = " ".join(content_lines)
        if should_skip_content(text, skip_filters):
            continue
        dedup_key = _normalize_for_dedup(text)
        if not dedup_key:
            continue

        entries.append(
            TranscriptEntry(
                timestamp=f"{start_str} --> {end_str}",
                start_time=parse_timestamp_to_seconds(start_str),
                end_time=parse_timestamp_to_seconds(end_str),
                speaker=speaker_label,
                content=text,
                dedup_key=dedup_key,
            )
        )

    return entries


def combine_transcripts(
    transcript_configs: list[TranscriptConfig],
    output_path: Path,
    skip_filters: Optional[list[str]] = None,
    include_timestamps: bool = True,
    chunks: int = 1,
) -> list[Path]:
    """Combine per-user VTTs into one (or more) chunked session transcripts.

    Dedup is always global across all speakers by (speaker, normalized_text).
    """
    skip_filters = skip_filters or []

    logger.info(f"Combining {len(transcript_configs)} transcript files...")
    all_entries: list[TranscriptEntry] = []
    for config in transcript_configs:
        logger.info(f"Processing {config.transcript_path.name} for {config.label}...")
        entries = parse_vtt_file(config.transcript_path, config.label, skip_filters)
        logger.info(f"  - Found {len(entries)} entries")
        all_entries.extend(entries)

    # Stable sort on (start, speaker) so ties across speakers are deterministic.
    all_entries.sort(key=lambda e: (e.start_time, e.speaker))
    logger.info(f"Total combined entries: {len(all_entries)}")

    deduped = _dedupe_entries(all_entries)

    summary_lines = ["Summary:"]
    seen_summary: set[str] = set()
    for config in transcript_configs:
        line = f"{config.name} - {config.label} - {config.description}"
        if line not in seen_summary:
            summary_lines.append(line)
            seen_summary.add(line)
    summary = "\n".join(summary_lines)

    min_per_chunk = 5
    actual_chunks = 1 if len(deduped) < max(chunks, min_per_chunk) else chunks
    chunk_size = len(deduped) // actual_chunks
    output_files: list[Path] = []

    for chunk_idx in range(actual_chunks):
        start = chunk_idx * chunk_size
        end = len(deduped) if chunk_idx == actual_chunks - 1 else start + chunk_size
        chunk_entries = deduped[start:end]
        chunk_path = (
            output_path
            if actual_chunks == 1
            else output_path.with_name(
                f"{output_path.stem}-{chunk_idx + 1}{output_path.suffix}"
            )
        )
        try:
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(summary + "\n\n")
                for entry in chunk_entries:
                    if include_timestamps:
                        f.write(
                            f"{entry.speaker}: {entry.content} [{entry.timestamp}]\n"
                        )
                    else:
                        f.write(f"{entry.speaker}: {entry.content}\n")
            output_files.append(chunk_path)
            logger.info(f"Generated: {chunk_path}")
        except Exception as e:
            raise CombineError(f"Failed to write output file {chunk_path}: {e}") from e

    return output_files


def load_speaker_mapping(
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, dict[str, str]]:
    """Build username -> {name, label, description} from TRANSCRIPT_N_* variables.

    Reads the process environment, or ``env`` when given (a parsed env file that
    is not the one this process runs with). Values are used as python-dotenv
    parsed them: it already removes the outer quotes, so quote marks left in a
    value are part of the name.
    """
    source: Mapping[str, Optional[str]] = os.environ if env is None else env

    def get(key: str) -> str:
        return source.get(key) or ""

    mapping: dict[str, dict[str, str]] = {}
    i = 1
    while True:
        username = get(f"TRANSCRIPT_{i}_USERNAME").strip()
        if not username:
            break

        name = (get(f"TRANSCRIPT_{i}_NAME") or get(f"TRANSCRIPT_{i}_PLAYER")).strip()
        label = (
            get(f"TRANSCRIPT_{i}_LABEL") or get(f"TRANSCRIPT_{i}_CHARACTER")
        ).strip()
        description = get(f"TRANSCRIPT_{i}_DESCRIPTION").strip()

        if all([name, label, description]):
            mapping[username] = {
                "name": name,
                "label": label,
                "description": description,
            }
        else:
            logger.warning(f"Incomplete configuration for TRANSCRIPT_{i}_* variables")
        i += 1

    return mapping


def _dedupe_entries(
    entries: list[TranscriptEntry],
    window_seconds: float = DEDUPE_WINDOW_SECONDS,
) -> list[TranscriptEntry]:
    """Drop an entry that repeats the previous kept entry for the same speaker
    within window_seconds. Entries must be sorted by start."""
    last_kept: dict[str, TranscriptEntry] = {}
    kept: list[TranscriptEntry] = []
    for entry in entries:
        prev = last_kept.get(entry.speaker)
        if (
            prev is not None
            and prev.dedup_key == entry.dedup_key
            and entry.start_time - prev.end_time <= window_seconds
        ):
            logger.debug(f"CONSECUTIVE SKIP: {entry.speaker}: '{entry.dedup_key}'")
            continue
        last_kept[entry.speaker] = entry
        kept.append(entry)
    return kept


def match_usernames(dir_name: str, mapping: dict[str, dict[str, str]]) -> list[str]:
    """Usernames that appear in a per-speaker dir name as a whole token.

    Token boundaries are non-alphanumerics or the string ends, so 'wulf' does
    not claim '5-wulfgar' and 'cadderly' does not claim '3-cadderly2', while
    '3-cadderly', '3-cadderly_16khz', and '2-catti_brie' still match.
    """
    return [
        u
        for u in mapping
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(u)}(?![A-Za-z0-9])", dir_name)
    ]


def validate_speaker_mapping(
    dir_names: Iterable[str],
    env: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, dict[str, str]]:
    """Check TRANSCRIPT_N_* covers every per-speaker directory name exactly once.

    Raises CombineError with the same messages the combine step would produce.
    Batch runs call this before transcription so a config typo fails in
    seconds rather than after hours of inference. Returns the loaded mapping.
    ``env`` checks a parsed env file instead of the process environment.
    """
    username_mapping = load_speaker_mapping(env)
    if not username_mapping:
        raise CombineError(
            "No transcript mappings found. Define TRANSCRIPT_N_USERNAME, "
            "TRANSCRIPT_N_NAME, TRANSCRIPT_N_LABEL, TRANSCRIPT_N_DESCRIPTION."
        )

    unmapped_dirs: list[str] = []
    for dir_name in dir_names:
        matches = match_usernames(dir_name, username_mapping)
        if not matches:
            unmapped_dirs.append(dir_name)
            continue
        if len(matches) > 1:
            raise CombineError(
                f"Ambiguous username match for directory '{dir_name}': {matches}. "
                "Make TRANSCRIPT_*_USERNAME values more specific."
            )

    if unmapped_dirs:
        raise CombineError(
            f"No mapping found for directories: {unmapped_dirs}. "
            "Update TRANSCRIPT_*_USERNAME environment variables."
        )
    return username_mapping


def combine_transcripts_from_env(
    base_dir: Path,
    session_subdir: Optional[str] = None,
    vtt_files: Optional[Iterable[Path]] = None,
) -> list[Path]:
    """Combine transcripts using TRANSCRIPT_N_* env vars for speaker mapping.

    ``vtt_files`` restricts the combine to exactly those transcripts. Batch
    runs pass the VTTs they produced so a stale per-speaker directory from an
    earlier export cannot double a speaker's lines. Without it every VTT
    under the session directory is used, in sorted order, and none may be
    left by a run that failed or was interrupted (``is_vtt_unfinished``).
    """
    search_dir = base_dir / session_subdir if session_subdir else base_dir
    output_dir = search_dir

    if vtt_files is not None:
        vtt_files = sorted(Path(p) for p in vtt_files)
        missing = [str(p) for p in vtt_files if not p.exists()]
        if missing:
            raise CombineError(f"Transcript files not found: {missing}")
    else:
        vtt_files = sorted(search_dir.glob("**/*.vtt"))
        # whisper writes the VTT even when some chunks failed; only a
        # finished run records it, so anything else may have gaps.
        incomplete = [str(p) for p in vtt_files if is_vtt_unfinished(p)]
        if incomplete:
            raise CombineError(
                f"Transcripts not recorded complete: {incomplete}. A run failed "
                "or was interrupted on these tracks; rerun it to finish them."
            )
    if not vtt_files:
        raise CombineError(f"No VTT files found in {search_dir}")

    username_mapping = validate_speaker_mapping(
        vtt_file.parent.name for vtt_file in vtt_files
    )

    transcript_configs: list[TranscriptConfig] = []
    for vtt_file in vtt_files:
        matches = match_usernames(vtt_file.parent.name, username_mapping)
        m = username_mapping[matches[0]]
        transcript_configs.append(
            TranscriptConfig(
                name=m["name"],
                label=m["label"],
                description=m["description"],
                transcript_path=vtt_file,
            )
        )

    if not transcript_configs:
        raise CombineError("No transcript configurations created.")

    include_timestamps = INCLUDE_TIMESTAMPS
    skip_filters = list(SKIP_FILTERS)
    chunks = CHUNKS

    output_filename = (
        f"{session_subdir}-combined.txt"
        if session_subdir
        else f"{output_dir.name}-combined.txt"
    )
    output_path = output_dir / output_filename

    return combine_transcripts(
        transcript_configs=transcript_configs,
        output_path=output_path,
        skip_filters=skip_filters,
        include_timestamps=include_timestamps,
        chunks=chunks,
    )
