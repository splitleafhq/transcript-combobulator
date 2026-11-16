#!/usr/bin/env python3
"""Tests for the transcript combination step."""

import json
from pathlib import Path

import pytest

from transcript_combobulator.combine import (
    CombineError,
    TranscriptConfig,
    _normalize_for_dedup,
    combine_transcripts,
    parse_vtt_file,
    should_skip_content,
    validate_speaker_mapping,
)
from transcript_combobulator.pipeline_cache import (
    FINAL_STAGE,
    get_manifest_path,
    get_progress_path,
    record_stage,
)

VTT_SAMPLE = """WEBVTT

00:00:01.000 --> 00:00:02.000
Hello, world!

00:00:02.500 --> 00:00:03.500
Hello, world!

00:00:04.000 --> 00:00:05.000
This is a test.
"""


def _write_vtt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_finished_vtt(path: Path, content: str) -> None:
    """A VTT plus the completion record a finished pipeline run leaves."""
    _write_vtt(path, content)
    record_stage(
        get_manifest_path(path.parent, path.parent.name), FINAL_STAGE, "done", [path]
    )


class TestNormalizeForDedup:
    def test_strips_punctuation_and_lowercases(self):
        assert _normalize_for_dedup("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_for_dedup("  a\t b\n c  ") == "a b c"

    def test_empty_returns_empty(self):
        assert _normalize_for_dedup("") == ""


class TestShouldSkipContent:
    def test_literal_match(self):
        assert should_skip_content("text with [BLANK_AUDIO] inside", ["[BLANK_AUDIO]"])

    def test_regex_match(self):
        assert should_skip_content("laughs laughs laughs", ["/(laughs\\s*){2,}/"])

    def test_no_match(self):
        assert not should_skip_content("normal speech", ["[BLANK_AUDIO]"])


class TestParseVttFile:
    def test_preserves_original_text(self, tmp_path):
        """Parsed entries keep the original capitalization and punctuation."""
        vtt = tmp_path / "sample.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        entries = parse_vtt_file(vtt, "Alice")
        assert entries[0].content == "Hello, world!"
        assert entries[2].content == "This is a test."

    def test_dedup_key_is_normalized(self, tmp_path):
        vtt = tmp_path / "sample.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        entries = parse_vtt_file(vtt, "Alice")
        assert entries[0].dedup_key == "hello world"
        assert entries[2].dedup_key == "this is a test"

    def test_skip_filter_drops_matching_entries(self, tmp_path):
        vtt = tmp_path / "sample.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n[BLANK_AUDIO]\n\n"
            "00:00:03.000 --> 00:00:04.000\nHello\n",
        )
        entries = parse_vtt_file(vtt, "Alice", skip_filters=["[BLANK_AUDIO]"])
        assert len(entries) == 1
        assert entries[0].content == "Hello"


class TestCombineTranscripts:
    def test_output_preserves_original_text(self, tmp_path):
        """Combined session transcript keeps capitalization and punctuation."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(vtt, VTT_SAMPLE)
        out = tmp_path / "combined.txt"

        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(
                    name="Alice Jones",
                    label="Alice",
                    description="Party leader",
                    transcript_path=vtt,
                )
            ],
            output_path=out,
        )

        text = out.read_text()
        assert "Alice: Hello, world!" in text
        assert "Alice: This is a test." in text

    def test_consecutive_repeat_within_window_is_deduped(self, tmp_path):
        """Whisper's repeated-line hallucination: same text, back to back."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nHello, world!\n\n"
            "00:00:02.500 --> 00:00:03.500\nhello world\n\n",
        )
        out = tmp_path / "combined.txt"
        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(
                    name="Alice", label="Alice", description="", transcript_path=vtt
                )
            ],
            output_path=out,
        )
        assert out.read_text().count("Alice: ") == 1

    def test_genuine_repeat_later_in_session_is_kept(self, tmp_path):
        """A speaker saying 'Yeah.' ten minutes apart is two lines, not one."""
        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nYeah.\n\n"
            "00:10:01.000 --> 00:10:02.000\nYeah.\n\n",
        )
        out = tmp_path / "combined.txt"
        combine_transcripts(
            transcript_configs=[
                TranscriptConfig(
                    name="Alice", label="Alice", description="", transcript_path=vtt
                )
            ],
            output_path=out,
        )
        assert out.read_text().count("Alice: Yeah.") == 2

    def test_window_bounds_the_dedup(self, tmp_path):
        from transcript_combobulator.combine import _dedupe_entries, parse_vtt_file

        vtt = tmp_path / "alice.vtt"
        _write_vtt(
            vtt,
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nYeah.\n\n"
            "00:00:03.500 --> 00:00:04.000\nYeah.\n\n",
        )
        entries = parse_vtt_file(vtt, "Alice")
        assert len(_dedupe_entries(entries, window_seconds=2.0)) == 1
        assert len(_dedupe_entries(entries, window_seconds=1.0)) == 2


def _set_mapping(monkeypatch, usernames: list[str]) -> None:
    """Point TRANSCRIPT_N_* at the given usernames and terminate the sequence."""
    for i, username in enumerate(usernames, start=1):
        monkeypatch.setenv(f"TRANSCRIPT_{i}_USERNAME", username)
        monkeypatch.setenv(f"TRANSCRIPT_{i}_NAME", f"Name {i}")
        monkeypatch.setenv(f"TRANSCRIPT_{i}_LABEL", f"Label {i}")
        monkeypatch.setenv(f"TRANSCRIPT_{i}_DESCRIPTION", f"Desc {i}")
    monkeypatch.setenv(f"TRANSCRIPT_{len(usernames) + 1}_USERNAME", "")


class TestValidateSpeakerMapping:
    """Up-front TRANSCRIPT_N_* validation shared with the combine step."""

    def test_every_dir_mapped_once_returns_mapping(self, monkeypatch):
        _set_mapping(monkeypatch, ["cadderly", "wulfgar"])
        mapping = validate_speaker_mapping(["1-cadderly", "2-wulfgar"])
        assert set(mapping) == {"cadderly", "wulfgar"}

    def test_no_mapping_configured(self, monkeypatch):
        _set_mapping(monkeypatch, [])
        with pytest.raises(CombineError, match="No transcript mappings found"):
            validate_speaker_mapping(["1-cadderly"])

    def test_unmapped_dir_names_all_reported(self, monkeypatch):
        _set_mapping(monkeypatch, ["cadderly"])
        with pytest.raises(CombineError, match="2-wulfgar.*3-bruenor"):
            validate_speaker_mapping(["1-cadderly", "2-wulfgar", "3-bruenor"])

    def test_ambiguous_token_match(self, monkeypatch):
        _set_mapping(monkeypatch, ["wulf", "gar"])
        with pytest.raises(CombineError, match="Ambiguous username match"):
            validate_speaker_mapping(["2-wulf-gar"])

    def test_username_must_match_as_whole_token(self, monkeypatch):
        """'wulf' must not silently claim wulfgar's lines."""
        _set_mapping(monkeypatch, ["wulf"])
        with pytest.raises(CombineError, match="No mapping found"):
            validate_speaker_mapping(["5-wulfgar"])

    def test_token_match_accepts_craig_and_converted_dir_names(self, monkeypatch):
        _set_mapping(monkeypatch, ["cadderly", "catti_brie"])
        mapping = validate_speaker_mapping(
            ["3-cadderly", "3-cadderly_16khz", "2-catti_brie"]
        )
        assert set(mapping) == {"cadderly", "catti_brie"}
        with pytest.raises(CombineError, match="No mapping found"):
            validate_speaker_mapping(["3-cadderly2"])

    def test_incomplete_entry_is_skipped_and_leaves_dir_unmapped(self, monkeypatch):
        _set_mapping(monkeypatch, ["cadderly"])
        monkeypatch.setenv("TRANSCRIPT_1_DESCRIPTION", "")
        with pytest.raises(CombineError, match="No transcript mappings found"):
            validate_speaker_mapping(["1-cadderly"])


def test_quote_marks_in_speaker_fields_survive(tmp_path, monkeypatch):
    """A nickname in quotes keeps both quote marks, in the summary and on each line."""
    from dotenv import dotenv_values

    from transcript_combobulator.combine import combine_transcripts_from_env

    env = tmp_path / "session.env"
    env.write_text(
        'TRANSCRIPT_1_USERNAME="regis"\n'
        'TRANSCRIPT_1_NAME="Danica"\n'
        "TRANSCRIPT_1_LABEL='Cadderly \"the Chosen\" Bonaduce'\n"
        "TRANSCRIPT_1_DESCRIPTION='Human Cleric'\n"
        'TRANSCRIPT_2_USERNAME="catti_brie"\n'
        'TRANSCRIPT_2_NAME="Ivan"\n'
        "TRANSCRIPT_2_LABEL='Pikel \"Doo-dad\" Bouldershoulder'\n"
        'TRANSCRIPT_2_DESCRIPTION="Dwarf Druid"\n',
        encoding="utf-8",
    )
    for key, value in dotenv_values(env).items():
        monkeypatch.setenv(key, value or "")
    monkeypatch.setenv("TRANSCRIPT_3_USERNAME", "")
    monkeypatch.setenv("CHUNKS", "1")
    for stem in ("1-regis", "4-catti_brie"):
        (tmp_path / "night" / stem).mkdir(parents=True)
        _write_finished_vtt(
            tmp_path / "night" / stem / f"{stem}.vtt",
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nRoll for initiative.\n\n",
        )

    text = combine_transcripts_from_env(tmp_path, "night")[0].read_text()

    assert 'Danica - Cadderly "the Chosen" Bonaduce - Human Cleric' in text
    assert 'Cadderly "the Chosen" Bonaduce: Roll for initiative.' in text
    assert 'Pikel "Doo-dad" Bouldershoulder: Roll for initiative.' in text


class TestCombineFromEnvExplicitFiles:
    """Batch runs hand combine the exact VTTs they produced."""

    def _session(self, tmp_path, monkeypatch):
        _set_mapping(monkeypatch, ["cadderly"])
        session = tmp_path / "night"
        current = session / "3-cadderly"
        stale = session / "3-cadderly-v0"
        for d in (current, stale):
            d.mkdir(parents=True)
            _write_finished_vtt(
                d / "3-cadderly.vtt",
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nRoll for initiative.\n\n",
            )
        monkeypatch.setenv("CHUNKS", "1")
        return session, current, stale

    def test_explicit_list_ignores_stale_sibling_dirs(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, current, _ = self._session(tmp_path, monkeypatch)
        out = combine_transcripts_from_env(
            tmp_path, "night", vtt_files=[current / "3-cadderly.vtt"]
        )
        assert out[0].read_text().count("Roll for initiative.") == 1

    def test_glob_fallback_sees_both(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, _, _ = self._session(tmp_path, monkeypatch)
        out = combine_transcripts_from_env(tmp_path, "night")
        assert out[0].read_text().count("Roll for initiative.") >= 1
        assert len(list(session.glob("**/*.vtt"))) == 2

    def _danica(self, session: Path, monkeypatch) -> Path:
        track = session / "5-danica"
        track.mkdir()
        _write_vtt(
            track / "5-danica.vtt",
            "WEBVTT\n\n00:00:03.000 --> 00:00:04.000\nI check for traps.\n\n",
        )
        _set_mapping(monkeypatch, ["cadderly", "danica"])
        return track

    @pytest.mark.parametrize("left_by", ["interrupted", "failed_chunks"])
    def test_glob_refuses_a_track_whose_run_did_not_finish(
        self, tmp_path, monkeypatch, left_by
    ):
        """whisper writes the VTT when some chunks fail; combine must not merge it."""
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, _, _ = self._session(tmp_path, monkeypatch)
        track = self._danica(session, monkeypatch)
        if left_by == "interrupted":
            # The run got as far as conversion; it never recorded the VTT.
            record_stage(
                get_manifest_path(track, track.name),
                "conversion",
                "done",
                [track / "5-danica.vtt"],
            )
        else:
            get_progress_path(track, track.name).write_text('{"key": "k"}\n')

        with pytest.raises(CombineError, match="5-danica.vtt"):
            combine_transcripts_from_env(tmp_path, "night")
        assert not (session / "night-combined.txt").exists()

    @pytest.mark.parametrize("manifest", [None, {"cache_version": 2}])
    def test_glob_combines_a_track_from_before_stage_records(
        self, tmp_path, monkeypatch, manifest
    ):
        """Sessions made before per-stage records have nothing to check."""
        from transcript_combobulator.combine import combine_transcripts_from_env

        session, _, _ = self._session(tmp_path, monkeypatch)
        track = self._danica(session, monkeypatch)
        if manifest is not None:
            get_manifest_path(track, track.name).write_text(json.dumps(manifest))

        out = combine_transcripts_from_env(tmp_path, "night")
        assert "I check for traps." in out[0].read_text()

    def test_missing_explicit_file_is_loud(self, tmp_path, monkeypatch):
        from transcript_combobulator.combine import combine_transcripts_from_env

        _, current, _ = self._session(tmp_path, monkeypatch)
        with pytest.raises(CombineError, match="Transcript files not found"):
            combine_transcripts_from_env(
                tmp_path, "night", vtt_files=[current / "nope.vtt"]
            )
