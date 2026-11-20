"""Console entry point: ``combobulator <command> ...``.

Every subcommand imports its module lazily so ``combobulator --help`` and the
light commands never load torch. Per-campaign settings come from ENV_FILE in
the environment, exactly as with the Makefile.
"""

from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "run": "Full pipeline for a session folder (convert, VAD, transcribe, combine)",
    "file": "Convert, VAD, and transcribe one file; no combine",
    "combine": "Merge a session's per-speaker VTTs into one transcript",
    "filter-vtt": "Rewrite a speaker's VTT from its saved JSON above a confidence",
    "compare": "Compare two combined transcripts of one session, per speaker",
    "session-env": "Write a session's env file from a recording bundle and a speakers file",
    "setup-model": "Download WHISPER_MODEL (or a named model) into models/ without loading it",
    "samples": "Build sample and test audio from samples/",
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {text}" for name, text in COMMANDS.items()]
    return (
        "usage: combobulator <command> [args]\n\n"
        + "\n".join(lines)
        + "\n\nPer-campaign settings: ENV_FILE=.env.<campaign> combobulator run <folder>\n"
        "Each command accepts --help."
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return
    command, rest = args[0], args[1:]

    if command == "run":
        from transcript_combobulator.batch import main as run_main

        run_main(rest)
    elif command == "file":
        parser = argparse.ArgumentParser(
            prog="combobulator file", description=COMMANDS["file"]
        )
        parser.add_argument("input_file")
        parser.add_argument(
            "--force", action="store_true", help="Ignore completed output"
        )
        opts = parser.parse_args(rest)
        from transcript_combobulator.pipeline import process_file

        process_file(opts.input_file, force=opts.force)
    elif command == "combine":
        parser = argparse.ArgumentParser(
            prog="combobulator combine", description=COMMANDS["combine"]
        )
        parser.add_argument("session", help="Session name under tmp/output/")
        opts = parser.parse_args(rest)
        from transcript_combobulator.combine import combine_transcripts_from_env
        from transcript_combobulator.config import OUTPUT_DIR

        for path in combine_transcripts_from_env(OUTPUT_DIR, opts.session):
            print(f"Combined transcript: {path}")
    elif command == "filter-vtt":
        from transcript_combobulator.vtt_filter import main as filter_main

        filter_main(rest)
    elif command == "compare":
        from transcript_combobulator.compare import main as compare_main

        compare_main(rest)
    elif command == "session-env":
        from transcript_combobulator.session_env import main as session_env_main

        session_env_main(rest)
    elif command == "setup-model":
        from transcript_combobulator.model_setup import main as setup_main

        setup_main(rest)
    elif command == "samples":
        from transcript_combobulator.samples import main as samples_main

        samples_main(rest)
    else:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
