.PHONY: help check-deps setup install setup-whisper run run-single \
        combine-transcripts filter-vtt session-env compare-transcripts create-sample-files sample-ami check-sample test test-fast build lint format typecheck lint-pyrefly \
        clean-output clean-tmp

ROOT_DIR := $(shell pwd)

# Interpreter used to create the virtualenv. Any CPython >= 3.11 works: the
# distro python3 on Linux, or Homebrew/pyenv/python.org on macOS. Override with
# `make setup PYTHON=/path/to/python3.12`.
PYTHON ?= python3
# pip extras to install. `mac` adds mlx-whisper on Apple Silicon.
EXTRAS ?= dev

# The installed console script; ENV_FILE passes through from the environment.
VENV_CMD = cd $(ROOT_DIR) && . .venv/bin/activate &&
CLI = $(VENV_CMD) combobulator

.SILENT:
.DEFAULT_GOAL := help

help: ## Show this help
	grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*## "}{printf "  %-20s %s\n", $$1, $$2}'
	echo
	echo "Per-campaign config: ENV_FILE=.env.<campaign> make run folder=tmp/input/<session>"

# ── Setup ──
setup: check-deps ## Create .venv from $(PYTHON), install [$(EXTRAS)], download the whisper model
	cd $(ROOT_DIR) && $(PYTHON) -m venv .venv
	$(VENV_CMD) pip install --upgrade pip
	$(MAKE) install
	$(MAKE) setup-whisper

# Verify host prerequisites without touching the venv. Recipes run under
# /bin/sh, so keep this POSIX (no bashisms like `&>`).
check-deps: ## Verify Python 3.11+, the venv module, and ffmpeg
	if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(PYTHON) not found. Install Python 3.11+ (Debian: apt install python3 python3-venv;" \
		     "macOS: brew install python) or point at one: make setup PYTHON=/path/to/python3"; \
		exit 1; \
	fi
	if ! $(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then \
		echo "Python 3.11+ required; $(PYTHON) is $$($(PYTHON) --version 2>&1)." \
		     "Point at a newer one: make setup PYTHON=/path/to/python3"; \
		exit 1; \
	fi
	if ! $(PYTHON) -c 'import venv, ensurepip' >/dev/null 2>&1; then \
		echo "The venv module is incomplete for $(PYTHON). Debian/Ubuntu: apt install python3-venv"; \
		exit 1; \
	fi
	if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "ffmpeg not found; it is required for audio decoding." \
		     "Debian/Raspberry Pi OS: apt install ffmpeg; macOS: brew install ffmpeg"; \
		exit 1; \
	fi
	echo "Prerequisites OK: $$($(PYTHON) --version 2>&1) at $$(command -v $(PYTHON)); $$(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"

install: ## Install the project into .venv (EXTRAS=dev or EXTRAS=dev,mac)
	$(VENV_CMD) pip install -e ".[$(EXTRAS)]"

setup-whisper: ## Download WHISPER_MODEL into models/ without loading it
	$(CLI) setup-model

# ── Pipeline ──
run: ## Full session pipeline: folder=tmp/input/<session> [force=1]
	if [ -z "$$folder" ]; then echo "Usage: make run folder=tmp/input/<session> [force=1]"; exit 1; fi
	if [ ! -d "$$folder" ]; then echo "Directory not found: $$folder"; exit 1; fi
	session_name=$$(basename "$$folder"); \
	force_flag=""; if [ "$$force" = "1" ]; then force_flag="--force"; fi; \
	$(CLI) run "$$folder" --session "$$session_name" $$force_flag

run-single: ## Convert -> VAD -> transcribe one file, no combine: file=path [force=1]
	if [ -z "$$file" ]; then echo "Usage: make run-single file=path/to/file.flac [force=1]"; exit 1; fi
	if [ ! -f "$$file" ]; then echo "File not found: $$file"; exit 1; fi
	echo "Processing $$(basename "$$file")..."
	force_flag=""; if [ "$$force" = "1" ]; then force_flag="--force"; fi; \
	$(CLI) file "$$file" $$force_flag

combine-transcripts: ## Merge per-speaker VTTs into one transcript: session=<name>
	if [ -z "$$session" ]; then echo "Usage: make combine-transcripts session=<name>"; exit 1; fi
	echo "Combining transcripts for $$session..."
	$(CLI) combine "$$session"

filter-vtt: ## Rewrite a speaker's VTT from saved JSON above a confidence: file=<input audio> [threshold=50]
	if [ -z "$$file" ]; then echo "Usage: make filter-vtt file=tmp/input/<session>/<speaker>.flac [threshold=50]"; exit 1; fi
	if [ -n "$$threshold" ]; then \
		$(CLI) filter-vtt "$$file" --threshold "$$threshold"; \
	else \
		$(CLI) filter-vtt "$$file"; \
	fi

session-env: ## Write a bundle's session env outside it: bundle=<dir> output=<env> [speakers=<json>] [base=<env>] [dm=<track_id>] [tz=<zone>] [force=1]
	if [ -z "$$bundle" ] || [ -z "$$output" ]; then echo "Usage: make session-env bundle=<dir> output=<env> [speakers=<json>] [base=<env>] [dm=<track_id>] [tz=<zone>] [force=1]"; exit 1; fi
	set -- --bundle "$$bundle" --output "$$output"; \
	if [ -n "$$speakers" ]; then set -- "$$@" --speakers "$$speakers"; fi; \
	if [ -n "$$base" ]; then set -- "$$@" --base "$$base"; fi; \
	if [ -n "$$dm" ]; then set -- "$$@" --dm "$$dm"; fi; \
	if [ -n "$$tz" ]; then set -- "$$@" --tz "$$tz"; fi; \
	if [ "$$force" = "1" ]; then set -- "$$@" --force; fi; \
	$(CLI) session-env "$$@"

compare-transcripts: ## Compare two combined transcripts of one session, per speaker: ref="a-1.txt a-2.txt" hyp="b-1.txt b-2.txt" [json=out.json]
	if [ -z "$$ref" ] || [ -z "$$hyp" ]; then echo "Usage: make compare-transcripts ref=\"<ref files>\" hyp=\"<new files>\" [json=out.json]"; exit 1; fi
	set -f; set -- --ref $$ref --hyp $$hyp; \
	if [ -n "$$json" ]; then set -- "$$@" --json "$$json"; fi; \
	$(CLI) compare "$$@"

# ── Dev ──
create-sample-files: ## Build tmp/input/jfk-sample/ from samples/
	if [ ! -d "$(ROOT_DIR)/samples" ]; then echo "Samples directory not found at samples/"; exit 1; fi
	$(CLI) samples

sample-ami: ## Fetch the 4-speaker AMI sample (checksum-verified) into tmp/input/ami-en2002b/
	$(VENV_CMD) combobulator samples --ami

# The reference was made with .env.ami-sample on CPU. Any change in output fails: use it
# for changes that should not change words (runtime, refactor, dependency bump).
# SAMPLE_ENV=<env> overrides it.
check-sample: sample-ami ## Transcribe the AMI sample and fail if it differs from samples/ami/
	env_file="$${SAMPLE_ENV:-.env.ami-sample}"; \
	$(VENV_CMD) ENV_FILE="$$env_file" combobulator run tmp/input/ami-en2002b --session ami-en2002b && \
	echo "Accuracy on this excerpt only (no gate):" && \
	{ ( $(VENV_CMD) ENV_FILE="$$env_file" combobulator compare --ref samples/ami/EN2002b-human.txt \
		--hyp tmp/output/ami-en2002b/ami-en2002b-combined.txt --wer ) || \
		echo "The accuracy report failed; the drift check still runs." >&2; } && \
	echo "Drift against this project's own output (the gate):" && \
	$(VENV_CMD) ENV_FILE="$$env_file" combobulator compare --ref samples/ami/EN2002b-reference.txt \
		--hyp tmp/output/ami-en2002b/ami-en2002b-combined.txt --strict

test: ## Run the suite scripts/check runs, including real whisper/VAD inference
	$(VENV_CMD) python -m pytest tests/ -v -m "not mlx"

test-mlx: ## Run the opt-in MLX inference test (needs the mlx checkpoint)
	$(VENV_CMD) python -m pytest tests/ -v -m mlx

test-fast: ## Run the suite without the slow inference tests
	$(VENV_CMD) python -m pytest tests/ -v -m "not slow and not mlx"

build: ## Build the sdist and wheel into dist/, as CI does
	$(VENV_CMD) python -m build

lint: ## Run ruff check and ruff format --check
	$(VENV_CMD) ruff check --output-format concise . && ruff format --check .

format: ## Format with ruff
	$(VENV_CMD) ruff format .

typecheck: ## Run mypy (strict)
	$(VENV_CMD) mypy transcript_combobulator

lint-pyrefly: ## Run pyrefly beside mypy (trial; mypy stays the gate)
	$(VENV_CMD) pyrefly check

# ── Cleanup ──
# These delete data. tmp/output holds every session's transcripts, metrics,
# and logs; nothing else ever removes it (make test does not).
clean-output: ## Delete tmp/output/* (all session transcripts, metrics, logs)
	cd $(ROOT_DIR) && find tmp/output -mindepth 1 -not -name ".gitkeep" -delete
	echo "Removed everything under tmp/output/"

clean-tmp: clean-output ## Delete tmp/output/* and tmp/input/* and caches
	cd $(ROOT_DIR) && find tmp -mindepth 1 -not -name ".gitkeep" -not -path "tmp/output" -not -path "tmp/input" -delete
	cd $(ROOT_DIR) && mkdir -p tmp/input tmp/output
	cd $(ROOT_DIR) && rm -rf __pycache__/ */__pycache__/ */*/__pycache__/ .pytest_cache/ .coverage .mypy_cache/
	echo "Removed everything under tmp/ and local caches"
