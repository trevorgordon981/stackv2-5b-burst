# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-27

### Added
- **Atomic generation-based checkpointing & full-state resume (`checkpoint_io.py`, `train.py`):**
  - Immutable `gen_<step:08d>_<id>/` generation directories with SHA-256 `manifest.json` integrity verification and single atomic `os.replace` `LATEST.json` pointer commits.
  - Unified full-state checkpoint publication across periodic saves, graceful `SIGINT`/`SIGTERM`/`STOP_AT_STEP` stops, and final saves (persisting model, optimizer, RNG states, deterministic `DeterministicBlockStream` data cursor, `completed_step`/`next_step`, and portable run identity), plus separate inference-only exports rejected for training resume.
  - Strict preflight identity verification (model config, global batch size `BS * ACC`, schedule, bundled tokenizer SHA-256, and shard-set fingerprint) executed before GPU/model allocation.
- **Concurrency safety & fail-closed burst orchestration (`go.sh`, `stage-ckpt.sh`, `setup-on-rental.sh`):**
  - Cross-process `fcntl.flock` serialization (`.latest.lock`) for `LATEST.json` monotonic read-check-swap, promotion, and retention pruning, paired with per-directory in-flight leases preventing premature deletion of active generations.
  - Monotonic `LATEST.json` rollback guard refusing backward step transitions unless explicitly authorized (`--allow-rollback` / `ALLOW_ROLLBACK=1`).
  - Fail-closed launcher (`go.sh`) and end-to-end NAS staging flow (`stage-ckpt.sh`: `to-nas`, `push-nas`, `from-nas`) with identity pins (`EXPECTED_STEP`, `EXPECTED_GENERATION`, `EXPECTED_SHA256`), bundled `./tokenizer/`, pinned `requirements.txt` (`transformers==5.10.4`), and explicit `ALLOW_COLD_START=1` gating.
- **Verification & documentation:**
  - CPU/stdlib local fixture round-trip CLI (`python checkpoint_io.py local-roundtrip`), shell syntax checks, and 37 regression tests wired into GitHub Actions CI.
  - Explicit `Python >= 3.11` prerequisites and verified measured results in `README.md`.
