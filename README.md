# stackv2-5b-burst

Burst-train a ~5B dense Llama-style model on The Stack v2 across a primary training host and rented GPUs (for example, 96GB Blackwell instances), with atomic generation-based checkpointing, fail-closed transfers, and full-state resumability.

## Checkpoint & Resume Contract

Every training checkpoint is published by `checkpoint_io.py` (`publish_checkpoint`) as an **immutable generation directory** plus a **single atomic commit pointer**:

```text
$CKPT_DIR/
  LATEST.json                        # Atomically replaced commit pointer (temp + fsync + os.replace)
  gen_00000200_<generation_id>/      # Immutable generation directory
    checkpoint.pt                    # Full training state payload
    manifest.json                    # Integrity manifest (version=2, sha256, size_bytes, step, identity)
  exports/                           # Optional inference-only exports (never pointed to by LATEST.json)
    inference_step_00000249_<id>/
      checkpoint.pt
      manifest.json                  # kind="inference_export" (rejected for training resume)
```

### Guaranteed Properties

1. **Atomic publication (no torn pairs):** `checkpoint.pt` and `manifest.json` are written and `fsync`ed inside a hidden `.staging_gen_*` directory, verified against SHA-256, promoted to an immutable `gen_<step:08d>_<id>/` directory, and only then committed via a single atomic `os.replace` of `LATEST.json`. Any crash before the final pointer swap leaves the previous generation and `LATEST.json` intact and usable—even when re-publishing the same step.
2. **Unified periodic, graceful-stop, and final checkpoints:** Periodic saves (`SAVE`), deliberate stops (`SIGINT`, `SIGTERM`, or `STOP_AT_STEP`), and end-of-run completion all publish through the exact same full-state generation path (`is_final=true` for final/stop saves). Tail progress after the last periodic interval is never lost to an unresumable weights-only `final.pt`. Optional inference-only exports (`EXPORT_INFERENCE_WEIGHTS=1`) are written under `$CKPT_DIR/exports/` with `kind="inference_export"` and are explicitly rejected if passed to training resume.
3. **Complete state persistence:** Every training checkpoint saves and restores:
   - `model` state dict
   - `opt` optimizer state dict
   - `rng_state` (Python `random` state, plus PyTorch CPU and CUDA RNG states when running under PyTorch)
   - `data_cursor` (`seed`, `epoch`, `shard_index`, `line_index`, `token_buffer`, `blocks_emitted`, `samples_consumed`) via `DeterministicBlockStream`
   - `completed_step` and `next_step` (`completed_step + 1`, never duplicating the last completed step)
   - `identity` (portable model, training schedule, tokenizer, and shard-set fingerprint)
4. **Strict preflight identity verification (before GPU allocation):** Before allocating `LlamaForCausalLM` or CUDA memory, `train.py` runs `preflight_resume(...)` and rejects any mismatch in:
   - **Model config:** `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`, `rms_norm_eps`, `rope_theta`
   - **Training config:** `seq_len`, `global_batch_size` (`BS * ACC`), `lr_max`, `lr_min`, `warmup_steps`, `total_steps`, `seed` (per-device `BS` and `ACC` may vary across hosts as long as `BS * ACC` stays identical, e.g. `6 * 8 = 48` and `16 * 3 = 48`)
   - **Tokenizer identity:** SHA-256 digest of bundled tokenizer files, `vocab_size`, and `eos_token_id`
   - **Data identity:** shard count, language filter, and deterministic per-shard relative-path/size/content fingerprint
5. **Fail-closed startup & transfers:**
   - `go.sh` is the **single supported launcher** (`setup-on-rental.sh` is a deprecated compatibility shim that delegates directly to `go.sh`).
   - Failed data or checkpoint transfers in `go.sh` abort non-zero immediately and **never** launch `train.py`, **never** silently restart from step 0, and **never** fall back to a stale preexisting checkpoint in `$CKPT_DIR`.
   - Remote checkpoint pulls and pushes stage into a fresh unique temporary directory on the destination and commit via `checkpoint_io.py commit-staged` (verifying `LATEST.json`, `manifest.json`, SHA-256, byte size, pre-existing destination generation collisions, and optional `EXPECTED_STEP` / `EXPECTED_GENERATION` / `EXPECTED_SHA256` pins) before atomically swapping `$CKPT_DIR/LATEST.json`.
   - **Monotonic pointer protection (no silent rollback):** `publish_checkpoint`, `commit-staged`, `go.sh`, and `stage-ckpt.sh` serialize destination `LATEST.json` read-check-swap critical sections under an exclusive cross-process file lock (`$CKPT_DIR/.latest.lock` via `fcntl.flock(LOCK_EX)`), re-reading on-disk `LATEST.json` inside the lock and refusing to roll `LATEST.json` backward to an older `completed_step` unless explicitly authorized with `--allow-rollback` (or `ALLOW_ROLLBACK=1`).
   - Starting from step 0 with an empty `$CKPT_DIR` is refused by default; it requires explicit `ALLOW_COLD_START=1` (and still refuses to run if a remote pull failed).

## Prerequisites

- **Hardware & CUDA runtime:** NVIDIA GPU with CUDA 12.8+ (for Blackwell `sm_120`, e.g. RTX PRO 6000). `go.sh` installs `torch==2.11.*` from `https://download.pytorch.org/whl/cu128` unless a compatible CUDA PyTorch (>= 2.7) is already installed. Override with `TORCH_SPEC` and `TORCH_INDEX_URL` if your driver requires a different CUDA wheel.
- **Pinned Python packages:** `requirements.txt` pins every dependency (`transformers==5.10.4`, `bitsandbytes==0.49.2`, `zstandard==0.25.0`, `huggingface_hub==1.4.1`, `accelerate==1.12.0`) so model and 8-bit AdamW optimizer state reload cleanly across hosts.
- **Bundled tokenizer:** `./tokenizer/` is included in this repository and is the default `TOKENIZER` for both `go.sh` and `train.py` (no Hugging Face token required).
- **Remote NAS mode (`BURST_MODE=nas`):** Requires:
  - A Tailscale auth key (`TS_AUTHKEY`, recommended reusable + ephemeral + tagged) and Tailscale ACL permitting the tag to reach `NAS_HOST`.
  - SSH authorization to `NAS_HOST` via either Tailscale SSH or an SSH deploy key (`NAS_KEY=/path/to/key`, required for key-only `sshd` hosts).
  - Explicit remote configuration via environment variables or a gitignored local `secrets.env` (see [secrets.env.example](secrets.env.example)): `NAS_HOST`, `NAS_USER`, `NAS_DATA`, `NAS_CKPT`.

## Quick Start

### 1. Local Fixture Round-Trip (CPU / stdlib-only, no GPU or NAS required)

Verify the bundled tokenizer identity, atomic generation publication, staging commit, final full-state checkpoint resume, and bit-for-bit continuity against an uninterrupted reference run:

```bash
python3 checkpoint_io.py local-roundtrip
```

Run the full regression test suite and shell syntax checks:

```bash
bash -n go.sh setup-on-rental.sh stage-ckpt.sh
python3 -m unittest discover -s tests -v
```

### 2. Local / Pre-Staged Burst Launch (`BURST_MODE=local`)

For an explicit cold start on locally staged shards in `$DATA_ROOT/<lang>/*.zst`:

```bash
BURST_MODE=local SKIP_DEPS=1 ALLOW_COLD_START=1 \
  DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpt \
  bash go.sh
```

To resume from an existing verified checkpoint generation (or atomically stage one from `LOCAL_CKPT_SRC`):

```bash
BURST_MODE=local SKIP_DEPS=1 EXPECTED_STEP=200 \
  DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpt \
  bash go.sh
```

### 3. Remote NAS Burst Loop (`BURST_MODE=nas`)

1. **Before the burst (primary -> NAS)** — atomically stage the primary host's latest verified generation to the NAS with identity pins (`EXPECTED_STEP`, `EXPECTED_GENERATION`, `EXPECTED_SHA256`):
   ```bash
   PRIMARY_HOST=user@primary.example PRIMARY_CKPT_DIR=/srv/runs/stackv2-5b \
   NAS_HOST=user@nas.example NAS_CKPT_DIR=/srv/checkpoints/burst \
   EXPECTED_STEP=1000 EXPECTED_GENERATION=gen_00001000_<id> EXPECTED_SHA256=<sha256> \
   ./stage-ckpt.sh to-nas
   ```
2. **On the rented GPU (NAS -> rental + train)** — copy `secrets.env.example` to `secrets.env` (gitignored) or pass env vars, pin the exact handoff identity, and launch:
   ```bash
   EXPECTED_STEP=1000 EXPECTED_GENERATION=gen_00001000_<id> EXPECTED_SHA256=<sha256> \
   bash go.sh
   ```
   Defaults (`BS=16 ACC=3 SEQ_LEN=8192 ROPE_THETA=500000`) keep `global_batch_size=48` aligned with the primary training schedule.
3. **End of burst — leg 1 (rented GPU -> NAS)** — stop cleanly with `SIGTERM`/`SIGINT` or `STOP_AT_STEP` so `train.py` publishes a final full-state numbered generation (for example step `1249`), inspect the final generation/SHA-256 via `python3 checkpoint_io.py inspect --pointer "$CKPT_DIR/LATEST.json"`, and atomically push it from the rented GPU back to the NAS with identity pins:
   ```bash
   CKPT_DIR="$HOME/stackv2-burst-run/ckpt" \
   NAS_HOST=user@nas.example NAS_CKPT_DIR=/srv/checkpoints/burst \
   EXPECTED_STEP=1249 EXPECTED_GENERATION=gen_00001249_<id> EXPECTED_SHA256=<sha256> \
   ./stage-ckpt.sh push-nas
   ```
   (On a rental host using `go.sh`'s userspace Tailscale SSH wrapper, pass `SSH_BIN="$HOME/stackv2-burst-run/_nassh"`. If pushing from an orchestrator host with SSH access to the rental host instead, equivalently run `PRIMARY_HOST=user@rental.example PRIMARY_CKPT_DIR=/path/to/ckpt ... ./stage-ckpt.sh to-nas`.)
4. **End of burst — leg 2 (NAS -> primary host)** — once the rental generation is verified on the NAS, atomically stage it from the NAS back onto the primary host with the exact same identity pins:
   ```bash
   PRIMARY_HOST=user@primary.example PRIMARY_CKPT_DIR=/srv/runs/stackv2-5b \
   NAS_HOST=user@nas.example NAS_CKPT_DIR=/srv/checkpoints/burst \
   EXPECTED_STEP=1249 EXPECTED_GENERATION=gen_00001249_<id> EXPECTED_SHA256=<sha256> \
   ./stage-ckpt.sh from-nas
   ```
   Destination promotion stages into a fresh unique temporary directory, runs full manifest + SHA-256 + byte-size + collision verification via `checkpoint_io.py commit-staged` on the destination host before atomically swapping `LATEST.json`, and refuses to roll `LATEST.json` backward to an older step unless `--allow-rollback` (or `ALLOW_ROLLBACK=1`) is explicitly passed.

## Explicit Limitations

- **Single-stream deterministic data cursor:** To guarantee exact `(epoch, shard_index, line_index, token_buffer)` continuity without prefetch skew, `train.py` consumes shards via in-process `DeterministicBlockStream` rather than asynchronous multi-worker `DataLoader` prefetching.
- **Shard fingerprinting threshold:** Shards `<= 16 MiB` are hashed in full with SHA-256. Larger multi-GB shards are fingerprinted by exact byte size plus SHA-256 of head, middle, and tail 64 KiB windows so startup preflight remains fast; a bit flip strictly outside those sampled windows in a multi-GB shard is not detected by startup preflight.
- **Cross-hardware optimizer & CUDA RNG compatibility:** Resuming 8-bit AdamW (`bitsandbytes`) and CUDA RNG states assumes compatible PyTorch, CUDA, and `bitsandbytes` versions (pinned in `requirements.txt` and `go.sh`). Changing GPU device count restores device-0 CUDA RNG state when device counts differ.
- **Strict shard-set invariance:** Because the persisted `data_cursor` indexes into the deterministic epoch permutation of discovered shards, adding, removing, or modifying shards under `DATA_ROOT` intentionally fails identity verification on resume.
- **Legacy v1 flat checkpoints:** Older `ckpt_*.pt` + `ckpt_*.pt.sha256` files remain readable via `load_latest_checkpoint(..., require_full_state=False)` for migration, but they lack `rng_state`, `data_cursor`, and `identity` metadata and cannot satisfy strict full-state resume checks.

## License

MIT, see [LICENSE](LICENSE).
