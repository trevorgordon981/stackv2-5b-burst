#!/usr/bin/env python3
"""Atomic generation-based checkpoint persistence, integrity & identity verification,
RNG state capture/restore, and deterministic data cursor streaming.

Stdlib-only at import time (PyTorch is imported lazily only when default
torch.save / torch.load or torch RNG capture/restore is invoked).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import hmac
import json
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import uuid
from typing import Any, Callable, Iterable

POINTER_FILENAME = "LATEST.json"
GENERATION_MANIFEST_FILENAME = "manifest.json"
GENERATION_PAYLOAD_FILENAME = "checkpoint.pt"
LEGACY_MANIFEST_SUFFIX = ".sha256"
MANIFEST_VERSION = 2
LEGACY_MANIFEST_VERSION = 1
POINTER_VERSION = 2
DATA_CURSOR_VERSION = 1
RNG_STATE_VERSION = 1
FULL_HASH_MAX_BYTES = 16 * 1024 * 1024
SAMPLED_CHUNK_BYTES = 64 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GEN_DIR_RE = re.compile(r"^gen_(\d+)_([0-9a-zA-Z_-]+)$")
_CKPT_STEP_RE = re.compile(r"^ckpt_(\d+)\.pt$")
_TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint, pointer, or integrity manifest is missing, corrupt, or inconsistent."""


class CheckpointIdentityError(CheckpointIntegrityError):
    """Raised when a checkpoint's model, training config, tokenizer, or data identity does not match."""


def manifest_path_for(ckpt_path: str) -> str:
    """Return the manifest path associated with a checkpoint file or generation directory."""
    if os.path.isdir(ckpt_path):
        return os.path.join(ckpt_path, GENERATION_MANIFEST_FILENAME)
    if os.path.basename(ckpt_path) == GENERATION_PAYLOAD_FILENAME:
        return os.path.join(os.path.dirname(ckpt_path), GENERATION_MANIFEST_FILENAME)
    return f"{ckpt_path}{LEGACY_MANIFEST_SUFFIX}"


def pointer_path_for(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, POINTER_FILENAME)


def _require_nonneg_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointIntegrityError(f"Invalid {field_name} in checkpoint metadata: {value!r}")
    return value


def _fsync_dir(dir_path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dfd = os.open(dir_path, flags)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _canonical_digest(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_file_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def compute_shard_fingerprint(path: str) -> dict[str, Any]:
    """Compute a deterministic fingerprint for a data shard.

    Files <= 16 MiB are hashed in full; larger files hash size plus head, middle,
    and tail 64 KiB windows so multi-GB shard validation stays fast while catching
    truncation, replacement, or boundary corruption.
    """
    size = os.path.getsize(path)
    if size <= FULL_HASH_MAX_BYTES:
        digest, _ = compute_file_sha256(path)
        return {"size_bytes": size, "mode": "full", "sha256": digest}

    h = hashlib.sha256()
    h.update(f"sampled:{size}:".encode("utf-8"))
    with open(path, "rb") as fh:
        fh.seek(0)
        h.update(fh.read(SAMPLED_CHUNK_BYTES))
        mid = max(0, (size // 2) - (SAMPLED_CHUNK_BYTES // 2))
        fh.seek(mid)
        h.update(fh.read(SAMPLED_CHUNK_BYTES))
        tail = max(0, size - SAMPLED_CHUNK_BYTES)
        fh.seek(tail)
        h.update(fh.read(SAMPLED_CHUNK_BYTES))
    return {"size_bytes": size, "mode": "sampled_64k", "sha256": h.hexdigest()}


def discover_shards(
    data_root: str,
    langs: list[str] | None = None,
    extensions: tuple[str, ...] = (".zst",),
) -> list[str]:
    """Return a sorted list of shard paths under data_root/<lang>/*<ext>."""
    found: list[str] = []
    if langs:
        for lang in sorted(langs):
            for ext in extensions:
                found.extend(glob.glob(os.path.join(data_root, lang, f"*{ext}")))
    else:
        for ext in extensions:
            found.extend(glob.glob(os.path.join(data_root, "*", f"*{ext}")))
    found = sorted(set(p for p in found if os.path.isfile(p)))
    return found


def compute_tokenizer_identity(
    tokenizer_path: str,
    vocab_size: int | None = None,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    """Compute a portable identity fingerprint for a local tokenizer directory or repo ID."""
    if os.path.isdir(tokenizer_path):
        file_hashes: dict[str, str] = {}
        for fname in _TOKENIZER_FILES:
            fpath = os.path.join(tokenizer_path, fname)
            if os.path.isfile(fpath):
                digest, _ = compute_file_sha256(fpath)
                file_hashes[fname] = digest
        if not file_hashes:
            raise CheckpointIntegrityError(
                f"Tokenizer directory contains no recognized tokenizer files: {tokenizer_path}"
            )
        cfg_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                tok_cfg = json.load(fh)
            if vocab_size is None and isinstance(tok_cfg.get("vocab_size"), int):
                vocab_size = tok_cfg["vocab_size"]
            if eos_token_id is None:
                eos_tok = tok_cfg.get("eos_token")
                decoder = tok_cfg.get("added_tokens_decoder", {})
                if isinstance(decoder, dict) and isinstance(eos_tok, str):
                    for idx_str, meta in decoder.items():
                        if isinstance(meta, dict) and meta.get("content") == eos_tok:
                            try:
                                eos_token_id = int(idx_str)
                                break
                            except ValueError:
                                pass
        return {
            "source": "local_dir",
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "files_sha256": _canonical_digest(file_hashes),
            "files": file_hashes,
        }

    return {
        "source": f"external:{tokenizer_path}",
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "files_sha256": hashlib.sha256(str(tokenizer_path).encode("utf-8")).hexdigest(),
        "files": {},
    }


def compute_data_identity(
    data_root: str,
    langs: list[str] | None = None,
    shard_paths: list[str] | None = None,
    extensions: tuple[str, ...] = (".zst",),
) -> dict[str, Any]:
    """Compute a portable fingerprint over the ordered shard set."""
    shards = list(shard_paths) if shard_paths is not None else discover_shards(data_root, langs=langs, extensions=extensions)
    shards = sorted(shards)
    abs_root = os.path.abspath(data_root)
    entries = []
    for path in shards:
        rel = os.path.relpath(os.path.abspath(path), abs_root).replace(os.sep, "/")
        fp = compute_shard_fingerprint(path)
        entries.append(
            {
                "relpath": rel,
                "size_bytes": fp["size_bytes"],
                "mode": fp["mode"],
                "sha256": fp["sha256"],
            }
        )
    entries.sort(key=lambda item: item["relpath"])
    return {
        "shard_count": len(entries),
        "langs": sorted(langs) if langs is not None else None,
        "shards_sha256": _canonical_digest(entries),
        "shards": entries,
    }


def extract_model_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture & positional-encoding invariants that must match on resume."""
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None and isinstance(cfg.get("rope_parameters"), dict):
        rope_theta = cfg["rope_parameters"].get("rope_theta")
    if rope_theta is None:
        rope_theta = 10000.0
    return {
        "vocab_size": int(cfg["vocab_size"]),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(rope_theta),
    }


def build_run_identity(
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    data_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build a canonical identity record binding model, training schedule, tokenizer, and data."""
    model_part = extract_model_identity(model_cfg)
    training_part = {
        "seq_len": int(training_cfg["seq_len"]),
        "global_batch_size": int(training_cfg["global_batch_size"]),
        "lr_max": float(training_cfg["lr_max"]),
        "lr_min": float(training_cfg["lr_min"]),
        "warmup_steps": int(training_cfg["warmup_steps"]),
        "total_steps": int(training_cfg["total_steps"]),
        "seed": int(training_cfg["seed"]),
    }
    tokenizer_part = {
        "source": tokenizer_identity.get("source"),
        "vocab_size": tokenizer_identity.get("vocab_size"),
        "eos_token_id": tokenizer_identity.get("eos_token_id"),
        "files_sha256": tokenizer_identity.get("files_sha256"),
    }
    data_part = {
        "shard_count": int(data_identity["shard_count"]),
        "langs": data_identity.get("langs"),
        "shards_sha256": data_identity["shards_sha256"],
    }
    body = {
        "model": model_part,
        "training": training_part,
        "tokenizer": tokenizer_part,
        "data": data_part,
    }
    return {
        **body,
        "identity_sha256": _canonical_digest(body),
    }


def verify_run_identity(saved_identity: Any, expected_identity: dict[str, Any]) -> None:
    """Strictly verify saved run identity against expected identity; report all mismatches clearly."""
    if not isinstance(saved_identity, dict):
        raise CheckpointIdentityError(
            f"Checkpoint is missing valid run identity metadata (got {type(saved_identity).__name__})"
        )

    mismatches: list[str] = []
    for section in ("model", "training", "tokenizer", "data"):
        exp_sec = expected_identity.get(section)
        sav_sec = saved_identity.get(section)
        if not isinstance(sav_sec, dict):
            mismatches.append(f"{section}: missing in checkpoint")
            continue
        if isinstance(exp_sec, dict):
            for key, exp_val in sorted(exp_sec.items()):
                sav_val = sav_sec.get(key)
                if sav_val != exp_val:
                    mismatches.append(f"{section}.{key}: checkpoint={sav_val!r} != expected={exp_val!r}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise CheckpointIdentityError(f"Checkpoint identity mismatch — refusing resume: {joined}")

    exp_hash = expected_identity.get("identity_sha256")
    sav_hash = saved_identity.get("identity_sha256")
    if exp_hash and sav_hash and not hmac.compare_digest(str(sav_hash), str(exp_hash)):
        raise CheckpointIdentityError(
            f"Checkpoint identity_sha256 mismatch: checkpoint={sav_hash} != expected={exp_hash}"
        )


def capture_rng_state(include_torch: bool = True, base_seed: int = 0) -> dict[str, Any]:
    """Capture Python RNG state and (when available/requested) PyTorch CPU & CUDA RNG states."""
    torch_cpu = None
    torch_cuda = None
    if include_torch:
        try:
            import torch

            torch_cpu = torch.get_rng_state().cpu().tolist()
            if torch.cuda.is_available():
                torch_cuda = [s.cpu().tolist() for s in torch.cuda.get_rng_state_all()]
        except ImportError:
            pass
    return {
        "version": RNG_STATE_VERSION,
        "base_seed": int(base_seed),
        "python": random.getstate(),
        "torch_cpu": torch_cpu,
        "torch_cuda": torch_cuda,
    }


def _normalize_python_rng_state(raw_state: Any) -> tuple[Any, ...]:
    if not isinstance(raw_state, (list, tuple)) or len(raw_state) != 3:
        raise CheckpointIntegrityError("Invalid Python RNG state structure in checkpoint")
    version, internal, gauss_next = raw_state
    if not isinstance(internal, (list, tuple)):
        raise CheckpointIntegrityError("Invalid Python RNG internal state in checkpoint")
    return (version, tuple(internal), gauss_next)


def restore_rng_state(rng_state: Any, include_torch: bool = True) -> None:
    """Restore Python RNG state and (when available/requested) PyTorch CPU & CUDA RNG states."""
    if not isinstance(rng_state, dict) or "python" not in rng_state:
        raise CheckpointIntegrityError("Checkpoint missing required 'rng_state' metadata")
    random.setstate(_normalize_python_rng_state(rng_state["python"]))
    if include_torch and rng_state.get("torch_cpu") is not None:
        try:
            import torch

            cpu_bytes = torch.tensor(rng_state["torch_cpu"], dtype=torch.uint8)
            torch.set_rng_state(cpu_bytes)
            cuda_states = rng_state.get("torch_cuda")
            if cuda_states is not None and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if len(cuda_states) == device_count:
                    torch.cuda.set_rng_state_all(
                        [torch.tensor(st, dtype=torch.uint8) for st in cuda_states]
                    )
                elif cuda_states:
                    torch.cuda.set_rng_state(torch.tensor(cuda_states[0], dtype=torch.uint8), device=0)
        except ImportError:
            pass


def validate_data_cursor(cursor: Any) -> dict[str, Any]:
    """Validate a serialized data cursor payload."""
    if not isinstance(cursor, dict):
        raise CheckpointIntegrityError(f"Checkpoint data_cursor must be a dict, got {type(cursor).__name__}")
    version = cursor.get("version")
    if version != DATA_CURSOR_VERSION:
        raise CheckpointIntegrityError(f"Unsupported data_cursor version: {version!r}")
    seed = _require_nonneg_int(cursor.get("seed"), "data_cursor.seed")
    epoch = _require_nonneg_int(cursor.get("epoch"), "data_cursor.epoch")
    shard_index = _require_nonneg_int(cursor.get("shard_index"), "data_cursor.shard_index")
    line_index = _require_nonneg_int(cursor.get("line_index"), "data_cursor.line_index")
    blocks_emitted = _require_nonneg_int(cursor.get("blocks_emitted"), "data_cursor.blocks_emitted")
    samples_consumed = _require_nonneg_int(cursor.get("samples_consumed", blocks_emitted), "data_cursor.samples_consumed")
    token_buffer = cursor.get("token_buffer")
    if not isinstance(token_buffer, list) or not all(
        isinstance(t, int) and not isinstance(t, bool) for t in token_buffer
    ):
        raise CheckpointIntegrityError("Invalid data_cursor.token_buffer in checkpoint")
    return {
        "version": DATA_CURSOR_VERSION,
        "seed": seed,
        "epoch": epoch,
        "shard_index": shard_index,
        "line_index": line_index,
        "token_buffer": list(token_buffer),
        "blocks_emitted": blocks_emitted,
        "samples_consumed": samples_consumed,
    }


def deterministic_shard_order(shards: list[str], seed: int, epoch: int) -> list[str]:
    """Return a deterministic epoch permutation of sorted shard paths."""
    ordered = list(shards)
    rng = random.Random(f"stackv2-burst-shard-order:{seed}:{epoch}")
    rng.shuffle(ordered)
    return ordered


def open_plain_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            yield line


class DeterministicBlockStream:
    """Deterministic, cursor-resumable packed token-block stream over JSONL shards.

    Because cursor state tracks (epoch, shard_index, line_index, token_buffer),
    restarting from snapshot_cursor() yields the exact same packed blocks bit-for-bit
    as an uninterrupted run across line, shard, and epoch boundaries.
    """

    def __init__(
        self,
        shards: list[str],
        tokenize_fn: Callable[[str], list[int]],
        eos_token_id: int,
        seq_len: int,
        seed: int = 0,
        cursor: dict[str, Any] | None = None,
        line_reader: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        if not shards:
            raise ValueError("DeterministicBlockStream requires at least one shard")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.shards = sorted(shards)
        self.tokenize_fn = tokenize_fn
        self.eos_token_id = int(eos_token_id)
        self.seq_len = int(seq_len)
        self.line_reader = line_reader or open_plain_jsonl

        if cursor is not None:
            validated = validate_data_cursor(cursor)
            if validated["seed"] != int(seed):
                raise CheckpointIdentityError(
                    f"Data cursor seed ({validated['seed']}) does not match stream seed ({seed})"
                )
            if validated["shard_index"] >= len(self.shards):
                raise CheckpointIdentityError(
                    f"Data cursor shard_index ({validated['shard_index']}) out of range for {len(self.shards)} shards"
                )
            self.seed = validated["seed"]
            self.epoch = validated["epoch"]
            self.shard_index = validated["shard_index"]
            self.line_index = validated["line_index"]
            self.token_buffer: list[int] = list(validated["token_buffer"])
            self.blocks_emitted = validated["blocks_emitted"]
            self.samples_consumed = validated["samples_consumed"]
        else:
            self.seed = int(seed)
            self.epoch = 0
            self.shard_index = 0
            self.line_index = 0
            self.token_buffer = []
            self.blocks_emitted = 0
            self.samples_consumed = 0

        self._epoch_shards = deterministic_shard_order(self.shards, self.seed, self.epoch)
        self._active_iter: Iterable[str] | None = None
        self._active_shard_index: int | None = None
        self._active_epoch: int | None = None

    def _open_current_shard(self) -> None:
        path = self._epoch_shards[self.shard_index]
        it = iter(self.line_reader(path))
        for _ in range(self.line_index):
            try:
                next(it)
            except StopIteration:
                break
        self._active_iter = it
        self._active_shard_index = self.shard_index
        self._active_epoch = self.epoch

    def _advance_shard(self) -> None:
        self._active_iter = None
        self._active_shard_index = None
        self._active_epoch = None
        self.shard_index += 1
        self.line_index = 0
        if self.shard_index >= len(self._epoch_shards):
            self.epoch += 1
            self.shard_index = 0
            self._epoch_shards = deterministic_shard_order(self.shards, self.seed, self.epoch)

    def next_block(self) -> list[int]:
        """Return the next packed token block of length seq_len."""
        empty_shard_cycles = 0
        while len(self.token_buffer) < self.seq_len:
            if (
                self._active_iter is None
                or self._active_shard_index != self.shard_index
                or self._active_epoch != self.epoch
            ):
                self._open_current_shard()
            try:
                raw_line = next(self._active_iter)  # type: ignore[arg-type]
                self.line_index += 1
            except StopIteration:
                self._advance_shard()
                empty_shard_cycles += 1
                if empty_shard_cycles > len(self.shards) * 2:
                    raise CheckpointIntegrityError(
                        "Data stream yielded no valid content across all shards; cannot form a sequence block"
                    )
                continue

            try:
                record = json.loads(raw_line)
                content = record["content"]
            except Exception:
                continue
            if not isinstance(content, str):
                continue
            ids = list(self.tokenize_fn(content))
            ids.append(self.eos_token_id)
            self.token_buffer.extend(ids)
            empty_shard_cycles = 0

        block = self.token_buffer[: self.seq_len]
        self.token_buffer = self.token_buffer[self.seq_len :]
        self.blocks_emitted += 1
        self.samples_consumed += 1
        return block

    def next_batch(self, batch_size: int) -> list[list[int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return [self.next_block() for _ in range(batch_size)]

    def snapshot_cursor(self) -> dict[str, Any]:
        return {
            "version": DATA_CURSOR_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "shard_index": self.shard_index,
            "line_index": self.line_index,
            "token_buffer": list(self.token_buffer),
            "blocks_emitted": self.blocks_emitted,
            "samples_consumed": self.samples_consumed,
        }


def build_checkpoint_payload(
    model_state: Any,
    opt_state: Any,
    completed_step: int,
    cfg: dict[str, Any],
    *,
    rng_state: dict[str, Any] | None = None,
    data_cursor: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
    is_final: bool = False,
) -> dict[str, Any]:
    completed_step = _require_nonneg_int(completed_step, "completed_step")
    payload: dict[str, Any] = {
        "kind": "final_training" if is_final else "training",
        "is_final": bool(is_final),
        "model": model_state,
        "opt": opt_state,
        "step": completed_step,
        "completed_step": completed_step,
        "next_step": completed_step + 1,
        "cfg": cfg,
    }
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if data_cursor is not None:
        payload["data_cursor"] = validate_data_cursor(data_cursor)
    if identity is not None:
        payload["identity"] = identity
    return payload


def resolve_resume_step(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        raise CheckpointIntegrityError(f"Checkpoint payload must be a dict, got {type(payload).__name__}")
    if "next_step" in payload:
        next_step = _require_nonneg_int(payload["next_step"], "next_step")
        if "completed_step" in payload:
            completed_step = _require_nonneg_int(payload["completed_step"], "completed_step")
            if next_step != completed_step + 1:
                raise CheckpointIntegrityError(
                    f"Inconsistent step metadata: completed_step={completed_step}, next_step={next_step}"
                )
            if "step" in payload:
                step = _require_nonneg_int(payload["step"], "step")
                if step != completed_step:
                    raise CheckpointIntegrityError(
                        f"Inconsistent step metadata: step={step}, completed_step={completed_step}"
                    )
        return next_step
    if "completed_step" in payload:
        completed_step = _require_nonneg_int(payload["completed_step"], "completed_step")
        return completed_step + 1
    if "step" in payload:
        completed_step = _require_nonneg_int(payload["step"], "step")
        return completed_step + 1
    raise CheckpointIntegrityError("Checkpoint payload missing step metadata")


def _write_json_atomic(path: str, data: dict[str, Any], prefix: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    raw = _canonical_json_bytes(data)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=prefix, suffix=".tmp", delete=False
        ) as fh:
            tmp_path = fh.name
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        _fsync_dir(parent)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _parse_generation_name(name: str) -> tuple[int, str] | None:
    m = _GEN_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def _validate_manifest_dict(manifest: Any, manifest_path: str, expected_filename: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CheckpointIntegrityError(f"Integrity manifest must be a JSON object: {manifest_path}")
    version = manifest.get("version")
    if version not in (LEGACY_MANIFEST_VERSION, MANIFEST_VERSION):
        raise CheckpointIntegrityError(
            f"Unsupported integrity manifest version in {manifest_path}: {version!r}"
        )
    if manifest.get("filename") != expected_filename:
        raise CheckpointIntegrityError(
            f"Manifest filename mismatch for {manifest_path}: expected {expected_filename!r}, got {manifest.get('filename')!r}"
        )
    expected_sha256 = manifest.get("sha256")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.match(expected_sha256):
        raise CheckpointIntegrityError(f"Invalid SHA-256 digest in manifest {manifest_path}")
    _require_nonneg_int(manifest.get("size_bytes"), "size_bytes")
    return manifest


def verify_checkpoint_integrity(ckpt_path: str) -> dict[str, Any]:
    """Verify a checkpoint file (either inside a generation directory or legacy flat file)."""
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, GENERATION_PAYLOAD_FILENAME)

    if not os.path.isfile(ckpt_path):
        raise CheckpointIntegrityError(f"Checkpoint file does not exist: {ckpt_path}")

    manifest_path = manifest_path_for(ckpt_path)
    if not os.path.isfile(manifest_path):
        raise CheckpointIntegrityError(
            f"Missing integrity manifest for checkpoint {ckpt_path} (expected {manifest_path})"
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(f"Unreadable integrity manifest {manifest_path}: {exc}") from exc

    expected_name = os.path.basename(ckpt_path)
    manifest = _validate_manifest_dict(manifest, manifest_path, expected_name)

    expected_size = manifest["size_bytes"]
    actual_size = os.path.getsize(ckpt_path)
    if actual_size != expected_size:
        raise CheckpointIntegrityError(
            f"Checkpoint size mismatch for {ckpt_path}: expected {expected_size} bytes, got {actual_size} bytes"
        )

    actual_sha256, read_size = compute_file_sha256(ckpt_path)
    if read_size != expected_size or not hmac.compare_digest(actual_sha256, manifest["sha256"]):
        raise CheckpointIntegrityError(
            f"Checkpoint SHA-256 mismatch for {ckpt_path}: expected {manifest['sha256']}, got {actual_sha256}"
        )
    return manifest


def read_pointer(pointer_path: str) -> dict[str, Any]:
    if not os.path.isfile(pointer_path):
        raise CheckpointIntegrityError(f"Checkpoint pointer does not exist: {pointer_path}")
    try:
        with open(pointer_path, "r", encoding="utf-8") as fh:
            pointer = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(f"Unreadable checkpoint pointer {pointer_path}: {exc}") from exc
    if not isinstance(pointer, dict):
        raise CheckpointIntegrityError(f"Checkpoint pointer must be a JSON object: {pointer_path}")
    if pointer.get("version") != POINTER_VERSION:
        raise CheckpointIntegrityError(
            f"Unsupported checkpoint pointer version in {pointer_path}: {pointer.get('version')!r}"
        )
    gen_name = pointer.get("generation")
    if not isinstance(gen_name, str) or not _GEN_DIR_RE.match(gen_name):
        raise CheckpointIntegrityError(f"Invalid generation name in pointer {pointer_path}: {gen_name!r}")
    expected_ckpt_rel = f"{gen_name}/{GENERATION_PAYLOAD_FILENAME}"
    expected_manifest_rel = f"{gen_name}/{GENERATION_MANIFEST_FILENAME}"
    if pointer.get("checkpoint_relpath") != expected_ckpt_rel:
        raise CheckpointIntegrityError(f"Invalid checkpoint_relpath in pointer {pointer_path}")
    if pointer.get("manifest_relpath") != expected_manifest_rel:
        raise CheckpointIntegrityError(f"Invalid manifest_relpath in pointer {pointer_path}")
    sha = pointer.get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise CheckpointIntegrityError(f"Invalid sha256 in pointer {pointer_path}")
    _require_nonneg_int(pointer.get("size_bytes"), "pointer.size_bytes")
    completed_step = _require_nonneg_int(pointer.get("completed_step"), "pointer.completed_step")
    next_step = _require_nonneg_int(pointer.get("next_step"), "pointer.next_step")
    if next_step != completed_step + 1:
        raise CheckpointIntegrityError(
            f"Inconsistent step metadata in pointer {pointer_path}: completed_step={completed_step}, next_step={next_step}"
        )
    return pointer


def verify_pointer(ckpt_dir: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Verify LATEST.json and its referenced immutable generation directory."""
    ptr_path = pointer_path_for(ckpt_dir)
    pointer = read_pointer(ptr_path)
    gen_name = pointer["generation"]
    gen_dir = os.path.join(ckpt_dir, gen_name)
    ckpt_path = os.path.join(gen_dir, GENERATION_PAYLOAD_FILENAME)
    manifest = verify_checkpoint_integrity(ckpt_path)

    if manifest.get("generation") != gen_name:
        raise CheckpointIntegrityError(
            f"Pointer generation ({gen_name}) does not match manifest generation ({manifest.get('generation')!r})"
        )
    if not hmac.compare_digest(manifest["sha256"], pointer["sha256"]):
        raise CheckpointIntegrityError(
            f"Pointer sha256 ({pointer['sha256']}) does not match manifest sha256 ({manifest['sha256']})"
        )
    if manifest["size_bytes"] != pointer["size_bytes"]:
        raise CheckpointIntegrityError(
            f"Pointer size_bytes ({pointer['size_bytes']}) does not match manifest ({manifest['size_bytes']})"
        )
    if manifest.get("completed_step") != pointer["completed_step"]:
        raise CheckpointIntegrityError(
            f"Pointer completed_step ({pointer['completed_step']}) does not match manifest ({manifest.get('completed_step')!r})"
        )
    if manifest.get("next_step") != pointer["next_step"]:
        raise CheckpointIntegrityError(
            f"Pointer next_step ({pointer['next_step']}) does not match manifest ({manifest.get('next_step')!r})"
        )
    return pointer, manifest, ckpt_path


def publish_checkpoint(
    ckpt_dir: str,
    payload: dict[str, Any],
    *,
    save_fn: Callable[[Any, Any], None] | None = None,
    is_final: bool | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a checkpoint into an immutable generation directory and swap LATEST.json.

    Publication order:
      1. Stage checkpoint.pt + manifest.json in a hidden directory (.staging_gen_...) and fsync both.
      2. Verify the staged pair.
      3. Promote staged directory to immutable gen_<step>_<uid>/ via os.rename + fsync ckpt_dir.
         (If a crash happens here, LATEST.json still points to the untouched prior generation.)
      4. Write .LATEST.json.<uid>.tmp, fsync, atomically os.replace -> LATEST.json, fsync ckpt_dir.
    """
    if save_fn is None:
        import torch

        save_fn = torch.save

    if not isinstance(payload, dict):
        raise CheckpointIntegrityError(f"Checkpoint payload must be a dict, got {type(payload).__name__}")

    next_step = resolve_resume_step(payload)
    completed_step = _require_nonneg_int(
        payload.get("completed_step", payload.get("step", next_step - 1)), "completed_step"
    )
    if next_step != completed_step + 1:
        raise CheckpointIntegrityError(
            f"Inconsistent step metadata: completed_step={completed_step}, next_step={next_step}"
        )

    final_flag = bool(payload.get("is_final", False) if is_final is None else is_final)
    kind = payload.get("kind") or ("final_training" if final_flag else "training")
    if kind == "inference_export":
        raise CheckpointIntegrityError(
            "Inference exports must use export_inference_weights(), not publish_checkpoint()"
        )

    os.makedirs(ckpt_dir, exist_ok=True)
    gen_uid = generation_id or uuid.uuid4().hex[:12]
    if not re.match(r"^[0-9a-zA-Z_-]+$", gen_uid):
        raise ValueError(f"Invalid generation_id: {gen_uid!r}")
    gen_name = f"gen_{completed_step:08d}_{gen_uid}"
    gen_dir = os.path.join(ckpt_dir, gen_name)
    if os.path.exists(gen_dir):
        raise CheckpointIntegrityError(
            f"Generation directory already exists (generations are immutable): {gen_dir}"
        )

    staging_dir = tempfile.mkdtemp(dir=ckpt_dir, prefix=f".staging_{gen_name}_")
    promoted = False
    tmp_pointer_path = None
    try:
        staged_ckpt = os.path.join(staging_dir, GENERATION_PAYLOAD_FILENAME)
        with open(staged_ckpt, "wb") as ckpt_fh:
            save_fn(payload, ckpt_fh)
            ckpt_fh.flush()
            os.fsync(ckpt_fh.fileno())

        digest, size_bytes = compute_file_sha256(staged_ckpt)
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "generation": gen_name,
            "generation_id": gen_uid,
            "kind": kind,
            "is_final": final_flag,
            "filename": GENERATION_PAYLOAD_FILENAME,
            "sha256": digest,
            "size_bytes": size_bytes,
            "completed_step": completed_step,
            "next_step": next_step,
        }
        if isinstance(payload.get("identity"), dict):
            manifest["identity"] = payload["identity"]

        staged_manifest = os.path.join(staging_dir, GENERATION_MANIFEST_FILENAME)
        with open(staged_manifest, "wb") as manifest_fh:
            manifest_fh.write(_canonical_json_bytes(manifest))
            manifest_fh.flush()
            os.fsync(manifest_fh.fileno())
        _fsync_dir(staging_dir)

        verify_checkpoint_integrity(staged_ckpt)

        os.rename(staging_dir, gen_dir)
        promoted = True
        _fsync_dir(ckpt_dir)

        prior_history: list[str] = []
        ptr_path = pointer_path_for(ckpt_dir)
        if os.path.isfile(ptr_path):
            try:
                old_ptr = read_pointer(ptr_path)
                hist = old_ptr.get("history")
                if isinstance(hist, list):
                    prior_history = [g for g in hist if isinstance(g, str)]
                elif isinstance(old_ptr.get("generation"), str):
                    prior_history = [old_ptr["generation"]]
            except CheckpointIntegrityError:
                prior_history = []
        history = [g for g in prior_history if g != gen_name] + [gen_name]

        pointer: dict[str, Any] = {
            "version": POINTER_VERSION,
            "generation": gen_name,
            "generation_id": gen_uid,
            "kind": kind,
            "is_final": final_flag,
            "checkpoint_relpath": f"{gen_name}/{GENERATION_PAYLOAD_FILENAME}",
            "manifest_relpath": f"{gen_name}/{GENERATION_MANIFEST_FILENAME}",
            "sha256": digest,
            "size_bytes": size_bytes,
            "completed_step": completed_step,
            "next_step": next_step,
            "history": history,
        }
        if isinstance(payload.get("identity"), dict):
            pointer["identity"] = payload["identity"]

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=ckpt_dir, prefix=f".{POINTER_FILENAME}.{gen_uid}.", suffix=".tmp", delete=False
        ) as ptr_fh:
            tmp_pointer_path = ptr_fh.name
            ptr_fh.write(_canonical_json_bytes(pointer))
            ptr_fh.flush()
            os.fsync(ptr_fh.fileno())

        os.replace(tmp_pointer_path, ptr_path)
        tmp_pointer_path = None
        _fsync_dir(ckpt_dir)
        return {
            **manifest,
            "generation_dir": gen_dir,
            "ckpt_path": os.path.join(gen_dir, GENERATION_PAYLOAD_FILENAME),
            "manifest_path": os.path.join(gen_dir, GENERATION_MANIFEST_FILENAME),
            "pointer_path": ptr_path,
        }
    except Exception:
        if tmp_pointer_path and os.path.exists(tmp_pointer_path):
            try:
                os.remove(tmp_pointer_path)
            except OSError:
                pass
        if not promoted and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def save_checkpoint_atomic(
    payload: dict[str, Any],
    ckpt_path: str,
    save_fn: Callable[[Any, Any], None] | None = None,
    *,
    is_final: bool | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a checkpoint into an immutable generation directory + LATEST.json pointer.

    Accepts either a checkpoint directory path or a legacy checkpoint file path (uses its parent dir).
    """
    if ckpt_path.endswith(".pt"):
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    else:
        ckpt_dir = os.path.abspath(ckpt_path)
    return publish_checkpoint(
        ckpt_dir,
        payload,
        save_fn=save_fn,
        is_final=is_final,
        generation_id=generation_id,
    )


def export_inference_weights(
    ckpt_dir: str,
    model_state: Any,
    completed_step: int,
    cfg: dict[str, Any],
    *,
    identity: dict[str, Any] | None = None,
    save_fn: Callable[[Any, Any], None] | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Write a clearly marked inference-only weights export separate from resumable generations."""
    if save_fn is None:
        import torch

        save_fn = torch.save

    completed_step = _require_nonneg_int(completed_step, "completed_step")
    exports_root = os.path.join(ckpt_dir, "exports")
    os.makedirs(exports_root, exist_ok=True)
    gen_uid = generation_id or uuid.uuid4().hex[:12]
    export_name = f"inference_step_{completed_step:08d}_{gen_uid}"
    export_dir = os.path.join(exports_root, export_name)
    staging_dir = tempfile.mkdtemp(dir=exports_root, prefix=f".staging_{export_name}_")
    try:
        payload = {
            "kind": "inference_export",
            "model": model_state,
            "completed_step": completed_step,
            "cfg": cfg,
            "identity": identity,
        }
        staged_weights = os.path.join(staging_dir, GENERATION_PAYLOAD_FILENAME)
        with open(staged_weights, "wb") as fh:
            save_fn(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        digest, size_bytes = compute_file_sha256(staged_weights)
        manifest = {
            "version": MANIFEST_VERSION,
            "generation": export_name,
            "generation_id": gen_uid,
            "kind": "inference_export",
            "is_final": True,
            "filename": GENERATION_PAYLOAD_FILENAME,
            "sha256": digest,
            "size_bytes": size_bytes,
            "completed_step": completed_step,
        }
        if identity is not None:
            manifest["identity"] = identity
        staged_manifest = os.path.join(staging_dir, GENERATION_MANIFEST_FILENAME)
        with open(staged_manifest, "wb") as fh:
            fh.write(_canonical_json_bytes(manifest))
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(staging_dir)
        os.rename(staging_dir, export_dir)
        _fsync_dir(exports_root)
        return {
            **manifest,
            "export_dir": export_dir,
            "weights_path": os.path.join(export_dir, GENERATION_PAYLOAD_FILENAME),
            "manifest_path": os.path.join(export_dir, GENERATION_MANIFEST_FILENAME),
        }
    except Exception:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def find_legacy_checkpoints(ckpt_dir: str) -> list[str]:
    paths = [p for p in glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")) if os.path.isfile(p)]

    def sort_key(path: str):
        name = os.path.basename(path)
        m = _CKPT_STEP_RE.match(name)
        return (0, int(m.group(1)), name) if m else (1, 0, name)

    paths.sort(key=sort_key)
    return paths


def find_generation_dirs(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    dirs = []
    for entry in os.listdir(ckpt_dir):
        parsed = _parse_generation_name(entry)
        full = os.path.join(ckpt_dir, entry)
        if parsed is not None and os.path.isdir(full):
            dirs.append((parsed[0], entry, full))
    dirs.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in dirs]


def find_checkpoints(ckpt_dir: str) -> list[str]:
    """Return discovered checkpoint payload paths (committed generations + legacy flat files)."""
    paths = [
        os.path.join(g, GENERATION_PAYLOAD_FILENAME)
        for g in find_generation_dirs(ckpt_dir)
        if os.path.isfile(os.path.join(g, GENERATION_PAYLOAD_FILENAME))
    ]
    paths.extend(find_legacy_checkpoints(ckpt_dir))
    return paths


def _check_expected_handoff(
    *,
    completed_step: int,
    sha256: str,
    generation: str | None,
    expected_step: int | None = None,
    expected_generation: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    if expected_step is not None and completed_step != expected_step:
        raise CheckpointIntegrityError(
            f"Handoff step mismatch: expected completed_step={expected_step}, found {completed_step}"
        )
    if expected_generation is not None and generation != expected_generation:
        raise CheckpointIntegrityError(
            f"Handoff generation mismatch: expected generation={expected_generation!r}, found {generation!r}"
        )
    if expected_sha256 is not None and not hmac.compare_digest(sha256, expected_sha256):
        raise CheckpointIntegrityError(
            f"Handoff SHA-256 mismatch: expected {expected_sha256}, found {sha256}"
        )


def preflight_resume(
    ckpt_dir: str,
    *,
    allow_cold_start: bool = False,
    expected_step: int | None = None,
    expected_generation: str | None = None,
    expected_sha256: str | None = None,
    expected_identity: dict[str, Any] | None = None,
    require_training_state: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint handoff state BEFORE expensive model/GPU allocation.

    Fails closed when:
      - No verified checkpoint exists and allow_cold_start is False.
      - Any expected_step / expected_generation / expected_sha256 pin is unsatisfied.
      - Pointer, manifest, SHA-256, or size is corrupt or inconsistent.
      - Saved identity (model, training config, tokenizer, data) disagrees with expected_identity.
    """
    ptr_path = pointer_path_for(ckpt_dir)
    if os.path.exists(ptr_path):
        pointer, manifest, ckpt_path = verify_pointer(ckpt_dir)
        if require_training_state and manifest.get("kind") == "inference_export":
            raise CheckpointIntegrityError(
                f"Checkpoint {ckpt_path} is an inference_export and cannot resume training"
            )
        _check_expected_handoff(
            completed_step=pointer["completed_step"],
            sha256=pointer["sha256"],
            generation=pointer["generation"],
            expected_step=expected_step,
            expected_generation=expected_generation,
            expected_sha256=expected_sha256,
        )
        if expected_identity is not None:
            verify_run_identity(manifest.get("identity"), expected_identity)
        return {
            "mode": "resume",
            "format": "generation",
            "generation": pointer["generation"],
            "ckpt_path": ckpt_path,
            "manifest": manifest,
            "pointer": pointer,
            "completed_step": pointer["completed_step"],
            "next_step": pointer["next_step"],
            "is_final": bool(pointer.get("is_final", False)),
            "sha256": pointer["sha256"],
        }

    legacy = find_legacy_checkpoints(ckpt_dir)
    if legacy:
        latest = legacy[-1]
        manifest = verify_checkpoint_integrity(latest)
        completed_step = manifest.get("completed_step")
        if completed_step is None:
            m = _CKPT_STEP_RE.match(os.path.basename(latest))
            completed_step = int(m.group(1)) if m else 0
        next_step = manifest.get("next_step", completed_step + 1)
        _check_expected_handoff(
            completed_step=completed_step,
            sha256=manifest["sha256"],
            generation=None,
            expected_step=expected_step,
            expected_generation=expected_generation,
            expected_sha256=expected_sha256,
        )
        if expected_identity is not None:
            verify_run_identity(manifest.get("identity"), expected_identity)
        return {
            "mode": "resume",
            "format": "legacy",
            "generation": None,
            "ckpt_path": latest,
            "manifest": manifest,
            "pointer": None,
            "completed_step": completed_step,
            "next_step": next_step,
            "is_final": False,
            "sha256": manifest["sha256"],
        }

    if expected_step is not None or expected_generation is not None or expected_sha256 is not None:
        raise CheckpointIntegrityError(
            f"Expected handoff checkpoint not found in {ckpt_dir} "
            f"(expected_step={expected_step}, expected_generation={expected_generation}, expected_sha256={expected_sha256})"
        )

    uncommitted = find_generation_dirs(ckpt_dir)
    if uncommitted:
        raise CheckpointIntegrityError(
            f"Found uncommitted generation directory without {POINTER_FILENAME} in {ckpt_dir}; "
            "refusing to guess or cold-start"
        )

    if not allow_cold_start:
        raise CheckpointIntegrityError(
            f"Refusing silent cold start from step 0: no verified checkpoint in {ckpt_dir}. "
            "Set ALLOW_COLD_START=1 to explicitly authorize a new training lineage."
        )

    return {
        "mode": "cold_start",
        "format": None,
        "generation": None,
        "ckpt_path": None,
        "manifest": None,
        "pointer": None,
        "completed_step": None,
        "next_step": 0,
        "is_final": False,
        "sha256": None,
    }


def load_checkpoint(
    ckpt_path: str,
    map_location: Any = "cpu",
    load_fn: Callable[..., Any] | None = None,
    require_training_state: bool = True,
    *,
    require_full_state: bool = False,
    expected_identity: dict[str, Any] | None = None,
    expected_step: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], int]:
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, GENERATION_PAYLOAD_FILENAME)

    manifest = verify_checkpoint_integrity(ckpt_path)

    if require_training_state and manifest.get("kind") == "inference_export":
        raise CheckpointIntegrityError(
            f"Checkpoint {ckpt_path} is marked as inference_export and cannot resume training"
        )

    if load_fn is None:
        import torch

        load_fn = torch.load

    try:
        sd = load_fn(ckpt_path, map_location=map_location)
    except Exception as exc:
        raise CheckpointIntegrityError(f"Failed to load checkpoint payload {ckpt_path}: {exc}") from exc

    if not isinstance(sd, dict):
        raise CheckpointIntegrityError(f"Checkpoint payload must be a dict, got {type(sd).__name__}")
    if require_training_state and sd.get("kind") == "inference_export":
        raise CheckpointIntegrityError(
            f"Checkpoint payload {ckpt_path} is an inference_export and cannot resume training"
        )
    if require_training_state and ("model" not in sd or "opt" not in sd):
        raise CheckpointIntegrityError(f"Checkpoint {ckpt_path} missing required 'model'/'opt' state")

    next_step = resolve_resume_step(sd)
    completed_step = _require_nonneg_int(
        sd.get("completed_step", sd.get("step", next_step - 1)), "completed_step"
    )

    if "next_step" in manifest:
        manifest_next = _require_nonneg_int(manifest["next_step"], "manifest.next_step")
        if manifest_next != next_step:
            raise CheckpointIntegrityError(
                f"Manifest next_step ({manifest_next}) does not match payload next_step ({next_step})"
            )
    if "completed_step" in manifest:
        manifest_completed = _require_nonneg_int(manifest["completed_step"], "manifest.completed_step")
        if manifest_completed != completed_step:
            raise CheckpointIntegrityError(
                f"Manifest completed_step ({manifest_completed}) does not match payload ({completed_step})"
            )

    _check_expected_handoff(
        completed_step=completed_step,
        sha256=manifest["sha256"],
        generation=manifest.get("generation"),
        expected_step=expected_step,
        expected_sha256=expected_sha256,
    )

    if require_full_state:
        if "rng_state" not in sd or not isinstance(sd["rng_state"], dict) or "python" not in sd["rng_state"]:
            raise CheckpointIntegrityError(f"Checkpoint {ckpt_path} missing required 'rng_state'")
        if "data_cursor" not in sd:
            raise CheckpointIntegrityError(f"Checkpoint {ckpt_path} missing required 'data_cursor'")
        validate_data_cursor(sd["data_cursor"])
        if "identity" not in sd or not isinstance(sd["identity"], dict):
            raise CheckpointIntegrityError(f"Checkpoint {ckpt_path} missing required 'identity'")

    if expected_identity is not None:
        if "identity" in manifest:
            verify_run_identity(manifest["identity"], expected_identity)
        verify_run_identity(sd.get("identity"), expected_identity)
    elif "identity" in manifest and "identity" in sd:
        verify_run_identity(sd["identity"], manifest["identity"])

    return sd, next_step


def load_latest_checkpoint(
    ckpt_dir: str,
    map_location: Any = "cpu",
    load_fn: Callable[..., Any] | None = None,
    require_training_state: bool = True,
    *,
    allow_cold_start: bool = False,
    require_full_state: bool = False,
    expected_step: int | None = None,
    expected_generation: str | None = None,
    expected_sha256: str | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None, int]:
    """Load the latest verified checkpoint from ckpt_dir.

    Defaults to allow_cold_start=False so an empty directory fails closed rather
    than silently restarting training from step 0.
    """
    info = preflight_resume(
        ckpt_dir,
        allow_cold_start=allow_cold_start,
        expected_step=expected_step,
        expected_generation=expected_generation,
        expected_sha256=expected_sha256,
        expected_identity=expected_identity,
        require_training_state=require_training_state,
    )
    if info["mode"] == "cold_start":
        return None, None, 0

    ckpt_path = info["ckpt_path"]
    sd, next_step = load_checkpoint(
        ckpt_path,
        map_location=map_location,
        load_fn=load_fn,
        require_training_state=require_training_state,
        require_full_state=require_full_state,
        expected_identity=expected_identity,
        expected_step=expected_step,
        expected_sha256=expected_sha256,
    )
    return ckpt_path, sd, next_step


def commit_staged_checkpoint(
    staging_dir: str,
    dest_ckpt_dir: str,
    *,
    expected_step: int | None = None,
    expected_generation: str | None = None,
    expected_sha256: str | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a staged transfer directory and atomically commit its generation to dest_ckpt_dir.

    If verification fails at any point, dest_ckpt_dir's existing LATEST.json and prior
    generations remain completely untouched.
    """
    pointer, manifest, staged_ckpt = verify_pointer(staging_dir)
    if manifest.get("kind") == "inference_export":
        raise CheckpointIntegrityError("Refusing to commit an inference_export as active training checkpoint")

    _check_expected_handoff(
        completed_step=pointer["completed_step"],
        sha256=pointer["sha256"],
        generation=pointer["generation"],
        expected_step=expected_step,
        expected_generation=expected_generation,
        expected_sha256=expected_sha256,
    )
    if expected_identity is not None:
        verify_run_identity(manifest.get("identity"), expected_identity)

    os.makedirs(dest_ckpt_dir, exist_ok=True)
    gen_name = pointer["generation"]
    staged_gen_dir = os.path.join(staging_dir, gen_name)
    dest_gen_dir = os.path.join(dest_ckpt_dir, gen_name)

    if os.path.isdir(dest_gen_dir):
        existing_manifest = verify_checkpoint_integrity(os.path.join(dest_gen_dir, GENERATION_PAYLOAD_FILENAME))
        if not hmac.compare_digest(existing_manifest["sha256"], manifest["sha256"]):
            raise CheckpointIntegrityError(
                f"Destination already contains conflicting generation {gen_name}"
            )
    else:
        temp_dest_gen = tempfile.mkdtemp(dir=dest_ckpt_dir, prefix=f".import_{gen_name}_")
        try:
            shutil.rmtree(temp_dest_gen)
            shutil.copytree(staged_gen_dir, temp_dest_gen)
            verify_checkpoint_integrity(os.path.join(temp_dest_gen, GENERATION_PAYLOAD_FILENAME))
            _fsync_dir(temp_dest_gen)
            os.rename(temp_dest_gen, dest_gen_dir)
            _fsync_dir(dest_ckpt_dir)
        except Exception:
            if os.path.exists(temp_dest_gen):
                shutil.rmtree(temp_dest_gen, ignore_errors=True)
            raise

    prior_history: list[str] = []
    dest_ptr_path = pointer_path_for(dest_ckpt_dir)
    if os.path.isfile(dest_ptr_path):
        try:
            old_ptr = read_pointer(dest_ptr_path)
            if isinstance(old_ptr.get("history"), list):
                prior_history = [g for g in old_ptr["history"] if isinstance(g, str)]
            elif isinstance(old_ptr.get("generation"), str):
                prior_history = [old_ptr["generation"]]
        except CheckpointIntegrityError:
            prior_history = []

    new_pointer = dict(pointer)
    new_pointer["history"] = [g for g in prior_history if g != gen_name] + [gen_name]
    _write_json_atomic(dest_ptr_path, new_pointer, prefix=f".{POINTER_FILENAME}.commit.")
    verify_pointer(dest_ckpt_dir)
    return {
        "generation": gen_name,
        "completed_step": pointer["completed_step"],
        "next_step": pointer["next_step"],
        "sha256": pointer["sha256"],
        "is_final": bool(pointer.get("is_final", False)),
        "ckpt_path": os.path.join(dest_gen_dir, GENERATION_PAYLOAD_FILENAME),
    }


def prune_old_checkpoints(ckpt_dir: str, keep: int = 3) -> list[str]:
    """Remove old checkpoint generations and stale staging artifacts while preserving active generation."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    removed: list[str] = []

    active_gen: str | None = None
    ptr_path = pointer_path_for(ckpt_dir)
    if os.path.isfile(ptr_path):
        try:
            active_gen = read_pointer(ptr_path)["generation"]
        except CheckpointIntegrityError:
            active_gen = None

    gen_dirs = find_generation_dirs(ckpt_dir)
    if gen_dirs:
        keep_set: set[str] = set()
        if active_gen:
            active_path = os.path.join(ckpt_dir, active_gen)
            if active_path in gen_dirs:
                keep_set.add(active_path)
        for cand in reversed(gen_dirs):
            if len(keep_set) >= keep:
                break
            keep_set.add(cand)
        for old_dir in gen_dirs:
            if old_dir not in keep_set:
                shutil.rmtree(old_dir, ignore_errors=True)
                removed.append(old_dir)

    if os.path.isdir(ckpt_dir):
        for entry in os.listdir(ckpt_dir):
            full = os.path.join(ckpt_dir, entry)
            if entry.startswith(".staging_") or entry.startswith(".import_"):
                shutil.rmtree(full, ignore_errors=True)
            elif entry.startswith(f".{POINTER_FILENAME}.") and entry.endswith(".tmp"):
                try:
                    os.remove(full)
                except OSError:
                    pass

    for old_ckpt in find_legacy_checkpoints(ckpt_dir)[:-keep]:
        for target in (old_ckpt, manifest_path_for(old_ckpt)):
            try:
                os.remove(target)
            except OSError:
                pass
        removed.append(old_ckpt)

    return removed


def run_local_fixture_roundtrip(repo_root: str, work_dir: str) -> dict[str, Any]:
    """Run a deterministic stdlib-only local round-trip using bundled tokenizer files.

    Simulates: cold start (explicitly authorized) -> steps 0..5 -> periodic save at step 4 ->
    final full-state save at step 6 -> stage into destination -> resume from step 7..9 ->
    verify bit-for-bit equality with an uninterrupted 0..9 reference run.
    """
    tok_dir = os.path.join(repo_root, "tokenizer")
    data_root = os.path.join(work_dir, "data")
    lang_dir = os.path.join(data_root, "Python")
    ckpt_a = os.path.join(work_dir, "ckpt_burst_a")
    ckpt_b = os.path.join(work_dir, "ckpt_burst_b")
    os.makedirs(lang_dir, exist_ok=True)

    for shard_idx in range(2):
        shard_path = os.path.join(lang_dir, f"fixture_{shard_idx:02d}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as fh:
            for line_idx in range(24):
                text = f"def fn_{shard_idx}_{line_idx}(x): return x + {shard_idx * 100 + line_idx}\n"
                fh.write(json.dumps({"content": text}) + "\n")

    def byte_tok(text: str) -> list[int]:
        return [b + 10 for b in text.encode("utf-8")]

    shards = discover_shards(data_root, langs=["Python"], extensions=(".jsonl",))
    tok_id = compute_tokenizer_identity(tok_dir)
    data_id = compute_data_identity(data_root, langs=["Python"], shard_paths=shards, extensions=(".jsonl",))
    model_cfg = {
        "vocab_size": tok_id["vocab_size"] or 49152,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    train_cfg = {
        "seq_len": 32,
        "global_batch_size": 4,
        "lr_max": 3e-4,
        "lr_min": 3e-5,
        "warmup_steps": 2,
        "total_steps": 10,
        "seed": 7,
    }
    identity = build_run_identity(model_cfg, train_cfg, tok_id, data_id)

    def pickle_save(obj: Any, fh: Any) -> None:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(path: str, map_location: Any = "cpu") -> Any:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def step_model(weight: int, opt_m: int, batch: list[list[int]]) -> tuple[int, int]:
        noise = random.randint(1, 97)
        batch_sum = sum(sum(seq) for seq in batch)
        new_m = (opt_m * 3 + batch_sum + noise) % 1_000_003
        new_w = (weight + new_m) % 1_000_003
        return new_w, new_m

    # Uninterrupted reference run (steps 0..9)
    random.seed(7)
    ref_stream = DeterministicBlockStream(shards, byte_tok, 0, 32, seed=7)
    ref_w, ref_m = 0, 0
    for _step in range(10):
        ref_w, ref_m = step_model(ref_w, ref_m, ref_stream.next_batch(4))
    ref_cursor = ref_stream.snapshot_cursor()

    # Segment 1 (steps 0..6, periodic save at 4, final full-state save at 6)
    pre0 = preflight_resume(ckpt_a, allow_cold_start=True, expected_identity=identity)
    assert pre0["mode"] == "cold_start"
    random.seed(7)
    stream_a = DeterministicBlockStream(shards, byte_tok, 0, 32, seed=7)
    w, m = 0, 0
    for step in range(0, 7):
        w, m = step_model(w, m, stream_a.next_batch(4))
        if step in (4, 6):
            payload = build_checkpoint_payload(
                {"w": w},
                {"m": m},
                step,
                model_cfg,
                rng_state=capture_rng_state(include_torch=False, base_seed=7),
                data_cursor=stream_a.snapshot_cursor(),
                identity=identity,
                is_final=(step == 6),
            )
            publish_checkpoint(ckpt_a, payload, save_fn=pickle_save, is_final=(step == 6))

    # Atomic stage ckpt_a -> ckpt_b with pinned expected step 6
    staged = commit_staged_checkpoint(ckpt_a, ckpt_b, expected_step=6, expected_identity=identity)

    # Segment 2 (resume from ckpt_b at step 7..9)
    _, sd, start = load_latest_checkpoint(
        ckpt_b,
        load_fn=pickle_load,
        allow_cold_start=False,
        require_full_state=True,
        expected_step=6,
        expected_generation=staged["generation"],
        expected_identity=identity,
    )
    assert sd is not None and start == 7
    restore_rng_state(sd["rng_state"], include_torch=False)
    stream_b = DeterministicBlockStream(shards, byte_tok, 0, 32, seed=7, cursor=sd["data_cursor"])
    w, m = sd["model"]["w"], sd["opt"]["m"]
    for _step in range(start, 10):
        w, m = step_model(w, m, stream_b.next_batch(4))

    if (w, m, stream_b.snapshot_cursor()) != (ref_w, ref_m, ref_cursor):
        raise CheckpointIntegrityError("Local round-trip diverged from uninterrupted reference run")

    return {
        "status": "ok",
        "resumed_from_generation": staged["generation"],
        "resumed_at_step": start,
        "final_weight": w,
        "final_opt_m": m,
        "blocks_emitted": stream_b.blocks_emitted,
        "tokenizer_files_sha256": tok_id["files_sha256"],
        "identity_sha256": identity["identity_sha256"],
    }


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint integrity, pointer, and staging utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Preflight-verify a checkpoint directory")
    p_verify.add_argument("--ckpt-dir", required=True)
    p_verify.add_argument("--allow-cold-start", action="store_true")
    p_verify.add_argument("--expected-step", type=int, default=None)
    p_verify.add_argument("--expected-generation", default=None)
    p_verify.add_argument("--expected-sha256", default=None)
    p_verify.add_argument("--json", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Inspect LATEST.json pointer")
    p_inspect.add_argument("--pointer", required=True)
    p_inspect.add_argument("--field", default=None)

    p_commit = sub.add_parser("commit-staged", help="Verify staged transfer and atomically commit")
    p_commit.add_argument("--staging-dir", required=True)
    p_commit.add_argument("--ckpt-dir", required=True)
    p_commit.add_argument("--expected-step", type=int, default=None)
    p_commit.add_argument("--expected-generation", default=None)
    p_commit.add_argument("--expected-sha256", default=None)

    p_rt = sub.add_parser("local-roundtrip", help="Run CPU/stdlib local fixture round-trip")
    p_rt.add_argument("--repo-root", default=os.path.dirname(os.path.abspath(__file__)))
    p_rt.add_argument("--work-dir", default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "verify":
            info = preflight_resume(
                args.ckpt_dir,
                allow_cold_start=args.allow_cold_start,
                expected_step=args.expected_step,
                expected_generation=args.expected_generation,
                expected_sha256=args.expected_sha256,
            )
            summary = {
                "mode": info["mode"],
                "format": info["format"],
                "generation": info["generation"],
                "completed_step": info["completed_step"],
                "next_step": info["next_step"],
                "is_final": info["is_final"],
                "sha256": info["sha256"],
            }
            if args.json:
                print(json.dumps(summary, sort_keys=True))
            else:
                print(
                    f"verified mode={summary['mode']} gen={summary['generation']} "
                    f"completed_step={summary['completed_step']} next_step={summary['next_step']}"
                )
            return 0

        if args.cmd == "inspect":
            ptr = read_pointer(args.pointer)
            if args.field:
                if args.field not in ptr:
                    raise CheckpointIntegrityError(f"Pointer missing field: {args.field}")
                print(ptr[args.field])
            else:
                print(json.dumps(ptr, sort_keys=True))
            return 0

        if args.cmd == "commit-staged":
            committed = commit_staged_checkpoint(
                args.staging_dir,
                args.ckpt_dir,
                expected_step=args.expected_step,
                expected_generation=args.expected_generation,
                expected_sha256=args.expected_sha256,
            )
            print(json.dumps(committed, sort_keys=True))
            return 0

        if args.cmd == "local-roundtrip":
            if args.work_dir:
                result = run_local_fixture_roundtrip(args.repo_root, args.work_dir)
            else:
                with tempfile.TemporaryDirectory(prefix="stackv2_roundtrip_") as tmp:
                    result = run_local_fixture_roundtrip(args.repo_root, tmp)
            print(json.dumps(result, sort_keys=True))
            return 0
    except CheckpointIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
