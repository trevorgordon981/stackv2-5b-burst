"""Atomic checkpoint persistence, SHA-256 integrity verification, and resume step progression."""

import glob
import hashlib
import hmac
import json
import os
import re
import tempfile

MANIFEST_SUFFIX = ".sha256"
MANIFEST_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CKPT_STEP_RE = re.compile(r"^ckpt_(\d+)\.pt$")


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint or its integrity manifest is missing, corrupt, or inconsistent."""


def manifest_path_for(ckpt_path: str) -> str:
    return f"{ckpt_path}{MANIFEST_SUFFIX}"


def _require_nonneg_int(value, field_name: str) -> int:
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


def build_checkpoint_payload(model_state, opt_state, completed_step: int, cfg: dict) -> dict:
    completed_step = _require_nonneg_int(completed_step, "completed_step")
    return {
        "model": model_state,
        "opt": opt_state,
        "step": completed_step,
        "completed_step": completed_step,
        "next_step": completed_step + 1,
        "cfg": cfg,
    }


def resolve_resume_step(payload: dict) -> int:
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
        # Legacy checkpoints persisted the last completed optimizer step in "step".
        completed_step = _require_nonneg_int(payload["step"], "step")
        return completed_step + 1
    raise CheckpointIntegrityError("Checkpoint payload missing step metadata")


def save_checkpoint_atomic(payload: dict, ckpt_path: str, save_fn=None) -> dict:
    if save_fn is None:
        import torch

        save_fn = torch.save

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    os.makedirs(ckpt_dir, exist_ok=True)
    base = os.path.basename(ckpt_path)
    manifest_path = manifest_path_for(ckpt_path)

    tmp_ckpt_path = None
    tmp_manifest_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=ckpt_dir, prefix=f".{base}.", suffix=".tmp", delete=False
        ) as ckpt_fh:
            tmp_ckpt_path = ckpt_fh.name
            save_fn(payload, ckpt_fh)
            ckpt_fh.flush()
            os.fsync(ckpt_fh.fileno())

        digest, size_bytes = compute_file_sha256(tmp_ckpt_path)
        manifest = {
            "version": MANIFEST_VERSION,
            "filename": base,
            "sha256": digest,
            "size_bytes": size_bytes,
        }
        if isinstance(payload, dict):
            if "completed_step" in payload:
                manifest["completed_step"] = _require_nonneg_int(payload["completed_step"], "completed_step")
            if "next_step" in payload:
                manifest["next_step"] = _require_nonneg_int(payload["next_step"], "next_step")

        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=ckpt_dir, prefix=f".{base}{MANIFEST_SUFFIX}.", suffix=".tmp", delete=False
        ) as manifest_fh:
            tmp_manifest_path = manifest_fh.name
            manifest_fh.write(manifest_bytes)
            manifest_fh.flush()
            os.fsync(manifest_fh.fileno())

        # Publish manifest before checkpoint so the checkpoint path never appears without a valid manifest.
        os.replace(tmp_manifest_path, manifest_path)
        tmp_manifest_path = None
        os.replace(tmp_ckpt_path, ckpt_path)
        tmp_ckpt_path = None
        _fsync_dir(ckpt_dir)
        return manifest
    except Exception:
        for leftover in (tmp_ckpt_path, tmp_manifest_path):
            if leftover and os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
        raise


def verify_checkpoint_integrity(ckpt_path: str) -> dict:
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

    if not isinstance(manifest, dict):
        raise CheckpointIntegrityError(f"Integrity manifest must be a JSON object: {manifest_path}")
    if manifest.get("version") != MANIFEST_VERSION:
        raise CheckpointIntegrityError(
            f"Unsupported integrity manifest version in {manifest_path}: {manifest.get('version')!r}"
        )

    expected_name = os.path.basename(ckpt_path)
    if manifest.get("filename") != expected_name:
        raise CheckpointIntegrityError(
            f"Manifest filename mismatch for {ckpt_path}: expected {expected_name!r}, got {manifest.get('filename')!r}"
        )

    expected_sha256 = manifest.get("sha256")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.match(expected_sha256):
        raise CheckpointIntegrityError(f"Invalid SHA-256 digest in manifest {manifest_path}")

    expected_size = _require_nonneg_int(manifest.get("size_bytes"), "size_bytes")
    actual_size = os.path.getsize(ckpt_path)
    if actual_size != expected_size:
        raise CheckpointIntegrityError(
            f"Checkpoint size mismatch for {ckpt_path}: expected {expected_size} bytes, got {actual_size} bytes"
        )

    actual_sha256, read_size = compute_file_sha256(ckpt_path)
    if read_size != expected_size or not hmac.compare_digest(actual_sha256, expected_sha256):
        raise CheckpointIntegrityError(
            f"Checkpoint SHA-256 mismatch for {ckpt_path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return manifest


def load_checkpoint(
    ckpt_path: str,
    map_location="cpu",
    load_fn=None,
    require_training_state: bool = True,
) -> tuple[dict, int]:
    manifest = verify_checkpoint_integrity(ckpt_path)

    if load_fn is None:
        import torch

        load_fn = torch.load

    try:
        sd = load_fn(ckpt_path, map_location=map_location)
    except Exception as exc:
        raise CheckpointIntegrityError(f"Failed to load checkpoint payload {ckpt_path}: {exc}") from exc

    if not isinstance(sd, dict):
        raise CheckpointIntegrityError(f"Checkpoint payload must be a dict, got {type(sd).__name__}")
    if require_training_state and ("model" not in sd or "opt" not in sd):
        raise CheckpointIntegrityError(f"Checkpoint {ckpt_path} missing required 'model'/'opt' state")

    next_step = resolve_resume_step(sd)
    if "next_step" in manifest:
        manifest_next = _require_nonneg_int(manifest["next_step"], "manifest.next_step")
        if manifest_next != next_step:
            raise CheckpointIntegrityError(
                f"Manifest next_step ({manifest_next}) does not match payload next_step ({next_step})"
            )
    if "completed_step" in manifest:
        manifest_completed = _require_nonneg_int(manifest["completed_step"], "manifest.completed_step")
        payload_completed = sd.get("completed_step", sd.get("step"))
        if payload_completed is not None and manifest_completed != payload_completed:
            raise CheckpointIntegrityError(
                f"Manifest completed_step ({manifest_completed}) does not match payload ({payload_completed})"
            )

    return sd, next_step


def find_checkpoints(ckpt_dir: str) -> list[str]:
    paths = [p for p in glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")) if os.path.isfile(p)]

    def sort_key(path: str):
        name = os.path.basename(path)
        m = _CKPT_STEP_RE.match(name)
        return (0, int(m.group(1)), name) if m else (1, 0, name)

    paths.sort(key=sort_key)
    return paths


def load_latest_checkpoint(
    ckpt_dir: str,
    map_location="cpu",
    load_fn=None,
    require_training_state: bool = True,
):
    cks = find_checkpoints(ckpt_dir)
    if not cks:
        return None, None, 0
    latest = cks[-1]
    sd, next_step = load_checkpoint(
        latest,
        map_location=map_location,
        load_fn=load_fn,
        require_training_state=require_training_state,
    )
    return latest, sd, next_step


def prune_old_checkpoints(ckpt_dir: str, keep: int = 3) -> list[str]:
    if keep < 1:
        raise ValueError("keep must be >= 1")
    removed = []
    for old_ckpt in find_checkpoints(ckpt_dir)[:-keep]:
        for target in (old_ckpt, manifest_path_for(old_ckpt)):
            try:
                os.remove(target)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        removed.append(old_ckpt)
    return removed
