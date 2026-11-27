#!/usr/bin/env python3
"""Dense Llama-style pretraining on The Stack v2 (streaming zstd JSONL).

SMOKE=1  -> tiny model (~6M) + 50 steps + Python-only, to verify the whole pipeline fast.
default  -> ~5B dense run over all staged languages.

Resumable contract:
- Preflight-verifies checkpoint integrity, expected handoff pins, and strict
  model/config/tokenizer/data identity BEFORE allocating the model or GPU memory.
- Refuses silent cold start from step 0 unless ALLOW_COLD_START=1 is explicitly set.
- Atomically publishes immutable generation directories (checkpoint.pt + manifest.json)
  followed by a single atomic LATEST.json pointer swap.
- Periodic, graceful-stop (SIGINT/SIGTERM), and end-of-run final checkpoints all save
  complete resumable state: model, optimizer, RNG states, deterministic data cursor,
  and run identity.
- Optional inference-only weight exports (EXPORT_INFERENCE_WEIGHTS=1) are written
  separately under $CKPT_DIR/exports/ and never replace the resumable training lineage.
"""

from __future__ import annotations

import io
import math
import os
import random
import signal
import time

import bitsandbytes as bnb
import torch
import zstandard as zstd
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from checkpoint_io import (
    CheckpointIdentityError,
    CheckpointIntegrityError,
    DeterministicBlockStream,
    build_checkpoint_payload,
    build_run_identity,
    capture_rng_state,
    compute_data_identity,
    compute_tokenizer_identity,
    discover_shards,
    export_inference_weights,
    load_latest_checkpoint,
    preflight_resume,
    prune_old_checkpoints,
    publish_checkpoint,
    restore_rng_state,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED_TOKENIZER = os.path.join(HERE, "tokenizer")

SMOKE = os.environ.get("SMOKE", "") == "1"
DATA_ROOT = os.path.expanduser(os.environ.get("DATA_ROOT", "~/stackv2-burst-run/data"))
CKPT_DIR = os.path.expanduser(
    os.environ.get("CKPT_DIR", "~/stackv2-burst-run/ckpt" if not SMOKE else "~/stackv2-burst-run/smoke-ckpt")
)
TOKENIZER = os.environ.get("TOKENIZER") or (BUNDLED_TOKENIZER if os.path.isdir(BUNDLED_TOKENIZER) else "bigcode/starcoder2-3b")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048"))
SEED = int(os.environ.get("SEED", "0"))
ALLOW_COLD_START = os.environ.get("ALLOW_COLD_START", "0") == "1"
EXPORT_INFERENCE = os.environ.get("EXPORT_INFERENCE_WEIGHTS", "0") == "1"
KEEP_CKPTS = int(os.environ.get("KEEP_CKPTS", "3"))
DEVICE = os.environ.get("DEVICE", "cuda")

LANGS = ["Python"] if SMOKE else None  # None => every language subdir under DATA_ROOT


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _optional_str_env(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw if raw else None


def list_shards() -> list[str]:
    return discover_shards(DATA_ROOT, langs=LANGS, extensions=(".zst",))


def open_zst_jsonl(path: str):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        rdr = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
        for line in rdr:
            yield line


def build_model_and_schedule(vocab: int):
    if SMOKE:
        cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=SEQ_LEN,
            rms_norm_eps=1e-5,
            rope_theta=float(os.environ.get("ROPE_THETA", "10000")),
        )
        steps = int(os.environ.get("STEPS", "50"))
        bs = int(os.environ.get("BS", "2"))
        acc = int(os.environ.get("ACC", "1"))
        save_every = int(os.environ.get("SAVE", "25"))
        log_every = 5
        warm = int(os.environ.get("WARM", "5"))
    else:
        cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=SEQ_LEN,
            rms_norm_eps=1e-5,
            rope_theta=float(os.environ.get("ROPE_THETA", "10000")),
        )
        steps = int(os.environ.get("STEPS", "500000"))
        bs = int(os.environ.get("BS", "4"))
        acc = int(os.environ.get("ACC", "8"))
        save_every = int(os.environ.get("SAVE", "1000"))
        log_every = 10
        warm = int(os.environ.get("WARM", "2000"))

    cfg.use_cache = False
    cfg._attn_implementation = "sdpa"
    return cfg, steps, bs, acc, save_every, log_every, warm


def main() -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    expected_step = _optional_int_env("EXPECTED_STEP")
    expected_generation = _optional_str_env("EXPECTED_GENERATION")
    expected_sha256 = _optional_str_env("EXPECTED_SHA256")
    stop_at_step = _optional_int_env("STOP_AT_STEP")

    log(
        f"SMOKE={SMOKE} tokenizer={TOKENIZER} seq={SEQ_LEN} "
        f"rope={os.environ.get('ROPE_THETA', '10000')} ckpt={CKPT_DIR} "
        f"allow_cold_start={ALLOW_COLD_START}"
    )

    # --- Preflight BEFORE expensive model or GPU allocation ---
    shards = list_shards()
    if not shards:
        raise RuntimeError(f"No .zst data shards found under DATA_ROOT={DATA_ROOT} (LANGS={LANGS})")

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    vocab = len(tok)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    cfg, steps, bs, acc, save_every, log_every, warm = build_model_and_schedule(vocab)
    lr_max, lr_min = 3e-4, 3e-5

    tokenizer_identity = compute_tokenizer_identity(TOKENIZER, vocab_size=vocab, eos_token_id=eos_id)
    data_identity = compute_data_identity(DATA_ROOT, langs=LANGS, shard_paths=shards, extensions=(".zst",))
    expected_identity = build_run_identity(
        model_cfg=cfg.to_dict(),
        training_cfg={
            "seq_len": SEQ_LEN,
            "global_batch_size": bs * acc,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "warmup_steps": warm,
            "total_steps": steps,
            "seed": SEED,
        },
        tokenizer_identity=tokenizer_identity,
        data_identity=data_identity,
    )

    preflight = preflight_resume(
        CKPT_DIR,
        allow_cold_start=ALLOW_COLD_START,
        expected_step=expected_step,
        expected_generation=expected_generation,
        expected_sha256=expected_sha256,
        expected_identity=expected_identity,
        require_training_state=True,
    )
    log(
        f"preflight OK: mode={preflight['mode']} generation={preflight['generation']} "
        f"next_step={preflight['next_step']} shards={len(shards)} "
        f"identity={expected_identity['identity_sha256'][:12]}"
    )

    # --- Model & optimizer allocation (only after preflight passes) ---
    dev = DEVICE
    torch.manual_seed(SEED)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    model = LlamaForCausalLM(cfg)
    log(
        f"model params: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B  "
        f"vocab={vocab} BS={bs} ACC={acc} global_bs={bs * acc} gckpt={os.environ.get('GRAD_CKPT', '1')}"
    )
    if os.environ.get("GRAD_CKPT", "1") == "1":
        model.gradient_checkpointing_enable()
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model.to(dev, dtype=dtype)
    if dev.startswith("cuda"):
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step_idx: int) -> float:
        if step_idx < warm:
            return lr_max * step_idx / max(1, warm)
        progress = min(1.0, (step_idx - warm) / max(1, steps - warm))
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

    initial_cursor = None
    start = 0
    if preflight["mode"] == "resume":
        ckpt_path, sd, start = load_latest_checkpoint(
            CKPT_DIR,
            map_location=dev,
            allow_cold_start=False,
            require_training_state=True,
            require_full_state=True,
            expected_step=expected_step,
            expected_generation=expected_generation,
            expected_sha256=expected_sha256,
            expected_identity=expected_identity,
        )
        assert sd is not None
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        restore_rng_state(sd["rng_state"], include_torch=True)
        initial_cursor = sd["data_cursor"]
        log(
            f"resumed {ckpt_path} (completed_step={start - 1}, next_step={start}, "
            f"epoch={initial_cursor['epoch']}, shard={initial_cursor['shard_index']}, "
            f"line={initial_cursor['line_index']})"
        )

    def tokenize_text(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    stream = DeterministicBlockStream(
        shards=shards,
        tokenize_fn=tokenize_text,
        eos_token_id=eos_id,
        seq_len=SEQ_LEN,
        seed=SEED,
        cursor=initial_cursor,
        line_reader=open_zst_jsonl,
    )

    stop_requested = False

    def _handle_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        log(f"received signal {signum}; will finish current step and publish a full-state checkpoint")

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    end_step = min(steps, stop_at_step) if stop_at_step is not None else steps
    if start >= end_step:
        log(f"nothing to run: start={start} >= end_step={end_step}")
        return

    def save_full_state(completed_step: int, *, is_final: bool = False) -> dict:
        payload = build_checkpoint_payload(
            model.state_dict(),
            opt.state_dict(),
            completed_step,
            cfg.to_dict(),
            rng_state=capture_rng_state(include_torch=True, base_seed=SEED),
            data_cursor=stream.snapshot_cursor(),
            identity=expected_identity,
            is_final=is_final,
        )
        record = publish_checkpoint(CKPT_DIR, payload, is_final=is_final)
        log(
            f"published {record['generation']} (completed_step={completed_step}, "
            f"next_step={completed_step + 1}, is_final={is_final}, sha256={record['sha256'][:12]})"
        )
        prune_old_checkpoints(CKPT_DIR, keep=KEEP_CKPTS)
        return record

    model.train()
    t0 = time.time()
    toks = 0
    last_completed_step: int | None = None
    last_saved_step: int | None = None
    last_saved_final = False

    for step in range(start, end_step):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        lacc = 0.0
        for _ in range(acc):
            batch_blocks = stream.next_batch(bs)
            batch = torch.tensor(batch_blocks, dtype=torch.long, device=dev)
            out = model(input_ids=batch, labels=batch)
            (out.loss / acc).backward()
            lacc += out.loss.item() / acc
            toks += batch.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_completed_step = step

        if step % log_every == 0:
            dt = time.time() - t0
            tps = toks / dt if dt > 0 else 0.0
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if dev.startswith("cuda") and torch.cuda.is_available()
                else 0.0
            )
            log(
                f"step {step}/{steps} loss {lacc:.4f} lr {lr_at(step):.2e} "
                f"tok/s {tps:,.0f} peakmem {peak_gb:.1f}GB"
            )
            t0 = time.time()
            toks = 0

        if stop_requested:
            break

        if step > 0 and step % save_every == 0:
            save_full_state(step, is_final=False)
            last_saved_step = step
            last_saved_final = False

    # Always publish a final numbered full-state checkpoint covering any tail steps
    # (including deliberate stop via SIGINT/SIGTERM or STOP_AT_STEP).
    if last_completed_step is not None and (last_saved_step != last_completed_step or not last_saved_final):
        final_record = save_full_state(last_completed_step, is_final=True)
        log(
            f"DONE (resumable final generation={final_record['generation']}, "
            f"completed_step={last_completed_step}, next_step={last_completed_step + 1}, "
            f"sha256={final_record['sha256'][:12]})"
        )
        if EXPORT_INFERENCE:
            export_rec = export_inference_weights(
                CKPT_DIR,
                model.state_dict(),
                last_completed_step,
                cfg.to_dict(),
                identity=expected_identity,
            )
            log(f"exported inference-only weights to {export_rec['export_dir']}")


if __name__ == "__main__":
    main()
