import ast
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import checkpoint_io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pickle_save(payload, fh):
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path, map_location="cpu"):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _write_executable(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_sample_shards(data_root: str, langs=("Python",), shards_per_lang=2, lines_per_shard=16) -> list[str]:
    paths = []
    for lang in langs:
        ldir = os.path.join(data_root, lang)
        os.makedirs(ldir, exist_ok=True)
        for sidx in range(shards_per_lang):
            p = os.path.join(ldir, f"shard_{sidx:02d}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                for lidx in range(lines_per_shard):
                    content = f"# {lang} s={sidx} l={lidx}\ndef f_{sidx}_{lidx}(a): return a * {lidx + 1} + {sidx}\n"
                    fh.write(json.dumps({"content": content}) + "\n")
            paths.append(p)
    return sorted(paths)


def _byte_tokenizer(text: str) -> list[int]:
    return [b + 10 for b in text.encode("utf-8")]


def _make_identity(
    data_root: str,
    *,
    tokenizer_dir: str | None = None,
    langs=None,
    rope_theta: float = 500000.0,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    seq_len: int = 32,
    global_batch_size: int = 4,
    lr_max: float = 3e-4,
    total_steps: int = 20,
    seed: int = 11,
    extensions=(".jsonl",),
) -> dict:
    tok_dir = tokenizer_dir or os.path.join(REPO_ROOT, "tokenizer")
    tok_id = checkpoint_io.compute_tokenizer_identity(tok_dir)
    data_id = checkpoint_io.compute_data_identity(data_root, langs=langs, extensions=extensions)
    model_cfg = {
        "vocab_size": tok_id["vocab_size"] or 49152,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": seq_len,
        "rms_norm_eps": 1e-5,
        "rope_theta": rope_theta,
    }
    train_cfg = {
        "seq_len": seq_len,
        "global_batch_size": global_batch_size,
        "lr_max": lr_max,
        "lr_min": 3e-5,
        "warmup_steps": 2,
        "total_steps": total_steps,
        "seed": seed,
    }
    return checkpoint_io.build_run_identity(model_cfg, train_cfg, tok_id, data_id)


def _make_full_payload(
    step: int,
    *,
    weight: int = 1,
    opt_val: int = 1,
    identity: dict | None = None,
    cursor: dict | None = None,
    rng_state: dict | None = None,
    is_final: bool = False,
) -> dict:
    cur = cursor or {
        "version": checkpoint_io.DATA_CURSOR_VERSION,
        "seed": 11,
        "epoch": 0,
        "shard_index": 0,
        "line_index": step + 1,
        "token_buffer": [12, 13, 14],
        "blocks_emitted": (step + 1) * 4,
        "samples_consumed": (step + 1) * 4,
    }
    rng = rng_state or checkpoint_io.capture_rng_state(include_torch=False, base_seed=11)
    return checkpoint_io.build_checkpoint_payload(
        model_state={"w": weight},
        opt_state={"m": opt_val},
        completed_step=step,
        cfg={"hidden_size": 256, "rope_theta": 500000.0},
        rng_state=rng,
        data_cursor=cur,
        identity=identity,
        is_final=is_final,
    )


class TestAtomicGenerationPublication(unittest.TestCase):
    """BURST-03: Immutable generation dirs + single atomic LATEST.json pointer swap."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        self.data_root = os.path.join(self._tmp.name, "data")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        _make_sample_shards(self.data_root)
        self.identity = _make_identity(self.data_root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_publish_creates_immutable_generation_and_atomic_pointer(self):
        payload = _make_full_payload(200, weight=200, opt_val=200, identity=self.identity)
        with mock.patch("checkpoint_io.os.fsync", wraps=os.fsync) as fsync_spy, mock.patch(
            "checkpoint_io.os.replace", wraps=os.replace
        ) as replace_spy:
            rec = checkpoint_io.publish_checkpoint(
                self.ckpt_dir, payload, save_fn=_pickle_save, generation_id="gen200"
            )

        self.assertGreaterEqual(fsync_spy.call_count, 3)
        self.assertEqual(replace_spy.call_count, 1)  # single atomic pointer swap

        gen_dir = os.path.join(self.ckpt_dir, "gen_00000200_gen200")
        self.assertTrue(os.path.isdir(gen_dir))
        self.assertTrue(os.path.isfile(os.path.join(gen_dir, "checkpoint.pt")))
        self.assertTrue(os.path.isfile(os.path.join(gen_dir, "manifest.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.ckpt_dir, "LATEST.json")))
        self.assertEqual(
            sorted(os.listdir(self.ckpt_dir)),
            [".latest.lock", "LATEST.json", "gen_00000200_gen200"],
        )

        with open(rec["ckpt_path"], "rb") as fh:
            raw = fh.read()
        self.assertEqual(rec["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(rec["size_bytes"], len(raw))

        loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir,
            load_fn=_pickle_load,
            require_full_state=True,
            expected_identity=self.identity,
        )
        self.assertEqual(loaded_path, rec["ckpt_path"])
        self.assertEqual(sd["completed_step"], 200)
        self.assertEqual(next_step, 201)

    def test_crash_during_serialization_preserves_prior_generation(self):
        rec1 = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(100, weight=100, opt_val=100, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g100",
        )

        def fail_save(_payload, fh):
            fh.write(b"partial-bytes")
            raise OSError("simulated power loss mid-write")

        with self.assertRaises(OSError):
            checkpoint_io.publish_checkpoint(
                self.ckpt_dir,
                _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
                save_fn=fail_save,
                generation_id="g200",
            )

        self.assertEqual(
            sorted(os.listdir(self.ckpt_dir)),
            [".latest.lock", "LATEST.json", "gen_00000100_g100"],
        )
        loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load, require_full_state=True, expected_identity=self.identity
        )
        self.assertEqual(loaded_path, rec1["ckpt_path"])
        self.assertEqual(sd["model"]["w"], 100)
        self.assertEqual(next_step, 101)

    def test_crash_between_dir_promotion_and_pointer_swap_preserves_prior_generation(self):
        """Inject crash right before LATEST.json os.replace (after gen dir rename), for BOTH new & same step."""
        rec1 = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(1000, weight=1000, opt_val=1000, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="orig1000",
        )

        # Case A: crash while publishing step 2000 right at the pointer swap.
        with mock.patch("checkpoint_io.os.replace", side_effect=OSError("crash at pointer swap")):
            with self.assertRaises(OSError):
                checkpoint_io.publish_checkpoint(
                    self.ckpt_dir,
                    _make_full_payload(2000, weight=2000, opt_val=2000, identity=self.identity),
                    save_fn=_pickle_save,
                    generation_id="uncommitted2000",
                )

        # Even though uncommitted gen_00002000_uncommitted2000 exists on disk, LATEST.json
        # still points atomically to gen_00001000_orig1000, which remains 100% valid and loadable!
        self.assertTrue(os.path.isdir(os.path.join(self.ckpt_dir, "gen_00002000_uncommitted2000")))
        loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load, require_full_state=True, expected_identity=self.identity
        )
        self.assertEqual(loaded_path, rec1["ckpt_path"])
        self.assertEqual(sd["model"]["w"], 1000)
        self.assertEqual(next_step, 1001)

        # Case B: crash while re-publishing the SAME step (1000) right at the pointer swap.
        # Prior generation MUST NOT be invalidated (the exact BURST-03 two-file overwrite flaw).
        with mock.patch("checkpoint_io.os.replace", side_effect=OSError("crash re-publishing step 1000")):
            with self.assertRaises(OSError):
                checkpoint_io.publish_checkpoint(
                    self.ckpt_dir,
                    _make_full_payload(1000, weight=9999, opt_val=9999, identity=self.identity),
                    save_fn=_pickle_save,
                    generation_id="retry1000",
                )

        loaded_path2, sd2, next_step2 = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load, require_full_state=True, expected_identity=self.identity
        )
        self.assertEqual(loaded_path2, rec1["ckpt_path"])
        self.assertEqual(sd2["model"]["w"], 1000)
        self.assertEqual(next_step2, 1001)

    def test_tamper_truncation_missing_and_corrupt_manifest_fail_closed(self):
        rec = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(500, weight=500, opt_val=500, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g500",
        )
        ckpt_path = rec["ckpt_path"]
        manifest_path = rec["manifest_path"]
        with open(ckpt_path, "rb") as fh:
            orig_ckpt_bytes = fh.read()
        with open(manifest_path, "rb") as fh:
            orig_manifest_bytes = fh.read()

        # Same-size single-byte tamper -> SHA-256 mismatch
        tampered = bytearray(orig_ckpt_bytes)
        tampered[-1] ^= 0xFF
        with open(ckpt_path, "wb") as fh:
            fh.write(tampered)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "SHA-256 mismatch"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

        # Truncation -> size mismatch
        with open(ckpt_path, "wb") as fh:
            fh.write(orig_ckpt_bytes[: len(orig_ckpt_bytes) // 2])
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "size mismatch"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

        # Restore checkpoint bytes, corrupt manifest JSON
        with open(ckpt_path, "wb") as fh:
            fh.write(orig_ckpt_bytes)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            fh.write("{broken json")
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Unreadable integrity manifest"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

        # Missing manifest
        os.remove(manifest_path)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Missing integrity manifest"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

        # Restore manifest, verify corrupt LATEST.json fails closed
        with open(manifest_path, "wb") as fh:
            fh.write(orig_manifest_bytes)
        with open(checkpoint_io.pointer_path_for(self.ckpt_dir), "w", encoding="utf-8") as fh:
            fh.write("{corrupt pointer")
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Unreadable checkpoint pointer"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

    def test_commit_staged_corrupt_transfer_leaves_destination_intact(self):
        rec_existing = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(300, weight=300, opt_val=300, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="existing300",
        )
        staging = os.path.join(self._tmp.name, "staging_bad")
        rec_new = checkpoint_io.publish_checkpoint(
            staging,
            _make_full_payload(400, weight=400, opt_val=400, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="bad400",
        )
        # Corrupt the staged checkpoint bytes to simulate broken rsync transfer
        with open(rec_new["ckpt_path"], "r+b") as fh:
            data = bytearray(fh.read())
            data[0] ^= 0xAA
            fh.seek(0)
            fh.write(data)

        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "SHA-256 mismatch"):
            checkpoint_io.commit_staged_checkpoint(staging, self.ckpt_dir, expected_step=400)

        # Destination still points cleanly to existing300
        loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load, require_full_state=True, expected_identity=self.identity
        )
        self.assertEqual(loaded_path, rec_existing["ckpt_path"])
        self.assertEqual(sd["model"]["w"], 300)
        self.assertEqual(next_step, 301)

    def test_prune_preserves_active_generation_and_cleans_staging(self):
        for step in (100, 200, 300, 400, 500):
            checkpoint_io.publish_checkpoint(
                self.ckpt_dir,
                _make_full_payload(step, weight=step, opt_val=step, identity=self.identity),
                save_fn=_pickle_save,
                generation_id=f"g{step}",
            )
        stale_stage = os.path.join(self.ckpt_dir, ".staging_gen_00000600_stale")
        os.makedirs(stale_stage, exist_ok=True)

        removed = checkpoint_io.prune_old_checkpoints(self.ckpt_dir, keep=3)
        self.assertEqual(len(removed), 2)
        self.assertFalse(os.path.exists(stale_stage))
        self.assertEqual(
            sorted(os.listdir(self.ckpt_dir)),
            [".latest.lock", "LATEST.json", "gen_00000300_g300", "gen_00000400_g400", "gen_00000500_g500"],
        )


class TestFinalCheckpointAndFullStateResume(unittest.TestCase):
    """BURST-02 & BURST-04: Final full-state checkpoint, RNG + cursor continuity, identity verification."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        self.data_root = os.path.join(self._tmp.name, "data")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.shards = _make_sample_shards(self.data_root, langs=("Python", "Go"), shards_per_lang=2, lines_per_shard=10)
        self.identity = _make_identity(self.data_root, langs=None, seq_len=24, global_batch_size=4, total_steps=18, seed=42)

    def tearDown(self):
        self._tmp.cleanup()

    def test_final_checkpoint_resumes_full_state_bit_for_bit_across_bursts(self):
        """Periodic save at step 5 + final full-state save at step 9 -> resume 10..17 matches uninterrupted 0..17."""
        seq_len = 24
        batch_size = 4
        seed = 42

        def step_fn(weight, opt_m, batch):
            r = random.randint(1, 1000)
            b_sum = sum(sum(row) for row in batch)
            new_m = (opt_m * 7 + b_sum + r) % 10_000_019
            new_w = (weight * 3 + new_m) % 10_000_019
            return new_w, new_m

        # Uninterrupted reference run (steps 0..17, crosses shards and epochs)
        random.seed(seed)
        ref_stream = checkpoint_io.DeterministicBlockStream(
            self.shards, _byte_tokenizer, eos_token_id=0, seq_len=seq_len, seed=seed
        )
        ref_w, ref_m = 0, 0
        ref_batches = []
        ref_rng_draws = []
        for _step in range(18):
            batch = ref_stream.next_batch(batch_size)
            ref_batches.append(batch)
            ref_w, ref_m = step_fn(ref_w, ref_m, batch)
            ref_rng_draws.append(random.randint(0, 999999))
        ref_cursor = ref_stream.snapshot_cursor()

        # Burst 1: cold start (explicitly authorized), periodic save at 5, final full-state save at 9
        pre = checkpoint_io.preflight_resume(
            self.ckpt_dir, allow_cold_start=True, expected_identity=self.identity
        )
        self.assertEqual(pre["mode"], "cold_start")
        random.seed(seed)
        stream1 = checkpoint_io.DeterministicBlockStream(
            self.shards, _byte_tokenizer, eos_token_id=0, seq_len=seq_len, seed=seed
        )
        w, m = 0, 0
        resumed_batches = []
        resumed_rng_draws = []
        for step in range(0, 10):
            batch = stream1.next_batch(batch_size)
            resumed_batches.append(batch)
            w, m = step_fn(w, m, batch)
            resumed_rng_draws.append(random.randint(0, 999999))
            if step in (5, 9):
                payload = checkpoint_io.build_checkpoint_payload(
                    model_state={"w": w},
                    opt_state={"m": m},
                    completed_step=step,
                    cfg={"hidden_size": 256, "rope_theta": 500000.0},
                    rng_state=checkpoint_io.capture_rng_state(include_torch=False, base_seed=seed),
                    data_cursor=stream1.snapshot_cursor(),
                    identity=self.identity,
                    is_final=(step == 9),
                )
                checkpoint_io.publish_checkpoint(
                    self.ckpt_dir, payload, save_fn=_pickle_save, is_final=(step == 9)
                )

        # Stage final checkpoint to a second host directory via stage-ckpt.sh local
        dest_ckpt_dir = os.path.join(self._tmp.name, "ckpt_burst_2")
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "local", self.ckpt_dir, dest_ckpt_dir],
            env={**os.environ, "EXPECTED_STEP": "9"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Burst 2: resume from final step 9 -> starts at step 10 (no lost tail steps 6..9!)
        _, sd, start_step = checkpoint_io.load_latest_checkpoint(
            dest_ckpt_dir,
            load_fn=_pickle_load,
            allow_cold_start=False,
            require_full_state=True,
            expected_step=9,
            expected_identity=self.identity,
        )
        self.assertEqual(start_step, 10)
        self.assertTrue(sd["is_final"])
        checkpoint_io.restore_rng_state(sd["rng_state"], include_torch=False)
        stream2 = checkpoint_io.DeterministicBlockStream(
            self.shards,
            _byte_tokenizer,
            eos_token_id=0,
            seq_len=seq_len,
            seed=seed,
            cursor=sd["data_cursor"],
        )
        w, m = sd["model"]["w"], sd["opt"]["m"]
        for _step in range(start_step, 18):
            batch = stream2.next_batch(batch_size)
            resumed_batches.append(batch)
            w, m = step_fn(w, m, batch)
            resumed_rng_draws.append(random.randint(0, 999999))

        self.assertEqual(resumed_batches, ref_batches)
        self.assertEqual(resumed_rng_draws, ref_rng_draws)
        self.assertEqual((w, m), (ref_w, ref_m))
        self.assertEqual(stream2.snapshot_cursor(), ref_cursor)

    def test_inference_export_is_separate_and_rejected_for_training_resume(self):
        rec = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(249, weight=249, opt_val=249, identity=self.identity, is_final=True),
            save_fn=_pickle_save,
            is_final=True,
            generation_id="final249",
        )
        exp = checkpoint_io.export_inference_weights(
            self.ckpt_dir,
            model_state={"w": 249},
            completed_step=249,
            cfg={"hidden_size": 256},
            identity=self.identity,
            save_fn=_pickle_save,
        )
        # LATEST.json still points to the resumable full-state final generation
        ptr = checkpoint_io.read_pointer(checkpoint_io.pointer_path_for(self.ckpt_dir))
        self.assertEqual(ptr["generation"], rec["generation"])

        # Attempting to load the inference export for training resume fails closed
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "inference_export"):
            checkpoint_io.load_checkpoint(
                exp["weights_path"], load_fn=_pickle_load, require_training_state=True
            )

    def test_strict_identity_mismatch_rejections(self):
        checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(50, weight=50, opt_val=50, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g50",
        )

        # 1. Shape-compatible RoPE drift (explicit BURST-04 case)
        bad_rope = _make_identity(self.data_root, seq_len=24, global_batch_size=4, total_steps=18, seed=42, rope_theta=10000.0)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "model\\.rope_theta"):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_rope)

        # 2. Model architecture mismatch
        bad_arch = _make_identity(self.data_root, seq_len=24, global_batch_size=4, total_steps=18, seed=42, hidden_size=512)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "model\\.hidden_size"):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_arch)

        # 3. Training config mismatch (global_batch_size, lr_max, seq_len, total_steps)
        bad_bs = _make_identity(self.data_root, seq_len=24, global_batch_size=8, total_steps=18, seed=42)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "training\\.global_batch_size"):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_bs)

        bad_lr = _make_identity(self.data_root, seq_len=24, global_batch_size=4, total_steps=18, seed=42, lr_max=1e-4)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "training\\.lr_max"):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_lr)

        # 4. Tokenizer mismatch (tampered copy of tokenizer_config.json)
        alt_tok_dir = os.path.join(self._tmp.name, "alt_tok")
        shutil.copytree(os.path.join(REPO_ROOT, "tokenizer"), alt_tok_dir)
        cfg_p = os.path.join(alt_tok_dir, "tokenizer_config.json")
        with open(cfg_p, "r", encoding="utf-8") as fh:
            tok_cfg = json.load(fh)
        tok_cfg["vocab_size"] = 32000
        with open(cfg_p, "w", encoding="utf-8") as fh:
            json.dump(tok_cfg, fh)
        bad_tok = _make_identity(
            self.data_root, tokenizer_dir=alt_tok_dir, seq_len=24, global_batch_size=4, total_steps=18, seed=42
        )
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "tokenizer\\."):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_tok)

        # 5. Data shard set mismatch (modified shard content)
        with open(self.shards[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"content": "extra_line = 1\n"}) + "\n")
        bad_data = _make_identity(self.data_root, seq_len=24, global_batch_size=4, total_steps=18, seed=42)
        with self.assertRaisesRegex(checkpoint_io.CheckpointIdentityError, "data\\.shards_sha256"):
            checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=bad_data)

    def test_cross_hardware_bs_acc_split_allowed_when_global_batch_matches(self):
        # BS=6, ACC=8 => global_batch_size=48 matches BS=16, ACC=3 => global_batch_size=48
        id_host_a = _make_identity(self.data_root, seq_len=24, global_batch_size=6 * 8, total_steps=18, seed=42)
        id_host_b = _make_identity(self.data_root, seq_len=24, global_batch_size=16 * 3, total_steps=18, seed=42)
        checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(100, weight=100, opt_val=100, identity=id_host_a),
            save_fn=_pickle_save,
            generation_id="g100",
        )
        info = checkpoint_io.preflight_resume(self.ckpt_dir, expected_identity=id_host_b)
        self.assertEqual(info["mode"], "resume")
        self.assertEqual(info["next_step"], 101)


class TestFailClosedLauncherAndColdStart(unittest.TestCase):
    """BURST-01 & BURST-05: Launcher fail-closed transfer refusal, cold-start gating, local fixture."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.work = os.path.join(self.root, "work")
        self.data_root = os.path.join(self.work, "data")
        self.ckpt_dir = os.path.join(self.work, "ckpt")
        self.remote_ckpt = os.path.join(self.root, "remote_ckpt")
        self.bin_dir = os.path.join(self.root, "bin")
        self.sentinel = os.path.join(self.root, "train_launched.json")
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.remote_ckpt, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)

        # Create a .zst placeholder shard so DATA_ROOT is already populated
        os.makedirs(os.path.join(self.data_root, "Python"), exist_ok=True)
        with open(os.path.join(self.data_root, "Python", "shard_00.zst"), "wb") as fh:
            fh.write(b"dummy-zst-fixture")

        self.fake_train = os.path.join(self.root, "fake_train.py")
        with open(self.fake_train, "w", encoding="utf-8") as fh:
            fh.write(
                "import json, os\n"
                f"with open({self.sentinel!r}, 'w') as f:\n"
                "    json.dump({'ckpt_dir': os.environ.get('CKPT_DIR'), "
                "'expected_generation': os.environ.get('EXPECTED_GENERATION'), "
                "'tokenizer': os.environ.get('TOKENIZER')}, f)\n"
            )

        self.fake_ssh = _write_executable(
            os.path.join(self.bin_dir, "fake_ssh"),
            "#!/usr/bin/env bash\nexit 0\n",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _base_env(self, rsync_bin: str, **extra) -> dict:
        env = dict(os.environ)
        env.update(
            {
                "BURST_MODE": "nas",
                "SKIP_DEPS": "1",
                "SKIP_TAILSCALE": "1",
                "WORK": self.work,
                "DATA_ROOT": self.data_root,
                "CKPT_DIR": self.ckpt_dir,
                "NAS_HOST": "nas.example.test",
                "NAS_USER": "burst",
                "NAS_DATA": "/remote/data",
                "NAS_CKPT": self.remote_ckpt,
                "SSH_BIN": self.fake_ssh,
                "RSYNC_BIN": rsync_bin,
                "PYTHON_BIN": sys.executable,
                "TRAIN_SCRIPT": self.fake_train,
            }
        )
        env.update(extra)
        return env

    def _make_copying_rsync(self) -> str:
        """Fake rsync that strips `user@host:` prefix and copies local files/dirs."""
        return _write_executable(
            os.path.join(self.bin_dir, "copy_rsync"),
            f"""#!{sys.executable}
import os, shutil, sys
args = [a for a in sys.argv[1:] if not a.startswith('-')]
# Skip value after -e
cleaned = []
skip = False
for a in sys.argv[1:]:
    if skip:
        skip = False
        continue
    if a == '-e':
        skip = True
        continue
    if a.startswith('-'):
        continue
    cleaned.append(a)
src, dst = cleaned[-2], cleaned[-1]
if ':' in src:
    src = src.split(':', 1)[1]
if not os.path.exists(src):
    sys.stderr.write(f"mock rsync missing: {{src}}\\n")
    sys.exit(23)
if os.path.isdir(src):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
else:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
""",
        )

    def test_load_latest_checkpoint_refuses_silent_cold_start_by_default(self):
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Refusing silent cold start"):
            checkpoint_io.load_latest_checkpoint(self.ckpt_dir, load_fn=_pickle_load)

        path, sd, step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load, allow_cold_start=True
        )
        self.assertIsNone(path)
        self.assertIsNone(sd)
        self.assertEqual(step, 0)

    def test_go_sh_fails_closed_on_failed_pull_empty_dir_and_never_launches(self):
        fail_rsync = _write_executable(
            os.path.join(self.bin_dir, "fail_rsync"),
            "#!/usr/bin/env bash\necho 'mock rsync failure' >&2\nexit 255\n",
        )
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._base_env(fail_rsync),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(self.sentinel), "train.py MUST NOT launch after failed pull")

    def test_go_sh_fails_closed_on_failed_pull_even_with_allow_cold_start_and_stale_ckpt(self):
        # Place a stale preexisting checkpoint in CKPT_DIR
        checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(50, weight=50, opt_val=50),
            save_fn=_pickle_save,
            generation_id="stale50",
        )
        fail_rsync = _write_executable(
            os.path.join(self.bin_dir, "fail_rsync_stale"),
            "#!/usr/bin/env bash\necho 'mock transfer timeout' >&2\nexit 12\n",
        )
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._base_env(fail_rsync, ALLOW_COLD_START="1"),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(
            os.path.exists(self.sentinel),
            "train.py MUST NOT launch on stale checkpoint or cold-start after failed pull",
        )

    def test_go_sh_rejects_corrupt_remote_pull_and_keeps_prior_ckpt_intact(self):
        rec_local = checkpoint_io.publish_checkpoint(
            self.ckpt_dir,
            _make_full_payload(100, weight=100, opt_val=100),
            save_fn=_pickle_save,
            generation_id="local100",
        )
        rec_remote = checkpoint_io.publish_checkpoint(
            self.remote_ckpt,
            _make_full_payload(200, weight=200, opt_val=200),
            save_fn=_pickle_save,
            generation_id="remote200",
        )
        # Tamper remote checkpoint bytes
        with open(rec_remote["ckpt_path"], "r+b") as fh:
            b = bytearray(fh.read())
            b[-1] ^= 0xFF
            fh.seek(0)
            fh.write(b)

        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._base_env(self._make_copying_rsync()),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(self.sentinel))
        # Prior valid local checkpoint remains intact
        loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            self.ckpt_dir, load_fn=_pickle_load
        )
        self.assertEqual(loaded_path, rec_local["ckpt_path"])
        self.assertEqual(sd["model"]["w"], 100)
        self.assertEqual(next_step, 101)

    def test_go_sh_commits_valid_remote_pull_and_pins_expected_generation(self):
        rec_remote = checkpoint_io.publish_checkpoint(
            self.remote_ckpt,
            _make_full_payload(400, weight=400, opt_val=400),
            save_fn=_pickle_save,
            generation_id="remote400",
        )
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._base_env(self._make_copying_rsync(), EXPECTED_STEP="400"),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertTrue(os.path.exists(self.sentinel))
        with open(self.sentinel, "r", encoding="utf-8") as fh:
            launched = json.load(fh)
        self.assertEqual(launched["expected_generation"], rec_remote["generation"])
        self.assertEqual(launched["tokenizer"], os.path.join(REPO_ROOT, "tokenizer"))

    def test_go_sh_rejects_expected_step_mismatch_and_never_launches(self):
        checkpoint_io.publish_checkpoint(
            self.remote_ckpt,
            _make_full_payload(400, weight=400, opt_val=400),
            save_fn=_pickle_save,
            generation_id="remote400",
        )
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._base_env(self._make_copying_rsync(), EXPECTED_STEP="999"),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(os.path.exists(self.sentinel), "train.py MUST NOT launch on EXPECTED_STEP mismatch")

    def test_go_sh_local_mode_requires_explicit_cold_start_authorization(self):
        env_no_cold = self._base_env(self._make_copying_rsync(), BURST_MODE="local", ALLOW_COLD_START="0")
        res1 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=env_no_cold,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res1.returncode, 0)
        self.assertFalse(os.path.exists(self.sentinel))

        env_cold = self._base_env(self._make_copying_rsync(), BURST_MODE="local", ALLOW_COLD_START="1")
        res2 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=env_cold,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res2.returncode, 0, msg=res2.stderr)
        self.assertTrue(os.path.exists(self.sentinel))

    def test_setup_on_rental_delegates_to_go_sh_with_bundled_tokenizer(self):
        env_cold = self._base_env(self._make_copying_rsync(), BURST_MODE="local", ALLOW_COLD_START="1")
        env_cold.pop("TOKENIZER", None)
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "setup-on-rental.sh")],
            env=env_cold,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        with open(self.sentinel, "r", encoding="utf-8") as fh:
            launched = json.load(fh)
        self.assertEqual(launched["tokenizer"], os.path.join(REPO_ROOT, "tokenizer"))


class TestRepoContractAndLegacyCompat(unittest.TestCase):
    """Static wiring, shell syntax, pinned deps, bundled-tokenizer round-trip, and legacy v1 compat."""

    def test_shell_scripts_pass_bash_n(self):
        for script in ("go.sh", "setup-on-rental.sh", "stage-ckpt.sh"):
            path = os.path.join(REPO_ROOT, script)
            res = subprocess.run(["bash", "-n", path], capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, msg=f"bash -n failed for {script}: {res.stderr}")

    def test_requirements_txt_pins_all_packages_and_matches_cuda_wheel(self):
        req_path = os.path.join(REPO_ROOT, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        self.assertGreaterEqual(len(lines), 4)
        for line in lines:
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.-]*$", msg=f"Unpinned dep: {line}")

        for rel in ("go.sh", "README.md", "requirements.txt"):
            with open(os.path.join(REPO_ROOT, rel), "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("cu128", text, msg=f"{rel} must document/use cu128 consistently")
            self.assertNotIn("cu130", text, msg=f"{rel} must not contain stale cu130 mismatch")

    def test_local_fixture_roundtrip_cli_uses_bundled_tokenizer(self):
        res = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "checkpoint_io.py"), "local-roundtrip"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        out = json.loads(res.stdout)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["resumed_at_step"], 7)
        self.assertRegex(out["tokenizer_files_sha256"], r"^[0-9a-f]{64}$")

    def test_legacy_v1_flat_checkpoint_compat_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_ckpt = os.path.join(tmp, "ckpt_00001000.pt")
            legacy_payload = {"model": {"w": 42}, "opt": {"lr": 3e-4}, "step": 1000, "cfg": {"hidden_size": 256}}
            with open(legacy_ckpt, "wb") as fh:
                _pickle_save(legacy_payload, fh)
            sha, size = checkpoint_io.compute_file_sha256(legacy_ckpt)
            manifest = {
                "version": 1,
                "filename": "ckpt_00001000.pt",
                "sha256": sha,
                "size_bytes": size,
                "completed_step": 1000,
                "next_step": 1001,
            }
            with open(f"{legacy_ckpt}.sha256", "w", encoding="utf-8") as fh:
                json.dump(manifest, fh)

            path, sd, next_step = checkpoint_io.load_latest_checkpoint(
                tmp, load_fn=_pickle_load, require_full_state=False
            )
            self.assertEqual(path, legacy_ckpt)
            self.assertEqual(sd["step"], 1000)
            self.assertEqual(next_step, 1001)

            # Strict full-state mode rejects legacy checkpoints missing rng_state/data_cursor/identity
            with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "rng_state"):
                checkpoint_io.load_latest_checkpoint(tmp, load_fn=_pickle_load, require_full_state=True)

    def test_train_py_wires_preflight_before_allocation_and_full_state_publication(self):
        train_path = os.path.join(REPO_ROOT, "train.py")
        with open(train_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=train_path)

        called_names = set()
        preflight_line = None
        model_alloc_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name:
                    called_names.add(name)
                    if name == "preflight_resume" and preflight_line is None:
                        preflight_line = node.lineno
                    if name == "LlamaForCausalLM" and model_alloc_line is None:
                        model_alloc_line = node.lineno

        for required in (
            "preflight_resume",
            "load_latest_checkpoint",
            "build_checkpoint_payload",
            "publish_checkpoint",
            "prune_old_checkpoints",
            "capture_rng_state",
            "restore_rng_state",
            "DeterministicBlockStream",
            "compute_tokenizer_identity",
            "compute_data_identity",
            "build_run_identity",
            "export_inference_weights",
        ):
            self.assertIn(required, called_names, msg=f"train.py missing call to {required}")

        self.assertIsNotNone(preflight_line)
        self.assertIsNotNone(model_alloc_line)
        self.assertLess(
            preflight_line,
            model_alloc_line,
            "preflight_resume must run BEFORE expensive LlamaForCausalLM allocation",
        )
        self.assertNotIn("save", called_names)
        self.assertNotIn("load", called_names)
        self.assertNotIn("final.pt", source, "train.py must not write unresumable final.pt")


class TestRemoteStagingAndMonotonicGuard(unittest.TestCase):
    """BURST-03 & BURST-05: Remote destination verification, monotonic pointer guard, full NAS roundtrip."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.bin_dir = os.path.join(self.root, "bin")
        self.data_root = os.path.join(self.root, "data")
        os.makedirs(self.bin_dir, exist_ok=True)
        _make_sample_shards(self.data_root)
        self.identity = _make_identity(self.data_root)

        # Isolated local fake SSH/rsync transport shim (network-free, sandboxed to self.root)
        self.fake_transport = _write_executable(
            os.path.join(self.bin_dir, "fake_transport"),
            f"""#!{sys.executable}
import os, pathlib, shlex, shutil, subprocess, sys

safe_root = pathlib.Path(os.environ["BURST_TEST_ROOT"]).resolve()

def strip_host(spec: str) -> pathlib.Path:
    raw = spec.split(":", 1)[1] if ":" in spec else spec
    p = pathlib.Path(raw).resolve()
    assert p.is_relative_to(safe_root), f"Refusing path outside sandbox: {{p}}"
    return p

def copy_entry(src: pathlib.Path, dst: pathlib.Path, src_trailing_slash: bool) -> None:
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(src):
            s = src / name
            d = dst / name
            if s.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

if len(sys.argv) == 3 and not sys.argv[1].startswith("-"):
    _host, command = sys.argv[1], sys.argv[2]
    if command.startswith("rsync "):
        tokens = shlex.split(command)
        src_raw, dst_raw = tokens[-2], tokens[-1]
        copy_entry(strip_host(src_raw), strip_host(dst_raw), src_raw.endswith("/"))
        sys.exit(0)
    res = subprocess.run(["/bin/bash", "-c", command], check=False)
    sys.exit(res.returncode)

cleaned = []
skip = False
for arg in sys.argv[1:]:
    if skip:
        skip = False
        continue
    if arg == "-e":
        skip = True
        continue
    if arg.startswith("-"):
        continue
    cleaned.append(arg)
src_raw, dst_raw = cleaned[-2], cleaned[-1]
copy_entry(strip_host(src_raw), strip_host(dst_raw), src_raw.endswith("/"))
""",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _remote_env(self, **extra) -> dict:
        env = dict(os.environ)
        env.update(
            {
                "BURST_TEST_ROOT": self.root,
                "TMPDIR": self.root,
                "SSH_BIN": self.fake_transport,
                "RSYNC_BIN": self.fake_transport,
                "PYTHON_BIN": sys.executable,
                "REMOTE_PYTHON_BIN": sys.executable,
            }
        )
        env.update(extra)
        return env

    def test_remote_to_nas_rejects_preexisting_corrupt_generation_and_preserves_prior_latest(self):
        """Exact BURST-03 counterexample: corrupt pre-existing generation collision on remote destination."""
        src = os.path.join(self.root, "src_primary")
        dst = os.path.join(self.root, "dst_nas")
        rec_old = checkpoint_io.publish_checkpoint(
            dst,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="safe150",
        )
        rec_new = checkpoint_io.publish_checkpoint(
            src,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="incoming200",
        )

        # Case 1: Pre-existing destination directory with appended garbage (size mismatch)
        existing_bad = os.path.join(dst, rec_new["generation"])
        shutil.copytree(rec_new["generation_dir"], existing_bad)
        with open(os.path.join(existing_bad, "checkpoint.pt"), "ab") as fh:
            fh.write(b"synthetic accidental corruption")

        env = self._remote_env(
            PRIMARY_HOST="fixture-primary",
            PRIMARY_CKPT_DIR=src,
            NAS_HOST="fixture-nas",
            NAS_CKPT_DIR=dst,
            EXPECTED_STEP="200",
        )
        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0, "Remote stage MUST fail loudly on corrupt pre-existing generation")
        self.assertIn("size mismatch", res.stderr)
        # Staging temp dir cleaned up; LATEST.json still verified and pointing to healthy step 150
        self.assertFalse(any(name.startswith(".stage_ckpt_") for name in os.listdir(dst)))
        ptr, _, _ = checkpoint_io.verify_pointer(dst)
        self.assertEqual(ptr["generation"], rec_old["generation"])
        self.assertEqual(ptr["completed_step"], 150)

        # Case 2: Same-size single-byte tamper in pre-existing destination directory (SHA-256 mismatch)
        shutil.rmtree(existing_bad)
        shutil.copytree(rec_new["generation_dir"], existing_bad)
        with open(os.path.join(existing_bad, "checkpoint.pt"), "r+b") as fh:
            buf = bytearray(fh.read())
            buf[-1] ^= 0xFF
            fh.seek(0)
            fh.write(buf)

        res2 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res2.returncode, 0)
        self.assertIn("SHA-256 mismatch", res2.stderr)
        ptr2, _, _ = checkpoint_io.verify_pointer(dst)
        self.assertEqual(ptr2["generation"], rec_old["generation"])
        self.assertEqual(ptr2["completed_step"], 150)

        # Case 3: Missing manifest in pre-existing destination directory
        shutil.rmtree(existing_bad)
        shutil.copytree(rec_new["generation_dir"], existing_bad)
        os.remove(os.path.join(existing_bad, "manifest.json"))

        res3 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res3.returncode, 0)
        self.assertIn("Missing integrity manifest", res3.stderr)
        ptr3, _, _ = checkpoint_io.verify_pointer(dst)
        self.assertEqual(ptr3["generation"], rec_old["generation"])

    def test_remote_to_nas_rejects_preexisting_conflicting_valid_generation(self):
        src = os.path.join(self.root, "src_conflict")
        dst = os.path.join(self.root, "dst_conflict")
        alt = os.path.join(self.root, "alt_conflict")

        rec_safe = checkpoint_io.publish_checkpoint(
            dst,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="safe150",
        )
        rec_src = checkpoint_io.publish_checkpoint(
            src,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="shared200",
        )
        rec_alt = checkpoint_io.publish_checkpoint(
            alt,
            _make_full_payload(200, weight=999, opt_val=999, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="shared200",
        )
        # Pre-populate destination with a different internally-valid generation sharing the same name
        shutil.copytree(rec_alt["generation_dir"], os.path.join(dst, rec_src["generation"]))

        res = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=self._remote_env(
                PRIMARY_HOST="fixture-primary",
                PRIMARY_CKPT_DIR=src,
                NAS_HOST="fixture-nas",
                NAS_CKPT_DIR=dst,
                EXPECTED_STEP="200",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("conflicting generation", res.stderr)
        ptr, _, _ = checkpoint_io.verify_pointer(dst)
        self.assertEqual(ptr["generation"], rec_safe["generation"])
        self.assertEqual(ptr["completed_step"], 150)

    def test_monotonic_latest_pointer_refuses_rollback_local_and_publish_unless_overridden(self):
        """Exact BURST-03 rollback counterexample: step 200 refused over healthy step 249 unless explicit override."""
        older = os.path.join(self.root, "older_200")
        newer = os.path.join(self.root, "newer_249")
        rec_old = checkpoint_io.publish_checkpoint(
            older,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g200",
        )
        rec_new = checkpoint_io.publish_checkpoint(
            newer,
            _make_full_payload(249, weight=249, opt_val=249, identity=self.identity, is_final=True),
            save_fn=_pickle_save,
            is_final=True,
            generation_id="g249",
        )

        # Direct Python commit_staged_checkpoint refuses rollback
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "roll back"):
            checkpoint_io.commit_staged_checkpoint(older, newer, expected_step=200)
        ptr, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr["completed_step"], 249)
        self.assertEqual(ptr["generation"], rec_new["generation"])

        # Direct publish_checkpoint also refuses rollback without allow_rollback=True
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "roll back"):
            checkpoint_io.publish_checkpoint(
                newer,
                _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
                save_fn=_pickle_save,
                generation_id="direct_back200",
            )
        ptr2, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr2["completed_step"], 249)

        # stage-ckpt.sh local refuses rollback by default
        res_refuse = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "local", older, newer],
            env={**os.environ, "EXPECTED_STEP": "200", "PYTHON_BIN": sys.executable},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res_refuse.returncode, 0)
        self.assertIn("roll back", res_refuse.stderr)
        ptr3, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr3["completed_step"], 249)

        # Explicit --allow-rollback succeeds
        res_ok = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "local", older, newer, "--allow-rollback"],
            env={**os.environ, "EXPECTED_STEP": "200", "PYTHON_BIN": sys.executable},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_ok.returncode, 0, msg=res_ok.stderr)
        ptr4, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr4["completed_step"], 200)
        self.assertEqual(ptr4["generation"], rec_old["generation"])

    def test_monotonic_latest_pointer_refuses_rollback_remote_and_go_sh_unless_overridden(self):
        older = os.path.join(self.root, "remote_older_200")
        newer = os.path.join(self.root, "remote_newer_249")
        rec_old = checkpoint_io.publish_checkpoint(
            older,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="rem200",
        )
        rec_new = checkpoint_io.publish_checkpoint(
            newer,
            _make_full_payload(249, weight=249, opt_val=249, identity=self.identity, is_final=True),
            save_fn=_pickle_save,
            is_final=True,
            generation_id="rem249",
        )

        env = self._remote_env(
            PRIMARY_HOST="fixture-primary",
            PRIMARY_CKPT_DIR=older,
            NAS_HOST="fixture-nas",
            NAS_CKPT_DIR=newer,
            EXPECTED_STEP="200",
        )
        # Remote to-nas refuses rollback by default
        res_refuse = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res_refuse.returncode, 0)
        self.assertIn("roll back", res_refuse.stderr)
        self.assertFalse(any(name.startswith(".stage_ckpt_") for name in os.listdir(newer)))
        ptr, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr["completed_step"], 249)
        self.assertEqual(ptr["generation"], rec_new["generation"])

        # go.sh refuses rolling local newer CKPT_DIR back to older NAS checkpoint without ALLOW_ROLLBACK=1
        work = os.path.join(self.root, "go_rollback_work")
        data_zst = os.path.join(work, "data", "Python")
        os.makedirs(data_zst, exist_ok=True)
        with open(os.path.join(data_zst, "shard_00.zst"), "wb") as fh:
            fh.write(b"dummy")
        sentinel = os.path.join(self.root, "go_rollback_launched")
        fake_train = os.path.join(self.root, "fake_train_rb.py")
        with open(fake_train, "w", encoding="utf-8") as fh:
            fh.write(f"open({sentinel!r}, 'w').write('ok')\n")

        go_env = self._remote_env(
            BURST_MODE="nas",
            SKIP_DEPS="1",
            SKIP_TAILSCALE="1",
            WORK=work,
            DATA_ROOT=os.path.join(work, "data"),
            CKPT_DIR=newer,
            NAS_HOST="fixture-nas",
            NAS_USER="burst",
            NAS_DATA="/remote/data",
            NAS_CKPT=older,
            TRAIN_SCRIPT=fake_train,
            EXPECTED_STEP="200",
        )
        res_go_refuse = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=go_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res_go_refuse.returncode, 0)
        self.assertFalse(os.path.exists(sentinel), "go.sh MUST NOT launch after rollback refusal")
        ptr_after_go, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr_after_go["completed_step"], 249)

        # Remote to-nas succeeds when --allow-rollback is explicitly supplied
        res_ok = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas", "--allow-rollback"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_ok.returncode, 0, msg=res_ok.stderr)
        ptr_ok, _, _ = checkpoint_io.verify_pointer(newer)
        self.assertEqual(ptr_ok["completed_step"], 200)
        self.assertEqual(ptr_ok["generation"], rec_old["generation"])

    def test_documented_full_nas_burst_roundtrip_with_identity_pins(self):
        """BURST-05: Primary -> NAS (to-nas) -> Rental (go.sh) -> NAS (push-nas) -> Primary (from-nas)."""
        primary_ckpt = os.path.join(self.root, "primary_ckpt")
        nas_ckpt = os.path.join(self.root, "nas_ckpt")
        rental_work = os.path.join(self.root, "rental_work")
        rental_ckpt = os.path.join(rental_work, "ckpt")
        rental_data = os.path.join(rental_work, "data")
        os.makedirs(os.path.join(rental_data, "Python"), exist_ok=True)
        with open(os.path.join(rental_data, "Python", "shard_00.zst"), "wb") as fh:
            fh.write(b"dummy-zst")

        # Step 0: Primary publishes step 1000
        rec_1000 = checkpoint_io.publish_checkpoint(
            primary_ckpt,
            _make_full_payload(1000, weight=1000, opt_val=1000, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="prim1000",
        )

        # Leg 1: Primary -> NAS with full identity pins (to-nas)
        res_leg1 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "to-nas"],
            env=self._remote_env(
                PRIMARY_HOST="fixture-primary",
                PRIMARY_CKPT_DIR=primary_ckpt,
                NAS_HOST="fixture-nas",
                NAS_CKPT_DIR=nas_ckpt,
                EXPECTED_STEP="1000",
                EXPECTED_GENERATION=rec_1000["generation"],
                EXPECTED_SHA256=rec_1000["sha256"],
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_leg1.returncode, 0, msg=res_leg1.stderr)
        nas_ptr1, _, _ = checkpoint_io.verify_pointer(nas_ckpt)
        self.assertEqual(nas_ptr1["generation"], rec_1000["generation"])

        # Leg 2: Rented GPU pulls step 1000 via go.sh and advances to final step 1249
        rental_out_json = os.path.join(self.root, "rental_final.json")
        fake_rental_train = os.path.join(self.root, "fake_rental_train.py")
        with open(fake_rental_train, "w", encoding="utf-8") as fh:
            fh.write(
                f"import json, os, pickle, sys\n"
                f"sys.path.insert(0, {REPO_ROOT!r})\n"
                f"import checkpoint_io\n"
                f"ckpt_dir = os.environ['CKPT_DIR']\n"
                f"def pload(p, map_location='cpu'):\n"
                f"    with open(p, 'rb') as f: return pickle.load(f)\n"
                f"def psave(o, fh): pickle.dump(o, fh, protocol=pickle.HIGHEST_PROTOCOL)\n"
                f"_, sd, next_step = checkpoint_io.load_latest_checkpoint(ckpt_dir, load_fn=pload, require_full_state=True)\n"
                f"assert next_step == 1001, next_step\n"
                f"payload = checkpoint_io.build_checkpoint_payload(\n"
                f"    {{'w': 1249}}, {{'m': 1249}}, 1249, sd['cfg'],\n"
                f"    rng_state=sd['rng_state'], data_cursor=sd['data_cursor'],\n"
                f"    identity=sd['identity'], is_final=True,\n"
                f")\n"
                f"rec = checkpoint_io.publish_checkpoint(ckpt_dir, payload, save_fn=psave, is_final=True, generation_id='rent1249')\n"
                f"with open({rental_out_json!r}, 'w') as out:\n"
                f"    json.dump({{'generation': rec['generation'], 'sha256': rec['sha256']}}, out)\n"
            )

        res_leg2 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "go.sh")],
            env=self._remote_env(
                BURST_MODE="nas",
                SKIP_DEPS="1",
                SKIP_TAILSCALE="1",
                WORK=rental_work,
                DATA_ROOT=rental_data,
                CKPT_DIR=rental_ckpt,
                NAS_HOST="fixture-nas",
                NAS_USER="burst",
                NAS_DATA="/remote/data",
                NAS_CKPT=nas_ckpt,
                EXPECTED_STEP="1000",
                EXPECTED_GENERATION=rec_1000["generation"],
                EXPECTED_SHA256=rec_1000["sha256"],
                TRAIN_SCRIPT=fake_rental_train,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_leg2.returncode, 0, msg=res_leg2.stderr)
        with open(rental_out_json, "r", encoding="utf-8") as fh:
            rec_1249 = json.load(fh)

        # Leg 3 (negative pin check): push-nas with mismatched SHA-256 fails closed
        res_bad_pin = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "push-nas"],
            env=self._remote_env(
                CKPT_DIR=rental_ckpt,
                NAS_HOST="fixture-nas",
                NAS_CKPT_DIR=nas_ckpt,
                EXPECTED_STEP="1249",
                EXPECTED_GENERATION=rec_1249["generation"],
                EXPECTED_SHA256="0" * 64,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res_bad_pin.returncode, 0)
        nas_ptr_still_1000, _, _ = checkpoint_io.verify_pointer(nas_ckpt)
        self.assertEqual(nas_ptr_still_1000["completed_step"], 1000)

        # Leg 3: Rented GPU -> NAS (push-nas) with exact identity pins
        res_leg3 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "push-nas"],
            env=self._remote_env(
                CKPT_DIR=rental_ckpt,
                NAS_HOST="fixture-nas",
                NAS_CKPT_DIR=nas_ckpt,
                EXPECTED_STEP="1249",
                EXPECTED_GENERATION=rec_1249["generation"],
                EXPECTED_SHA256=rec_1249["sha256"],
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_leg3.returncode, 0, msg=res_leg3.stderr)
        nas_ptr2, _, _ = checkpoint_io.verify_pointer(nas_ckpt)
        self.assertEqual(nas_ptr2["generation"], rec_1249["generation"])
        self.assertEqual(nas_ptr2["completed_step"], 1249)
        self.assertEqual(nas_ptr2["sha256"], rec_1249["sha256"])

        # Leg 4: NAS -> Primary (from-nas) with exact identity pins
        res_leg4 = subprocess.run(
            ["bash", os.path.join(REPO_ROOT, "stage-ckpt.sh"), "from-nas"],
            env=self._remote_env(
                PRIMARY_HOST="fixture-primary",
                PRIMARY_CKPT_DIR=primary_ckpt,
                NAS_HOST="fixture-nas",
                NAS_CKPT_DIR=nas_ckpt,
                EXPECTED_STEP="1249",
                EXPECTED_GENERATION=rec_1249["generation"],
                EXPECTED_SHA256=rec_1249["sha256"],
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_leg4.returncode, 0, msg=res_leg4.stderr)

        _, sd_final, next_step = checkpoint_io.load_latest_checkpoint(
            primary_ckpt,
            load_fn=_pickle_load,
            require_full_state=True,
            expected_step=1249,
            expected_generation=rec_1249["generation"],
            expected_sha256=rec_1249["sha256"],
            expected_identity=self.identity,
        )
        self.assertEqual(next_step, 1250)
        self.assertEqual(sd_final["model"]["w"], 1249)
        self.assertTrue(sd_final["is_final"])

        # Verify README documents both return legs and identity pins
        with open(os.path.join(REPO_ROOT, "README.md"), "r", encoding="utf-8") as fh:
            readme = fh.read()
        self.assertIn("./stage-ckpt.sh to-nas", readme)
        self.assertIn("./stage-ckpt.sh push-nas", readme)
        self.assertIn("./stage-ckpt.sh from-nas", readme)
        self.assertIn("EXPECTED_GENERATION=", readme)
        self.assertIn("EXPECTED_SHA256=", readme)
        self.assertIn("--allow-rollback", readme)


def _wait_for_file(path: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for signal file: {path}")
        time.sleep(0.01)


class TestConcurrentPointerSerialization(unittest.TestCase):
    """BURST-06: Cross-process & multi-thread serialization of LATEST.json read-check-swap."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.data_root = os.path.join(self.root, "data")
        _make_sample_shards(self.data_root)
        self.identity = _make_identity(self.data_root)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_150_200_300(self, prefix: str) -> tuple[str, str, str, dict, dict, dict]:
        dest = os.path.join(self.root, f"{prefix}_dest")
        stage_200 = os.path.join(self.root, f"{prefix}_stage_200")
        stage_300 = os.path.join(self.root, f"{prefix}_stage_300")
        rec_150 = checkpoint_io.publish_checkpoint(
            dest,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g150",
        )
        rec_200 = checkpoint_io.publish_checkpoint(
            stage_200,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g200",
        )
        rec_300 = checkpoint_io.publish_checkpoint(
            stage_300,
            _make_full_payload(300, weight=300, opt_val=300, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g300",
        )
        return dest, stage_200, stage_300, rec_150, rec_200, rec_300

    def test_concurrent_processes_older_commit_paused_before_lock_refuses_rollback_after_newer_commits(self):
        """Exact BURST-06 counterexample across two OS processes: step 200 passes precheck on 150,
        step 300 commits first, step 200 resumes and MUST refuse rollback loudly."""
        dest, stage_200, stage_300, _r150, rec_200, rec_300 = self._seed_150_200_300("mp_before_lock")
        ready_file = os.path.join(self.root, "p1_ready")
        go_file = os.path.join(self.root, "p1_go")

        worker_code = f"""
import contextlib, os, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

real_lock = checkpoint_io._pointer_commit_lock

@contextlib.contextmanager
def paused_lock(ckpt_dir):
    with open({ready_file!r}, "w", encoding="utf-8") as fh:
        fh.write("ready\\n")
    deadline = time.monotonic() + 15.0
    while not os.path.exists({go_file!r}):
        if time.monotonic() >= deadline:
            raise TimeoutError("worker timed out waiting for go_file")
        time.sleep(0.01)
    with real_lock(ckpt_dir) as lp:
        yield lp

checkpoint_io._pointer_commit_lock = paused_lock
try:
    checkpoint_io.commit_staged_checkpoint({stage_200!r}, {dest!r})
except checkpoint_io.CheckpointIntegrityError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    raise SystemExit(1)
"""
        proc1 = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_file(ready_file)
            # Process 1 has already read step 150 in fast-fail precheck and copied stage_200.
            # Now Process 2 commits step 300 cleanly through the real CLI.
            proc2 = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "checkpoint_io.py"),
                    "commit-staged",
                    "--staging-dir",
                    stage_300,
                    "--ckpt-dir",
                    dest,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr)
            ptr_mid, _, _ = checkpoint_io.verify_pointer(dest)
            self.assertEqual(ptr_mid["completed_step"], 300)
            self.assertEqual(ptr_mid["generation"], rec_300["generation"])

            # Release Process 1 to acquire .latest.lock and attempt its commit.
            with open(go_file, "w", encoding="utf-8") as fh:
                fh.write("go\n")
            _out1, err1 = proc1.communicate(timeout=15)
        finally:
            if proc1.poll() is None:
                proc1.kill()
                proc1.communicate()

        self.assertNotEqual(proc1.returncode, 0, "Older step-200 process MUST fail loudly after step-300 commits")
        self.assertIn("roll back", err1)
        self.assertIn("completed_step=300", err1)
        self.assertIn("completed_step=200", err1)

        # Destination LATEST.json MUST remain verified at step 300 with no leftover import temps.
        ptr_final, _, _ = checkpoint_io.verify_pointer(dest)
        self.assertEqual(ptr_final["completed_step"], 300)
        self.assertEqual(ptr_final["generation"], rec_300["generation"])
        self.assertFalse(os.path.exists(os.path.join(dest, rec_200["generation"])))
        self.assertFalse(any(name.startswith(".import_") for name in os.listdir(dest)))

    def test_concurrent_processes_newer_commit_waits_for_older_lock_holder_and_ends_at_300(self):
        """Second BURST-06 interleaving across two OS processes: step 200 acquires .latest.lock first
        and pauses right before os.replace; step 300 MUST block on flock(LOCK_EX) until step 200
        releases, then advance cleanly 150 -> 200 -> 300."""
        dest, stage_200, stage_300, rec_150, rec_200, rec_300 = self._seed_150_200_300("mp_in_lock")
        in_lock_file = os.path.join(self.root, "p1_in_lock")
        release_file = os.path.join(self.root, "p1_release")

        worker_code = f"""
import os, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

real_write = checkpoint_io._write_json_atomic

def paused_write(path, data, prefix):
    if path == checkpoint_io.pointer_path_for({dest!r}) and data["completed_step"] == 200:
        with open({in_lock_file!r}, "w", encoding="utf-8") as fh:
            fh.write("locked\\n")
        deadline = time.monotonic() + 15.0
        while not os.path.exists({release_file!r}):
            if time.monotonic() >= deadline:
                raise TimeoutError("worker timed out waiting for release_file")
            time.sleep(0.01)
    return real_write(path, data, prefix)

checkpoint_io._write_json_atomic = paused_write
checkpoint_io.commit_staged_checkpoint({stage_200!r}, {dest!r})
"""
        proc1 = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc2 = None
        try:
            _wait_for_file(in_lock_file)
            # Launch Process 2 while Process 1 holds fcntl.flock(LOCK_EX) on dest/.latest.lock.
            proc2 = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "checkpoint_io.py"),
                    "commit-staged",
                    "--staging-dir",
                    stage_300,
                    "--ckpt-dir",
                    dest,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Process 2 must remain blocked waiting on the exclusive cross-process lock.
            time.sleep(0.25)
            self.assertIsNone(
                proc2.poll(),
                "Process 2 must block on dest/.latest.lock while Process 1 holds the critical section",
            )

            # Release Process 1 so 150 -> 200 finishes, unlocking Process 2 for 200 -> 300.
            with open(release_file, "w", encoding="utf-8") as fh:
                fh.write("release\n")
            _out1, err1 = proc1.communicate(timeout=15)
            _out2, err2 = proc2.communicate(timeout=15)
        finally:
            for p in (proc1, proc2):
                if p is not None and p.poll() is None:
                    p.kill()
                    p.communicate()

        self.assertEqual(proc1.returncode, 0, msg=err1)
        self.assertEqual(proc2.returncode, 0, msg=err2)
        ptr_final, _, _ = checkpoint_io.verify_pointer(dest)
        self.assertEqual(ptr_final["completed_step"], 300)
        self.assertEqual(ptr_final["generation"], rec_300["generation"])
        self.assertEqual(
            ptr_final["history"],
            [rec_150["generation"], rec_200["generation"], rec_300["generation"]],
        )

    def test_concurrent_processes_publish_and_commit_staged_share_pointer_lock(self):
        """publish_checkpoint (step 200) and commit_staged_checkpoint (step 300) share .latest.lock."""
        dest, _stage_200, stage_300, _r150, _r200, rec_300 = self._seed_150_200_300("mp_pub_commit")
        ready_file = os.path.join(self.root, "pub_ready")
        go_file = os.path.join(self.root, "pub_go")

        worker_code = f"""
import contextlib, os, pickle, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

real_lock = checkpoint_io._pointer_commit_lock

@contextlib.contextmanager
def paused_lock(ckpt_dir):
    with open({ready_file!r}, "w", encoding="utf-8") as fh:
        fh.write("ready\\n")
    deadline = time.monotonic() + 15.0
    while not os.path.exists({go_file!r}):
        if time.monotonic() >= deadline:
            raise TimeoutError("publisher timed out waiting for go_file")
        time.sleep(0.01)
    with real_lock(ckpt_dir) as lp:
        yield lp

checkpoint_io._pointer_commit_lock = paused_lock
payload = checkpoint_io.build_checkpoint_payload({{"w": 200}}, {{"m": 200}}, 200, {{"hidden_size": 256}})
try:
    checkpoint_io.publish_checkpoint(
        {dest!r},
        payload,
        save_fn=lambda obj, fh: pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL),
        generation_id="pub200",
    )
except checkpoint_io.CheckpointIntegrityError as exc:
    print(f"ERROR: {{exc}}", file=sys.stderr)
    raise SystemExit(1)
"""
        proc_pub = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_file(ready_file)
            proc_commit = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "checkpoint_io.py"),
                    "commit-staged",
                    "--staging-dir",
                    stage_300,
                    "--ckpt-dir",
                    dest,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc_commit.returncode, 0, msg=proc_commit.stderr)

            with open(go_file, "w", encoding="utf-8") as fh:
                fh.write("go\n")
            _out, err = proc_pub.communicate(timeout=15)
        finally:
            if proc_pub.poll() is None:
                proc_pub.kill()
                proc_pub.communicate()

        self.assertNotEqual(proc_pub.returncode, 0, "Stale publish_checkpoint(200) MUST refuse rollback over 300")
        self.assertIn("roll back", err)
        ptr_final, _, _ = checkpoint_io.verify_pointer(dest)
        self.assertEqual(ptr_final["completed_step"], 300)
        self.assertEqual(ptr_final["generation"], rec_300["generation"])

    def test_concurrent_threads_racing_step_300_and_200_always_end_at_300(self):
        """Two concurrent threads racing step 200 and step 300 over step 150 NEVER end at step 200."""
        for trial in range(3):
            dest, stage_200, stage_300, rec_150, rec_200, rec_300 = self._seed_150_200_300(f"thr_{trial}")
            barrier = threading.Barrier(2)
            real_lock = checkpoint_io._pointer_commit_lock
            outcomes: dict[int, tuple[str, object]] = {}

            def synced_lock(ckpt_dir):
                barrier.wait(timeout=10)
                return real_lock(ckpt_dir)

            def run_commit(step: int, stage_dir: str):
                try:
                    res = checkpoint_io.commit_staged_checkpoint(stage_dir, dest)
                    outcomes[step] = ("ok", res)
                except Exception as exc:
                    outcomes[step] = ("err", exc)

            with mock.patch.object(checkpoint_io, "_pointer_commit_lock", side_effect=synced_lock):
                t200 = threading.Thread(target=run_commit, args=(200, stage_200))
                t300 = threading.Thread(target=run_commit, args=(300, stage_300))
                t200.start()
                t300.start()
                t200.join(timeout=15)
                t300.join(timeout=15)

            self.assertEqual(outcomes[300][0], "ok", msg=f"trial {trial}: step 300 failed: {outcomes[300]}")
            ptr_final, _, _ = checkpoint_io.verify_pointer(dest)
            self.assertEqual(ptr_final["completed_step"], 300)
            self.assertEqual(ptr_final["generation"], rec_300["generation"])
            if outcomes[200][0] == "err":
                self.assertIsInstance(outcomes[200][1], checkpoint_io.CheckpointIntegrityError)
                self.assertIn("roll back", str(outcomes[200][1]))
            else:
                # If 200 won the lock first, 300 serialized after it and preserved full history.
                self.assertEqual(
                    ptr_final["history"],
                    [rec_150["generation"], rec_200["generation"], rec_300["generation"]],
                )

    def test_reentrant_same_thread_interleave_fails_closed_without_regression(self):
        """Re-entering commit_staged_checkpoint on the same thread inside _write_json_atomic
        fails closed instead of deadlocking or regressing LATEST.json."""
        dest, stage_200, stage_300, rec_150, _r200, _r300 = self._seed_150_200_300("reentrant")
        real_write = checkpoint_io._write_json_atomic

        def interleave(path, data, prefix):
            if path == checkpoint_io.pointer_path_for(dest) and data["completed_step"] == 200:
                checkpoint_io.commit_staged_checkpoint(stage_300, dest)
            return real_write(path, data, prefix)

        with mock.patch.object(checkpoint_io, "_write_json_atomic", side_effect=interleave):
            with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Re-entrant"):
                checkpoint_io.commit_staged_checkpoint(stage_200, dest)

        ptr, _, _ = checkpoint_io.verify_pointer(dest)
        self.assertEqual(ptr["completed_step"], 150)
        self.assertEqual(ptr["generation"], rec_150["generation"])


class TestConcurrentRetentionProtection(unittest.TestCase):
    """BURST-07: Retention pruning coordinates with pointer publication and protects in-flight generations."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.data_root = os.path.join(self.root, "data")
        _make_sample_shards(self.data_root)
        self.identity = _make_identity(self.data_root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_concurrent_retention_keep1_waits_for_inflight_publish_before_latest_swap(self):
        """Exact BURST-07 counterexample across two OS processes:
        publish step 200 from step 150, pause right before LATEST.json swap while holding real
        .latest.lock, run keep=1 retention concurrently -> retention waits on lock, re-reads
        LATEST.json inside the lock, preserves promoted step 200, and verify_pointer passes."""
        ckpt_dir = os.path.join(self.root, "burst07_publish")
        rec_150 = checkpoint_io.publish_checkpoint(
            ckpt_dir,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g150",
        )
        in_lock_file = os.path.join(self.root, "pub200_in_lock")
        release_file = os.path.join(self.root, "pub200_release")

        pub_worker = f"""
import os, pickle, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

real_write = checkpoint_io._write_json_atomic

def paused_write(path, data, prefix):
    if path == checkpoint_io.pointer_path_for({ckpt_dir!r}) and data["completed_step"] == 200:
        with open({in_lock_file!r}, "w", encoding="utf-8") as fh:
            fh.write("locked\\n")
        deadline = time.monotonic() + 15.0
        while not os.path.exists({release_file!r}):
            if time.monotonic() >= deadline:
                raise TimeoutError("publisher timed out waiting for release_file")
            time.sleep(0.01)
    return real_write(path, data, prefix)

checkpoint_io._write_json_atomic = paused_write
payload = checkpoint_io.build_checkpoint_payload(
    {{"w": 200}}, {{"m": 200}}, 200, {{"hidden_size": 256, "rope_theta": 500000.0}},
    rng_state=checkpoint_io.capture_rng_state(include_torch=False, base_seed=11),
    data_cursor={{
        "version": checkpoint_io.DATA_CURSOR_VERSION,
        "seed": 11, "epoch": 0, "shard_index": 0, "line_index": 201,
        "token_buffer": [12, 13, 14], "blocks_emitted": 804, "samples_consumed": 804,
    }},
)
checkpoint_io.publish_checkpoint(
    {ckpt_dir!r},
    payload,
    save_fn=lambda obj, fh: pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL),
    generation_id="g200",
)
"""
        prune_worker = f"""
import json, sys
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

removed = checkpoint_io.prune_old_checkpoints({ckpt_dir!r}, keep=1)
print(json.dumps(removed))
"""
        proc_pub = subprocess.Popen(
            [sys.executable, "-c", pub_worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc_prune = None
        try:
            _wait_for_file(in_lock_file)
            gen_200_dir = os.path.join(ckpt_dir, "gen_00000200_g200")
            self.assertTrue(
                os.path.isdir(gen_200_dir),
                "Step-200 generation must already be promoted before paused LATEST.json swap",
            )

            proc_prune = subprocess.Popen(
                [sys.executable, "-c", prune_worker],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Retention MUST block on .latest.lock while publisher holds the promotion critical section.
            time.sleep(0.25)
            self.assertIsNone(
                proc_prune.poll(),
                "prune_old_checkpoints(keep=1) must wait on .latest.lock while publication is in flight",
            )
            self.assertTrue(
                os.path.isdir(gen_200_dir),
                "Promoted step-200 directory MUST NOT be removed while publication is in flight",
            )

            with open(release_file, "w", encoding="utf-8") as fh:
                fh.write("release\n")
            _out_pub, err_pub = proc_pub.communicate(timeout=15)
            out_prune, err_prune = proc_prune.communicate(timeout=15)
        finally:
            for p in (proc_pub, proc_prune):
                if p is not None and p.poll() is None:
                    p.kill()
                    p.communicate()

        self.assertEqual(proc_pub.returncode, 0, msg=err_pub)
        self.assertEqual(proc_prune.returncode, 0, msg=err_prune)
        removed = json.loads(out_prune)
        self.assertEqual(removed, [rec_150["generation_dir"]])

        ptr, manifest, ckpt_path = checkpoint_io.verify_pointer(ckpt_dir)
        self.assertEqual(ptr["completed_step"], 200)
        self.assertEqual(ptr["generation"], "gen_00000200_g200")
        self.assertEqual(manifest["generation"], "gen_00000200_g200")
        self.assertTrue(os.path.isfile(ckpt_path))
        self.assertEqual(
            sorted(os.listdir(ckpt_dir)),
            [".latest.lock", "LATEST.json", "gen_00000200_g200"],
        )
        self.assertEqual(
            sorted(os.listdir(os.path.join(ckpt_dir, "gen_00000200_g200"))),
            ["checkpoint.pt", "manifest.json"],
        )
        _loaded_path, sd, next_step = checkpoint_io.load_latest_checkpoint(
            ckpt_dir, load_fn=_pickle_load
        )
        self.assertEqual(sd["completed_step"], 200)
        self.assertEqual(next_step, 201)

    def test_concurrent_retention_preserves_live_inflight_staging_and_cleans_stale_staging(self):
        """Retention running while a publisher is still serializing into .staging_* MUST skip the
        live leased staging directory while still cleaning up genuinely stale staging leftovers."""
        ckpt_dir = os.path.join(self.root, "burst07_staging")
        checkpoint_io.publish_checkpoint(
            ckpt_dir,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g150",
        )
        # Seed a stale crashed staging directory containing an unlocked .inflight.lock.
        stale_stage = os.path.join(ckpt_dir, ".staging_gen_00000175_crashed")
        os.makedirs(stale_stage, exist_ok=True)
        with open(os.path.join(stale_stage, checkpoint_io.INFLIGHT_LOCK_FILENAME), "w", encoding="utf-8") as fh:
            json.dump({"pid": 999999, "generation": "gen_00000175_crashed"}, fh)

        staging_ready = os.path.join(self.root, "staging_ready")
        staging_go = os.path.join(self.root, "staging_go")

        pub_worker = f"""
import os, pickle, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

def paused_save(obj, fh):
    pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    fh.flush()
    with open({staging_ready!r}, "w", encoding="utf-8") as sf:
        sf.write("staging\\n")
    deadline = time.monotonic() + 15.0
    while not os.path.exists({staging_go!r}):
        if time.monotonic() >= deadline:
            raise TimeoutError("publisher timed out waiting for staging_go")
        time.sleep(0.01)

payload = checkpoint_io.build_checkpoint_payload({{"w": 200}}, {{"m": 200}}, 200, {{"hidden_size": 256}})
checkpoint_io.publish_checkpoint({ckpt_dir!r}, payload, save_fn=paused_save, generation_id="g200")
"""
        proc_pub = subprocess.Popen(
            [sys.executable, "-c", pub_worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_file(staging_ready)
            # Run keep=1 retention while Process 1 is paused mid-staging (before acquiring .latest.lock).
            checkpoint_io.prune_old_checkpoints(ckpt_dir, keep=1)

            self.assertFalse(os.path.exists(stale_stage), "Stale unlocked staging dir must be pruned")
            live_stages = [
                name for name in os.listdir(ckpt_dir) if name.startswith(".staging_gen_00000200_g200_")
            ]
            self.assertEqual(len(live_stages), 1, "Live leased staging dir MUST survive concurrent retention")

            with open(staging_go, "w", encoding="utf-8") as fh:
                fh.write("go\n")
            _out, err = proc_pub.communicate(timeout=15)
        finally:
            if proc_pub.poll() is None:
                proc_pub.kill()
                proc_pub.communicate()

        self.assertEqual(proc_pub.returncode, 0, msg=err)
        ptr, _, _ = checkpoint_io.verify_pointer(ckpt_dir)
        self.assertEqual(ptr["completed_step"], 200)
        self.assertEqual(ptr["generation"], "gen_00000200_g200")

    def test_concurrent_retention_keep1_waits_for_commit_staged_and_preserves_promoted_generation(self):
        """commit_staged_checkpoint and prune_old_checkpoints(keep=1) coordinate on .latest.lock."""
        dest = os.path.join(self.root, "burst07_commit_dest")
        stage_200 = os.path.join(self.root, "burst07_commit_stage200")
        rec_150 = checkpoint_io.publish_checkpoint(
            dest,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g150",
        )
        checkpoint_io.publish_checkpoint(
            stage_200,
            _make_full_payload(200, weight=200, opt_val=200, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g200",
        )
        in_lock_file = os.path.join(self.root, "commit200_in_lock")
        release_file = os.path.join(self.root, "commit200_release")

        commit_worker = f"""
import os, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

real_write = checkpoint_io._write_json_atomic

def paused_write(path, data, prefix):
    if path == checkpoint_io.pointer_path_for({dest!r}) and data["completed_step"] == 200:
        with open({in_lock_file!r}, "w", encoding="utf-8") as fh:
            fh.write("locked\\n")
        deadline = time.monotonic() + 15.0
        while not os.path.exists({release_file!r}):
            if time.monotonic() >= deadline:
                raise TimeoutError("commit worker timed out")
            time.sleep(0.01)
    return real_write(path, data, prefix)

checkpoint_io._write_json_atomic = paused_write
checkpoint_io.commit_staged_checkpoint({stage_200!r}, {dest!r})
"""
        prune_worker = f"""
import json, sys
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

print(json.dumps(checkpoint_io.prune_old_checkpoints({dest!r}, keep=1)))
"""
        proc_commit = subprocess.Popen(
            [sys.executable, "-c", commit_worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc_prune = None
        try:
            _wait_for_file(in_lock_file)
            proc_prune = subprocess.Popen(
                [sys.executable, "-c", prune_worker],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.25)
            self.assertIsNone(proc_prune.poll(), "Retention must block until commit_staged releases .latest.lock")
            with open(release_file, "w", encoding="utf-8") as fh:
                fh.write("release\n")
            _out_c, err_c = proc_commit.communicate(timeout=15)
            out_p, err_p = proc_prune.communicate(timeout=15)
        finally:
            for p in (proc_commit, proc_prune):
                if p is not None and p.poll() is None:
                    p.kill()
                    p.communicate()

        self.assertEqual(proc_commit.returncode, 0, msg=err_c)
        self.assertEqual(proc_prune.returncode, 0, msg=err_p)
        self.assertEqual(json.loads(out_p), [rec_150["generation_dir"]])
        ptr, _, _ = checkpoint_io.verify_pointer(dest)
        self.assertEqual(ptr["completed_step"], 200)
        self.assertEqual(ptr["generation"], "gen_00000200_g200")

    def test_active_inflight_lease_protects_generation_and_stale_lease_is_prunable(self):
        """Even independently of .latest.lock, a live owner lease keeps a generation out of the
        prune candidate set until the lease ends or becomes demonstrably stale; corrupt LATEST.json
        fails closed without pruning."""
        ckpt_dir = os.path.join(self.root, "burst07_lease")
        rec_150 = checkpoint_io.publish_checkpoint(
            ckpt_dir,
            _make_full_payload(150, weight=150, opt_val=150, identity=self.identity),
            save_fn=_pickle_save,
            generation_id="g150",
        )
        live_gen = os.path.join(ckpt_dir, "gen_00000200_leased")
        stale_gen = os.path.join(ckpt_dir, "gen_00000175_stale")
        os.makedirs(live_gen, exist_ok=True)
        os.makedirs(stale_gen, exist_ok=True)
        with open(os.path.join(stale_gen, checkpoint_io.INFLIGHT_LOCK_FILENAME), "w", encoding="utf-8") as fh:
            json.dump({"pid": 999999, "generation": "gen_00000175_stale"}, fh)

        lease_ready = os.path.join(self.root, "lease_ready")
        lease_release = os.path.join(self.root, "lease_release")
        lease_holder = f"""
import os, sys, time
sys.path.insert(0, {REPO_ROOT!r})
import checkpoint_io

lease = checkpoint_io._acquire_inflight_lease({live_gen!r}, generation="gen_00000200_leased")
with open({lease_ready!r}, "w", encoding="utf-8") as fh:
    fh.write("held\\n")
deadline = time.monotonic() + 15.0
while not os.path.exists({lease_release!r}):
    if time.monotonic() >= deadline:
        break
    time.sleep(0.01)
lease.release(unlink=False)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", lease_holder],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_file(lease_ready)
            removed1 = checkpoint_io.prune_old_checkpoints(ckpt_dir, keep=1)
            self.assertEqual(removed1, [stale_gen])
            self.assertTrue(os.path.isdir(rec_150["generation_dir"]))
            self.assertTrue(os.path.isdir(live_gen), "Live-leased generation must not be pruned")

            with open(lease_release, "w", encoding="utf-8") as fh:
                fh.write("done\n")
            _out, err = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, msg=err)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()

        # Once the owner releases the lease (or dies), the uncommitted generation is stale and prunable.
        removed2 = checkpoint_io.prune_old_checkpoints(ckpt_dir, keep=1)
        self.assertEqual(removed2, [live_gen])
        ptr, _, _ = checkpoint_io.verify_pointer(ckpt_dir)
        self.assertEqual(ptr["generation"], rec_150["generation"])

        # Corrupt LATEST.json must fail closed rather than pruning blindly.
        with open(checkpoint_io.pointer_path_for(ckpt_dir), "w", encoding="utf-8") as fh:
            fh.write("{corrupt")
        with self.assertRaisesRegex(checkpoint_io.CheckpointIntegrityError, "Unreadable checkpoint pointer"):
            checkpoint_io.prune_old_checkpoints(ckpt_dir, keep=1)
        self.assertTrue(os.path.isdir(rec_150["generation_dir"]))


if __name__ == "__main__":
    unittest.main()
