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
        self.assertEqual(sorted(os.listdir(self.ckpt_dir)), ["LATEST.json", "gen_00000200_gen200"])

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

        self.assertEqual(sorted(os.listdir(self.ckpt_dir)), ["LATEST.json", "gen_00000100_g100"])
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
            ["LATEST.json", "gen_00000300_g300", "gen_00000400_g400", "gen_00000500_g500"],
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


if __name__ == "__main__":
    unittest.main()
